use std::{
    collections::BTreeMap,
    io::{self, Read, Write},
    path::Path,
};

use anyhow::{Context, Result};
use synapse_core::{
    worker_framing_sync::{read_frame, read_json_frame, write_frame, write_json_frame},
    EngineIdentity, WorkerHello, WorkerHelloAck, WorkerRequest, WorkerResponse,
    DEFAULT_MAX_FRAME_BYTES, WORKER_PROTOCOL_VERSION,
};
use synapse_parity::{
    manifest::{Lane, Model, Operation, Profile},
    safetensors::{parse_header_json, Header},
};

use crate::{
    admission::{requirements, select, Adapter},
    KERNEL_REVISION, MANIFEST_DIGEST,
};

pub fn hello(nonce: &str) -> WorkerHello {
    WorkerHello {
        v: WORKER_PROTOCOL_VERSION,
        nonce: nonce.into(),
        pid: std::process::id(),
        max_frame: DEFAULT_MAX_FRAME_BYTES,
        engine: EngineIdentity {
            engine: synapse_core::worker_engine_names::VULKAN_WORKER_ENGINE.into(),
            version: env!("CARGO_PKG_VERSION").into(),
            build_flags: BTreeMap::from([
                ("vulkan".into(), cfg!(feature = "vulkan").to_string()),
                ("kernel_revision".into(), KERNEL_REVISION.into()),
                ("dtype".into(), "f16".into()),
            ]),
        },
        manifest_digest: Some(MANIFEST_DIGEST.into()),
        kernel_revision: Some(KERNEL_REVISION.into()),
    }
}

fn err(req_id: Option<String>, code: &str, msg: impl Into<String>) -> WorkerResponse {
    WorkerResponse::Err {
        req_id,
        code: code.into(),
        msg: msg.into(),
    }
}
fn digest(value: &str) -> &str {
    value.strip_prefix("sha256:").unwrap_or(value)
}

pub fn preflight(
    config: &BTreeMap<String, String>,
    artifact_digest: &str,
    lookup: impl FnOnce() -> std::result::Result<Vec<Adapter>, String>,
    open_header: impl FnOnce() -> std::result::Result<Header, String>,
) -> std::result::Result<(Model, Profile, Adapter, Header), String> {
    let manifest = crate::manifest();
    let profile = config
        .get("profile")
        .and_then(|id| manifest.profiles.get(id))
        .filter(|p| p.lane == Lane::OwnedVulkan)
        .ok_or("model_unsupported")?
        .clone();
    let model = manifest
        .model(&profile.model)
        .map_err(|_| "model_unsupported")?
        .clone();
    let operation = match model.operation {
        Operation::Embed => "embed",
        Operation::Rerank => "rerank",
    };
    if config.get("operation").map(String::as_str) != Some(operation) {
        return Err("operation_mismatch".into());
    }
    if profile.converted_package_digest.as_deref().map(digest) != Some(digest(artifact_digest)) {
        return Err("package_digest_mismatch".into());
    }
    let required = requirements(&manifest, Some(&profile.model))?;
    let adapter = select(lookup()?, required)?;
    let header = open_header()?;
    if let Some(head) = &model.head {
        if head
            .required_tensor_keys
            .iter()
            .any(|key| !header.tensors.contains_key(key))
        {
            return Err("head_tensor_missing".into());
        }
        if head
            .forbidden_tensor_keys
            .iter()
            .any(|key| header.tensors.contains_key(key))
        {
            return Err("package_digest_mismatch".into());
        }
    }
    Ok((model, profile, adapter, header))
}

struct Loaded {
    model_ref: String,
    operation: Operation,
    dims: usize,
    #[cfg(feature = "vulkan")]
    engine: crate::runtime::Engine,
}

pub fn session<S: Read + Write>(
    stream: &mut S,
    nonce: &str,
    lookup: impl FnMut() -> std::result::Result<Vec<Adapter>, String>,
) -> Result<()> {
    write_json_frame(stream, &hello(nonce), DEFAULT_MAX_FRAME_BYTES)?;
    let ack: WorkerHelloAck = read_json_frame(stream, DEFAULT_MAX_FRAME_BYTES)?;
    anyhow::ensure!(
        ack.v == WORKER_PROTOCOL_VERSION && ack.accept,
        "worker HELLO rejected"
    );
    anyhow::ensure!(
        ack.max_frame <= DEFAULT_MAX_FRAME_BYTES && ack.max_frame > 0,
        "invalid frame limit"
    );
    request_loop(stream, ack.max_frame, lookup)
}

pub fn request_loop<S: Read + Write>(
    stream: &mut S,
    max_frame: u32,
    mut lookup: impl FnMut() -> std::result::Result<Vec<Adapter>, String>,
) -> Result<()> {
    let mut loaded: Option<Loaded> = None;
    loop {
        let frame = match read_frame(stream, max_frame) {
            Ok(frame) => frame,
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.into()),
        };
        let request: WorkerRequest =
            serde_json::from_slice(&frame).context("decode worker request")?;
        let raw = if request.carries_raw_frame() {
            Some(read_frame(stream, max_frame)?)
        } else {
            None
        };
        let response = match &request {
            WorkerRequest::Shutdown {} => break,
            WorkerRequest::Ping { req_id } => WorkerResponse::Pong {
                req_id: req_id.clone(),
                rss_mb: 0,
                models_loaded: usize::from(loaded.is_some()),
                placement_share: None,
                buckets: None,
                ane_resident_shapes: None,
            },
            WorkerRequest::Load {
                req_id,
                artifact_path,
                artifact_digest,
                runtime_config,
                ..
            } => {
                // A worker retains one model arena. Release it before constructing a
                // replacement so repeated LOAD cannot double device-local residency.
                loaded = None;
                match load(artifact_path, artifact_digest, runtime_config, &mut lookup) {
                    Ok(model) => {
                        let response = WorkerResponse::Loaded {
                            req_id: req_id.clone(),
                            model_ref: model.model_ref.clone(),
                            dims: model.dims,
                            cold_load_ms: 0,
                            buckets: None,
                        };
                        loaded = Some(model);
                        response
                    }
                    Err(code) => err(Some(req_id.clone()), &code, &code),
                }
            }
            WorkerRequest::Unload { req_id, model_ref } => {
                if loaded.as_ref().is_some_and(|m| &m.model_ref == model_ref) {
                    loaded = None;
                    WorkerResponse::Unloaded {
                        req_id: req_id.clone(),
                    }
                } else {
                    err(
                        Some(req_id.clone()),
                        "model_not_loaded",
                        "model reference is not loaded",
                    )
                }
            }
            WorkerRequest::EmbedBatch {
                req_id,
                model_ref,
                items,
                ..
            } => {
                match infer(
                    loaded.as_ref(),
                    model_ref,
                    Operation::Embed,
                    &items.iter().map(|i| i.n_tokens).collect::<Vec<_>>(),
                    raw.as_deref().unwrap(),
                ) {
                    Ok(values) => {
                        let model = loaded.as_ref().unwrap();
                        write_json_frame(
                            stream,
                            &WorkerResponse::Vectors {
                                req_id: req_id.clone(),
                                dims: model.dims,
                                n: items.len(),
                            },
                            max_frame,
                        )?;
                        write_frame(stream, &synapse_core::encode_f32_frame(&values), max_frame)?;
                        continue;
                    }
                    Err(code) => err(Some(req_id.clone()), &code, &code),
                }
            }
            WorkerRequest::RerankSequences {
                req_id,
                model_ref,
                sequences,
            } => {
                match infer(
                    loaded.as_ref(),
                    model_ref,
                    Operation::Rerank,
                    &sequences.iter().map(|s| s.n_tokens).collect::<Vec<_>>(),
                    raw.as_deref().unwrap(),
                ) {
                    Ok(values) => {
                        write_json_frame(
                            stream,
                            &WorkerResponse::Scores {
                                req_id: req_id.clone(),
                            },
                            max_frame,
                        )?;
                        write_frame(stream, &synapse_core::encode_f32_frame(&values), max_frame)?;
                        continue;
                    }
                    Err(code) => err(Some(req_id.clone()), &code, &code),
                }
            }
            _ => WorkerResponse::unsupported_request(&request),
        };
        write_json_frame(stream, &response, max_frame)?;
    }
    Ok(())
}

fn load(
    path: &str,
    artifact_digest: &str,
    config: &BTreeMap<String, String>,
    lookup: impl FnOnce() -> std::result::Result<Vec<Adapter>, String>,
) -> std::result::Result<Loaded, String> {
    let mut package_path = Path::new(path).to_path_buf();
    let preflight = preflight(config, artifact_digest, lookup, || {
        if package_path.is_dir() {
            package_path = package_path.join("model.safetensors");
        }
        let header = synapse_parity::safetensors::read_header_bytes(&package_path)
            .map_err(|_| "artifact_invalid".to_owned())?;
        parse_header_json(&header).map_err(|_| "artifact_invalid".into())
    })?;
    #[cfg(feature = "vulkan")]
    {
        use sha2::{Digest, Sha256};
        let (model, profile, adapter, header) = preflight;
        let bytes = std::fs::read(&package_path).map_err(|_| "artifact_invalid".to_owned())?;
        if format!("{:x}", Sha256::digest(&bytes)) != digest(artifact_digest) {
            return Err("package_digest_mismatch".into());
        }
        let profile_id = config.get("profile").unwrap();
        synapse_parity::safetensors::verify_package_structure(&bytes, profile_id)
            .map_err(|_| "package_digest_mismatch".to_owned())?;
        let expected = synapse_parity::arch::expected_tensors(&model)
            .map_err(|_| "model_unsupported".to_owned())?;
        if header.tensors.len() != expected.len()
            || expected.iter().any(|(name, shape)| {
                header.tensors.get(name).is_none_or(|t| {
                    &t.shape != shape || t.dtype != synapse_parity::safetensors::StDType::F16
                })
            })
        {
            return Err("package_digest_mismatch".into());
        }
        let (_, data) = synapse_parity::safetensors::split_file(&bytes)
            .map_err(|_| "artifact_invalid".to_owned())?;
        if header
            .tensors
            .values()
            .any(|t| t.data_offsets.1 > data.len())
        {
            return Err("artifact_invalid".into());
        }
        let dims = model.output.dimension as usize;
        let operation = model.operation;
        let required = requirements(&crate::manifest(), Some(&profile.model))?;
        let engine = crate::runtime::Engine::load(model, profile, adapter, required, &header, data)
            .map_err(|e| {
                if e.to_string().contains("vulkan_insufficient_memory") {
                    "vulkan_insufficient_memory".to_owned()
                } else if e.to_string().contains("vulkan_no_device") {
                    "vulkan_no_device".to_owned()
                } else {
                    "vulkan_load_failed".to_owned()
                }
            })?;
        Ok(Loaded {
            model_ref: digest(artifact_digest).into(),
            operation,
            dims,
            engine,
        })
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = preflight;
        Err("vulkan_no_device".into())
    }
}

fn infer(
    loaded: Option<&Loaded>,
    model_ref: &str,
    operation: Operation,
    lengths: &[usize],
    raw: &[u8],
) -> std::result::Result<Vec<f32>, String> {
    let model = loaded
        .filter(|m| m.model_ref == model_ref)
        .ok_or("model_not_loaded")?;
    if model.operation != operation {
        return Err("operation_mismatch".into());
    }
    if lengths.iter().any(|n| *n > 8192) {
        return Err("sequence_too_long".into());
    }
    if lengths.is_empty() || lengths.len() > 256 || lengths.contains(&0) {
        return Err("invalid_tokens".into());
    }
    let ids = synapse_core::decode_i32_frame(raw).map_err(|_| "invalid_tokens".to_owned())?;
    if lengths.iter().sum::<usize>() != ids.len() {
        return Err("invalid_tokens".into());
    }
    let mut cursor = 0;
    let sequences: Vec<_> = lengths
        .iter()
        .map(|n| {
            let sequence = ids[cursor..cursor + n].to_vec();
            cursor += n;
            sequence
        })
        .collect();
    #[cfg(feature = "vulkan")]
    {
        model.engine.infer(&sequences).map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = sequences;
        Err("vulkan_no_device".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, io::Cursor};
    struct Script {
        input: Cursor<Vec<u8>>,
        output: Vec<u8>,
    }
    impl Read for Script {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            self.input.read(b)
        }
    }
    impl Write for Script {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.output.extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn hello_ping_and_unsupported_rerank_survive_missing_loader() {
        let mut input = Vec::new();
        write_json_frame(
            &mut input,
            &WorkerHelloAck {
                v: WORKER_PROTOCOL_VERSION,
                accept: true,
                max_frame: DEFAULT_MAX_FRAME_BYTES,
            },
            DEFAULT_MAX_FRAME_BYTES,
        )
        .unwrap();
        write_json_frame(
            &mut input,
            &WorkerRequest::Rerank {
                req_id: "unsupported".into(),
                model_ref: "absent".into(),
                query_n_tokens: 1,
                candidates: vec![],
            },
            DEFAULT_MAX_FRAME_BYTES,
        )
        .unwrap();
        write_frame(&mut input, &[1, 0, 0, 0], DEFAULT_MAX_FRAME_BYTES).unwrap();
        write_json_frame(
            &mut input,
            &WorkerRequest::Ping {
                req_id: "ping".into(),
            },
            DEFAULT_MAX_FRAME_BYTES,
        )
        .unwrap();
        write_json_frame(
            &mut input,
            &WorkerRequest::Shutdown {},
            DEFAULT_MAX_FRAME_BYTES,
        )
        .unwrap();
        let calls = Cell::new(0);
        let mut script = Script {
            input: Cursor::new(input),
            output: Vec::new(),
        };
        session(&mut script, "nonce", || {
            calls.set(calls.get() + 1);
            Err("vulkan_no_device".into())
        })
        .unwrap();
        assert_eq!(calls.get(), 0, "HELLO/PING must not resolve the loader");
        let mut output = Cursor::new(script.output);
        let hello: WorkerHello = read_json_frame(&mut output, DEFAULT_MAX_FRAME_BYTES).unwrap();
        assert_eq!(hello.manifest_digest.as_deref(), Some(MANIFEST_DIGEST));
        let refusal: WorkerResponse =
            read_json_frame(&mut output, DEFAULT_MAX_FRAME_BYTES).unwrap();
        assert!(matches!(refusal,WorkerResponse::Err {code,..} if code=="unsupported_request"));
        let pong: WorkerResponse = read_json_frame(&mut output, DEFAULT_MAX_FRAME_BYTES).unwrap();
        assert!(matches!(pong,WorkerResponse::Pong {req_id,..} if req_id=="ping"));
    }
    fn eligible() -> Adapter {
        Adapter {
            index: 0,
            discrete: true,
            cpu: false,
            vendor: 0x1002,
            api_major: 1,
            api_minor: 2,
            shader_float16: true,
            storage_buffer16_bit_access: true,
            subgroup_arithmetic: true,
            subgroup_compute_stage: true,
            max_storage_buffer_range: u64::MAX,
            device_local_heaps: vec![1, u64::MAX],
            cooperative_matrix: false,
        }
    }
    #[test]
    fn every_floor_refusal_precedes_artifact_open() {
        let manifest = crate::manifest();
        let profile = &manifest.profiles["gte-modernbert-base.owned-vulkan"];
        let config = BTreeMap::from([
            ("profile".into(), "gte-modernbert-base.owned-vulkan".into()),
            ("operation".into(), "embed".into()),
        ]);
        let required = requirements(&manifest, Some(&profile.model)).unwrap();
        let mut devices = Vec::new();
        let mut d = eligible();
        d.cpu = true;
        devices.push((d, "vulkan_software_device"));
        let mut d = eligible();
        d.vendor = 0x8086;
        devices.push((d, "vulkan_unsupported_vendor"));
        let mut d = eligible();
        d.api_minor = 1;
        devices.push((d, "vulkan_api_too_old"));
        let mut d = eligible();
        d.shader_float16 = false;
        devices.push((d, "vulkan_missing_feature:shaderFloat16"));
        let mut d = eligible();
        d.storage_buffer16_bit_access = false;
        devices.push((d, "vulkan_missing_feature:storageBuffer16BitAccess"));
        let mut d = eligible();
        d.subgroup_arithmetic = false;
        devices.push((d, "vulkan_missing_feature:subgroupArithmetic"));
        let mut d = eligible();
        d.subgroup_compute_stage = false;
        devices.push((d, "vulkan_missing_feature:subgroupComputeStage"));
        let mut d = eligible();
        d.max_storage_buffer_range = required.min_storage_buffer_range - 1;
        devices.push((d, "vulkan_insufficient_memory"));
        let mut d = eligible();
        d.device_local_heaps = vec![
            required.min_device_local_bytes / 2,
            required.min_device_local_bytes - 1,
        ];
        devices.push((d, "vulkan_insufficient_memory"));
        for (device, code) in devices {
            let opened = Cell::new(false);
            let result = preflight(
                &config,
                profile.converted_package_digest.as_ref().unwrap(),
                || Ok(vec![device]),
                || {
                    opened.set(true);
                    Err("sentinel".into())
                },
            );
            assert_eq!(result.unwrap_err(), code);
            assert!(!opened.get());
        }
        assert_eq!(
            preflight(
                &config,
                profile.converted_package_digest.as_ref().unwrap(),
                || Err("vulkan_no_device".into()),
                || panic!("artifact opened")
            )
            .unwrap_err(),
            "vulkan_no_device"
        );
    }
    #[test]
    fn load_contract_refusals_precede_device_and_weight_reads() {
        let mut config = BTreeMap::from([
            ("profile".into(), "unknown".into()),
            ("operation".into(), "embed".into()),
        ]);
        assert_eq!(
            preflight(&config, "bad", || panic!("lookup"), || panic!("artifact")).unwrap_err(),
            "model_unsupported"
        );
        config.insert(
            "profile".into(),
            "gte-reranker-modernbert-base.owned-vulkan".into(),
        );
        assert_eq!(
            preflight(&config, "bad", || panic!("lookup"), || panic!("artifact")).unwrap_err(),
            "operation_mismatch"
        );
        config.insert("operation".into(), "rerank".into());
        assert_eq!(
            preflight(&config, "bad", || panic!("lookup"), || panic!("artifact")).unwrap_err(),
            "package_digest_mismatch"
        );
        let manifest = crate::manifest();
        let profile = &manifest.profiles[&config["profile"]];
        assert_eq!(
            preflight(
                &config,
                profile.converted_package_digest.as_ref().unwrap(),
                || Ok(vec![eligible()]),
                || Ok(Header {
                    tensors: BTreeMap::new(),
                    metadata: None,
                    key_order: vec![]
                })
            )
            .unwrap_err(),
            "head_tensor_missing"
        );
    }
}
