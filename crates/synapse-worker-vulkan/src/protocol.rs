use std::{
    collections::BTreeMap,
    io::{self, Read, Write},
    path::Path,
};

use anyhow::{Context, Result};
use synapse_core::{
    worker_framing_sync::{read_frame, read_json_frame, write_frame, write_json_frame},
    EngineIdentity, WorkerHello, WorkerHelloAck, WorkerProgressPhase, WorkerRequest,
    WorkerResponse, DEFAULT_MAX_FRAME_BYTES, WORKER_PROTOCOL_VERSION,
};
use synapse_parity::{
    manifest::{Lane, Model, Operation, Profile},
    safetensors::{parse_header_json, Header},
};

use crate::{
    admission::{requirements, select, Adapter},
    KERNEL_REVISION, MANIFEST_DIGEST,
};

#[cfg(all(test, feature = "vulkan"))]
#[path = "transcribe_hardware.rs"]
mod transcribe_hardware;

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
        packed: None,
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

fn loaded_response(req_id: String, model: &Loaded) -> WorkerResponse {
    let mut response =
        WorkerResponse::loaded(req_id, model.model_ref.clone(), model.dims, 0, None, None);
    if model.operation == Operation::Transcribe {
        if let WorkerResponse::Loaded {
            transcribe_revision,
            ..
        } = &mut response
        {
            *transcribe_revision = Some(synapse_core::TranscribeRevision {
                kernel_revision:
                    synapse_core::worker_engine_names::VULKAN_TRANSCRIBE_KERNEL_REVISION.into(),
                decode_rule: synapse_speech::DECODE_RULE_REVISION.into(),
            });
        }
    }
    response
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
        Operation::Transcribe => "transcribe",
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
    engine: LoadedEngine,
}

#[cfg(feature = "vulkan")]
enum LoadedEngine {
    Tokens(Box<crate::runtime::Engine>),
    Speech(Box<crate::runtime::transcribe::Engine>),
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
        if let Some(raw) = &raw {
            request
                .validate_raw_frame(raw)
                .context("invalid raw frame")?;
        }
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
                match load(
                    artifact_path,
                    artifact_digest,
                    runtime_config,
                    max_frame,
                    &mut lookup,
                ) {
                    Ok(model) => {
                        let response = loaded_response(req_id.clone(), &model);
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
            WorkerRequest::Transcribe {
                req_id,
                model_ref,
                mel_frames,
                valid_mel_frames,
                observe,
            } => {
                serve_transcribe(
                    stream,
                    max_frame,
                    loaded.as_mut(),
                    req_id,
                    model_ref,
                    *mel_frames,
                    *valid_mel_frames,
                    raw.as_deref().unwrap(),
                    *observe,
                )?;
                continue;
            }
            WorkerRequest::EmbedBatch {
                req_id,
                model_ref,
                items,
                ..
            } => {
                let mut writer = synapse_core::worker_framing_sync::ProgressWriter::new(
                    stream, req_id, max_frame,
                );
                let result = infer_progress(
                    loaded.as_ref(),
                    model_ref,
                    Operation::Embed,
                    &items.iter().map(|i| i.n_tokens).collect::<Vec<_>>(),
                    raw.as_deref().unwrap(),
                    &mut || {
                        writer
                            .completed(WorkerProgressPhase::Inference)
                            .map_err(Into::into)
                    },
                );
                if writer.failed() {
                    return Err(anyhow::anyhow!("progress stream write failed"));
                }
                match result {
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
                let mut writer = synapse_core::worker_framing_sync::ProgressWriter::new(
                    stream, req_id, max_frame,
                );
                let result = infer_progress(
                    loaded.as_ref(),
                    model_ref,
                    Operation::Rerank,
                    &sequences.iter().map(|s| s.n_tokens).collect::<Vec<_>>(),
                    raw.as_deref().unwrap(),
                    &mut || {
                        writer
                            .completed(WorkerProgressPhase::Inference)
                            .map_err(Into::into)
                    },
                );
                if writer.failed() {
                    return Err(anyhow::anyhow!("progress stream write failed"));
                }
                match result {
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

#[allow(clippy::too_many_arguments)]
fn serve_transcribe(
    stream: &mut impl Write,
    max_frame: u32,
    loaded: Option<&mut Loaded>,
    req_id: &str,
    model_ref: &str,
    mel_frames: usize,
    valid: usize,
    raw: &[u8],
    observe: bool,
) -> Result<()> {
    let mut writer =
        synapse_core::worker_framing_sync::ProgressWriter::new(stream, req_id, max_frame);
    let result = transcribe_progress(
        loaded,
        model_ref,
        mel_frames,
        valid,
        raw,
        observe,
        &mut || {
            writer
                .completed(WorkerProgressPhase::Inference)
                .map_err(Into::into)
        },
    );
    if writer.failed() {
        return Err(anyhow::anyhow!("progress stream write failed"));
    }
    match result {
        Ok(output) => crate::transcribe::write_output(stream, req_id, &output, observe, max_frame)?,
        Err(code) => write_json_frame(stream, &err(Some(req_id.into()), &code, &code), max_frame)?,
    }
    Ok(())
}

fn load(
    path: &str,
    artifact_digest: &str,
    config: &BTreeMap<String, String>,
    max_frame: u32,
    lookup: impl FnOnce() -> std::result::Result<Vec<Adapter>, String>,
) -> std::result::Result<Loaded, String> {
    if let Some(profile) = config
        .get("profile")
        .and_then(|id| crate::manifest().profiles.get(id).cloned())
        .filter(|p| p.transcribe.is_some())
    {
        let maximum = crate::transcribe::maximum(&profile).map_err(|e| e.to_string())?;
        crate::transcribe::validate_frame_envelope(maximum, max_frame)
            .map_err(|e| e.to_string())?;
        if config
            .get("max_encoder_frames")
            .is_some_and(|m| m.parse::<usize>().ok() != Some(maximum))
        {
            return Err("invalid_runtime_config".into());
        }
    }
    let mut package_path = Path::new(path).to_path_buf();
    #[cfg(feature = "vulkan")]
    let mut prepared = None;
    #[cfg(feature = "vulkan")]
    let lookup = || {
        let manifest = crate::manifest();
        let profile = &manifest.profiles[&config["profile"]];
        let required = requirements(&manifest, Some(&profile.model))?;
        select(lookup()?, required)?;
        // Keep the admitted logical device alive before opening the package;
        // loader or instance startup must not occur again after artifact reads.
        let context = crate::runtime::prepare(required).map_err(runtime_error_code)?;
        let adapters = vec![context.adapter.clone()];
        prepared = Some(context);
        Ok(adapters)
    };
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
        let (model, profile, _adapter, _header) = preflight;
        let bytes = std::fs::read(&package_path).map_err(|_| "artifact_invalid".to_owned())?;
        if format!("{:x}", Sha256::digest(&bytes)) != digest(artifact_digest) {
            return Err("package_digest_mismatch".into());
        }
        let profile_id = config.get("profile").unwrap();
        synapse_parity::safetensors::verify_package_structure(&bytes, profile_id)
            .map_err(|_| "package_digest_mismatch".to_owned())?;
        // Tensor offsets must come from the same verified bytes as the data,
        // not from the earlier header-only read of a replaceable path.
        let (header_bytes, data) = synapse_parity::safetensors::split_file(&bytes)
            .map_err(|_| "artifact_invalid".to_owned())?;
        let header = parse_header_json(header_bytes).map_err(|_| "artifact_invalid".to_owned())?;
        let expected = synapse_parity::arch::expected_tensors(&model)
            .map_err(|_| "model_unsupported".to_owned())?;
        if header.tensors.len() != expected.len()
            || expected.iter().any(|(name, shape)| {
                header.tensors.get(name).is_none_or(|t| {
                    &t.shape != shape
                        || t.dtype
                            != if name.ends_with("num_batches_tracked") {
                                synapse_parity::safetensors::StDType::I64
                            } else {
                                synapse_parity::safetensors::StDType::F16
                            }
                })
            })
        {
            return Err("package_digest_mismatch".into());
        }
        if header
            .tensors
            .values()
            .any(|t| t.data_offsets.1 > data.len())
        {
            return Err("artifact_invalid".into());
        }
        let dims = if model.operation == Operation::Transcribe {
            0
        } else {
            model.output().map_err(|_| "model_unsupported")?.dimension as usize
        };
        let operation = model.operation;
        let required = requirements(&crate::manifest(), Some(&profile.model))?;
        let engine = if operation == Operation::Transcribe {
            LoadedEngine::Speech(Box::new(
                crate::runtime::transcribe::Engine::load(
                    &model,
                    &profile,
                    prepared.take().expect("device prepared before header"),
                    required,
                    &header,
                    data,
                )
                .map_err(runtime_error_code)?,
            ))
        } else {
            LoadedEngine::Tokens(Box::new(
                crate::runtime::Engine::load(
                    model,
                    profile,
                    prepared.take().expect("device prepared before header"),
                    required,
                    &header,
                    data,
                )
                .map_err(runtime_error_code)?,
            ))
        };
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

#[cfg(feature = "vulkan")]
fn runtime_error_code(error: anyhow::Error) -> String {
    let code = error.to_string();
    if matches!(
        code.as_str(),
        "vulkan_no_device"
            | "vulkan_api_too_old"
            | "vulkan_software_device"
            | "vulkan_unsupported_vendor"
            | "vulkan_insufficient_memory"
    ) || code.starts_with("vulkan_missing_feature:")
    {
        code
    } else {
        "vulkan_load_failed".into()
    }
}

fn infer_progress(
    loaded: Option<&Loaded>,
    model_ref: &str,
    operation: Operation,
    lengths: &[usize],
    raw: &[u8],
    progress: &mut impl FnMut() -> Result<()>,
) -> std::result::Result<Vec<f32>, String> {
    let model = loaded
        .filter(|m| m.model_ref == model_ref)
        .ok_or("model_not_loaded")?;
    if model.operation != operation {
        return Err("operation_mismatch".into());
    }
    let max_context = crate::manifest().admission.max_context_tokens as usize;
    if lengths.iter().any(|n| *n > max_context) {
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
        let LoadedEngine::Tokens(engine) = &model.engine else {
            return Err("operation_mismatch".into());
        };
        engine
            .infer_progress(&sequences, progress)
            .map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (sequences, progress);
        Err("vulkan_no_device".into())
    }
}

#[allow(clippy::too_many_arguments)]
fn transcribe_progress(
    loaded: Option<&mut Loaded>,
    model_ref: &str,
    mel_frames: usize,
    valid: usize,
    raw: &[u8],
    observe: bool,
    progress: &mut impl FnMut() -> Result<()>,
) -> std::result::Result<crate::transcribe::Output, String> {
    let model = loaded
        .filter(|m| m.model_ref == model_ref)
        .ok_or("model_not_loaded")?;
    if model.operation != Operation::Transcribe {
        return Err("operation_mismatch".into());
    }
    let mel = synapse_core::decode_f32_frame(raw).map_err(|_| "invalid_mel_frames".to_owned())?;
    #[cfg(feature = "vulkan")]
    {
        let LoadedEngine::Speech(engine) = &mut model.engine else {
            return Err("operation_mismatch".into());
        };
        engine
            .transcribe(&mel, mel_frames, valid, observe, progress)
            .map_err(|e| e.to_string())
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (mel, mel_frames, valid, observe, progress);
        Err("vulkan_no_device".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcribe_profile_is_admitted_before_weights_are_opened() {
        let manifest = crate::manifest();
        let profile = &manifest.profiles["parakeet-tdt-0.6b-v3.owned-vulkan"];
        let config = BTreeMap::from([
            ("profile".into(), "parakeet-tdt-0.6b-v3.owned-vulkan".into()),
            ("operation".into(), "transcribe".into()),
        ]);
        let mut opened = false;
        let (model, _, _, _) = preflight(
            &config,
            profile.converted_package_digest.as_ref().unwrap(),
            || Ok(vec![eligible()]),
            || {
                opened = true;
                parse_header_json(b"{}").map_err(|e| e.to_string())
            },
        )
        .unwrap();
        assert!(opened);
        assert_eq!(model.operation, Operation::Transcribe);
        let mut too_small = eligible();
        too_small.device_local_heaps = vec![profile.vulkan_min_device_local_bytes.unwrap() - 1];
        assert_eq!(
            preflight(
                &config,
                profile.converted_package_digest.as_ref().unwrap(),
                || Ok(vec![too_small]),
                || panic!("floor refusal must precede artifact access")
            )
            .unwrap_err(),
            "vulkan_insufficient_memory"
        );
    }

    #[test]
    fn transcribe_raw_count_and_negotiated_limit_are_protocol_faults() {
        for (raw, limit) in [
            (vec![0; 511], 4096),
            (vec![0; 513], 4096),
            (vec![0; 512], 256),
        ] {
            let mut input = Vec::new();
            write_json_frame(
                &mut input,
                &WorkerRequest::Transcribe {
                    req_id: "speech".into(),
                    model_ref: "missing".into(),
                    mel_frames: 1,
                    valid_mel_frames: 0,
                    observe: false,
                },
                4096,
            )
            .unwrap();
            write_frame(&mut input, &raw, 4096).unwrap();
            let mut script = Script {
                input: std::io::Cursor::new(input),
                output: Vec::new(),
            };
            assert!(request_loop(&mut script, limit, || panic!(
                "no device access for invalid frame"
            ))
            .is_err());
            assert!(script.output.is_empty());
        }
    }

    #[test]
    fn transcribe_refusal_consumes_raw_frame_before_ping() {
        let mut input = Vec::new();
        write_json_frame(
            &mut input,
            &WorkerRequest::Transcribe {
                req_id: "speech".into(),
                model_ref: "missing".into(),
                mel_frames: 2,
                valid_mel_frames: 1,
                observe: true,
            },
            4096,
        )
        .unwrap();
        write_frame(&mut input, &[0; 1024], 4096).unwrap();
        write_json_frame(
            &mut input,
            &WorkerRequest::Ping {
                req_id: "alive".into(),
            },
            4096,
        )
        .unwrap();
        let mut script = Script {
            input: std::io::Cursor::new(input),
            output: Vec::new(),
        };
        request_loop(&mut script, 4096, || panic!("no device access before LOAD")).unwrap();
        let mut responses = std::io::Cursor::new(script.output);
        assert!(
            matches!(read_json_frame::<_,WorkerResponse>(&mut responses,4096).unwrap(), WorkerResponse::Err {code,..} if code=="model_not_loaded")
        );
        assert!(
            matches!(read_json_frame::<_,WorkerResponse>(&mut responses,4096).unwrap(), WorkerResponse::Pong {req_id,..} if req_id=="alive")
        );
    }

    #[test]
    #[cfg(not(feature = "vulkan"))]
    fn transcribe_load_revision_is_separate_from_legacy_loaded_bytes() {
        let mut model = Loaded {
            model_ref: "speech".into(),
            operation: Operation::Transcribe,
            dims: 0,
        };
        let response = loaded_response("load".into(), &model);
        let WorkerResponse::Loaded {
            transcribe_revision: Some(revision),
            ..
        } = response
        else {
            panic!("missing revision");
        };
        assert_eq!(
            revision.kernel_revision,
            synapse_core::worker_engine_names::VULKAN_TRANSCRIBE_KERNEL_REVISION
        );
        assert_eq!(revision.decode_rule, synapse_speech::DECODE_RULE_REVISION);
        model.operation = Operation::Embed;
        model.dims = 768;
        assert_eq!(
            serde_json::to_string(&loaded_response("load".into(), &model)).unwrap(),
            r#"{"type":"LOADED","req_id":"load","model_ref":"speech","dims":768,"cold_load_ms":0}"#
        );
    }
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
    #[cfg(feature = "vulkan")]
    #[test]
    fn retained_device_startup_preserves_floor_refusal_codes() {
        for code in [
            "vulkan_no_device",
            "vulkan_api_too_old",
            "vulkan_software_device",
            "vulkan_unsupported_vendor",
            "vulkan_insufficient_memory",
            "vulkan_missing_feature:shaderFloat16",
            "vulkan_missing_feature:storageBuffer16BitAccess",
            "vulkan_missing_feature:subgroupArithmetic",
            "vulkan_missing_feature:subgroupComputeStage",
        ] {
            assert_eq!(runtime_error_code(anyhow::anyhow!(code)), code);
        }
        assert_eq!(
            runtime_error_code(anyhow::anyhow!("pipeline creation failed")),
            "vulkan_load_failed"
        );
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
