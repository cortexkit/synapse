#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
#[cfg(unix)]
use synapse_core::worker_framing_sync::read_json_frame;
use synapse_core::worker_framing_sync::{read_frame, write_frame, write_json_frame};
#[cfg(unix)]
use synapse_core::WorkerHelloAck;
use synapse_core::{
    decode_i32_frame, encode_f32_frame, owned_cuda_engine_identity, WorkerHello, WorkerRequest,
    WorkerResponse, DEFAULT_MAX_FRAME_BYTES, WORKER_PROTOCOL_VERSION,
};
#[cfg(feature = "cuda")]
use synapse_core::{EmbedEngine, RuntimeConfig, TokenBatch, ValidatedArtifact};
#[cfg(feature = "cuda")]
use synapse_engine_cuda::manifest::Profile;
use synapse_engine_cuda::manifest::{floor_envelope, manifest_digest};
#[cfg(feature = "cuda")]
use synapse_engine_cuda::{ModelFamily, OwnedCudaEmbedEngine};

const KERNEL_REVISION: &str = "4d0ded67c30286fe2be37cc7413359ad745dd751";

#[derive(Parser, Debug)]
#[command(name = "ck-synapse-worker-cuda")]
struct Args {
    #[arg(long)]
    socket: Option<PathBuf>,
    #[cfg(windows)]
    #[arg(long)]
    pipe: Option<String>,
    #[arg(long)]
    nonce: String,
    #[arg(long = "test-abort", hide = true)]
    test_abort: bool,
    #[arg(long = "test-abort-on-request", hide = true)]
    test_abort_on_request: bool,
}

#[derive(Default)]
struct WorkerState {
    loaded: Option<LoadedModel>,
}

struct LoadedModel {
    model_ref: String,
    #[cfg(feature = "cuda")]
    dims: usize,
    #[cfg(feature = "cuda")]
    operation: String,
    #[cfg(feature = "cuda")]
    engine: OwnedCudaEmbedEngine,
    #[cfg(feature = "cuda")]
    engine_model: synapse_core::LoadedModel,
}

fn version_probe() -> bool {
    if std::env::args().skip(1).any(|arg| arg == "--version") {
        println!(
            "{} {} features={} manifest_digest={}",
            env!("CARGO_BIN_NAME"),
            env!("CARGO_PKG_VERSION"),
            if cfg!(feature = "cuda") {
                "cuda"
            } else {
                "none"
            },
            manifest_digest()
        );
        true
    } else {
        false
    }
}

fn probe_floor() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let model = args
        .windows(2)
        .find(|pair| pair[0] == "--model")
        .map(|pair| pair[1].as_str());
    let envelope = floor_envelope(synapse_engine_cuda::probe_hardware_floor(), model);
    println!("{}", envelope);
    if envelope["status"] != "ok" {
        std::process::exit(2);
    }
    Ok(())
}

/// Build the identity announced in the worker HELLO handshake.
pub fn engine_identity() -> synapse_core::EngineIdentity {
    owned_cuda_engine_identity("worker", "f16", KERNEL_REVISION)
}

fn main() -> Result<()> {
    if version_probe() {
        return Ok(());
    }
    if std::env::args().skip(1).any(|arg| arg == "--probe-floor") {
        return probe_floor();
    }
    let args = Args::parse();
    let hello = worker_hello(args.nonce.clone());
    #[cfg(unix)]
    {
        let socket = args
            .socket
            .as_ref()
            .context("owned-CUDA worker requires --socket on Unix")?;
        let mut stream = std::os::unix::net::UnixStream::connect(socket)
            .with_context(|| format!("connect worker socket {}", socket.display()))?;
        write_json_frame(&mut stream, &hello, DEFAULT_MAX_FRAME_BYTES)?;
        let ack: WorkerHelloAck = read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES)?;
        validate_ack(&ack)?;
        worker_request_loop(&mut stream, ack.max_frame, &args)
    }
    #[cfg(windows)]
    {
        let pipe = args
            .pipe
            .as_deref()
            .context("owned-CUDA worker requires --pipe on Windows")?;
        let (mut stream, max_frame) =
            synapse_core::worker_transport::windows_client::connect_and_handshake(
                pipe,
                &hello,
                DEFAULT_MAX_FRAME_BYTES,
            )?;
        worker_request_loop(&mut stream, max_frame, &args)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (args, hello);
        anyhow::bail!("owned-CUDA worker transport is unsupported on this target");
    }
}

#[cfg(unix)]
fn validate_ack(ack: &WorkerHelloAck) -> Result<()> {
    if ack.v != WORKER_PROTOCOL_VERSION {
        anyhow::bail!("module replied with unsupported protocol v{}", ack.v);
    }
    if !ack.accept {
        anyhow::bail!("module rejected owned-CUDA worker handshake");
    }
    Ok(())
}

fn worker_request_loop<S: Read + Write>(stream: &mut S, max_frame: u32, args: &Args) -> Result<()> {
    worker_request_loop_with_probe(
        stream,
        max_frame,
        args,
        synapse_engine_cuda::probe_hardware_floor,
    )
}
fn worker_request_loop_with_probe<S: Read + Write>(
    stream: &mut S,
    max_frame: u32,
    args: &Args,
    probe: fn() -> Result<synapse_engine_cuda::HardwareFloorProbe>,
) -> Result<()> {
    let mut state = WorkerState::default();
    loop {
        let frame = match read_frame(stream, max_frame) {
            Ok(frame) => frame,
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error).context("read owned-CUDA request frame"),
        };
        let request: WorkerRequest =
            serde_json::from_slice(&frame).context("decode request JSON")?;
        if args.test_abort || args.test_abort_on_request {
            std::process::abort();
        }
        let (response, vectors) = match request {
            WorkerRequest::Load {
                req_id,
                artifact_path,
                artifact_digest,
                format,
                runtime_config,
            } => (
                handle_load_with_probe(
                    &mut state,
                    req_id,
                    artifact_path,
                    artifact_digest,
                    format,
                    runtime_config,
                    probe,
                ),
                None,
            ),
            WorkerRequest::EmbedBatch {
                req_id,
                model_ref,
                items,
                ..
            } => {
                let ids = match read_frame(stream, max_frame) {
                    Ok(frame) => decode_i32_frame(&frame)
                        .map_err(|error| error.to_string())
                        .and_then(|ids| {
                            ids.into_iter()
                                .map(|id| {
                                    u32::try_from(id)
                                        .map_err(|_| format!("token id {id} is negative"))
                                })
                                .collect::<Result<Vec<_>, _>>()
                        }),
                    Err(error) => Err(format!("read token-id frame: {error}")),
                };
                match ids {
                    Ok(ids) => handle_sequences(&state, req_id, &model_ref, &items, &ids, false),
                    Err(message) => (
                        error_response(Some(req_id), "invalid_request", &message),
                        None,
                    ),
                }
            }
            unsupported @ WorkerRequest::Rerank { .. } => {
                read_frame(stream, max_frame).context("discard unsupported rerank frame")?;
                (WorkerResponse::unsupported_request(&unsupported), None)
            }
            WorkerRequest::Generate { req_id, .. } => (
                error_response(
                    Some(req_id),
                    "backend_missing",
                    "owned-CUDA worker v1 does not expose generation",
                ),
                None,
            ),
            WorkerRequest::Unload { req_id, model_ref } => {
                if state
                    .loaded
                    .as_ref()
                    .is_some_and(|model| model.model_ref == model_ref)
                {
                    #[cfg(feature = "cuda")]
                    if let Some(mut model) = state.loaded.take() {
                        model.engine.unload(&model.engine_model);
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        state.loaded = None;
                    }
                    (WorkerResponse::Unloaded { req_id }, None)
                } else {
                    (
                        error_response(Some(req_id), "model_not_loaded", "unknown model reference"),
                        None,
                    )
                }
            }
            WorkerRequest::Ping { req_id } => (
                WorkerResponse::Pong {
                    req_id,
                    rss_mb: 0,
                    models_loaded: usize::from(state.loaded.is_some()),
                    placement_share: None,
                    // This lane accepts any sequence up to the model's
                    // configured maximum, so it advertises no bucket ladder.
                    buckets: None,
                    ane_resident_shapes: None,
                },
                None,
            ),
            WorkerRequest::Shutdown {} => break,
            // Not served by this worker yet. Read and discard the raw frame a
            // refused request carries so the connection stays usable.
            WorkerRequest::RerankSequences {
                req_id,
                model_ref,
                sequences,
            } => {
                let raw = read_frame(stream, max_frame).context("read rerank sequences")?;
                match decode_i32_frame(&raw) {
                    Ok(ids) => match ids
                        .into_iter()
                        .map(|id| u32::try_from(id).map_err(|_| "negative token id"))
                        .collect::<Result<Vec<_>, _>>()
                    {
                        Ok(ids) => {
                            let items: Vec<_> = sequences
                                .iter()
                                .enumerate()
                                .map(|(i, s)| synapse_core::WorkerTokenItem {
                                    id: i.to_string(),
                                    n_tokens: s.n_tokens,
                                })
                                .collect();
                            handle_sequences(&state, req_id, &model_ref, &items, &ids, true)
                        }
                        Err(error) => {
                            (error_response(Some(req_id), "invalid_request", error), None)
                        }
                    },
                    Err(error) => (
                        error_response(Some(req_id), "invalid_request", &error.to_string()),
                        None,
                    ),
                }
            }
            unsupported @ (WorkerRequest::AneAdmitShape { .. }
            | WorkerRequest::AneEvictShape { .. }) => {
                if unsupported.carries_raw_frame() {
                    read_frame(stream, max_frame)
                        .context("discard the raw frame of an unsupported request")?;
                }
                (WorkerResponse::unsupported_request(&unsupported), None)
            }
        };
        write_json_frame(stream, &response, max_frame)?;
        if let Some(vectors) = vectors {
            write_frame(stream, &encode_f32_frame(&vectors), max_frame)?;
        }
    }
    Ok(())
}

fn handle_load_with_probe(
    state: &mut WorkerState,
    req_id: String,
    artifact_path: String,
    artifact_digest: String,
    format: String,
    runtime_config: BTreeMap<String, String>,
    probe: fn() -> Result<synapse_engine_cuda::HardwareFloorProbe>,
) -> WorkerResponse {
    if artifact_path.trim().is_empty() || format.trim().is_empty() {
        return error_response(
            Some(req_id),
            "artifact_invalid",
            "model artifact path and format are required",
        );
    }
    if artifact_digest.trim().is_empty() {
        return error_response(
            Some(req_id),
            "artifact_invalid",
            "model artifact digest is required",
        );
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            state,
            artifact_path,
            artifact_digest,
            format,
            runtime_config,
            probe,
        );
        error_response(
            Some(req_id),
            "backend_missing",
            "owned-CUDA worker was built without the cuda feature",
        )
    }

    #[cfg(feature = "cuda")]
    {
        let profile = match Profile::select(
            runtime_config
                .get("profile")
                .map(String::as_str)
                .unwrap_or(""),
            Some(
                runtime_config
                    .get("operation")
                    .map(String::as_str)
                    .unwrap_or(""),
            ),
        ) {
            Ok(p) => p,
            Err(e) => return error_response(Some(req_id), &e.to_string(), &e.to_string()),
        };
        let envelope = synapse_engine_cuda::manifest::floor_envelope_for_profile(probe(), &profile);
        if envelope["status"] != "ok" {
            return error_response(
                Some(req_id),
                envelope["code"].as_str().unwrap(),
                "CUDA floor refused load",
            );
        }
        let family = match profile.model["architecture"]["family"].as_str() {
            Some("modernbert") => ModelFamily::GteModernBert,
            Some("qwen3") => ModelFamily::Qwen3,
            _ => {
                return error_response(
                    Some(req_id),
                    "model_unsupported",
                    "unsupported manifest family",
                )
            }
        };
        let mut engine = OwnedCudaEmbedEngine::serving(family);
        let mut runtime = RuntimeConfig {
            values: runtime_config,
        };
        runtime
            .values
            .insert("model_path".into(), artifact_path.clone());
        runtime
            .values
            .insert("artifact_path".into(), artifact_path.clone());
        let started = Instant::now();
        let engine_model = match engine.load(
            &ValidatedArtifact {
                digest: artifact_digest,
                format,
            },
            &runtime,
        ) {
            Ok(model) => model,
            Err(error) => {
                let code = match error.message.as_str() {
                    "model_unsupported"
                    | "operation_mismatch"
                    | "package_digest_mismatch"
                    | "head_tensor_missing" => error.message.as_str(),
                    _ => "artifact_invalid",
                };
                return error_response(Some(req_id), code, &error.message);
            }
        };
        let Some(_) = engine.dimensions(&engine_model) else {
            return error_response(
                Some(req_id),
                "artifact_invalid",
                "owned-CUDA engine did not report embedding dimensions",
            );
        };
        let dims = profile.model["output"]["dimension"].as_u64().unwrap() as usize;
        let model_ref = engine_model.model_id.clone();
        state.loaded = Some(LoadedModel {
            model_ref: model_ref.clone(),
            dims,
            operation: profile.operation().into(),
            engine,
            engine_model,
        });
        WorkerResponse::Loaded {
            req_id,
            model_ref,
            dims,
            cold_load_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
            buckets: None,
        }
    }
}

fn handle_sequences(
    state: &WorkerState,
    req_id: String,
    model_ref: &str,
    items: &[synapse_core::WorkerTokenItem],
    ids: &[u32],
    rerank: bool,
) -> (WorkerResponse, Option<Vec<f32>>) {
    let Some(model) = state.loaded.as_ref() else {
        return (
            error_response(
                Some(req_id),
                "model_not_loaded",
                "no model specification is loaded",
            ),
            None,
        );
    };
    if model.model_ref != model_ref {
        return (
            error_response(Some(req_id), "model_not_loaded", "unknown model reference"),
            None,
        );
    }
    let expected_ids = items.iter().map(|item| item.n_tokens).sum::<usize>();
    if expected_ids != ids.len() || items.iter().any(|item| item.n_tokens == 0) {
        return (
            error_response(
                Some(req_id),
                "invalid_request",
                "token-id frame does not match non-empty item lengths",
            ),
            None,
        );
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (model, ids, rerank);
        (
            error_response(
                Some(req_id),
                "backend_missing",
                "owned-CUDA worker was built without the cuda feature",
            ),
            None,
        )
    }

    #[cfg(feature = "cuda")]
    {
        if (model.operation == "rerank") != rerank {
            return (
                error_response(
                    Some(req_id),
                    "operation_mismatch",
                    "request does not match manifest operation",
                ),
                None,
            );
        }
        if items
            .iter()
            .any(|item| item.n_tokens > synapse_engine_cuda::manifest::max_context_tokens())
        {
            return (
                error_response(
                    Some(req_id),
                    "sequence_too_long",
                    "input exceeds 8192 tokens",
                ),
                None,
            );
        }
        let mut offset = 0;
        let batch = TokenBatch {
            items: items
                .iter()
                .map(|item| {
                    let end = offset + item.n_tokens;
                    let tokens = ids[offset..end].to_vec();
                    offset = end;
                    tokens
                })
                .collect(),
        };
        match model.engine.embed_batch(&model.engine_model, batch) {
            Ok(vectors)
                if vectors.len() == items.len()
                    && vectors.iter().all(|vector| vector.len() == model.dims) =>
            {
                let values = vectors.into_iter().flatten().collect();
                (
                    if rerank {
                        WorkerResponse::Scores { req_id }
                    } else {
                        WorkerResponse::Vectors {
                            req_id,
                            dims: model.dims,
                            n: items.len(),
                        }
                    },
                    Some(values),
                )
            }
            Ok(_) => (
                error_response(
                    Some(req_id),
                    "engine_crashed",
                    "owned-CUDA engine returned an unexpected vector shape",
                ),
                None,
            ),
            Err(error) => (
                error_response(Some(req_id), "engine_crashed", &error.message),
                None,
            ),
        }
    }
}

fn error_response(req_id: Option<String>, code: &str, msg: &str) -> WorkerResponse {
    WorkerResponse::Err {
        req_id,
        code: code.to_string(),
        msg: msg.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_core::{worker_engine_names::CUDA_WORKER_ENGINE, OWNED_CUDA_PTX_VIRTUAL_ARCH};

    #[test]
    fn producer_identity_matches_catalog_engine_and_owned_cuda_ptx() {
        let identity = engine_identity();
        assert_eq!(identity.engine, CUDA_WORKER_ENGINE);
        assert_eq!(identity.build_flags["backend"], "cuda-ptx");
        assert_eq!(
            identity.build_flags["ptx_virtual_arch"],
            OWNED_CUDA_PTX_VIRTUAL_ARCH
        );
        assert_eq!(identity.build_flags["risk_class"], "abort_capable");
    }

    #[test]
    fn producer_identity_matches_shared_catalog_constant() {
        assert_eq!(engine_identity().engine, CUDA_WORKER_ENGINE);
    }

    #[test]
    fn missing_feature_refuses_load_without_creating_model_state() {
        let mut state = WorkerState::default();
        let response = handle_load_with_probe(
            &mut state,
            "load-1".to_string(),
            "/tmp/model".to_string(),
            "sha256:abc".to_string(),
            "safetensors".to_string(),
            BTreeMap::new(),
            synapse_engine_cuda::probe_hardware_floor,
        );
        if !cfg!(feature = "cuda") {
            assert!(
                matches!(response, WorkerResponse::Err { ref code, .. } if code == "backend_missing")
            );
            assert!(state.loaded.is_none());
        }
    }
}

fn worker_hello(nonce: String) -> WorkerHello {
    WorkerHello {
        v: WORKER_PROTOCOL_VERSION,
        nonce,
        engine: engine_identity(),
        pid: std::process::id(),
        max_frame: DEFAULT_MAX_FRAME_BYTES,
        manifest_digest: Some(manifest_digest()),
        kernel_revision: Some(KERNEL_REVISION.into()),
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    struct Duplex {
        input: std::io::Cursor<Vec<u8>>,
        output: Vec<u8>,
    }
    impl Read for Duplex {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            self.input.read(bytes)
        }
    }
    impl Write for Duplex {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.output.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    fn missing_driver() -> Result<synapse_engine_cuda::HardwareFloorProbe> {
        anyhow::bail!("cuda_no_driver")
    }
    #[test]
    fn hello_and_ping_succeed_with_injected_missing_driver() {
        assert_eq!(
            floor_envelope(missing_driver(), None)["code"],
            "cuda_no_driver"
        );
        let hello = worker_hello("driverless".into());
        assert_eq!(hello.nonce, "driverless");
        assert!(hello.manifest_digest.is_some());
        let mut input = Vec::new();
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
        let args = Args {
            socket: None,
            nonce: "driverless".into(),
            test_abort: false,
            test_abort_on_request: false,
            #[cfg(windows)]
            pipe: None,
        };
        let mut stream = Duplex {
            input: std::io::Cursor::new(input),
            output: Vec::new(),
        };
        worker_request_loop_with_probe(&mut stream, DEFAULT_MAX_FRAME_BYTES, &args, missing_driver)
            .unwrap();
        let mut output = std::io::Cursor::new(stream.output);
        assert!(
            matches!(synapse_core::worker_framing_sync::read_json_frame::<_,WorkerResponse>(&mut output,DEFAULT_MAX_FRAME_BYTES).unwrap(),WorkerResponse::Pong{req_id,..} if req_id=="ping")
        );
    }
}

#[cfg(all(test, feature = "cuda"))]
mod load_floor_tests {
    use super::*;
    #[test]
    fn missing_driver_refuses_before_artifact_open() {
        fn no_driver() -> Result<synapse_engine_cuda::HardwareFloorProbe> {
            anyhow::bail!("cuda_no_driver")
        }
        let p = Profile::select("gte-modernbert-base.owned-cuda", Some("embed")).unwrap();
        let response = handle_load_with_probe(
            &mut WorkerState::default(),
            "sentinel".into(),
            "artifact-open-must-not-happen/model.safetensors".into(),
            p.package_digest,
            "safetensors-package".into(),
            BTreeMap::from([
                ("profile".into(), p.id),
                ("operation".into(), "embed".into()),
            ]),
            no_driver,
        );
        assert!(matches!(response,WorkerResponse::Err{code,..} if code=="cuda_no_driver"));
    }
}
