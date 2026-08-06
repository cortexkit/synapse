#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use synapse_core::worker_framing_sync::{
    read_frame, read_json_frame, write_frame, write_json_frame,
};
use synapse_core::{
    decode_i32_frame, encode_f32_frame, owned_cuda_engine_identity, WorkerHello, WorkerHelloAck,
    WorkerRequest, WorkerResponse, DEFAULT_MAX_FRAME_BYTES, WORKER_PROTOCOL_VERSION,
};
#[cfg(feature = "cuda")]
use synapse_core::{EmbedEngine, RuntimeConfig, TokenBatch, ValidatedArtifact};
#[cfg(feature = "cuda")]
use synapse_engine_cuda::{detect_family, OwnedCudaEmbedEngine};

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
    engine: OwnedCudaEmbedEngine,
    #[cfg(feature = "cuda")]
    engine_model: synapse_core::LoadedModel,
}

fn version_probe() -> bool {
    if std::env::args().skip(1).any(|arg| arg == "--version") {
        println!(concat!(
            env!("CARGO_BIN_NAME"),
            " ",
            env!("CARGO_PKG_VERSION")
        ));
        true
    } else {
        false
    }
}

fn engine_identity() -> synapse_core::EngineIdentity {
    owned_cuda_engine_identity("worker", "f16", KERNEL_REVISION)
}

fn main() -> Result<()> {
    if version_probe() {
        return Ok(());
    }
    let args = Args::parse();
    let hello = WorkerHello {
        v: WORKER_PROTOCOL_VERSION,
        nonce: args.nonce.clone(),
        engine: engine_identity(),
        pid: std::process::id(),
        max_frame: DEFAULT_MAX_FRAME_BYTES,
    };
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
        bail!("owned-CUDA worker transport is unsupported on this target");
    }
}

fn validate_ack(ack: &WorkerHelloAck) -> Result<()> {
    if ack.v != WORKER_PROTOCOL_VERSION {
        bail!("module replied with unsupported protocol v{}", ack.v);
    }
    if !ack.accept {
        bail!("module rejected owned-CUDA worker handshake");
    }
    Ok(())
}

fn worker_request_loop<S: Read + Write>(stream: &mut S, max_frame: u32, args: &Args) -> Result<()> {
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
                handle_load(
                    &mut state,
                    req_id,
                    artifact_path,
                    artifact_digest,
                    format,
                    runtime_config,
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
                    Ok(ids) => handle_embed(&state, req_id, &model_ref, &items, &ids),
                    Err(message) => (
                        error_response(Some(req_id), "invalid_request", &message),
                        None,
                    ),
                }
            }
            WorkerRequest::Rerank { req_id, .. } => (
                error_response(
                    Some(req_id),
                    "backend_missing",
                    "owned-CUDA worker v1 does not expose rerank",
                ),
                None,
            ),
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
                },
                None,
            ),
            WorkerRequest::Shutdown {} => break,
        };
        write_json_frame(stream, &response, max_frame)?;
        if let Some(vectors) = vectors {
            write_frame(stream, &encode_f32_frame(&vectors), max_frame)?;
        }
    }
    Ok(())
}

fn handle_load(
    state: &mut WorkerState,
    req_id: String,
    artifact_path: String,
    artifact_digest: String,
    format: String,
    runtime_config: BTreeMap<String, String>,
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
        );
        error_response(
            Some(req_id),
            "backend_missing",
            "owned-CUDA worker was built without the cuda feature",
        )
    }

    #[cfg(feature = "cuda")]
    {
        let family = match detect_family(&artifact_path) {
            Ok(family) => family,
            Err(error) => {
                return error_response(Some(req_id), "artifact_invalid", &error.to_string());
            }
        };
        let mut engine = OwnedCudaEmbedEngine::serving(family);
        let mut runtime = RuntimeConfig {
            values: runtime_config,
        };
        runtime
            .values
            .entry("model_path".to_string())
            .or_insert_with(|| artifact_path.clone());
        runtime
            .values
            .entry("artifact_path".to_string())
            .or_insert_with(|| artifact_path.clone());
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
                return error_response(Some(req_id), "artifact_invalid", &error.message);
            }
        };
        let Some(dims) = engine.dimensions(&engine_model) else {
            return error_response(
                Some(req_id),
                "artifact_invalid",
                "owned-CUDA engine did not report embedding dimensions",
            );
        };
        let model_ref = engine_model.model_id.clone();
        state.loaded = Some(LoadedModel {
            model_ref: model_ref.clone(),
            dims,
            engine,
            engine_model,
        });
        WorkerResponse::Loaded {
            req_id,
            model_ref,
            dims,
            cold_load_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
        }
    }
}

fn handle_embed(
    state: &WorkerState,
    req_id: String,
    model_ref: &str,
    items: &[synapse_core::WorkerTokenItem],
    ids: &[u32],
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
        let _ = (model, ids);
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
                    WorkerResponse::Vectors {
                        req_id,
                        dims: model.dims,
                        n: items.len(),
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
    use synapse_core::{OWNED_CUDA_ENGINE, OWNED_CUDA_PTX_VIRTUAL_ARCH};

    #[test]
    fn handshake_identity_is_owned_cuda_ptx() {
        let identity = engine_identity();
        assert_eq!(identity.engine, OWNED_CUDA_ENGINE);
        assert_eq!(identity.build_flags["backend"], "cuda-ptx");
        assert_eq!(
            identity.build_flags["ptx_virtual_arch"],
            OWNED_CUDA_PTX_VIRTUAL_ARCH
        );
        assert_eq!(identity.build_flags["risk_class"], "abort_capable");
    }

    #[test]
    fn missing_feature_refuses_load_without_creating_model_state() {
        let mut state = WorkerState::default();
        let response = handle_load(
            &mut state,
            "load-1".to_string(),
            "/tmp/model".to_string(),
            "sha256:abc".to_string(),
            "safetensors".to_string(),
            BTreeMap::new(),
        );
        if !cfg!(feature = "cuda") {
            assert!(
                matches!(response, WorkerResponse::Err { ref code, .. } if code == "backend_missing")
            );
            assert!(state.loaded.is_none());
        }
    }
}
