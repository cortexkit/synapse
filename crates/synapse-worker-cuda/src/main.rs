#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use synapse_core::worker_framing_sync::{
    read_frame, read_json_frame, write_frame, write_json_frame,
};
use synapse_core::{
    encode_f32_frame, owned_cuda_engine_identity, WorkerHello, WorkerHelloAck, WorkerRequest,
    WorkerResponse, DEFAULT_MAX_FRAME_BYTES, WORKER_PROTOCOL_VERSION,
};

const DEFAULT_VECTOR_DIMS: usize = 384;
const KERNEL_REVISION: &str = "cuda-kernel-v1";

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
    dims: usize,
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
        return worker_request_loop(&mut stream, ack.max_frame, &args);
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
        return worker_request_loop(&mut stream, max_frame, &args);
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
        let response = match request {
            WorkerRequest::Load {
                req_id,
                artifact_path,
                artifact_digest,
                format,
                runtime_config,
            } => handle_load(
                &mut state,
                req_id,
                artifact_path,
                artifact_digest,
                format,
                runtime_config,
            ),
            WorkerRequest::EmbedBatch { req_id, items, .. } => {
                handle_embed(&state, req_id, items.len())
            }
            WorkerRequest::Rerank { req_id, .. } => error_response(
                Some(req_id),
                "backend_missing",
                "owned-CUDA worker v1 does not expose rerank",
            ),
            WorkerRequest::Generate { req_id, .. } => error_response(
                Some(req_id),
                "backend_missing",
                "owned-CUDA worker v1 does not expose generation",
            ),
            WorkerRequest::Unload { req_id, model_ref } => {
                if state
                    .loaded
                    .as_ref()
                    .is_some_and(|model| model.model_ref == model_ref)
                {
                    state.loaded = None;
                    WorkerResponse::Unloaded { req_id }
                } else {
                    error_response(Some(req_id), "model_not_loaded", "unknown model reference")
                }
            }
            WorkerRequest::Ping { req_id } => WorkerResponse::Pong {
                req_id,
                rss_mb: 0,
                models_loaded: usize::from(state.loaded.is_some()),
                placement_share: None,
            },
            WorkerRequest::Shutdown {} => break,
        };
        write_json_frame(stream, &response, max_frame)?;
        if let WorkerResponse::Vectors { n, dims, .. } = response {
            write_frame(
                stream,
                &encode_f32_frame(&vec![0.0; n.saturating_mul(dims)]),
                max_frame,
            )?;
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
    if !cfg!(feature = "cuda") {
        return error_response(
            Some(req_id),
            "backend_missing",
            "owned-CUDA worker was built without the cuda feature",
        );
    }
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
    let dims = runtime_config
        .get("dims")
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_VECTOR_DIMS);
    let model_ref = format!(
        "owned-cuda-model-{}",
        state.loaded.as_ref().map_or(0, |_| 1)
    );
    state.loaded = Some(LoadedModel {
        model_ref: model_ref.clone(),
        dims,
    });
    WorkerResponse::Loaded {
        req_id,
        model_ref,
        dims,
        cold_load_ms: 0,
    }
}

fn handle_embed(state: &WorkerState, req_id: String, n: usize) -> WorkerResponse {
    let Some(model) = state.loaded.as_ref() else {
        return error_response(
            Some(req_id),
            "model_not_loaded",
            "no model specification is loaded",
        );
    };
    WorkerResponse::Vectors {
        req_id,
        dims: model.dims,
        n,
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
