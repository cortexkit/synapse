//! Worker protocol v2 against the built llama worker: HELLO, PING to PONG, and
//! the unsupported-request exchange. The llama worker does not serve
//! `RERANK_SEQUENCES` or `ANE_ADMIT_SHAPE` (only the owned CUDA, Vulkan and
//! direct-ANE workers do); each must be refused with `unsupported_request`,
//! and a PING on the same connection must still return PONG.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use synapse_core::{
    accept_worker_handshake_with_engine_and_protocol_version, encode_i32_frame, prepare_listener,
    read_json, worker_engine_names::LLAMA_WORKER_ENGINE, write_json, write_raw, WorkerRequest,
    WorkerResponse, WorkerSequence, WorkerTransportStream, DEFAULT_MAX_FRAME_BYTES,
    ERR_UNSUPPORTED_REQUEST,
};

const MAX_FRAME: u32 = DEFAULT_MAX_FRAME_BYTES;

fn unique_suffix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(0);
    format!("{:016x}", nanos ^ u64::from(std::process::id()))
}

async fn ping(stream: &mut WorkerTransportStream, req_id: &str) {
    write_json(
        stream,
        &WorkerRequest::Ping {
            req_id: req_id.to_string(),
        },
        MAX_FRAME,
    )
    .await
    .unwrap();
    let response: WorkerResponse = read_json(stream, MAX_FRAME).await.unwrap();
    match response {
        WorkerResponse::Pong {
            req_id: got,
            ane_resident_shapes,
            ..
        } => {
            assert_eq!(got, req_id);
            assert_eq!(
                ane_resident_shapes, None,
                "llama PONG carries no ANE shapes"
            );
        }
        other => panic!("PING must return PONG, got {other:?}"),
    }
}

async fn expect_unsupported(stream: &mut WorkerTransportStream, req_id: &str) {
    let response: WorkerResponse = read_json(stream, MAX_FRAME).await.unwrap();
    match response {
        WorkerResponse::Err {
            req_id: got, code, ..
        } => {
            assert_eq!(got.as_deref(), Some(req_id));
            assert_eq!(code, ERR_UNSUPPORTED_REQUEST);
        }
        other => panic!("expected ERR unsupported_request, got {other:?}"),
    }
}

#[tokio::test]
async fn llama_worker_speaks_protocol_v2_and_refuses_owned_worker_requests() {
    let suffix = unique_suffix();
    // macOS temp dirs are too long for a Unix socket path (SUN_LEN), so the
    // socket lives under /tmp there, as the host's own tests do.
    #[cfg(unix)]
    let temp_root = std::path::PathBuf::from("/tmp");
    #[cfg(not(unix))]
    let temp_root = std::env::temp_dir();
    let runtime_dir = temp_root.join(format!("synh-llama-v2-{suffix}"));
    let (endpoint, listener) =
        prepare_listener(&runtime_dir, &format!("llama-protocol-v2-{suffix}")).unwrap();
    let nonce = suffix.clone();
    let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_ck-synapse-worker-llama"));
    #[cfg(unix)]
    command.arg("--socket").arg(&endpoint);
    #[cfg(windows)]
    command.arg("--pipe").arg(&endpoint);
    command.arg("--nonce").arg(&nonce).kill_on_drop(true);
    let mut child = command.spawn().expect("spawn the built llama worker");

    let mut stream = accept_worker_handshake_with_engine_and_protocol_version(
        listener,
        &nonce,
        MAX_FRAME,
        Duration::from_secs(30),
        Some(LLAMA_WORKER_ENGINE),
        None,
        None,
    )
    .await
    .expect("the llama HELLO at protocol v2 is accepted");

    ping(&mut stream, "ping-0").await;

    // RERANK_SEQUENCES carries a raw frame the worker must read and discard.
    write_json(
        &mut stream,
        &WorkerRequest::RerankSequences {
            req_id: "rerank-sequences-1".to_string(),
            model_ref: "llama:0".to_string(),
            sequences: vec![
                WorkerSequence { n_tokens: 3 },
                WorkerSequence { n_tokens: 2 },
            ],
        },
        MAX_FRAME,
    )
    .await
    .unwrap();
    write_raw(&mut stream, &encode_i32_frame(&[1, 10, 2, 1, 2]), MAX_FRAME)
        .await
        .unwrap();
    expect_unsupported(&mut stream, "rerank-sequences-1").await;
    ping(&mut stream, "ping-1").await;

    // ANE_ADMIT_SHAPE has no raw frame after its JSON request.
    write_json(
        &mut stream,
        &WorkerRequest::AneAdmitShape {
            req_id: "admit-1".to_string(),
            model_ref: "llama:0".to_string(),
            shape: 128,
        },
        MAX_FRAME,
    )
    .await
    .unwrap();
    expect_unsupported(&mut stream, "admit-1").await;
    ping(&mut stream, "ping-2").await;

    write_json(&mut stream, &WorkerRequest::Shutdown {}, MAX_FRAME)
        .await
        .unwrap();
    let status = tokio::time::timeout(Duration::from_secs(30), child.wait())
        .await
        .expect("the worker exits after SHUTDOWN")
        .unwrap();
    assert!(status.success(), "worker exit status {status}");
    let _ = std::fs::remove_dir_all(runtime_dir);
}
