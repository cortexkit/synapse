#![cfg(target_os = "macos")]

use std::{
    io::Write,
    os::unix::net::UnixListener,
    path::PathBuf,
    process::Command,
    sync::{Mutex, MutexGuard, OnceLock},
    time::SystemTime,
};

use synapse_core::{
    worker_framing_sync::{read_frame, write_frame, write_json_frame},
    worker_protocol::{
        WorkerHello, WorkerHelloAck, WorkerResponse, DEFAULT_MAX_FRAME_BYTES,
        WORKER_PROTOCOL_VERSION,
    },
};
use synapse_worker_host::worker_host::{WorkerEngine, WorkerHostConfig};

fn worker_process_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

fn runtime_dir(label: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    PathBuf::from(format!("/tmp/sdw-{label}-{}-{suffix}", std::process::id()))
}

#[test]
fn fleet_binary_version_uses_decode_worker_name() {
    let output = Command::new(
        synapse_core::dev_binary::ckdev_binary(
            env!("CARGO_BIN_EXE_ck-synapse-worker-decode"),
            std::env::temp_dir().join(format!("decode-transport-{}", std::process::id())),
        )
        .unwrap(),
    )
    .arg("--version")
    .output()
    .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("ck-synapse-worker-decode "));
}

#[test]
fn worker_completes_standard_nonce_handshake_and_ping() {
    let _worker_process_guard = worker_process_lock();
    let runtime_dir = runtime_dir("ping");
    std::fs::create_dir_all(&runtime_dir).unwrap();
    let config = WorkerHostConfig::new(
        synapse_core::dev_binary::ckdev_binary(
            env!("CARGO_BIN_EXE_ck-synapse-worker-decode"),
            std::env::temp_dir().join(format!("decode-transport-{}", std::process::id())),
        )
        .unwrap(),
        &runtime_dir,
    );
    let engine = WorkerEngine::new(config).unwrap();
    let ping = engine.ping().unwrap();
    assert_eq!(ping.models_loaded, 0);
    drop(engine);
    let _ = std::fs::remove_dir_all(runtime_dir);
}

#[test]
fn normal_requests_cannot_select_a_certification_forced_fault() {
    let _worker_process_guard = worker_process_lock();
    let runtime_dir = runtime_dir("certification-fault-refusal");
    std::fs::create_dir_all(&runtime_dir).unwrap();
    let socket_path = runtime_dir.join("worker.sock");
    let listener = UnixListener::bind(&socket_path).unwrap();
    let mut child = Command::new(
        synapse_core::dev_binary::ckdev_binary(
            env!("CARGO_BIN_EXE_ck-synapse-worker-decode"),
            std::env::temp_dir().join(format!("decode-transport-{}", std::process::id())),
        )
        .unwrap(),
    )
    .arg("--socket")
    .arg(&socket_path)
    .arg("--nonce")
    .arg("0123456789abcdef")
    .env_remove("CK_ANE_PREFILL_CERTIFICATION_PROBE")
    .spawn()
    .unwrap();
    let (mut stream, _) = listener.accept().unwrap();
    let hello: WorkerHello =
        serde_json::from_slice(&read_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap()).unwrap();
    assert_eq!(hello.nonce, "0123456789abcdef");
    write_json_frame(
        &mut stream,
        &WorkerHelloAck {
            v: WORKER_PROTOCOL_VERSION,
            accept: true,
            max_frame: DEFAULT_MAX_FRAME_BYTES,
        },
        DEFAULT_MAX_FRAME_BYTES,
    )
    .unwrap();
    write_frame(
        &mut stream,
        &serde_json::to_vec(&serde_json::json!({
            "type": "PING",
            "req_id": "normal-request",
            "forced_fault": "handoff_budget",
        }))
        .unwrap(),
        DEFAULT_MAX_FRAME_BYTES,
    )
    .unwrap();
    let response: WorkerResponse =
        serde_json::from_slice(&read_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap()).unwrap();
    match response {
        WorkerResponse::Err { req_id, code, .. } => {
            assert_eq!(req_id.as_deref(), Some("normal-request"));
            assert_eq!(code, "certification_probe_required");
        }
        other => panic!("normal request unexpectedly reached the certification probe: {other:?}"),
    }
    write_frame(
        &mut stream,
        &serde_json::to_vec(&serde_json::json!({
            "type": "CERTIFICATION_ANE_PREFILL_FALLBACK_PROBE",
            "req_id": "disabled-probe",
            "case_id": "artifact_warm",
            "prompt_ids": [1],
        }))
        .unwrap(),
        DEFAULT_MAX_FRAME_BYTES,
    )
    .unwrap();
    let disabled_probe: serde_json::Value =
        serde_json::from_slice(&read_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES).unwrap()).unwrap();
    assert_eq!(
        disabled_probe
            .get("type")
            .and_then(serde_json::Value::as_str),
        Some("CERTIFICATION_ANE_PREFILL_FALLBACK_TIMING_ERROR")
    );
    assert_eq!(
        disabled_probe
            .get("code")
            .and_then(serde_json::Value::as_str),
        Some("certification_probe_disabled")
    );
    assert!(disabled_probe.get("consumed_components_ms").is_none());
    assert!(disabled_probe.get("total_ttft_ms").is_none());
    write_frame(
        &mut stream,
        &serde_json::to_vec(&serde_json::json!({ "type": "SHUTDOWN" })).unwrap(),
        DEFAULT_MAX_FRAME_BYTES,
    )
    .unwrap();
    let _ = stream.flush();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(runtime_dir);
}
