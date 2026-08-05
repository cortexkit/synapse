#![cfg(target_os = "macos")]

use std::{path::PathBuf, process::Command, time::SystemTime};

use synapse_module::worker_host::{WorkerEngine, WorkerHostConfig};

fn runtime_dir(label: &str) -> PathBuf {
    let suffix = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    PathBuf::from(format!("/tmp/sdw-{label}-{}-{suffix}", std::process::id()))
}

#[test]
fn fleet_binary_version_uses_decode_worker_name() {
    let output = Command::new(env!("CARGO_BIN_EXE_ck-synapse-worker-decode"))
        .arg("--version")
        .output()
        .unwrap();
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("ck-synapse-worker-decode "));
}

#[test]
fn worker_completes_standard_nonce_handshake_and_ping() {
    let runtime_dir = runtime_dir("ping");
    std::fs::create_dir_all(&runtime_dir).unwrap();
    let config =
        WorkerHostConfig::new(env!("CARGO_BIN_EXE_ck-synapse-worker-decode"), &runtime_dir);
    let engine = WorkerEngine::new(config).unwrap();
    let ping = engine.ping().unwrap();
    assert_eq!(ping.models_loaded, 0);
    drop(engine);
    let _ = std::fs::remove_dir_all(runtime_dir);
}
