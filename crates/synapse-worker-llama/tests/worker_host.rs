use std::path::PathBuf;
use std::time::Duration;

use synapse_core::{RuntimeConfig, ValidatedArtifact, WorkerPooling};
use synapse_module::worker_host::{CrashBudget, WorkerHost, WorkerHostConfig, WorkerHostError};

#[tokio::test]
async fn host_classifies_crashes_and_quarantines_after_budget() {
    let mut config = worker_config("abort-budget");
    config
        .extra_args
        .push("--test-abort-on-request".to_string());
    config.request_timeout = Duration::from_secs(2);
    config.crash_budget = CrashBudget { max_crashes: 2 };
    let mut host = WorkerHost::new(config);
    let mut runtime_config = RuntimeConfig::default();
    runtime_config.values.insert(
        "artifact_path".to_string(),
        "/tmp/abort-before-load.gguf".to_string(),
    );

    let first = host.load_model(&gguf_artifact(), &runtime_config).await;
    assert!(matches!(first, Err(WorkerHostError::EngineCrashed { .. })));
    let second = host.load_model(&gguf_artifact(), &runtime_config).await;
    assert!(matches!(second, Err(WorkerHostError::EngineCrashed { .. })));
    let third = host.load_model(&gguf_artifact(), &runtime_config).await;
    assert!(matches!(third, Err(WorkerHostError::Quarantined { .. })));
}

fn worker_config(worker_id: &str) -> WorkerHostConfig {
    let mut config = WorkerHostConfig::new(
        synapse_core::dev_binary::ckdev_binary(
            env!("CARGO_BIN_EXE_ck-synapse-worker-llama"),
            PathBuf::from(format!("/tmp/synw-{}", std::process::id())),
        )
        .unwrap(),
        PathBuf::from(format!("/tmp/synw-{}", std::process::id())),
    );
    config.worker_id = format!("{worker_id}-{}", short_suffix());
    config.pooling = WorkerPooling::Mean;
    config.normalize = true;
    config.handshake_timeout = Duration::from_secs(5);
    config.request_timeout = Duration::from_secs(120);
    config
}

fn gguf_artifact() -> ValidatedArtifact {
    ValidatedArtifact {
        digest: String::new(),
        format: "gguf".to_string(),
    }
}

fn short_suffix() -> String {
    format!(
        "{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    )
}
