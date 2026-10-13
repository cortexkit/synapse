#![cfg(unix)]

use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use synapse_core::{RuntimeConfig, TokenBatch, ValidatedArtifact};
use synapse_module::worker_host::{WorkerHost, WorkerHostConfig};

#[tokio::test]
async fn catalog_inference_uses_fixed_idle_window_not_owned_decode_timeout() {
    let short_timeout = Duration::from_millis(50);
    let sleep = Duration::from_millis(125);
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after the Unix epoch")
        .as_nanos();
    let mut config = WorkerHostConfig::new(
        synapse_core::dev_binary::ckdev_binary(
            env!("CARGO_BIN_EXE_synapse-worker-timeout-mock"),
            PathBuf::from(format!(
                "/tmp/synapse-timeout-{}-{suffix}",
                std::process::id()
            )),
        )
        .unwrap(),
        PathBuf::from(format!(
            "/tmp/synapse-timeout-{}-{suffix}",
            std::process::id()
        )),
    );
    config.worker_id = format!("timeout-test-{}-{suffix}", std::process::id());
    config.handshake_timeout = Duration::from_secs(30);
    config.request_timeout = short_timeout;

    config.extra_args = vec![
        "--load-sleep-ms".to_string(),
        sleep.as_millis().to_string(),
        "--embed-sleep-ms".to_string(),
        sleep.as_millis().to_string(),
    ];

    let mut host = WorkerHost::new(config);
    let mut runtime_config = RuntimeConfig::default();
    runtime_config
        .values
        .insert("artifact_path".to_string(), "/tmp/mock-model".to_string());
    let model = host
        .load_model(
            &ValidatedArtifact {
                digest: String::new(),
                format: "mock".to_string(),
            },
            &runtime_config,
        )
        .await
        .expect("LOAD should use the long timeout budget");

    let result = host
        .embed_batch(
            &model,
            TokenBatch {
                items: vec![vec![1]],
            },
        )
        .await
        .expect("catalog inference has no configurable total-time cap");
    assert_eq!(result, vec![vec![1.0]]);
    assert_eq!(host.health_snapshot().crash_count_window, 0);
}
