#![cfg(unix)]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cortexkit_log::{Config, Lane, Retention};
use synapse_core::{RuntimeConfig, ValidatedArtifact};
use synapse_module::worker_host::{WorkerHost, WorkerHostConfig};
use synapse_module::LOG_TAGS;

fn unique_root(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "synapse-{label}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ))
}

static RUNTIME_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn runtime_root() -> PathBuf {
    PathBuf::from(format!(
        "/tmp/sw-{}-{}",
        std::process::id(),
        RUNTIME_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ))
}

fn runtime_config() -> RuntimeConfig {
    let mut config = RuntimeConfig::default();
    config
        .values
        .insert("artifact_path".to_string(), "/tmp/mock-model".to_string());
    config
}

async fn wait_for_log(path: &std::path::Path, needle: &str) -> String {
    for _ in 0..100 {
        let contents = fs::read_to_string(path).unwrap_or_default();
        if contents.contains(needle) {
            return contents;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    panic!("timed out waiting for '{needle}' in {}", path.display());
}

async fn start_fixture_worker(
    label: &str,
    model_id: &str,
    cap: u32,
    stderr_lines: usize,
    final_delay_ms: Option<u64>,
) -> WorkerHost {
    let mut config = WorkerHostConfig::new(
        env!("CARGO_BIN_EXE_synapse-worker-timeout-mock"),
        runtime_root(),
    );
    config.worker_id = label.to_string();
    config.model_id = Some(model_id.to_string());
    config.worker_forward_lines_per_sec = cap;
    config.handshake_timeout = Duration::from_secs(5);
    config.load_timeout = Duration::from_secs(5);
    config.extra_args = vec!["--stderr-lines".to_string(), stderr_lines.to_string()];
    if let Some(delay_ms) = final_delay_ms {
        config
            .extra_args
            .extend(["--stderr-final-delay-ms".to_string(), delay_ms.to_string()]);
    }
    let mut host = WorkerHost::new(config);
    host.load_model(
        &ValidatedArtifact {
            digest: String::new(),
            format: "mock".to_string(),
        },
        &runtime_config(),
    )
    .await
    .expect("fixture worker loads");
    host
}

#[tokio::test]
async fn worker_lines_are_forwarded_and_rate_limited() {
    let log_root = unique_root("worker-log");
    cortexkit_log::declared_tags(LOG_TAGS);
    let handle = cortexkit_log::init(Config {
        module_id: "synapse".to_string(),
        logs_dir: log_root,
        lane: Lane::Module,
        spec: Some("info".to_string()),
        retention: Retention::default(),
        redactor: None,
        clock: Some(Arc::new(SystemTime::now)),
    })
    .expect("fleet logger initializes");

    let _three = start_fixture_worker("worker-three", "model-three", 50, 3, None).await;
    let contents = wait_for_log(handle.path(), "fixture stderr line 2").await;
    let three_lines = contents
        .lines()
        .filter(|line| line.contains("tag=worker") && line.contains("worker=worker-three"))
        .collect::<Vec<_>>();
    assert_eq!(three_lines.len(), 3, "{three_lines:#?}");
    assert!(three_lines
        .iter()
        .all(|line| { line.contains("stream=stderr") && line.contains("model_id=model-three") }));

    let _flood = start_fixture_worker("worker-flood", "model-flood", 3, 500, Some(1_100)).await;
    let contents = wait_for_log(handle.path(), "fixture stderr after flood").await;
    let flood_lines = contents
        .lines()
        .filter(|line| line.contains("tag=worker") && line.contains("worker=worker-flood"))
        .collect::<Vec<_>>();
    let initial_forwarded = flood_lines
        .iter()
        .filter(|line| line.contains("fixture stderr line"))
        .count();
    assert!(initial_forwarded <= 3, "{flood_lines:#?}");
    assert!(initial_forwarded > 0, "{flood_lines:#?}");
    let next_line = flood_lines
        .iter()
        .find(|line| line.contains("fixture stderr after flood"))
        .expect("post-window line is forwarded");
    assert!(next_line.contains("dropped="), "{next_line}");
    assert!(next_line.contains("stream=stderr"), "{next_line}");
    assert!(next_line.contains("model_id=model-flood"), "{next_line}");
}
