#![cfg(unix)]

use std::{
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use synapse_core::{RuntimeConfig, TokenBatch, ValidatedArtifact};
use synapse_module::worker_host::{WorkerHost, WorkerHostConfig, WorkerHostError};

fn config(extra: &[&str]) -> (WorkerHostConfig, PathBuf) {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    // The worker's Unix socket lives in this folder, and macOS caps a socket
    // path at 104 bytes. Under the per-user temp folder (/var/folders/.../T/)
    // a long folder name pushes `wk-<digest>.sock` past that cap, so keep it short.
    let root = std::env::temp_dir().join(format!(
        "wl-{}-{:x}",
        std::process::id(),
        id % 0x1_0000_0000
    ));
    std::fs::create_dir_all(&root).unwrap();
    let binary = synapse_core::dev_binary::ckdev_binary(
        env!("CARGO_BIN_EXE_synapse-worker-timeout-mock"),
        &root,
    )
    .unwrap();
    let mut config = WorkerHostConfig::new(binary, &root);
    config.extra_args = extra.iter().map(|arg| (*arg).into()).collect();
    (config, root)
}

async fn load(host: &mut WorkerHost) -> synapse_core::LoadedModel {
    let mut cfg = RuntimeConfig::default();
    cfg.values
        .insert("artifact_path".into(), "/tmp/progress-fake-model".into());
    host.load_model(
        &ValidatedArtifact {
            digest: String::new(),
            format: "mock".into(),
        },
        &cfg,
    )
    .await
    .unwrap()
}

fn batch(value: u32) -> TokenBatch {
    TokenBatch {
        items: vec![vec![value]],
    }
}

#[tokio::test]
async fn slow_progress_worker_exceeds_30_seconds_without_crash_strike() {
    let (config, root) = config(&[
        "--embed-sleep-ms",
        "45000",
        "--progress-ms",
        "1000",
        "--echo-tokens",
        "1",
    ]);
    let mut host = WorkerHost::new(config);
    let model = load(&mut host).await;
    let started = Instant::now();
    let result = host.embed_batch(&model, batch(7)).await.unwrap();
    assert!(started.elapsed() >= Duration::from_secs(45));
    assert_eq!(result[0], vec![7.0]);
    assert_eq!(host.health_snapshot().crash_count_window, 0);
    assert_eq!(host.health_snapshot().quarantined_models, 0);
    drop(host);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn stalled_progress_worker_strikes_within_idle_window() {
    let (config, root) = config(&["--hang-on-embed", "1"]);
    let mut host = WorkerHost::new(config);
    let model = load(&mut host).await;
    let started = Instant::now();
    let error = tokio::time::timeout(Duration::from_secs(125), host.embed_batch(&model, batch(1)))
        .await
        .expect("idle watchdog must detect the stall")
        .unwrap_err();
    assert!(
        matches!(error, WorkerHostError::EngineCrashed { .. }),
        "{error:?}"
    );
    assert!(
        started.elapsed() >= Duration::from_secs(120),
        "a progress frame is not a terminal reply"
    );
    assert!(started.elapsed() < Duration::from_secs(125));
    assert_eq!(host.health_snapshot().crash_count_window, 1);
    drop(host);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn two_sequential_hangs_quarantine() {
    let (config, root) = config(&["--hang-on-embed", "1"]);
    let mut host = WorkerHost::new(config);
    let model = load(&mut host).await;
    for _ in 0..2 {
        let error = host.embed_batch(&model, batch(1)).await.unwrap_err();
        assert!(
            matches!(error, WorkerHostError::EngineCrashed { .. }),
            "{error:?}"
        );
    }
    assert_eq!(host.health_snapshot().crash_count_window, 2);
    assert!(matches!(
        host.embed_batch(&model, batch(1)).await,
        Err(WorkerHostError::Quarantined { .. })
    ));
    drop(host);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn hang_success_hang_does_not_quarantine() {
    let (mut config, root) = config(&["--echo-tokens", "1"]);
    let flag = root.join("behavior");
    config
        .extra_args
        .extend(["--hang-file".into(), flag.to_string_lossy().into_owned()]);
    std::fs::write(&flag, "hang").unwrap();
    let mut host = WorkerHost::new(config);
    let model = load(&mut host).await;
    assert!(matches!(
        host.embed_batch(&model, batch(1)).await,
        Err(WorkerHostError::EngineCrashed { .. })
    ));
    std::fs::write(&flag, "serve").unwrap();
    assert_eq!(
        host.embed_batch(&model, batch(9)).await.unwrap()[0],
        vec![9.0]
    );
    assert_eq!(
        host.health_snapshot().crash_count_window,
        0,
        "drained serving success must clear strikes"
    );
    std::fs::write(&flag, "hang").unwrap();
    assert!(matches!(
        host.embed_batch(&model, batch(1)).await,
        Err(WorkerHostError::EngineCrashed { .. })
    ));
    assert_eq!(host.health_snapshot().crash_count_window, 1);
    assert_eq!(host.health_snapshot().quarantined_models, 0);
    drop(host);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn cancelled_low_level_host_starts_fresh_worker_without_strike() {
    let (mut config, root) = config(&[
        "--embed-sleep-ms",
        "1000",
        "--progress-ms",
        "250",
        "--echo-tokens",
        "1",
    ]);
    let pids = root.join("pids");
    config
        .extra_args
        .extend(["--pid-log".into(), pids.to_string_lossy().into_owned()]);
    let mut host = WorkerHost::new(config);
    let model = load(&mut host).await;
    assert!(tokio::time::timeout(
        Duration::from_millis(100),
        host.embed_batch(&model, batch(7))
    )
    .await
    .is_err());
    let result = host.embed_batch(&model, batch(9)).await.unwrap();
    assert_eq!(
        result,
        vec![vec![9.0]],
        "a cancelled caller's result 7 must never reach caller 2"
    );
    let starts = std::fs::read_to_string(&pids).unwrap();
    let starts: Vec<_> = starts.lines().collect();
    assert_eq!(
        starts.len(),
        2,
        "the unusable connection must launch a fresh worker"
    );
    assert_ne!(starts[0], starts[1]);
    assert_eq!(
        host.health_snapshot().crash_count_window,
        0,
        "cancellation teardown is lifecycle, not a fault strike"
    );
    assert_eq!(host.health_snapshot().quarantined_models, 0);
    drop(host);
    std::fs::remove_dir_all(root).unwrap();
}
