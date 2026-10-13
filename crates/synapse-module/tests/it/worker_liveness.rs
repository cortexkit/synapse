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
    let root = std::env::temp_dir().join(format!("worker-liveness-{}-{id}", std::process::id()));
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
