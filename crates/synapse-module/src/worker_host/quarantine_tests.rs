use super::*;
use synapse_core::{decode_i32_frame, encode_f32_frame};

fn host(catalog: bool) -> (WorkerHost, PathBuf) {
    let root = std::env::temp_dir().join(format!(
        "quarantine-{}-{}",
        std::process::id(),
        nonce_hex16()
    ));
    let mut config = WorkerHostConfig::new("/bin/false", &root);
    config.handshake_timeout = Duration::from_millis(20);
    config.catalog_lane = catalog.then(|| "catalog-test-vulkan".into());
    let mut host = WorkerHost::new(config);
    host.clock = Some(Instant::now());
    (host, root)
}

// Use the host's actual framed stream and crash handling. A faulting peer
// closes after reading inference; its successful LOAD must not hide that fault.
fn peer(host: &mut WorkerHost, generation: u64, crash: bool) -> tokio::task::JoinHandle<()> {
    let (module, mut worker) = tokio::net::UnixStream::pair().unwrap();
    host.connection = Some(WorkerConnection {
        stream: module,
        child: Command::from(synapse_core::without_launch_nonce(
            std::process::Command::new("sleep"),
        ))
        .arg("30")
        .kill_on_drop(true)
        .spawn()
        .unwrap(),
        logs: LogRing::new(),
        worker_generation: generation,
        packed: None,
        unusable: false,
    });
    tokio::spawn(async move {
        while let Ok(request) =
            read_json::<WorkerRequest, _>(&mut worker, DEFAULT_MAX_FRAME_BYTES).await
        {
            match request {
                WorkerRequest::Load { req_id, .. } => {
                    write_json(
                        &mut worker,
                        &WorkerResponse::Loaded {
                            req_id,
                            model_ref: "mock-ref".into(),
                            dims: 1,
                            cold_load_ms: 0,
                            buckets: None,
                            ane_cached_shapes: None,
                        },
                        DEFAULT_MAX_FRAME_BYTES,
                    )
                    .await
                    .unwrap();
                }
                WorkerRequest::EmbedBatch { req_id, .. } => {
                    let raw = read_raw(&mut worker, DEFAULT_MAX_FRAME_BYTES)
                        .await
                        .unwrap();
                    if crash {
                        return;
                    }
                    let value = decode_i32_frame(&raw).unwrap().iter().sum::<i32>() as f32;
                    write_json(
                        &mut worker,
                        &WorkerResponse::Vectors {
                            req_id,
                            dims: 1,
                            n: 1,
                        },
                        DEFAULT_MAX_FRAME_BYTES,
                    )
                    .await
                    .unwrap();
                    write_raw(
                        &mut worker,
                        &encode_f32_frame(&[value]),
                        DEFAULT_MAX_FRAME_BYTES,
                    )
                    .await
                    .unwrap();
                }
                _ => panic!("unexpected fake request"),
            }
        }
    })
}

fn batch() -> TokenBatch {
    TokenBatch {
        items: vec![vec![9]],
    }
}

async fn quarantined_host(catalog: bool) -> (WorkerHost, LoadedModel, PathBuf) {
    let (mut host, root) = host(catalog);
    let first = peer(&mut host, 1, true);
    let mut cfg = RuntimeConfig::default();
    cfg.values
        .insert("artifact_path".into(), "/tmp/quarantine-fixture".into());
    let model = host
        .load_model(
            &ValidatedArtifact {
                digest: String::new(),
                format: "mock".into(),
            },
            &cfg,
        )
        .await
        .unwrap();
    assert!(matches!(
        host.embed_batch(&model, batch()).await,
        Err(WorkerHostError::EngineCrashed { .. })
    ));
    first.await.unwrap();
    let second = peer(&mut host, 2, true);
    assert!(matches!(
        host.embed_batch(&model, batch()).await,
        Err(WorkerHostError::EngineCrashed { .. })
    ));
    second.await.unwrap();
    assert_eq!(host.health_snapshot().quarantined_models, 1);
    (host, model, root)
}

#[tokio::test]
async fn quarantined_catalog_lane_returns_transient_engine_crashed_with_remaining_cooldown() {
    let (mut host, model, root) = quarantined_host(true).await;
    let before = host.request_count();
    let error = host.embed_batch(&model, batch()).await.unwrap_err();
    let wire = crate::worker_error_to_wire(
        error.to_engine_error(EngineErrorStage::Inference),
        Some("catalog-test-vulkan"),
    );
    assert_eq!(wire.code, "engine_crashed");
    assert!(matches!(wire.class, crate::ErrorClass::Transient));
    assert_eq!(wire.retry_after_ms, Some(600_000));
    assert_eq!(
        wire.details,
        Some(serde_json::json!({"lane_id":"catalog-test-vulkan","quarantined":true}))
    );
    assert!(wire.safe_to_retry_same_request);
    assert_eq!(
        host.request_count(),
        before,
        "quarantine must refuse without worker dispatch"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn catalog_quarantine_cooldown_reopens_serving_with_zero_strikes() {
    let (mut host, model, root) = quarantined_host(true).await;
    let start = host.clock.unwrap();
    host.clock = Some(start + CATALOG_QUARANTINE_COOLDOWN - Duration::from_millis(1));
    let refused = host.embed_batch(&model, batch()).await.unwrap_err();
    assert!(matches!(
        refused,
        WorkerHostError::Quarantined {
            retry_after_ms: Some(1),
            ..
        }
    ));
    host.clock = Some(start + CATALOG_QUARANTINE_COOLDOWN);
    let fresh = peer(&mut host, 3, false);
    let before = host.request_count();
    assert_eq!(
        host.embed_batch(&model, batch()).await.unwrap(),
        vec![vec![9.0]]
    );
    assert_eq!(
        host.request_count(),
        before + 2,
        "the fresh generation reloads then serves"
    );
    assert_eq!(host.health_snapshot().quarantined_models, 0);
    assert_eq!(host.health_snapshot().crash_count_window, 0);
    assert_eq!(host.connection.as_ref().unwrap().worker_generation, 3);
    host.kill_current().await;
    fresh.await.unwrap();
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn quarantined_legacy_lane_keeps_probe_required_without_auto_recovery() {
    let (mut host, model, root) = quarantined_host(false).await;
    host.clock = Some(host.clock.unwrap() + CATALOG_QUARANTINE_COOLDOWN);
    let error = host.embed_batch(&model, batch()).await.unwrap_err();
    let wire =
        crate::worker_error_to_wire(error.to_engine_error(EngineErrorStage::Inference), None);
    assert_eq!(wire.code, "probe_required");
    assert!(matches!(wire.class, crate::ErrorClass::Permanent));
    assert_eq!(wire.retry_after_ms, None);
    assert_eq!(wire.details, None);
    std::fs::remove_dir_all(root).unwrap();
}

#[tokio::test]
async fn catalog_unload_clears_quarantine_without_waiting_for_cooldown() {
    let (mut host, model, root) = quarantined_host(true).await;
    host.unload(&model).await.unwrap();
    assert_eq!(host.health_snapshot().quarantined_models, 0);
    assert!(host.quarantine_until.is_empty());
    assert!(host.crashes.is_empty());
    std::fs::remove_dir_all(root).unwrap();
}
