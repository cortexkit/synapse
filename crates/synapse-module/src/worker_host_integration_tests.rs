use crate::worker_host::ane_residency::{AneResidencyError, AneShapeWorker};
use crate::worker_host::test_support::{progress::*, quarantine::*};
use crate::worker_host::*;
use crate::*;
use std::time::{Duration, Instant};
use synapse_core::{EngineErrorStage, TokenBatch};

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
async fn quarantined_legacy_lane_keeps_probe_required_without_auto_recovery() {
    let (mut host, model, root) = quarantined_host(false).await;
    host.advance_clock_for_test(CATALOG_QUARANTINE_COOLDOWN);
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
async fn live_worker_deadline_drains_before_next_correct_response() {
    // Three 50 ms progress units: the work outlasts the 100 ms deadline.
    let (host, model, peer) = live_host_with_work(3, Duration::from_millis(50)).await;
    let owner_host = Arc::clone(&host);
    let first_model = model.clone();
    let started = Instant::now();
    let owner = tokio::spawn(async move {
        owner_host
            .lock()
            .await
            .embed_batch(
                &first_model,
                TokenBatch {
                    items: vec![vec![7]],
                },
            )
            .await
            .map_err(|error| error.to_engine_error(EngineErrorStage::Inference))
    });
    let error = crate::await_worker_completion(
        owner,
        Some(tokio::time::Instant::now() + Duration::from_millis(100)),
        None,
    )
    .await
    .unwrap_err();
    assert_eq!(error.code, "deadline_exceeded");
    // The deadline_exceeded code is what shows the caller was answered before
    // the work finished; the bound only catches a hang on a loaded runner.
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "deadline must answer before native work finishes"
    );
    let mut host = host.lock().await;
    let result = host
        .embed_batch(
            &model,
            TokenBatch {
                items: vec![vec![9]],
            },
        )
        .await
        .unwrap();
    assert_eq!(
        result,
        vec![vec![9.0]],
        "the second caller must not receive the abandoned result 7"
    );
    assert_eq!(host.health_snapshot().crash_count_window, 0);
    assert_eq!(host.worker_generation_for_test(), 1);
    host.shutdown_for_test().await;
    peer.await.unwrap();
}

#[tokio::test]
async fn two_consecutive_ane_hangs_use_catalog_quarantine() {
    let (channel, _, starts) = worker_host::ane_residency::ane_quarantine_support::channel().await;
    for _ in 0..2 {
        assert!(matches!(
            worker_host::ane_residency::ane_quarantine_support::infer(&channel).await,
            Err(AneResidencyError::Channel(_))
        ));
        channel.restart().await.unwrap();
    }
    assert_eq!(
        starts.load(Ordering::Relaxed),
        2,
        "quarantine confirms exit without spawning again"
    );
    let error = worker_host::ane_residency::ane_quarantine_support::infer(&channel)
        .await
        .unwrap_err();
    assert!(matches!(error, AneResidencyError::Quarantined { .. }));
    let wire = crate::ane_lane_error_to_wire(error, "catalog-ane-test");
    assert_eq!(wire.code, "engine_crashed");
    assert!(wire.retry_after_ms.unwrap() > 0);
    assert_eq!(
        wire.details,
        Some(serde_json::json!({"lane_id":"catalog-ane-test","quarantined":true}))
    );
}
