use super::test_support::progress::*;
use super::*;

#[tokio::test]
async fn real_transport_progress_renews_multiple_idle_windows() {
    let window = Duration::from_millis(200);
    let (host, model, peer) = live_host_with_work(30, Duration::from_millis(50)).await;
    let mut host = host.lock().await;
    host.test_idle_window = Some(window);
    let started = Instant::now();
    let result = host
        .embed_batch(
            &model,
            TokenBatch {
                items: vec![vec![7]],
            },
        )
        .await
        .unwrap();
    assert!(
        started.elapsed() > window * 6,
        "real exchange must outlive several idle windows"
    );
    assert_eq!(result, vec![vec![7.0]]);
    assert_eq!(host.health_snapshot().crash_count_window, 0);
    assert_eq!(host.connection.as_ref().unwrap().worker_generation, 1);
    host.kill_current().await;
    peer.await.unwrap();
}

#[tokio::test]
async fn cuda_inference_transition_uses_short_window_before_first_layer() {
    let inference = Duration::from_millis(200);
    // The setup allowance is far longer than the 5 s bound below, so a host
    // that kept it after the inference transition fails the bound.
    let setup = Duration::from_secs(10);
    // CUDA now announces inference before entering the opaque native forward.
    // A blocked upload, plan setup or first layer must all use the short window.
    let (host, model, peer) = live_host_with_script(1, Duration::ZERO, true).await;
    let mut host = host.lock().await;
    host.config.engine_identity = Some(EngineIdentity {
        engine: synapse_core::worker_engine_names::CUDA_WORKER_ENGINE.into(),
        version: "test".into(),
        build_flags: BTreeMap::new(),
    });
    host.config.crash_budget.max_crashes = 1;
    host.test_idle_window = Some(inference);
    host.test_setup_window = Some(setup);
    let started = Instant::now();
    let error = tokio::time::timeout(
        Duration::from_secs(5),
        host.embed_batch(
            &model,
            TokenBatch {
                items: vec![vec![7]],
            },
        ),
    )
    .await
    .expect("inference transition must not retain the setup allowance")
    .unwrap_err();
    assert!(matches!(error, WorkerHostError::EngineCrashed { .. }));
    assert!(started.elapsed() >= inference && started.elapsed() < Duration::from_secs(5));
    assert_eq!(host.health_snapshot().crash_count_window, 1);
    assert!(host.connection.is_none());
    peer.await.unwrap();
}

#[tokio::test]
async fn cuda_cold_setup_keeps_distinct_initial_silence_allowance() {
    let (host, model, peer) = live_host_with_work(1, Duration::from_millis(400)).await;
    let mut host = host.lock().await;
    host.config.engine_identity = Some(EngineIdentity {
        engine: synapse_core::worker_engine_names::CUDA_WORKER_ENGINE.into(),
        version: "test".into(),
        build_flags: BTreeMap::new(),
    });
    host.test_idle_window = Some(Duration::from_millis(200));
    host.test_setup_window = Some(Duration::from_secs(2));
    assert_eq!(
        host.embed_batch(
            &model,
            TokenBatch {
                items: vec![vec![7]]
            }
        )
        .await
        .unwrap(),
        vec![vec![7.0]]
    );
    assert_eq!(host.health_snapshot().crash_count_window, 0);
    host.kill_current().await;
    peer.await.unwrap();
}
