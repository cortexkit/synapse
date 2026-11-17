use super::test_support::quarantine::*;
use super::*;

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
async fn catalog_unload_clears_quarantine_without_waiting_for_cooldown() {
    let (mut host, model, root) = quarantined_host(true).await;
    host.unload(&model).await.unwrap();
    assert_eq!(host.health_snapshot().quarantined_models, 0);
    assert!(host.quarantine_until.is_empty());
    assert!(host.crashes.is_empty());
    std::fs::remove_dir_all(root).unwrap();
}

// Strikes count consecutive faults: a request served between two faults
// resets the count, so fault, success, fault leaves one strike and no
// quarantine.
#[tokio::test]
async fn a_served_request_between_two_faults_clears_strikes_without_quarantine() {
    let (mut host, root) = host(false);
    let first = peer(&mut host, 1, true);
    let model = load(&mut host).await;
    assert!(matches!(
        host.embed_batch(&model, batch()).await,
        Err(WorkerHostError::EngineCrashed { .. })
    ));
    first.await.unwrap();
    assert_eq!(host.health_snapshot().crash_count_window, 1);

    let serving = peer(&mut host, 2, false);
    assert_eq!(
        host.embed_batch(&model, batch()).await.unwrap(),
        vec![vec![9.0]]
    );
    assert_eq!(
        host.health_snapshot().crash_count_window,
        0,
        "a served request must clear strikes"
    );
    host.kill_current().await;
    serving.await.unwrap();

    let last = peer(&mut host, 3, true);
    assert!(matches!(
        host.embed_batch(&model, batch()).await,
        Err(WorkerHostError::EngineCrashed { .. })
    ));
    last.await.unwrap();
    assert_eq!(host.health_snapshot().crash_count_window, 1);
    assert_eq!(host.health_snapshot().quarantined_models, 0);
    std::fs::remove_dir_all(root).unwrap();
}
