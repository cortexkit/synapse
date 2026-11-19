use super::test_support::*;
use super::*;
use std::time::Duration;

/// A caller that goes away while its request waits for the worker stream,
/// before anything reached the wire, must withdraw that request: it must not
/// run once the stream frees up.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_caller_withdraws_queued_inference() {
    let (engine, control) = module_mock_engine_controlled();
    let sequence = || vec![vec![7u32; 10]];
    // Make the shape resident first, so later requests wait only for the stream.
    engine.infer(sequence(), false, None).await.unwrap();
    assert_eq!(control.inferences(), 1);

    let (started, release) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
    control.hold_next_inference(started.clone(), release.clone());
    let holding = tokio::spawn({
        let engine = engine.clone();
        async move { engine.infer(sequence(), false, None).await }
    });
    started.notified().await;

    // This one queues behind the held inference, then its caller goes away.
    let queued = tokio::spawn({
        let engine = engine.clone();
        async move { engine.infer(sequence(), false, None).await }
    });
    tokio::time::sleep(Duration::from_millis(200)).await;
    queued.abort();
    let _ = queued.await;

    release.notify_one();
    holding.await.unwrap().unwrap();
    // The stream serves waiters in order, so a withdrawn request that still
    // ran would do so before this one.
    engine.infer(sequence(), false, None).await.unwrap();
    assert_eq!(
        control.inferences(),
        3,
        "the abandoned request reached the worker after its caller went away"
    );
}
