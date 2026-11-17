use super::ane_quarantine_support::*;
use super::*;

#[tokio::test]
async fn ane_hang_success_hang_does_not_quarantine() {
    let (channel, hung, _) = channel().await;
    assert!(matches!(
        infer(&channel).await,
        Err(AneResidencyError::Channel(_))
    ));
    channel.restart().await.unwrap();
    hung.store(false, Ordering::Relaxed);
    assert!(matches!(
        infer(&channel).await.unwrap(),
        WorkerResponse::Vectors { .. }
    ));
    assert_eq!(channel.faults.lock().unwrap().strikes, 0);
    hung.store(true, Ordering::Relaxed);
    assert!(matches!(
        infer(&channel).await,
        Err(AneResidencyError::Channel(_))
    ));
    assert_eq!(channel.faults.lock().unwrap().strikes, 1);
    assert_eq!(channel.cooldown_remaining(), None);
    channel.shutdown().await.unwrap();
}
