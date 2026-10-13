use super::*;
use std::sync::atomic::AtomicBool;
use std::time::Duration;
use synapse_core::{
    decode_i32_frame, encode_f32_frame, encode_i32_frame, WorkerPooling, WorkerProgressPhase,
    WorkerTokenItem,
};
use tokio::io::DuplexStream;

async fn channel() -> (
    AneWorkerChannel<DuplexStream>,
    Arc<AtomicBool>,
    Arc<AtomicU64>,
) {
    let hung = Arc::new(AtomicBool::new(true));
    let starts = Arc::new(AtomicU64::new(0));
    let flag = hung.clone();
    let count = starts.clone();
    let connect: AneWorkerConnector<DuplexStream> = Box::new(move || {
        let flag = flag.clone();
        let count = count.clone();
        Box::pin(async move {
            count.fetch_add(1, Ordering::Relaxed);
            let (module, mut worker) = tokio::io::duplex(4096);
            let peer = tokio::spawn(async move {
                while let Ok(request) = read_json::<WorkerRequest, _>(
                    &mut worker,
                    synapse_core::DEFAULT_MAX_FRAME_BYTES,
                )
                .await
                {
                    let WorkerRequest::EmbedBatch { req_id, .. } = request else {
                        panic!("unexpected fake request");
                    };
                    let raw = read_raw(&mut worker, synapse_core::DEFAULT_MAX_FRAME_BYTES)
                        .await
                        .unwrap();
                    write_json(
                        &mut worker,
                        &WorkerResponse::Progress {
                            req_id: req_id.clone(),
                            completed: 1,
                            phase: WorkerProgressPhase::Inference,
                        },
                        synapse_core::DEFAULT_MAX_FRAME_BYTES,
                    )
                    .await
                    .unwrap();
                    if flag.load(Ordering::Relaxed) {
                        continue;
                    }
                    let value = decode_i32_frame(&raw).unwrap().iter().sum::<i32>() as f32;
                    write_json(
                        &mut worker,
                        &WorkerResponse::Vectors {
                            req_id,
                            dims: 1,
                            n: 1,
                        },
                        synapse_core::DEFAULT_MAX_FRAME_BYTES,
                    )
                    .await
                    .unwrap();
                    write_raw(
                        &mut worker,
                        &encode_f32_frame(&[value]),
                        synapse_core::DEFAULT_MAX_FRAME_BYTES,
                    )
                    .await
                    .unwrap();
                }
            });
            Ok(AneWorkerSession {
                stream: module,
                packed: None,
                confirm_exit: Box::pin(async move {
                    peer.await
                        .map_err(|error| AneResidencyError::Channel(error.to_string()))
                }),
            })
        })
    });
    (
        AneWorkerChannel::connect(
            "catalog-ane-test",
            synapse_core::DEFAULT_MAX_FRAME_BYTES,
            connect,
        )
        .await
        .unwrap(),
        hung,
        starts,
    )
}

async fn infer(
    channel: &AneWorkerChannel<DuplexStream>,
) -> Result<WorkerResponse, AneResidencyError> {
    let request = WorkerRequest::EmbedBatch {
        req_id: channel.next_req_id("embed"),
        model_ref: "model".into(),
        pooling: WorkerPooling::Last,
        normalize: true,
        items: vec![WorkerTokenItem {
            id: "0".into(),
            n_tokens: 1,
        }],
    };
    channel
        .exchange_inference(
            &request,
            Some(&encode_i32_frame(&[9])),
            None,
            Duration::from_millis(20),
            async { Ok(Box::new(()) as ExchangePermit) },
        )
        .await
        .map(|(response, _)| response)
}

#[tokio::test]
async fn two_consecutive_ane_hangs_use_catalog_quarantine() {
    let (channel, _, starts) = channel().await;
    for _ in 0..2 {
        assert!(matches!(
            infer(&channel).await,
            Err(AneResidencyError::Channel(_))
        ));
        channel.restart().await.unwrap();
    }
    assert_eq!(
        starts.load(Ordering::Relaxed),
        2,
        "quarantine confirms exit without spawning again"
    );
    let error = infer(&channel).await.unwrap_err();
    assert!(matches!(error, AneResidencyError::Quarantined { .. }));
    let wire = crate::ane_lane_error_to_wire(error, "catalog-ane-test");
    assert_eq!(wire.code, "engine_crashed");
    assert!(wire.retry_after_ms.unwrap() > 0);
    assert_eq!(
        wire.details,
        Some(serde_json::json!({"lane_id":"catalog-ane-test","quarantined":true}))
    );
}

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
