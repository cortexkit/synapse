use super::*;
use synapse_core::WorkerProgressPhase;
use tokio::sync::Mutex as AsyncMutex;

async fn live_host() -> (
    Arc<AsyncMutex<WorkerHost>>,
    LoadedModel,
    tokio::task::JoinHandle<()>,
) {
    live_host_with_work(45, Duration::from_secs(1)).await
}

async fn live_host_with_work(
    units: u64,
    interval: Duration,
) -> (
    Arc<AsyncMutex<WorkerHost>>,
    LoadedModel,
    tokio::task::JoinHandle<()>,
) {
    let (module, mut worker) = tokio::net::UnixStream::pair().unwrap();
    let mut host = WorkerHost::new(WorkerHostConfig::new("/bin/false", std::env::temp_dir()));
    host.connection = Some(WorkerConnection {
        stream: module,
        child: Command::from(synapse_core::without_launch_nonce(
            std::process::Command::new("sleep"),
        ))
        .arg("600")
        .kill_on_drop(true)
        .spawn()
        .unwrap(),
        logs: LogRing::new(),
        worker_generation: 1,
        packed: None,
    });
    let peer = tokio::spawn(async move {
        let mut inferences = 0;
        while let Ok(request) =
            read_json::<WorkerRequest, _>(&mut worker, DEFAULT_MAX_FRAME_BYTES).await
        {
            match request {
                WorkerRequest::Load { req_id, .. } => {
                    write_json(
                        &mut worker,
                        &WorkerResponse::Loaded {
                            req_id,
                            model_ref: "live".into(),
                            dims: 1,
                            cold_load_ms: 0,
                            buckets: None,
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
                    let value = synapse_core::decode_i32_frame(&raw)
                        .unwrap()
                        .iter()
                        .sum::<i32>() as f32;
                    inferences += 1;
                    if inferences == 1 {
                        for completed in 1..=units {
                            tokio::time::sleep(interval).await;
                            if write_json(
                                &mut worker,
                                &WorkerResponse::Progress {
                                    req_id: req_id.clone(),
                                    completed,
                                    phase: WorkerProgressPhase::Inference,
                                },
                                DEFAULT_MAX_FRAME_BYTES,
                            )
                            .await
                            .is_err()
                            {
                                return;
                            }
                        }
                    }
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
                        &synapse_core::encode_f32_frame(&[value]),
                        DEFAULT_MAX_FRAME_BYTES,
                    )
                    .await
                    .unwrap();
                }
                _ => panic!("unexpected fake request"),
            }
        }
    });
    let mut cfg = RuntimeConfig::default();
    cfg.values
        .insert("artifact_path".into(), "/tmp/live-fixture".into());
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
    (Arc::new(AsyncMutex::new(host)), model, peer)
}

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
async fn live_worker_deadline_drains_before_next_correct_response() {
    let (host, model, peer) = live_host().await;
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
    assert!(
        started.elapsed() < Duration::from_secs(1),
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
    assert_eq!(host.connection.as_ref().unwrap().worker_generation, 1);
    host.kill_current().await;
    peer.await.unwrap();
}
