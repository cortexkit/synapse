use super::super::*;
use synapse_core::WorkerProgressPhase;
use tokio::sync::Mutex as AsyncMutex;

pub async fn live_host_with_work(
    units: u64,
    interval: Duration,
) -> (
    Arc<AsyncMutex<WorkerHost>>,
    LoadedModel,
    tokio::task::JoinHandle<()>,
) {
    live_host_with_script(units, interval, false).await
}

pub(crate) async fn live_host_with_script(
    units: u64,
    interval: Duration,
    stall_after_progress: bool,
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
        max_frame: synapse_core::DEFAULT_MAX_FRAME_BYTES,
        unusable: false,
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
                            ane_cached_shapes: None,
                            transcribe_revision: None,
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
                        if stall_after_progress {
                            // Stop request progress while still observing EOF,
                            // so owned-child teardown is independently confirmed.
                            let _ =
                                read_json::<WorkerRequest, _>(&mut worker, DEFAULT_MAX_FRAME_BYTES)
                                    .await;
                            return;
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
