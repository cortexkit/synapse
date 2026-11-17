use super::super::*;
use synapse_core::{decode_i32_frame, encode_f32_frame};

pub(crate) fn host(catalog: bool) -> (WorkerHost, PathBuf) {
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
pub(crate) fn peer(
    host: &mut WorkerHost,
    generation: u64,
    crash: bool,
) -> tokio::task::JoinHandle<()> {
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

pub fn batch() -> TokenBatch {
    TokenBatch {
        items: vec![vec![9]],
    }
}

pub(crate) async fn load(host: &mut WorkerHost) -> LoadedModel {
    let mut cfg = RuntimeConfig::default();
    cfg.values
        .insert("artifact_path".into(), "/tmp/quarantine-fixture".into());
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

pub async fn quarantined_host(catalog: bool) -> (WorkerHost, LoadedModel, PathBuf) {
    let (mut host, root) = host(catalog);
    let first = peer(&mut host, 1, true);
    let model = load(&mut host).await;
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
