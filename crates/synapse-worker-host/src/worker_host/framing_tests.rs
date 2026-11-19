use super::*;
use synapse_core::{encode_f32_frame, WorkerProgressPhase};
use tokio::io::AsyncWriteExt;

#[derive(Clone, Copy)]
enum Corruption {
    FragmentedProgress,
    ShortRawEof,
    ExtraRaw,
    ProgressInsideRawPair,
    LateRawAfterHeader,
}

async fn malformed_reply_faults(corruption: Corruption) {
    let (module, mut worker) = tokio::net::UnixStream::pair().unwrap();
    let mut config = WorkerHostConfig::new("/bin/false", std::env::temp_dir());
    config.crash_budget.max_crashes = 1;
    let mut host = WorkerHost::new(config);
    let window = Duration::from_millis(200);
    host.test_idle_window = Some(window);
    host.connection = Some(WorkerConnection {
        stream: module,
        child: Command::from(synapse_core::without_launch_nonce(
            std::process::Command::new("sleep"),
        ))
        .arg("60")
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
        let load: WorkerRequest = read_json(&mut worker, DEFAULT_MAX_FRAME_BYTES)
            .await
            .unwrap();
        let WorkerRequest::Load { req_id, .. } = load else {
            panic!("expected load");
        };
        write_json(
            &mut worker,
            &WorkerResponse::Loaded {
                req_id,
                model_ref: "mock".into(),
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
        let request: WorkerRequest = read_json(&mut worker, DEFAULT_MAX_FRAME_BYTES)
            .await
            .unwrap();
        let WorkerRequest::EmbedBatch { req_id, .. } = request else {
            panic!("expected inference");
        };
        read_raw(&mut worker, DEFAULT_MAX_FRAME_BYTES)
            .await
            .unwrap();
        let progress = WorkerResponse::Progress {
            req_id: req_id.clone(),
            completed: 1,
            phase: WorkerProgressPhase::Inference,
        };
        if matches!(corruption, Corruption::FragmentedProgress) {
            let body = serde_json::to_vec(&progress).unwrap();
            let mut frame = (body.len() as u32).to_le_bytes().to_vec();
            frame.extend(body);
            // Valid bytes, but no complete progress frame within the idle window.
            // Small writes must not each create a new inactivity allowance. One
            // byte per 150 ms keeps each write inside the window but spreads the
            // frame over many seconds, so a host that renewed per write would
            // fault far later than the bound below. A correct host faults after
            // one window and drops the stream, which ends this loop.
            for chunk in frame.chunks(1) {
                if worker.write_all(chunk).await.is_err() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
            return;
        }
        if matches!(corruption, Corruption::LateRawAfterHeader) {
            tokio::time::sleep(Duration::from_millis(150)).await;
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
        match corruption {
            Corruption::ShortRawEof => {
                worker.write_all(&4u32.to_le_bytes()).await.unwrap();
                worker.write_all(&[0, 0]).await.unwrap();
                // EOF arrives while two bytes are still missing from the raw body.
            }
            Corruption::ExtraRaw => {
                let _ = write_raw(
                    &mut worker,
                    &encode_f32_frame(&[7.0, 9.0]),
                    DEFAULT_MAX_FRAME_BYTES,
                )
                .await;
            }
            Corruption::ProgressInsideRawPair => {
                let _ = write_json(&mut worker, &progress, DEFAULT_MAX_FRAME_BYTES).await;
                let _ = write_raw(
                    &mut worker,
                    &encode_f32_frame(&[7.0]),
                    DEFAULT_MAX_FRAME_BYTES,
                )
                .await;
            }
            Corruption::LateRawAfterHeader => {
                tokio::time::sleep(Duration::from_millis(150)).await;
                let _ = write_raw(
                    &mut worker,
                    &encode_f32_frame(&[7.0]),
                    DEFAULT_MAX_FRAME_BYTES,
                )
                .await;
            }
            Corruption::FragmentedProgress => unreachable!(),
        }
    });
    let mut cfg = RuntimeConfig::default();
    cfg.values
        .insert("artifact_path".into(), "/tmp/framing-fixture".into());
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
    let started = Instant::now();
    let error = host
        .embed_batch(
            &model,
            TokenBatch {
                items: vec![vec![7]],
            },
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, WorkerHostError::EngineCrashed { .. }),
        "malformed traffic must fault the transport: {error:?}"
    );
    if matches!(
        corruption,
        Corruption::FragmentedProgress | Corruption::LateRawAfterHeader
    ) {
        assert!(
            started.elapsed() >= window && started.elapsed() < Duration::from_secs(5),
            "partial bytes and a terminal header must not renew the original idle deadline"
        );
    }
    assert!(
        host.connection.is_none(),
        "a malformed exchange cannot leave a reusable stream"
    );
    assert_eq!(host.health_snapshot().crash_count_window, 1);
    peer.await.unwrap();
}

#[tokio::test]
async fn fragmented_progress_crossing_idle_deadline_faults_real_host() {
    malformed_reply_faults(Corruption::FragmentedProgress).await;
}

#[tokio::test]
async fn terminal_short_raw_then_eof_faults_real_host() {
    malformed_reply_faults(Corruption::ShortRawEof).await;
}

#[tokio::test]
async fn raw_payload_longer_than_terminal_header_faults_real_host() {
    malformed_reply_faults(Corruption::ExtraRaw).await;
}

#[tokio::test]
async fn progress_between_terminal_header_and_raw_faults_real_host() {
    malformed_reply_faults(Corruption::ProgressInsideRawPair).await;
}

#[tokio::test]
async fn terminal_header_does_not_renew_raw_deadline_real_host() {
    malformed_reply_faults(Corruption::LateRawAfterHeader).await;
}

/// A request too large for the frame limit agreed with this worker is the
/// caller's error: it is refused before anything is written, the worker keeps
/// running, and no crash is charged.
#[tokio::test]
async fn oversized_request_is_refused_locally_without_faulting_the_worker() {
    let mut host = WorkerHost::new(WorkerHostConfig::new("/bin/false", std::env::temp_dir()));
    let peer = super::test_support::attach_mock_worker(&mut host, |request, raw| match request {
        WorkerRequest::Load { req_id, .. } => (
            WorkerResponse::Loaded {
                req_id: req_id.clone(),
                model_ref: "mock".into(),
                dims: 1,
                cold_load_ms: 0,
                buckets: None,
                ane_cached_shapes: None,
                transcribe_revision: None,
            },
            None,
        ),
        WorkerRequest::EmbedBatch { req_id, .. } => {
            let ids = synapse_core::decode_i32_frame(raw.unwrap()).unwrap();
            (
                WorkerResponse::Vectors {
                    req_id: req_id.clone(),
                    dims: 1,
                    n: 1,
                },
                Some(encode_f32_frame(&[ids.len() as f32])),
            )
        }
        other => panic!("unexpected request {other:?}"),
    });
    let mut cfg = RuntimeConfig::default();
    cfg.values
        .insert("artifact_path".into(), "/tmp/framing-fixture".into());
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
    // The worker agreed to a smaller limit than the host's configured one:
    // the header fits, the raw token frame (400 bytes) does not.
    host.connection.as_mut().unwrap().max_frame = 256;
    let error = host
        .embed_batch(
            &model,
            TokenBatch {
                items: vec![vec![7; 100]],
            },
        )
        .await
        .unwrap_err();
    assert!(
        matches!(&error, WorkerHostError::WorkerErr { code, .. } if code == "invalid_request"),
        "{error:?}"
    );
    assert!(
        !error
            .to_engine_error(EngineErrorStage::Inference)
            .safe_to_retry_same_request
    );
    assert_eq!(host.health_snapshot().crash_count_window, 0);
    assert!(host.connection.as_ref().is_some_and(|c| !c.unusable));
    // The same worker still serves a request that fits.
    let vectors = host
        .embed_batch(
            &model,
            TokenBatch {
                items: vec![vec![7; 3]],
            },
        )
        .await
        .unwrap();
    assert_eq!(vectors, vec![vec![3.0]]);
    host.kill_current().await;
    assert_eq!(
        peer.await.unwrap().len(),
        2,
        "only LOAD and the fitting request were sent"
    );
}

/// A worker start that fails (here the binary exits before connecting)
/// removes the socket file it created, as a successful start does.
#[tokio::test]
async fn failed_worker_start_leaves_no_socket_file() {
    let dir = std::env::temp_dir().join(format!("synapse-sock-{}", nonce_hex16()));
    let mut config = WorkerHostConfig::new("/bin/false", &dir);
    config.handshake_timeout = Duration::from_millis(200);
    config.worker_id = "socket-cleanup".into();
    let mut host = WorkerHost::new(config);
    assert!(host.start_worker().await.is_err());
    let socket = synapse_core::worker_socket_path(&dir, "socket-cleanup");
    assert!(!socket.exists(), "{} was left behind", socket.display());
    let _ = std::fs::remove_dir_all(&dir);
}
