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
            // Small writes must not each create a new inactivity allowance.
            for chunk in frame.chunks(2) {
                if worker.write_all(chunk).await.is_err() {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(30)).await;
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
            started.elapsed() >= window && started.elapsed() < Duration::from_millis(700),
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
