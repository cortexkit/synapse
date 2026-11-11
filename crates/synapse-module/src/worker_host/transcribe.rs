use super::*;
use crate::transcribe_host::TranscribeObservationFrames;
use synapse_core::{
    decode_decode_trace_frame, decode_transcript_frame, encode_f32_frame, TranscribeObservation,
    TranscribeOutput, TranscribeRequest,
};

pub(crate) struct WorkerTranscription {
    pub output: TranscribeOutput,
    pub transcript_frame: Vec<u8>,
    pub observation: Option<TranscribeObservationFrames>,
    pub progress: Vec<WorkerResponse>,
}

pub(super) fn decode_transcription(
    response: WorkerResponse,
    raw: Option<Vec<u8>>,
    frames: liveness::TranscribeFrames,
    expected_frames: usize,
    observe: bool,
) -> Result<WorkerTranscription, WorkerHostError> {
    let fail = |msg: &str| WorkerHostError::Protocol(msg.into());
    let (valid_encoder_frames, count) = match response {
        WorkerResponse::Transcript {
            valid_encoder_frames,
            count,
            ..
        } => (valid_encoder_frames, count),
        WorkerResponse::Err { code, msg, .. } => {
            return Err(WorkerHostError::WorkerErr { code, msg })
        }
        _ => return Err(fail("expected TRANSCRIPT")),
    };
    if valid_encoder_frames != expected_frames {
        return Err(fail(
            "TRANSCRIPT valid_encoder_frames differs from shared speech core",
        ));
    }
    let transcript_frame = raw.ok_or_else(|| fail("missing TRANSCRIPT raw frame"))?;
    let triples =
        decode_transcript_frame(&transcript_frame, count).map_err(|e| fail(&e.to_string()))?;
    if triples.iter().any(|t| {
        t.token_id < 0
            || t.token_id >= synapse_speech::BLANK as i32
            || t.frame_index < 0
            || t.frame_index as usize >= expected_frames
            || !(0..=4).contains(&t.duration)
    }) {
        return Err(fail("invalid transcript triple"));
    }
    let mut output = TranscribeOutput {
        valid_encoder_frames,
        triples,
        observation: None,
    };
    let observation = if observe {
        let mut parts = frames.parts.into_iter();
        let Some((WorkerResponse::EncoderOutput { frames, .. }, encoder)) = parts.next() else {
            return Err(fail("missing ENCODER_OUTPUT"));
        };
        let Some((WorkerResponse::DecodeTrace { steps, .. }, trace)) = parts.next() else {
            return Err(fail("missing DECODE_TRACE"));
        };
        if frames != expected_frames || parts.next().is_some() {
            return Err(fail(
                "ENCODER_OUTPUT frames differs from shared speech core",
            ));
        }
        let encoder_output = decode_f32_frame(&encoder).map_err(|e| fail(&e.to_string()))?;
        let decode_trace =
            decode_decode_trace_frame(&trace, steps).map_err(|e| fail(&e.to_string()))?;
        if encoder_output.iter().any(|v| !v.is_finite())
            || decode_trace.iter().any(|t| {
                [
                    t.token_logit1,
                    t.token_logit2,
                    t.duration_logit1,
                    t.duration_logit2,
                ]
                .iter()
                .any(|v| !v.is_finite())
            })
        {
            return Err(fail("nonfinite transcription observation"));
        }
        output.observation = Some(TranscribeObservation {
            encoder_output,
            decode_trace,
        });
        Some(TranscribeObservationFrames {
            encoder_output: encoder,
            decode_trace: trace,
        })
    } else {
        None
    };
    Ok(WorkerTranscription {
        output,
        transcript_frame,
        observation,
        progress: frames.progress,
    })
}

impl WorkerHost {
    pub(crate) async fn transcribe(
        &mut self,
        model: &LoadedModel,
        request: TranscribeRequest,
        expected_frames: usize,
    ) -> Result<WorkerTranscription, WorkerHostError> {
        let (model_ref, crash_key) = self.ensure_worker_model(model).await?;
        let wire = WorkerRequest::Transcribe {
            req_id: self.next_req_id("transcribe"),
            model_ref,
            mel_frames: request.mel_frames,
            valid_mel_frames: request.valid_mel_frames,
            observe: request.observe,
        };
        let mut frames = liveness::TranscribeFrames::default();
        let result = self
            .send_request_collect(
                wire,
                Some(encode_f32_frame(&request.mel)),
                Some(&mut frames),
            )
            .await
            .and_then(|(response, raw)| {
                decode_transcription(response, raw, frames, expected_frames, request.observe)
            });
        match result {
            Ok(output) => {
                self.crashes.remove(&crash_key);
                Ok(output)
            }
            Err(error @ WorkerHostError::Protocol(_)) => {
                let stderr_tail = self.kill_current().await;
                self.record_crash_and_maybe_restart(crash_key).await;
                Err(WorkerHostError::EngineCrashed {
                    stage: "inference".into(),
                    detail: error.to_string(),
                    stderr_tail,
                })
            }
            Err(error @ WorkerHostError::EngineCrashed { .. }) => {
                self.record_crash_and_maybe_restart(crash_key).await;
                Err(error)
            }
            Err(error) => Err(error),
        }
    }
}

impl WorkerEngine {
    pub(crate) fn load_transcribe(
        &self,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
    ) -> Result<LoadedModel, WorkerHostError> {
        self.runtime()
            .block_on(self.lock_host()?.load_model(artifact, cfg))
    }

    pub(crate) fn transcribe_with_frames(
        &self,
        model: &LoadedModel,
        request: TranscribeRequest,
        expected_frames: usize,
    ) -> Result<WorkerTranscription, WorkerHostError> {
        self.runtime().block_on(
            self.lock_host()?
                .transcribe(model, request, expected_frames),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use synapse_core::TranscribeRevision;
    use synapse_core::{write_raw, DecodeTraceRecord, WorkerProgressPhase};

    fn request(observe: bool) -> WorkerRequest {
        WorkerRequest::Transcribe {
            req_id: "speech".into(),
            model_ref: "speech:0".into(),
            mel_frames: 1,
            valid_mel_frames: 1,
            observe,
        }
    }

    fn parts() -> Vec<(WorkerResponse, Vec<u8>)> {
        vec![
            (
                WorkerResponse::EncoderOutput {
                    req_id: "speech".into(),
                    frames: 1,
                },
                vec![0; 4096],
            ),
            (
                WorkerResponse::DecodeTrace {
                    req_id: "speech".into(),
                    steps: 1,
                },
                synapse_core::encode_decode_trace_frame(&[DecodeTraceRecord {
                    frame_index: 0,
                    token_top1: 8192,
                    token_top2: 1,
                    duration_top1: 1,
                    duration_top2: 0,
                    token_logit1: 2.0,
                    token_logit2: 1.0,
                    duration_logit1: 2.0,
                    duration_logit2: 1.0,
                    flags: 0,
                }]),
            ),
            (
                WorkerResponse::Transcript {
                    req_id: "speech".into(),
                    valid_encoder_frames: 1,
                    count: 0,
                },
                vec![],
            ),
        ]
    }

    #[tokio::test]
    async fn transcribe_observation_collects_ordered_raw_parts_and_progress_then_stays_aligned() {
        let (mut module, mut worker) = tokio::io::duplex(8192);
        let mut expected = parts();
        expected[0] = (
            WorkerResponse::EncoderOutput {
                req_id: "speech".into(),
                frames: 129,
            },
            encode_f32_frame(&(0..129 * 1024).map(|n| (n % 31) as f32).collect::<Vec<_>>()),
        );
        expected[2].0 = WorkerResponse::Transcript {
            req_id: "speech".into(),
            valid_encoder_frames: 129,
            count: 0,
        };
        let sent = expected.clone();
        let peer = tokio::spawn(async move {
            assert_eq!(
                read_json::<WorkerRequest, _>(&mut worker, DEFAULT_MAX_FRAME_BYTES)
                    .await
                    .unwrap(),
                request(true)
            );
            assert_eq!(
                synapse_core::read_raw(&mut worker, DEFAULT_MAX_FRAME_BYTES)
                    .await
                    .unwrap(),
                vec![0; 512]
            );
            for completed in [24, 149] {
                write_json(
                    &mut worker,
                    &WorkerResponse::Progress {
                        req_id: "speech".into(),
                        completed,
                        phase: WorkerProgressPhase::Inference,
                    },
                    DEFAULT_MAX_FRAME_BYTES,
                )
                .await
                .unwrap();
            }
            for (response, raw) in sent {
                write_json(&mut worker, &response, DEFAULT_MAX_FRAME_BYTES)
                    .await
                    .unwrap();
                write_raw(&mut worker, &raw, DEFAULT_MAX_FRAME_BYTES)
                    .await
                    .unwrap();
            }
            let ping: WorkerRequest = read_json(&mut worker, DEFAULT_MAX_FRAME_BYTES)
                .await
                .unwrap();
            write_json(
                &mut worker,
                &WorkerResponse::unsupported_request(&ping),
                DEFAULT_MAX_FRAME_BYTES,
            )
            .await
            .unwrap();
        });
        let mut collected = liveness::TranscribeFrames::default();
        let (response, raw) = liveness::exchange_collect(
            &mut module,
            &request(true),
            Some(&[0; 512]),
            DEFAULT_MAX_FRAME_BYTES,
            false,
            None,
            Some(&mut collected),
        )
        .await
        .unwrap();
        assert_eq!(collected.progress.len(), 2);
        assert_eq!(collected.parts, expected[..2]);
        let result = decode_transcription(response, raw, collected, 129, true).unwrap();
        let observed = result.observation.unwrap();
        assert!(observed.encoder_output.len() > 512 * 1024);
        assert_eq!(observed.encoder_output, expected[0].1);
        assert_eq!(observed.decode_trace, expected[1].1);
        assert_eq!(result.output.observation.unwrap().decode_trace.len(), 1);
        liveness::exchange(
            &mut module,
            &WorkerRequest::Ping {
                req_id: "ping".into(),
            },
            None,
            DEFAULT_MAX_FRAME_BYTES,
            false,
            None,
        )
        .await
        .unwrap();
        peer.await.unwrap();
    }

    #[tokio::test]
    async fn transcribe_observation_missing_reordered_or_malformed_parts_are_protocol_faults() {
        // Missing encoder, missing trace, early transcript, duplicate encoder,
        // trace before encoder, malformed raw for every part, and a wrong req_id.
        for case in 0..10 {
            let mut sent = parts();
            match case {
                0 => {
                    sent.remove(0);
                }
                1 => {
                    sent.remove(1);
                }
                2 => {
                    sent.swap(0, 2);
                }
                3 => {
                    sent.insert(1, sent[0].clone());
                }
                4 => {
                    sent.swap(0, 1);
                }
                5..=7 => {
                    sent[case - 5].1.push(0);
                }
                8 => {
                    sent[0].0 = WorkerResponse::EncoderOutput {
                        req_id: "wrong".into(),
                        frames: 1,
                    };
                }
                9 => {
                    sent.pop();
                }
                _ => unreachable!(),
            }
            let (mut module, mut worker) = tokio::io::duplex(32 * 1024);
            let peer = tokio::spawn(async move {
                let _: WorkerRequest = read_json(&mut worker, DEFAULT_MAX_FRAME_BYTES)
                    .await
                    .unwrap();
                synapse_core::read_raw(&mut worker, DEFAULT_MAX_FRAME_BYTES)
                    .await
                    .unwrap();
                for (response, raw) in sent {
                    if write_json(&mut worker, &response, DEFAULT_MAX_FRAME_BYTES)
                        .await
                        .is_err()
                    {
                        break;
                    }
                    if write_raw(&mut worker, &raw, DEFAULT_MAX_FRAME_BYTES)
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            });
            let error = liveness::exchange_collect(
                &mut module,
                &request(true),
                Some(&[0; 512]),
                DEFAULT_MAX_FRAME_BYTES,
                false,
                None,
                Some(&mut liveness::TranscribeFrames::default()),
            )
            .await
            .err();
            assert!(error.is_some(), "case {case} must be a protocol fault");
            drop(module);
            peer.await.unwrap();
        }
    }

    #[tokio::test]
    async fn transcribe_serving_rejects_unsolicited_observation() {
        let (mut module, mut worker) = tokio::io::duplex(8192);
        let peer = tokio::spawn(async move {
            let _: WorkerRequest = read_json(&mut worker, 8192).await.unwrap();
            synapse_core::read_raw(&mut worker, 8192).await.unwrap();
            write_json(&mut worker, &parts()[0].0, 8192).await.unwrap();
        });
        assert!(liveness::exchange(
            &mut module,
            &request(false),
            Some(&[0; 512]),
            8192,
            false,
            None
        )
        .await
        .is_err());
        peer.await.unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn transcribe_uses_existing_120_second_inference_and_600_second_setup_hang_windows() {
        for (cuda, delay, expected_ok) in [
            (false, 119, true),
            (false, 121, false),
            (true, 599, true),
            (true, 601, false),
        ] {
            let (mut module, mut worker) = tokio::io::duplex(8192);
            let peer = tokio::spawn(async move {
                let _: WorkerRequest = read_json(&mut worker, 8192).await.unwrap();
                synapse_core::read_raw(&mut worker, 8192).await.unwrap();
                for completed in 1..=2 {
                    tokio::time::sleep(Duration::from_secs(delay)).await;
                    if write_json(
                        &mut worker,
                        &WorkerResponse::Progress {
                            req_id: "speech".into(),
                            completed,
                            phase: if cuda {
                                WorkerProgressPhase::Setup
                            } else {
                                WorkerProgressPhase::Inference
                            },
                        },
                        8192,
                    )
                    .await
                    .is_err()
                    {
                        return;
                    }
                }
                if cuda {
                    write_json(
                        &mut worker,
                        &WorkerResponse::Progress {
                            req_id: "speech".into(),
                            completed: 3,
                            phase: WorkerProgressPhase::Inference,
                        },
                        8192,
                    )
                    .await
                    .unwrap();
                    tokio::time::sleep(Duration::from_secs(119)).await;
                }
                write_json(&mut worker, &parts()[2].0, 8192).await.unwrap();
                write_raw(&mut worker, &[], 8192).await.unwrap();
            });
            let result = liveness::exchange(
                &mut module,
                &request(false),
                Some(&[0; 512]),
                8192,
                cuda,
                None,
            )
            .await;
            assert_eq!(result.is_ok(), expected_ok, "cuda={cuda}, delay={delay}");
            drop(module);
            peer.await.unwrap();
        }
    }

    #[tokio::test(start_paused = true)]
    async fn transcribe_observation_frames_do_not_renew_completed_work_watchdog() {
        let (mut module, mut worker) = tokio::io::duplex(8192);
        let peer = tokio::spawn(async move {
            let _: WorkerRequest = read_json(&mut worker, 8192).await.unwrap();
            synapse_core::read_raw(&mut worker, 8192).await.unwrap();
            tokio::time::sleep(Duration::from_secs(100)).await;
            let (header, raw) = &parts()[0];
            write_json(&mut worker, header, 8192).await.unwrap();
            write_raw(&mut worker, raw, 8192).await.unwrap();
            tokio::time::sleep(Duration::from_secs(21)).await;
        });
        let error = liveness::exchange_collect(
            &mut module,
            &request(true),
            Some(&[0; 512]),
            8192,
            false,
            None,
            Some(&mut liveness::TranscribeFrames::default()),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(error, WorkerHostError::Protocol(ref detail) if detail.contains("no-progress hang"))
        );
        drop(module);
        peer.await.unwrap();
    }

    #[test]
    fn transcribe_response_valid_frames_are_checked_against_shared_core() {
        let mut parts = parts();
        let (response, raw) = parts.pop().unwrap();
        let frames = liveness::TranscribeFrames {
            parts,
            progress: vec![],
        };
        assert!(matches!(
            decode_transcription(response, Some(raw), frames, 2, true),
            Err(WorkerHostError::Protocol(_))
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn transcribe_v4_worker_with_old_kernel_is_refused_at_load_before_inference() {
        let (module, mut worker) = tokio::net::UnixStream::pair().unwrap();
        let mut host = WorkerHost::new(WorkerHostConfig::new("/bin/false", "unused"));
        host.connection = Some(WorkerConnection {
            stream: module,
            child: Command::new("sleep")
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
            let request: WorkerRequest = read_json(&mut worker, DEFAULT_MAX_FRAME_BYTES)
                .await
                .unwrap();
            let WorkerRequest::Load { req_id, .. } = request else {
                panic!("expected LOAD")
            };
            write_json(
                &mut worker,
                &WorkerResponse::Loaded {
                    req_id,
                    model_ref: "speech:0".into(),
                    dims: 1024,
                    cold_load_ms: 0,
                    buckets: None,
                    ane_cached_shapes: None,
                    transcribe_revision: Some(TranscribeRevision {
                        kernel_revision: "old-kernel".into(),
                        decode_rule: synapse_speech::DECODE_RULE_REVISION.into(),
                    }),
                },
                DEFAULT_MAX_FRAME_BYTES,
            )
            .await
            .unwrap();
            use tokio::io::AsyncReadExt;
            let mut remaining = Vec::new();
            worker.read_to_end(&mut remaining).await.unwrap();
            assert!(
                remaining.is_empty(),
                "no self-check or transcription may follow refused LOAD"
            );
        });
        let profile =
            crate::transcribe_host::TranscribeProfile::pinned("parakeet-tdt-0.6b-v3.owned-cuda")
                .unwrap();
        let mut cfg = profile.runtime_config(&RuntimeConfig::default());
        cfg.values
            .insert("artifact_path".into(), "unused.safetensors".into());
        let error = host
            .load_model(
                &ValidatedArtifact {
                    digest: "fake".into(),
                    format: "safetensors-package".into(),
                },
                &cfg,
            )
            .await
            .unwrap_err();
        assert_eq!(error.code(), Some("transcribe_revision_mismatch"));
        assert!(host.loaded_models.is_empty());
        host.kill_current().await;
        peer.await.unwrap();
    }
}
