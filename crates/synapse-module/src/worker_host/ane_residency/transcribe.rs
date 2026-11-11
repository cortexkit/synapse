use super::*;
use crate::worker_host::{
    liveness::TranscribeFrames,
    transcribe::{decode_transcription, WorkerTranscription},
};
use synapse_core::{encode_f32_frame, RuntimeConfig, TranscribeRequest};

pub(super) fn validate_loaded_request(
    request: &WorkerRequest,
    response: &WorkerResponse,
    max_frame: u32,
) -> Result<(), AneResidencyError> {
    if let (
        WorkerRequest::Load { runtime_config, .. },
        WorkerResponse::Loaded {
            transcribe_revision,
            ..
        },
    ) = (request, response)
    {
        crate::transcribe_host::validate_worker_load(
            &RuntimeConfig {
                values: runtime_config.clone(),
            },
            max_frame,
            transcribe_revision.as_ref(),
        )
        .map_err(|error| match error {
            crate::worker_host::WorkerHostError::WorkerErr { code, msg } => {
                AneResidencyError::WorkerErr { code, msg }
            }
            error => AneResidencyError::Channel(error.to_string()),
        })?;
    }
    Ok(())
}

impl<S: AsyncRead + AsyncWrite + Unpin + Send + 'static> AneWorkerChannel<S> {
    async fn exchange_transcription(
        &self,
        model_ref: &str,
        request: TranscribeRequest,
        expected_frames: usize,
    ) -> Result<WorkerTranscription, AneResidencyError> {
        let generation = self.generation();
        let mut guard = cancellable_wait(self.stream.lock()).await?;
        let wire = WorkerRequest::Transcribe {
            req_id: self.next_req_id("transcribe"),
            model_ref: model_ref.into(),
            mel_frames: request.mel_frames,
            valid_mel_frames: request.valid_mel_frames,
            observe: request.observe,
        };
        let raw = encode_f32_frame(&request.mel);
        let mut frames = TranscribeFrames::default();
        let result = self
            .exchange_locked_collect(
                &mut guard,
                &wire,
                Some(&raw),
                generation,
                None,
                Some(&mut frames),
            )
            .await
            .and_then(|(response, raw)| {
                decode_transcription(response, raw, frames, expected_frames, request.observe)
                    .map_err(|error| match error {
                        crate::worker_host::WorkerHostError::WorkerErr { code, msg } => {
                            AneResidencyError::WorkerErr { code, msg }
                        }
                        error => AneResidencyError::Channel(error.to_string()),
                    })
            });
        if self.generation() == generation {
            match &result {
                Ok(_) if !self.faulted.load(Ordering::Acquire) => {
                    let mut faults = self.faults.lock().unwrap_or_else(|p| p.into_inner());
                    faults.strikes = 0;
                    faults.last_generation = None;
                    faults.until = None;
                }
                Err(AneResidencyError::Channel(_))
                    if !self.faulted.swap(true, Ordering::AcqRel) =>
                {
                    self.charge_fault(generation);
                }
                _ => {}
            }
        }
        result
    }
}

impl<S: AsyncRead + AsyncWrite + Unpin + Send + 'static> AneServing<S> {
    pub(crate) async fn transcribe_with_frames(
        self: &Arc<Self>,
        request: TranscribeRequest,
        expected_frames: usize,
    ) -> Result<WorkerTranscription, AneResidencyError> {
        if self.channel.retired.load(Ordering::Acquire) {
            return Err(AneResidencyError::Channel("worker is retired".into()));
        }
        let serving = self.clone();
        self.active_requests.fetch_add(1, Ordering::AcqRel);
        let activity = RequestActivity(self.active_requests.clone());
        let waits = Arc::new(RequestWaits::new());
        tokio::spawn(REQUEST_WAITS.scope(waits, async move {
            let _activity = activity;
            let worker: Arc<dyn AneShapeWorker> = serving.channel.clone();
            if let Some(retry_after_ms) = serving.channel.cooldown_remaining() {
                return Err(AneResidencyError::Quarantined { retry_after_ms });
            }
            if serving.channel.faulted.load(Ordering::Acquire) {
                serving.supervisor.recover(&worker).await?;
            }
            let generation = worker.generation();
            let result = serving
                .supervisor
                .run_by_published_rung_until(
                    &worker,
                    &serving.metadata.model_ref,
                    // Parakeet residency is measured in encoder frames, never mel frames.
                    &[expected_frames],
                    &serving.metadata.buckets,
                    None,
                    |_, _| {
                        let serving = serving.clone();
                        let request = request.clone();
                        async move {
                            serving
                                .channel
                                .exchange_transcription(
                                    &serving.metadata.model_ref,
                                    request,
                                    expected_frames,
                                )
                                .await
                                .map(|output| vec![output])
                        }
                    },
                )
                .await;
            if matches!(&result, Err(AneResidencyError::Channel(_))) {
                let _ = serving
                    .supervisor
                    .recover_at_generation(&worker, generation)
                    .await;
            }
            result.map(|mut results| results.remove(0))
        }))
        .await
        .map_err(|error| AneResidencyError::Channel(error.to_string()))?
    }
}

#[cfg(unix)]
impl DirectAneEngine {
    pub(crate) fn transcribe_with_frames(
        &self,
        request: TranscribeRequest,
        expected_frames: usize,
    ) -> Result<WorkerTranscription, AneResidencyError> {
        self.runtime.as_ref().expect("live ANE runtime").block_on(
            self.serving
                .transcribe_with_frames(request, expected_frames),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use synapse_core::{AneExecutable, DEFAULT_MAX_FRAME_BYTES};

    type MockChannel = (
        Arc<AneWorkerChannel<tokio::io::DuplexStream>>,
        Arc<Mutex<Vec<(bool, usize)>>>,
    );

    async fn mock_channel(id: &str) -> MockChannel {
        let admitted = Arc::new(Mutex::new(Vec::new()));
        let seen = admitted.clone();
        let connector: AneWorkerConnector<tokio::io::DuplexStream> = Box::new(move || {
            let seen = seen.clone();
            Box::pin(async move {
                let (module, mut worker) = tokio::io::duplex(8192);
                let peer = tokio::spawn(async move {
                    let mut compiled = HashSet::new();
                    while let Ok(request) =
                        read_json::<WorkerRequest, _>(&mut worker, DEFAULT_MAX_FRAME_BYTES).await
                    {
                        let response = match request {
                            WorkerRequest::AneAdmitShape {
                                req_id,
                                model_ref,
                                shape,
                            } => {
                                seen.lock().unwrap().push((true, shape));
                                let layers = pinned_model_param(&model_ref, "num_hidden_layers")
                                    .unwrap() as u32;
                                let cold = compiled.insert((model_ref, shape));
                                WorkerResponse::Admitted {
                                    req_id,
                                    inventory: AnePlacementInventory {
                                        executables: (0..layers)
                                            .map(|n| AneExecutable {
                                                id: format!("{shape}-{n}"),
                                                layers: vec![n],
                                            })
                                            .collect(),
                                        cpu_stages: vec![],
                                    },
                                    cache_hit_layers: Some(if cold { 0 } else { layers }),
                                    compiled_layers: Some(if cold { layers } else { 0 }),
                                }
                            }
                            WorkerRequest::AneEvictShape { req_id, shape, .. } => {
                                seen.lock().unwrap().push((false, shape));
                                WorkerResponse::Evicted { req_id }
                            }
                            WorkerRequest::Transcribe {
                                req_id,
                                mel_frames,
                                valid_mel_frames,
                                observe,
                                ..
                            } => {
                                assert_eq!(mel_frames, 1033);
                                assert_eq!(valid_mel_frames, 1032);
                                assert!(!observe);
                                assert_eq!(
                                    read_raw(&mut worker, DEFAULT_MAX_FRAME_BYTES)
                                        .await
                                        .unwrap()
                                        .len(),
                                    1033 * 512
                                );
                                write_json(
                                    &mut worker,
                                    &WorkerResponse::Transcript {
                                        req_id,
                                        valid_encoder_frames: 129,
                                        count: 0,
                                    },
                                    DEFAULT_MAX_FRAME_BYTES,
                                )
                                .await
                                .unwrap();
                                write_raw(&mut worker, &[], DEFAULT_MAX_FRAME_BYTES)
                                    .await
                                    .unwrap();
                                continue;
                            }
                            _ => panic!("unexpected request"),
                        };
                        write_json(&mut worker, &response, DEFAULT_MAX_FRAME_BYTES)
                            .await
                            .unwrap();
                    }
                });
                Ok(AneWorkerSession {
                    stream: module,
                    pid: None,
                    packed: None,
                    confirm_exit: Box::pin(async move {
                        peer.await
                            .map_err(|e| AneResidencyError::Channel(e.to_string()))
                    }),
                })
            })
        });
        let channel = Arc::new(
            AneWorkerChannel::connect(id, DEFAULT_MAX_FRAME_BYTES, connector)
                .await
                .unwrap(),
        );
        (channel, admitted)
    }

    #[tokio::test]
    async fn transcribe_parakeet_encoder_ladder_four_shapes_fit_an_otherwise_empty_budget() {
        let (channel, admitted) = mock_channel("speech-only").await;
        let worker: Arc<dyn AneShapeWorker> = channel.clone();
        let model_ref = "ane-direct:parakeet-tdt-0.6b-v3.ane-direct-worker:0";
        assert_eq!(worker.executable_count(model_ref).unwrap(), 24);
        let supervisor = AneResidencySupervisor::new(AneResidencyLimits::default());
        let mut leases = Vec::new();
        for shape in [128, 256, 512, 1024] {
            leases.push(supervisor.lease(&worker, model_ref, shape).await.unwrap());
        }
        assert_eq!(supervisor.inner.lock().reserved_executables(), 96);
        assert_eq!(
            *admitted.lock().unwrap(),
            [(true, 128), (true, 256), (true, 512), (true, 1024)]
        );
        drop(leases);
        let serving = Arc::new(
            AneServing::new(
                channel,
                supervisor.clone(),
                AneServingMetadata {
                    model_ref: model_ref.into(),
                    dims: 1024,
                    buckets: ANE_SHAPE_LADDER.to_vec(),
                    pooling: synapse_core::WorkerPooling::Mean,
                    normalize: false,
                    timeout: std::time::Duration::from_secs(30),
                },
            )
            .unwrap(),
        );
        let result = serving
            .transcribe_with_frames(
                TranscribeRequest {
                    mel: vec![0.0; 1033 * 128],
                    mel_frames: 1033,
                    valid_mel_frames: 1032,
                    observe: false,
                },
                129,
            )
            .await
            .unwrap();
        assert_eq!(result.output.valid_encoder_frames, 129);
        assert!(result.observation.is_none());
        // The 1033 mel frames would choose 2048; encoder frames reuse 256.
        assert_eq!(
            *admitted.lock().unwrap(),
            [(true, 128), (true, 256), (true, 512), (true, 1024)]
        );
        serving.retire().await.unwrap();
        assert_eq!(supervisor.inner.lock().reserved_executables(), 0);
    }

    #[tokio::test]
    async fn transcribe_parakeet_and_embedding_share_budget_and_evict_idle_lru_shapes() {
        let (embedding, embed_events) = mock_channel("embedding-only").await;
        let (speech, speech_events) = mock_channel("speech-only").await;
        let embed_worker: Arc<dyn AneShapeWorker> = embedding.clone();
        let speech_worker: Arc<dyn AneShapeWorker> = speech.clone();
        let embed_ref = "ane-direct:qwen3-embedding-0.6b.ane-direct-worker:pinned";
        let speech_ref = "ane-direct:parakeet-tdt-0.6b-v3.ane-direct-worker:pinned";
        let supervisor = AneResidencySupervisor::new(AneResidencyLimits::default());
        for shape in [128, 256, 512] {
            drop(
                supervisor
                    .lease(&embed_worker, embed_ref, shape)
                    .await
                    .unwrap(),
            );
        }
        assert_eq!(supervisor.inner.lock().reserved_executables(), 84);
        drop(
            supervisor
                .lease(&speech_worker, speech_ref, 128)
                .await
                .unwrap(),
        );
        assert_eq!(supervisor.inner.lock().reserved_executables(), 80);
        assert_eq!(
            *embed_events.lock().unwrap(),
            [(true, 128), (true, 256), (true, 512), (false, 128)]
        );
        // Embedding can continue to admit shapes; once speech is least recently
        // used, its idle shape is evicted just like an idle embedding shape.
        for shape in [1024, 2048, 4096] {
            drop(
                supervisor
                    .lease(&embed_worker, embed_ref, shape)
                    .await
                    .unwrap(),
            );
            assert!(supervisor.inner.lock().reserved_executables() <= 100);
        }
        assert_eq!(supervisor.inner.lock().reserved_executables(), 84);
        assert_eq!(*speech_events.lock().unwrap(), [(true, 128), (false, 128)]);
        drop(
            supervisor
                .lease(&speech_worker, speech_ref, 128)
                .await
                .unwrap(),
        );
        assert_eq!(
            *speech_events.lock().unwrap(),
            [(true, 128), (false, 128), (true, 128)]
        );
        assert_eq!(supervisor.inner.lock().reserved_executables(), 80);
        supervisor
            .retire(&embed_worker, embedding.shutdown())
            .await
            .unwrap();
        supervisor
            .retire(&speech_worker, speech.shutdown())
            .await
            .unwrap();
        assert_eq!(supervisor.inner.lock().reserved_executables(), 0);
    }
}
