use super::*;
#[cfg(unix)]
use crate::worker_host::{
    liveness::TranscribeFrames,
    transcribe::{decode_transcription, WorkerTranscription},
};
use synapse_core::RuntimeConfig;
#[cfg(unix)]
use synapse_core::{encode_f32_frame, TranscribeRequest};

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
        crate::exchange::validate_worker_load(
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

// Transcription on the direct Neural Engine is reached only through
// `DirectAneEngine`, which exists on Unix; elsewhere these paths are unused.
#[cfg(unix)]
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

#[cfg(unix)]
impl<S: AsyncRead + AsyncWrite + Unpin + Send + 'static> AneServing<S> {
    pub fn transcribe_quarantine_retry_after_ms(&self) -> Option<u64> {
        self.channel.cooldown_remaining()
    }

    async fn admit_self_check(&self, frames: usize) -> Result<AneShapeLease, AneResidencyError> {
        if self.channel.retired.load(Ordering::Acquire) {
            return Err(AneResidencyError::Channel("worker is retired".into()));
        }
        let worker: Arc<dyn AneShapeWorker> = self.channel.clone();
        if let Some(retry_after_ms) = self.channel.cooldown_remaining() {
            return Err(AneResidencyError::Quarantined { retry_after_ms });
        }
        if self.channel.faulted.load(Ordering::Acquire) {
            self.supervisor.recover(&worker).await?;
        }
        let rung = self
            .metadata
            .buckets
            .iter()
            .copied()
            .find(|r| *r >= frames)
            .ok_or(AneResidencyError::SequenceTooLong {
                n_tokens: frames,
                max: self.metadata.buckets.last().copied().unwrap_or_default(),
            })?;
        self.supervisor
            .lease(&worker, &self.metadata.model_ref, rung)
            .await
    }

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
    pub fn transcribe_shapes(&self) -> Vec<serde_json::Value> {
        self.serving.supervisor.transcribe_shapes(
            self.serving.channel.worker_id(),
            &self.serving.metadata.model_ref,
        )
    }

    pub fn admit_self_check(&self, frames: usize) -> Result<AneShapeLease, AneResidencyError> {
        self.runtime
            .as_ref()
            .expect("live ANE runtime")
            .block_on(self.serving.admit_self_check(frames))
    }

    // The caller holds the ordinary scheduler lease across this exchange, so
    // no compile or second scheduler wait belongs to the timed self-check.
    pub fn transcribe_self_check_admitted(
        &self,
        request: TranscribeRequest,
        expected_frames: usize,
    ) -> Result<WorkerTranscription, AneResidencyError> {
        self.runtime.as_ref().expect("live ANE runtime").block_on(
            self.serving.channel.exchange_transcription(
                &self.serving.metadata.model_ref,
                request,
                expected_frames,
            ),
        )
    }

    pub fn transcribe_with_frames(
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

    #[cfg(unix)]
    const PARAKEET_PROFILE: &str = "parakeet-tdt-0.6b-v3.ane-direct-worker";
    #[cfg(unix)]
    const PARAKEET_REF: &str = "ane-direct:parakeet-tdt-0.6b-v3.ane-direct-worker:0";

    type MockChannel = (
        Arc<AneWorkerChannel<tokio::io::DuplexStream>>,
        Arc<Mutex<Vec<(bool, usize)>>>,
    );

    async fn mock_channel(id: &str) -> MockChannel {
        // The worker advertises the full ladder even though serving caps audio
        // duration. It reports no embedding dimension for transcription.
        mock_channel_with_load(id, 0, &[128, 256, 512, 1024, 2048, 4096, 8192]).await
    }

    async fn mock_channel_with_load(
        id: &str,
        dims: usize,
        buckets: &'static [usize],
    ) -> MockChannel {
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
                            WorkerRequest::Load {
                                req_id,
                                artifact_digest,
                                runtime_config,
                                ..
                            } => WorkerResponse::Loaded {
                                req_id,
                                model_ref: format!(
                                    "ane-direct:{}:{artifact_digest}",
                                    runtime_config["profile"]
                                ),
                                dims,
                                cold_load_ms: 0,
                                buckets: Some(buckets.to_vec()),
                                ane_cached_shapes: Some(vec![]),
                                transcribe_revision: (runtime_config["operation"] == "transcribe")
                                    .then(|| synapse_core::TranscribeRevision {
                                        kernel_revision:
                                            synapse_core::ANE_DIRECT_TRANSCRIBE_KERNEL_REVISION
                                                .into(),
                                        decode_rule: synapse_speech::DECODE_RULE_REVISION.into(),
                                    }),
                            },
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
                                        peak_device_memory_bytes: None,
                                        peak_storage_buffer_bytes: None,
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

    #[cfg(unix)]
    async fn load_mock_serving(
        channel: &Arc<AneWorkerChannel<tokio::io::DuplexStream>>,
        supervisor: &AneResidencySupervisor,
        config: RuntimeConfig,
    ) -> Result<AneServing<tokio::io::DuplexStream>, AneResidencyError> {
        channel
            .load_serving(
                supervisor,
                &WorkerRequest::Load {
                    req_id: channel.next_req_id("load"),
                    artifact_path: "mock.safetensors".into(),
                    artifact_digest: "0".into(),
                    format: "safetensors-package".into(),
                    runtime_config: config.values,
                },
                synapse_core::WorkerPooling::Mean,
                false,
                std::time::Duration::from_secs(30),
            )
            .await
    }

    #[cfg(unix)]
    fn parakeet_config() -> RuntimeConfig {
        crate::exchange::TranscribeProfile::pinned(PARAKEET_PROFILE)
            .unwrap()
            .runtime_config(&RuntimeConfig::default())
    }

    // Runs the Unix serving path against a mock worker, without ANE hardware.
    #[cfg(unix)]
    #[tokio::test]
    async fn transcribe_parakeet_encoder_ladder_four_shapes_fit_an_otherwise_empty_budget() {
        let (channel, admitted) = mock_channel("speech-only").await;
        let worker: Arc<dyn AneShapeWorker> = channel.clone();
        assert_eq!(worker.executable_count(PARAKEET_REF).unwrap(), 24);
        let supervisor = AneResidencySupervisor::new(AneResidencyLimits::default());
        let serving = Arc::new(
            load_mock_serving(&channel, &supervisor, parakeet_config())
                .await
                .unwrap(),
        );
        let mut leases = Vec::new();
        for shape in [128, 256, 512, 1024] {
            leases.push(
                supervisor
                    .lease(&worker, PARAKEET_REF, shape)
                    .await
                    .unwrap(),
            );
        }
        assert_eq!(supervisor.inner.lock().reserved_executables(), 96);
        assert_eq!(
            *admitted.lock().unwrap(),
            [(true, 128), (true, 256), (true, 512), (true, 1024)]
        );
        drop(leases);
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

    #[cfg(unix)]
    #[tokio::test]
    async fn transcribe_load_requires_a_nonzero_increasing_ladder_covering_the_frame_limit() {
        let pinned = parakeet_config();
        let mut shorter = pinned.clone();
        // A shorter request envelope may be served by a shorter ladder.
        shorter
            .values
            .insert("max_encoder_frames".into(), "1024".into());
        let supervisor = AneResidencySupervisor::new(AneResidencyLimits::default());
        for (config, buckets, accepted) in [
            (&shorter, &[128, 256, 512, 1024][..], true),
            (&pinned, &[128, 256, 512, 1024][..], false),
            (&shorter, &[128, 256, 512][..], false),
            (&shorter, &[][..], false),
            (&shorter, &[0, 128, 1024][..], false),
            (&shorter, &[128, 512, 256, 1024][..], false),
            (&shorter, &[128, 128, 1024][..], false),
        ] {
            let (channel, _) = mock_channel_with_load("speech-ladder", 0, buckets).await;
            let result = load_mock_serving(&channel, &supervisor, config.clone()).await;
            if accepted {
                result.unwrap().retire().await.unwrap();
            } else {
                assert!(
                    matches!(result, Err(AneResidencyError::Channel(msg))
                    if msg == "LOAD returned invalid bucket or dimension metadata"),
                    "{buckets:?}"
                );
                channel.shutdown().await.unwrap();
            }
        }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn embedding_load_refuses_a_ladder_ending_below_8192() {
        let (channel, _) = mock_channel_with_load("embedding-ladder", 384, &[128, 256, 1024]).await;
        let supervisor = AneResidencySupervisor::new(AneResidencyLimits::default());
        let config = RuntimeConfig {
            values: BTreeMap::from([
                (
                    "profile".into(),
                    "gte-modernbert-base.ane-direct-worker".into(),
                ),
                ("operation".into(), "embed".into()),
            ]),
        };
        assert!(
            matches!(load_mock_serving(&channel, &supervisor, config).await,
            Err(AneResidencyError::Channel(msg)) if msg == "LOAD returned invalid bucket or dimension metadata")
        );
        channel.shutdown().await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn speech_self_check_admission_holds_only_an_ordinary_temporary_lease() {
        let (channel, events) = mock_channel("speech-self-check").await;
        let supervisor = AneResidencySupervisor::new(AneResidencyLimits::default());
        let serving = Arc::new(
            load_mock_serving(&channel, &supervisor, parakeet_config())
                .await
                .unwrap(),
        );
        let lease = serving.admit_self_check(129).await.unwrap();
        assert_eq!(*events.lock().unwrap(), [(true, 256)]);
        assert_eq!(supervisor.inner.lock().reserved_executables(), 24);
        assert_eq!(
            supervisor
                .inner
                .lock()
                .slots
                .values()
                .map(|s| s.leases)
                .sum::<u32>(),
            1
        );
        let result = channel
            .exchange_transcription(
                &serving.metadata.model_ref,
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
        assert!(result.output.triples.is_empty());
        assert_eq!(
            *events.lock().unwrap(),
            [(true, 256)],
            "timed exchange never readmits"
        );
        drop(lease);
        assert_eq!(
            supervisor
                .inner
                .lock()
                .slots
                .values()
                .map(|s| s.leases)
                .sum::<u32>(),
            0
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
        let mut supervisor = AneResidencySupervisor::new(AneResidencyLimits::default());
        supervisor.enable_observations();
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
        let frozen_shapes = supervisor.transcribe_shapes("speech-only", speech_ref);
        assert_eq!(frozen_shapes.len(), 1);
        assert_eq!(frozen_shapes[0]["encoder_frames"], 128);
        let executables = frozen_shapes[0]["executables"].as_array().unwrap();
        assert_eq!(executables.len(), 24);
        assert_eq!(
            executables
                .iter()
                .map(|e| e["id"].as_str().unwrap())
                .collect::<HashSet<_>>()
                .len(),
            24
        );
        assert_eq!(
            executables
                .iter()
                .flat_map(|e| e["layers"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap()))
                .collect::<std::collections::BTreeSet<_>>(),
            (0..24).collect()
        );
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
        assert_eq!(
            supervisor.transcribe_shapes("speech-only", speech_ref),
            frozen_shapes,
            "eviction cannot erase admitted shapes"
        );
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
        assert_eq!(
            supervisor.transcribe_shapes("speech-only", speech_ref),
            frozen_shapes,
            "readmission deduplicates each encoder rung"
        );
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
