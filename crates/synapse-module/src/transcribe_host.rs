//! Audio admission and serving seam shared by catalog self-checks and media jobs.
//! Workers only see normalized features; container bytes and text stay here.

use std::sync::atomic::{AtomicU64, Ordering};

use synapse_core::{
    worker_engine_names::*, LoadedModel, RuntimeConfig, TranscribeEngine, TranscribeOutput,
    TranscribeRequest, ValidatedArtifact, WorkerResponse,
};
use synapse_engine_owned::transcribe::{OwnedMetalTranscribeEngine, Progress};
use synapse_speech::{Frontend, Tokenizer, Transcript};

use crate::worker_host::{WorkerEngine, WorkerHostConfig, WorkerHostError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationMode {
    Serving,
    Certification,
}

#[cfg(test)]
use synapse_core::EngineError;
pub use synapse_worker_host::exchange::{
    DurationLimit, ProgressTiming, TranscribeHostError, TranscribeObservationFrames,
    TranscribeProfile,
};

pub struct PreparedTranscription {
    pub request: TranscribeRequest,
    pub duration_ms: u64,
    pub valid_encoder_frames: usize,
    audio_duration_ms: f64,
}

pub fn prepare_wav(
    profile: &TranscribeProfile,
    audio: &[u8],
    mode: ObservationMode,
) -> Result<PreparedTranscription, TranscribeHostError> {
    let audio = synapse_speech::decode_wav(audio)?;
    let duration_ms = audio.duration_ms();
    // The resampler's ceiling length is known before allocating/resampling. Use
    // that exact count for admission, including the input-duration refusal.
    let samples = (audio.samples.len() as u64 * 16_000).div_ceil(audio.sample_rate as u64) as usize;
    let valid_encoder_frames = synapse_speech::valid_encoder_frames(samples as u64) as usize;
    // A clip shorter than one mel hop leaves no valid frame for any engine to
    // encode. That is a property of the audio, so refuse it as invalid audio
    // here rather than let every engine reject it as a retryable fault.
    if synapse_speech::valid_mel_frames(samples as u64) == 0 {
        return Err(synapse_speech::Error::InvalidAudio("shorter than one mel hop").into());
    }
    // Keep the numerical profile intact for identity and worker negotiation;
    // serving must also obey the independently certified catalog duration.
    let max_duration_ms = crate::catalog::compiled_catalog()
        .map_err(|error| TranscribeHostError::Profile(error.to_string()))?
        .models
        .iter()
        .flat_map(|entry| &entry.backends)
        .find(|backend| backend.profile.as_deref() == Some(profile.id.as_str()))
        .and_then(crate::catalog::CatalogBackend::effective_max_duration_ms)
        .ok_or_else(|| TranscribeHostError::Profile("missing catalog serving duration".into()))?
        .min(profile.max_duration_ms);
    if duration_ms > max_duration_ms || valid_encoder_frames > profile.max_encoder_frames {
        return Err(TranscribeHostError::DurationLimit(DurationLimit {
            duration_ms,
            max_duration_ms,
            valid_encoder_frames,
            max_encoder_frames: profile.max_encoder_frames,
        }));
    }
    let features = Frontend::new().process_samples(&audio.samples_16khz()?)?;
    Ok(PreparedTranscription {
        request: TranscribeRequest {
            mel: features.values,
            mel_frames: features.mel_frames,
            valid_mel_frames: features.valid_mel_frames,
            observe: mode == ObservationMode::Certification,
        },
        duration_ms,
        valid_encoder_frames,
        audio_duration_ms: audio.samples.len() as f64 * 1000.0 / audio.sample_rate as f64,
    })
}

#[derive(Debug)]
pub struct TranscribeResult {
    pub transcript: Transcript,
    pub output: TranscribeOutput,
    pub transcript_frame: Vec<u8>,
    pub observation: Option<TranscribeObservationFrames>,
    pub progress: Vec<WorkerResponse>,
    pub(crate) timing: ProgressTiming,
    pub(crate) peak_device_memory_bytes: Option<u64>,
    pub(crate) peak_storage_buffer_bytes: Option<u64>,
}

enum Backend {
    Metal(OwnedMetalTranscribeEngine, LoadedModel),
    Worker(WorkerEngine, LoadedModel),
    #[cfg(unix)]
    Ane(crate::worker_host::ane_residency::DirectAneEngine),
    #[cfg(test)]
    InProcessTest(
        Box<dyn Fn(TranscribeRequest) -> Result<TranscribeOutput, EngineError> + Send>,
        Option<Box<dyn FnMut() + Send>>,
    ),
}

pub(crate) enum CertifyCounter {
    WorkerRequests(String, u64),
    InProcessEngineInvocations(u64),
}

/// A normal, temporary shape lease. It is released after checking, never pinned
/// for the model's lifetime. Keeping it alive keeps compilation out of the timed check.
pub struct SelfCheckAdmission<'a> {
    host: &'a TranscribeHost,
    frames: usize,
    #[cfg(unix)]
    _lease: Option<crate::worker_host::ane_residency::AneShapeLease>,
}

impl SelfCheckAdmission<'_> {
    pub fn transcribe(&self, wav: &[u8]) -> Result<TranscribeResult, TranscribeHostError> {
        let prepared = prepare_wav(&self.host.profile, wav, ObservationMode::Serving)?;
        if prepared.valid_encoder_frames > self.frames {
            return Err(TranscribeHostError::Profile(
                "self-check exceeds admitted shape".into(),
            ));
        }
        #[cfg(unix)]
        if let Backend::Ane(engine) = &self.host.backend {
            if crate::worker_host::ane_residency::ladder_rung(prepared.valid_encoder_frames)
                != crate::worker_host::ane_residency::ladder_rung(self.frames)
            {
                return Err(TranscribeHostError::Profile(
                    "self-check input requires a different admitted rung".into(),
                ));
            }
            let result = engine
                .transcribe_self_check_admitted(prepared.request, prepared.valid_encoder_frames)?;
            return finish(
                result.output,
                Some(result.transcript_frame),
                result.observation,
                result.progress,
                prepared.valid_encoder_frames,
                prepared.audio_duration_ms,
            );
        }
        self.host.transcribe_prepared(prepared, |_| {})
    }
}

/// One loaded transcription lane. A worker backend is constructed here, never
/// borrowed from an embedding or reranking lane's process pool.
pub struct TranscribeHost {
    profile: TranscribeProfile,
    mode: ObservationMode,
    backend: Backend,
    in_process_engine_invocations: Option<AtomicU64>,
}

impl TranscribeHost {
    /// Call from the lane's blocking task after load and revision validation.
    /// Direct ANE uses the shared compile scheduler and its usual leased LRU budget.
    pub fn admit_self_check(
        &self,
        frames: usize,
    ) -> Result<SelfCheckAdmission<'_>, TranscribeHostError> {
        if frames == 0 || frames > self.profile.max_encoder_frames {
            return Err(TranscribeHostError::Profile(
                "invalid self-check frame envelope".into(),
            ));
        }
        #[cfg(unix)]
        let lease = if let Backend::Ane(engine) = &self.backend {
            Some(engine.admit_self_check(frames)?)
        } else {
            None
        };
        Ok(SelfCheckAdmission {
            host: self,
            frames,
            #[cfg(unix)]
            _lease: lease,
        })
    }

    #[cfg(test)]
    pub(crate) fn in_process_for_test(
        profile: TranscribeProfile,
        run: impl Fn(TranscribeRequest) -> Result<TranscribeOutput, EngineError> + Send + 'static,
    ) -> Self {
        Self {
            profile,
            mode: ObservationMode::Certification,
            backend: Backend::InProcessTest(Box::new(run), None),
            in_process_engine_invocations: Some(AtomicU64::new(0)),
        }
    }

    /// Runs `hook` inside `unload`, so a test can hold an unload in progress.
    #[cfg(test)]
    pub(crate) fn with_unload_hook_for_test(mut self, hook: impl FnMut() + Send + 'static) -> Self {
        if let Backend::InProcessTest(_, unload) = &mut self.backend {
            *unload = Some(Box::new(hook));
        }
        self
    }

    #[cfg(all(test, unix))]
    pub(crate) fn worker_for_test(
        profile: TranscribeProfile,
        engine: WorkerEngine,
        model: LoadedModel,
    ) -> Self {
        Self {
            profile,
            mode: ObservationMode::Certification,
            backend: Backend::Worker(engine, model),
            in_process_engine_invocations: None,
        }
    }

    pub fn load_metal(
        profile: TranscribeProfile,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
        mode: ObservationMode,
    ) -> Result<Self, TranscribeHostError> {
        if profile.lane != "owned-metal" {
            return Err(TranscribeHostError::Profile(
                "expected Metal profile".into(),
            ));
        }
        let mut engine = OwnedMetalTranscribeEngine::new();
        let model = engine
            .load(artifact, &profile.runtime_config(cfg))
            .map_err(TranscribeHostError::Engine)?;
        Ok(Self {
            profile,
            mode,
            backend: Backend::Metal(engine, model),
            in_process_engine_invocations: (mode == ObservationMode::Certification)
                .then(|| AtomicU64::new(1)), // This host's successful load is its first invocation.
        })
    }

    pub fn load_worker(
        profile: TranscribeProfile,
        mut config: WorkerHostConfig,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
        mode: ObservationMode,
    ) -> Result<Self, TranscribeHostError> {
        let expected = match profile.lane.as_str() {
            "owned-cuda" => CUDA_WORKER_ENGINE,
            "owned-vulkan" => VULKAN_WORKER_ENGINE,
            _ => {
                return Err(TranscribeHostError::Profile(
                    "expected CUDA or Vulkan profile".into(),
                ))
            }
        };
        if config.engine_identity.as_ref().map(|i| i.engine.as_str()) != Some(expected) {
            return Err(TranscribeHostError::Profile(
                "worker identity does not match transcribe lane".into(),
            ));
        }
        profile.validate_max_frame(config.max_frame)?;
        isolate_worker(&mut config);
        let engine = WorkerEngine::new(config)?;
        let model = engine.load_transcribe(artifact, &profile.runtime_config(cfg))?;
        Ok(Self {
            profile,
            mode,
            backend: Backend::Worker(engine, model),
            in_process_engine_invocations: None,
        })
    }

    #[cfg(unix)]
    /// Use the same machine-wide supervisor as the embedding/reranking lanes.
    /// Transcription leases only its current request's shape, without pinning it.
    pub fn load_ane(
        profile: TranscribeProfile,
        mut config: WorkerHostConfig,
        supervisor: crate::worker_host::ane_residency::AneResidencySupervisor,
        artifact: &ValidatedArtifact,
        cfg: &RuntimeConfig,
        mode: ObservationMode,
    ) -> Result<Self, TranscribeHostError> {
        if profile.lane != "ane-direct-worker"
            || config.engine_identity.as_ref().map(|i| i.engine.as_str())
                != Some(ANE_DIRECT_WORKER_ENGINE)
        {
            return Err(TranscribeHostError::Profile(
                "expected direct-ANE identity and profile".into(),
            ));
        }
        profile.validate_max_frame(config.max_frame)?;
        isolate_worker(&mut config);
        let engine = crate::worker_host::ane_residency::DirectAneEngine::load(
            config,
            supervisor,
            artifact,
            &profile.runtime_config(cfg),
        )?;
        Ok(Self {
            profile,
            mode,
            backend: Backend::Ane(engine),
            in_process_engine_invocations: None,
        })
    }

    pub(crate) fn certify_counter(&self) -> Option<CertifyCounter> {
        match &self.backend {
            Backend::Worker(engine, _) => engine
                .request_count()
                .ok()
                .map(|(id, count)| CertifyCounter::WorkerRequests(id, count)),
            #[cfg(unix)]
            Backend::Ane(engine) => {
                use crate::worker_host::ane_residency::AneShapeWorker;
                let channel = &engine.serving.channel;
                Some(CertifyCounter::WorkerRequests(
                    channel.worker_id().to_owned(),
                    channel.request_count(),
                ))
            }
            _ => self.in_process_engine_invocations.as_ref().map(|count| {
                CertifyCounter::InProcessEngineInvocations(count.load(Ordering::Relaxed))
            }),
        }
    }

    pub(crate) fn quarantine_retry_after_ms(&self) -> Option<u64> {
        match &self.backend {
            Backend::Worker(engine, model) => engine.transcribe_quarantine_retry_after_ms(model),
            #[cfg(unix)]
            Backend::Ane(engine) => engine.serving.transcribe_quarantine_retry_after_ms(),
            _ => None,
        }
    }

    pub(crate) fn unload(&mut self) -> Result<(), TranscribeHostError> {
        match &mut self.backend {
            Backend::Metal(engine, model) => TranscribeEngine::unload(engine, model),
            Backend::Worker(engine, _) => engine.shutdown_for_pause()?,
            #[cfg(unix)]
            Backend::Ane(engine) => engine.unload()?,
            #[cfg(test)]
            Backend::InProcessTest(_, unload) => {
                if let Some(unload) = unload {
                    unload();
                }
            }
        }
        Ok(())
    }

    /// Run on a dedicated blocking thread that owns the engine call until it
    /// completes. A media job may stop waiting, but the thread must still drain
    /// the worker response before discarding that job's result.
    pub fn transcribe(
        &self,
        wav: &[u8],
        metal_progress: impl FnMut(Progress),
    ) -> Result<TranscribeResult, TranscribeHostError> {
        self.transcribe_mode(wav, self.mode, metal_progress)
    }

    /// Self-checks use the same audio and engine path but never request
    /// certification tensors, even on a certification-mode host.
    pub fn transcribe_self_check(
        &self,
        wav: &[u8],
        metal_progress: impl FnMut(Progress),
    ) -> Result<TranscribeResult, TranscribeHostError> {
        self.transcribe_mode(wav, ObservationMode::Serving, metal_progress)
    }

    pub fn profile(&self) -> &TranscribeProfile {
        &self.profile
    }

    pub(crate) fn ane_shapes(&self) -> Vec<serde_json::Value> {
        match &self.backend {
            #[cfg(unix)]
            Backend::Ane(engine) => engine.transcribe_shapes(),
            _ => Vec::new(),
        }
    }

    fn transcribe_mode(
        &self,
        wav: &[u8],
        mode: ObservationMode,
        metal_progress: impl FnMut(Progress),
    ) -> Result<TranscribeResult, TranscribeHostError> {
        let start = std::time::Instant::now();
        let prepared = prepare_wav(&self.profile, wav, mode)?;
        let frontend = start.elapsed();
        let mut result = self.transcribe_prepared(prepared, metal_progress)?;
        result
            .timing
            .boundaries(frontend, std::time::Duration::ZERO);
        Ok(result)
    }

    fn transcribe_prepared(
        &self,
        prepared: PreparedTranscription,
        mut metal_progress: impl FnMut(Progress),
    ) -> Result<TranscribeResult, TranscribeHostError> {
        let mut timing = ProgressTiming::default();
        let mut peak_device_memory_bytes = None;
        let mut peak_storage_buffer_bytes = None;
        let mut last = std::time::Instant::now();
        if let Some(count) = &self.in_process_engine_invocations {
            count.fetch_add(1, Ordering::Relaxed);
        }
        let (output, transcript_frame, observation, progress) = match &self.backend {
            #[cfg(test)]
            Backend::InProcessTest(run, _) => (
                run(prepared.request).map_err(TranscribeHostError::Engine)?,
                None,
                None,
                Vec::new(),
            ),
            Backend::Metal(engine, model) => {
                let output = engine
                    .transcribe_with_progress(model, prepared.request, &mut |progress| {
                        timing.signal(last.elapsed());
                        last = std::time::Instant::now();
                        metal_progress(progress);
                    })
                    .map_err(TranscribeHostError::Engine)?;
                let observation =
                    output
                        .observation
                        .as_ref()
                        .map(|o| TranscribeObservationFrames {
                            encoder_output: synapse_core::encode_f32_frame(&o.encoder_output),
                            decode_trace: synapse_core::encode_decode_trace_frame(&o.decode_trace),
                        });
                timing.finish(last.elapsed());
                (output, None, observation, Vec::new())
            }
            Backend::Worker(engine, model) => {
                let result = engine.transcribe_with_frames(
                    model,
                    prepared.request,
                    prepared.valid_encoder_frames,
                )?;
                timing = result.timing;
                peak_device_memory_bytes = result.peak_device_memory_bytes;
                peak_storage_buffer_bytes = result.peak_storage_buffer_bytes;
                (
                    result.output,
                    Some(result.transcript_frame),
                    result.observation,
                    result.progress,
                )
            }
            #[cfg(unix)]
            Backend::Ane(engine) => {
                let result = engine
                    .transcribe_with_frames(prepared.request, prepared.valid_encoder_frames)?;
                timing = result.timing;
                (
                    result.output,
                    Some(result.transcript_frame),
                    result.observation,
                    result.progress,
                )
            }
        };
        #[cfg(test)]
        if matches!(&self.backend, Backend::InProcessTest(..)) {
            timing.finish(last.elapsed());
        }
        let tail = std::time::Instant::now();
        let mut result = finish(
            output,
            transcript_frame,
            observation,
            progress,
            prepared.valid_encoder_frames,
            prepared.audio_duration_ms,
        )?;
        timing.boundaries(std::time::Duration::ZERO, tail.elapsed());
        result.timing = timing;
        result.peak_device_memory_bytes = peak_device_memory_bytes;
        result.peak_storage_buffer_bytes = peak_storage_buffer_bytes;
        Ok(result)
    }
}

fn isolate_worker(config: &mut WorkerHostConfig) {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let next = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    config.worker_id = format!("{}-transcribe-{next}", config.worker_id);
    config.isolate_crash_key_by_worker_id = true;
    // Check the task-specific revision in the load reply (LOADED), rather than
    // the connection handshake (HELLO), whose identity covers existing tasks.
    config.expected_hello_binding = None;
}

fn finish(
    output: TranscribeOutput,
    transcript_frame: Option<Vec<u8>>,
    observation: Option<TranscribeObservationFrames>,
    progress: Vec<WorkerResponse>,
    expected_frames: usize,
    duration_ms: f64,
) -> Result<TranscribeResult, TranscribeHostError> {
    if output.valid_encoder_frames != expected_frames {
        return Err(WorkerHostError::Protocol(
            "valid_encoder_frames differs from shared speech core".into(),
        )
        .into());
    }
    let triples = output
        .triples
        .iter()
        .map(|t| {
            if t.token_id < 0
                || t.frame_index < 0
                || t.frame_index as usize >= expected_frames
                || t.duration < 0
            {
                return Err(synapse_speech::Error::InvalidJoint(
                    "invalid emitted triple",
                ));
            }
            Ok([t.token_id as u32, t.frame_index as u32, t.duration as u32])
        })
        .collect::<synapse_speech::Result<Vec<_>>>()?;
    let transcript = Tokenizer::pinned().transcript(&triples, duration_ms)?;
    Ok(TranscribeResult {
        transcript,
        transcript_frame: transcript_frame
            .unwrap_or_else(|| synapse_core::encode_transcript_frame(&output.triples)),
        output,
        observation,
        progress,
        timing: ProgressTiming::default(),
        peak_device_memory_bytes: None,
        peak_storage_buffer_bytes: None,
    })
}

#[cfg(test)]
mod tests;
