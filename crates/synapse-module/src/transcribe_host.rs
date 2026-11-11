//! Audio admission and serving seam shared by catalog self-checks and media jobs.
//! Workers only see normalized features; container bytes and text stay here.

use serde::Serialize;
use synapse_core::{
    worker_engine_names::*, EngineError, LoadedModel, RuntimeConfig, TranscribeEngine,
    TranscribeOutput, TranscribeRequest, TranscribeRevision, ValidatedArtifact, WorkerResponse,
};
use synapse_engine_owned::transcribe::{OwnedMetalTranscribeEngine, Progress};
use synapse_speech::{Frontend, Tokenizer, Transcript, DECODE_RULE_REVISION, MEL_BINS};

use crate::worker_host::{WorkerEngine, WorkerHostConfig, WorkerHostError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationMode {
    Serving,
    Certification,
}

#[derive(Clone, Debug)]
pub struct TranscribeProfile {
    pub id: String,
    pub lane: String,
    pub max_duration_ms: u64,
    pub max_encoder_frames: usize,
    pub kernel_revision: String,
    pub decode_rule: String,
    storage_dtype: String,
}

impl TranscribeProfile {
    /// Resolve the numerical envelope from the pinned profile, not token limits.
    pub fn pinned(id: &str) -> Result<Self, TranscribeHostError> {
        let manifest: serde_json::Value =
            serde_json::from_str(include_str!("../../../bench/parity/models.json"))
                .expect("pinned parity manifest");
        let profile = &manifest["profiles"][id];
        let speech = &profile["transcribe"];
        let lane = profile["lane"].as_str().unwrap_or_default();
        let kernel = match lane {
            "owned-metal" => METAL_TRANSCRIBE_KERNEL_REVISION,
            "owned-cuda" => CUDA_TRANSCRIBE_KERNEL_REVISION,
            "owned-vulkan" => VULKAN_TRANSCRIBE_KERNEL_REVISION,
            "ane-direct-worker" => ANE_DIRECT_TRANSCRIBE_KERNEL_REVISION,
            _ => {
                return Err(TranscribeHostError::Profile(
                    "unknown transcribe lane".into(),
                ))
            }
        };
        let duration = speech["max_duration_ms"]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or_else(|| TranscribeHostError::Profile("missing duration envelope".into()))?;
        let decode_rule = speech["confirmations"]["decode_rule"]["revision"]
            .as_str()
            .ok_or_else(|| TranscribeHostError::Profile("missing decode rule".into()))?;
        if profile["model"] != "parakeet-tdt-0.6b-v3" || decode_rule != DECODE_RULE_REVISION {
            return Err(TranscribeHostError::Profile(
                "unsupported speech model or decode rule".into(),
            ));
        }
        let samples = usize::try_from(
            duration
                .checked_mul(16)
                .ok_or_else(|| TranscribeHostError::Profile("duration overflow".into()))?,
        )
        .map_err(|_| TranscribeHostError::Profile("duration overflow".into()))?;
        let max_encoder_frames = synapse_speech::valid_encoder_frames(samples as u64) as usize;
        if crate::worker_host::ane_residency::ladder_rung(max_encoder_frames).is_none() {
            return Err(TranscribeHostError::Profile(
                "envelope exceeds encoder ladder".into(),
            ));
        }
        Ok(Self {
            id: id.into(),
            lane: lane.into(),
            max_duration_ms: duration,
            max_encoder_frames,
            kernel_revision: kernel.into(),
            decode_rule: decode_rule.into(),
            storage_dtype: profile["storage_dtype"]
                .as_str()
                .ok_or_else(|| TranscribeHostError::Profile("missing storage dtype".into()))?
                .into(),
        })
    }

    pub fn runtime_config(&self, base: &RuntimeConfig) -> RuntimeConfig {
        let mut cfg = base.clone();
        cfg.values.extend([
            ("operation".into(), "transcribe".into()),
            ("profile".into(), self.id.clone()),
            ("family".into(), "parakeet-tdt".into()),
            ("dtype".into(), self.storage_dtype.clone()),
            (
                "max_encoder_frames".into(),
                self.max_encoder_frames.to_string(),
            ),
            (
                "execution".into(),
                if self.lane == "owned-metal" {
                    "explicit"
                } else {
                    "supervised"
                }
                .into(),
            ),
        ]);
        cfg
    }

    /// Include the engine's largest rung capacity even though requests themselves
    /// are never rung-padded. Reserve the front end's trailing mel frame as well.
    pub fn validate_max_frame(&self, max_frame: u32) -> Result<(), WorkerHostError> {
        let rung = crate::worker_host::ane_residency::ladder_rung(self.max_encoder_frames)
            .ok_or_else(|| named("transcribe_frame_limit", "envelope exceeds ladder"))?;
        let bytes = (rung * 8 + 1).checked_mul(MEL_BINS * 4);
        if bytes.is_none_or(|bytes| bytes > max_frame as usize) {
            return Err(named(
                "transcribe_frame_limit",
                "largest padded mel frame exceeds negotiated max_frame",
            ));
        }
        Ok(())
    }
}

fn named(code: &str, message: &str) -> WorkerHostError {
    WorkerHostError::WorkerErr {
        code: code.into(),
        msg: message.into(),
    }
}

pub(crate) fn validate_worker_load(
    cfg: &RuntimeConfig,
    max_frame: u32,
    revision: Option<&TranscribeRevision>,
) -> Result<(), WorkerHostError> {
    if cfg.values.get("operation").map(String::as_str) != Some("transcribe") {
        return Ok(());
    }
    let profile = TranscribeProfile::pinned(
        cfg.values
            .get("profile")
            .map(String::as_str)
            .unwrap_or_default(),
    )
    .map_err(|e| named("transcribe_profile_invalid", &e.to_string()))?;
    profile.validate_max_frame(max_frame)?;
    if revision.is_none_or(|r| {
        r.kernel_revision != profile.kernel_revision || r.decode_rule != profile.decode_rule
    }) {
        return Err(named(
            "transcribe_revision_mismatch",
            "LOADED transcription kernel_revision or decode_rule differs from profile",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DurationLimit {
    pub duration_ms: u64,
    pub max_duration_ms: u64,
    pub valid_encoder_frames: usize,
    pub max_encoder_frames: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum TranscribeHostError {
    #[error("invalid_params: audio: {0}")]
    Audio(#[from] synapse_speech::Error),
    #[error("invalid_params: audio duration limit: {0:?}")]
    DurationLimit(DurationLimit),
    #[error("transcribe_profile_invalid: {0}")]
    Profile(String),
    #[error("{0}")]
    Worker(#[from] WorkerHostError),
    #[error("{0:?}")]
    Engine(EngineError),
    #[cfg(unix)]
    #[error("{0}")]
    Ane(#[from] crate::worker_host::ane_residency::AneResidencyError),
}

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
    if duration_ms > profile.max_duration_ms || valid_encoder_frames > profile.max_encoder_frames {
        return Err(TranscribeHostError::DurationLimit(DurationLimit {
            duration_ms,
            max_duration_ms: profile.max_duration_ms,
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

/// Raw observation frames are retained exactly as received so callers can split
/// them into bounded result pages and hash the original bytes. They are absent
/// outside certification observation mode.
#[derive(Default, Debug)]
pub struct TranscribeObservationFrames {
    pub encoder_output: Vec<u8>,
    pub decode_trace: Vec<u8>,
}

#[derive(Debug)]
pub struct TranscribeResult {
    pub transcript: Transcript,
    pub output: TranscribeOutput,
    pub transcript_frame: Vec<u8>,
    pub observation: Option<TranscribeObservationFrames>,
    pub progress: Vec<WorkerResponse>,
}

enum Backend {
    Metal(OwnedMetalTranscribeEngine, LoadedModel),
    Worker(WorkerEngine, LoadedModel),
    #[cfg(unix)]
    Ane(crate::worker_host::ane_residency::DirectAneEngine),
    #[cfg(test)]
    InProcessTest(Box<dyn Fn(TranscribeRequest) -> Result<TranscribeOutput, EngineError> + Send>),
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
            backend: Backend::InProcessTest(Box::new(run)),
        }
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
        })
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

    fn transcribe_mode(
        &self,
        wav: &[u8],
        mode: ObservationMode,
        metal_progress: impl FnMut(Progress),
    ) -> Result<TranscribeResult, TranscribeHostError> {
        let prepared = prepare_wav(&self.profile, wav, mode)?;
        self.transcribe_prepared(prepared, metal_progress)
    }

    fn transcribe_prepared(
        &self,
        prepared: PreparedTranscription,
        mut metal_progress: impl FnMut(Progress),
    ) -> Result<TranscribeResult, TranscribeHostError> {
        let (output, transcript_frame, observation, progress) = match &self.backend {
            #[cfg(test)]
            Backend::InProcessTest(run) => (
                run(prepared.request).map_err(TranscribeHostError::Engine)?,
                None,
                None,
                Vec::new(),
            ),
            Backend::Metal(engine, model) => {
                let output = engine
                    .transcribe_with_progress(model, prepared.request, &mut metal_progress)
                    .map_err(TranscribeHostError::Engine)?;
                let observation =
                    output
                        .observation
                        .as_ref()
                        .map(|o| TranscribeObservationFrames {
                            encoder_output: synapse_core::encode_f32_frame(&o.encoder_output),
                            decode_trace: synapse_core::encode_decode_trace_frame(&o.decode_trace),
                        });
                (output, None, observation, Vec::new())
            }
            Backend::Worker(engine, model) => {
                let result = engine.transcribe_with_frames(
                    model,
                    prepared.request,
                    prepared.valid_encoder_frames,
                )?;
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
                (
                    result.output,
                    Some(result.transcript_frame),
                    result.observation,
                    result.progress,
                )
            }
        };
        finish(
            output,
            transcript_frame,
            observation,
            progress,
            prepared.valid_encoder_frames,
            prepared.audio_duration_ms,
        )
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
    })
}

#[cfg(test)]
mod tests;
