//! Catalog-independent transcription profiles and worker exchange validation.

use serde::Serialize;
use synapse_core::{worker_engine_names::*, EngineError, RuntimeConfig, TranscribeRevision};
use synapse_speech::{DECODE_RULE_REVISION, MEL_BINS};

use crate::worker_host::WorkerHostError;

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

/// Raw observation frames are retained exactly as received so callers can split
/// them into bounded result pages and hash the original bytes. They are absent
/// outside certification observation mode.
#[derive(Default, Debug)]
pub struct TranscribeObservationFrames {
    pub encoder_output: Vec<u8>,
    pub decode_trace: Vec<u8>,
}

#[derive(Default, Debug)]
pub struct ProgressTiming {
    first: Option<std::time::Duration>,
    longest: std::time::Duration,
    last: std::time::Duration,
}

impl ProgressTiming {
    pub fn signal(&mut self, gap: std::time::Duration) {
        self.first.get_or_insert(gap);
        self.longest = self.longest.max(gap);
    }

    pub fn finish(&mut self, gap: std::time::Duration) {
        self.last = gap;
        self.longest = self.longest.max(gap);
    }

    pub fn boundaries(&mut self, before: std::time::Duration, after: std::time::Duration) {
        self.last += after;
        self.longest = if let Some(first) = self.first.as_mut() {
            *first += before;
            self.longest.max(*first).max(self.last)
        } else {
            self.longest + before + after
        };
    }

    pub fn longest_ms(&self) -> u64 {
        self.longest.as_millis().try_into().unwrap_or(u64::MAX)
    }
}

#[cfg(test)]
#[path = "exchange_tests.rs"]
mod tests;
