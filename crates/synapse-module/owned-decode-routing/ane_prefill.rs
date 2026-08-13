//! ANE-prefill routing, completed-processing identity, and split-arm health.
//!
//! This module is deliberately a pure routing boundary: it selects a certified
//! ANE-prefill arm before dispatch and translates the terminal attempt outcome
//! into provenance. The worker owns CoreML execution; this layer owns the
//! global-first gate order, bucket escalation, artifact eligibility, deadline
//! feasibility, and persistent-in-shape per-arm health semantics.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use synapse_core::Fingerprint;

use crate::owned_decode_routing::family::Family;
use crate::owned_decode_routing::request::SamplingMode;

/// The machine-level prefill mode. `Gpu` is the portable default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DecodePrefill {
    #[default]
    Gpu,
    AneSplit,
}

/// Whether this process can execute an ANE split arm.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AnePlatform {
    #[default]
    Supported,
    Unsupported,
}

/// The three fixed full-window CoreML packages. They are ordered so routing can
/// always escalate from the smallest fitting window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrefillBucket {
    W128,
    W256,
    W512,
}

impl PrefillBucket {
    /// All included windows in mandatory selection order.
    pub const ALL: [Self; 3] = [Self::W128, Self::W256, Self::W512];

    /// Fixed prompt width represented by this package.
    pub const fn tokens(self) -> u32 {
        match self {
            Self::W128 => 128,
            Self::W256 => 256,
            Self::W512 => 512,
        }
    }

    /// Completed-prefill provenance name for this exact package.
    pub const fn engine_name(self) -> &'static str {
        match self {
            Self::W128 => "ane-w128",
            Self::W256 => "ane-w256",
            Self::W512 => "ane-w512",
        }
    }
}

/// The decode-engine configuration in the split-arm key. This remains more
/// precise than the compiled package: one package can serve both configurations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SplitDecodeConfig {
    F16Step,
    Q8Step,
}

impl SplitDecodeConfig {
    /// Canonical evidence and ops label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16Step => "f16-step",
            Self::Q8Step => "q8-step",
        }
    }
}

/// The exact identity for certification, health, and quarantine. No state is
/// inherited between buckets, decode configurations, or machine profiles.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitArmKey {
    pub machine_profile: String,
    pub family: Family,
    pub bucket: PrefillBucket,
    pub decode_config: SplitDecodeConfig,
}

impl SplitArmKey {
    /// Build one exact arm key.
    #[must_use]
    pub fn new(
        machine_profile: impl Into<String>,
        family: Family,
        bucket: PrefillBucket,
        decode_config: SplitDecodeConfig,
    ) -> Self {
        Self {
            machine_profile: machine_profile.into(),
            family,
            bucket,
            decode_config,
        }
    }
}

/// The two immutable identities recorded by a green certification row.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertifiedArtifactIdentity {
    /// Source checkpoint digest used for certification.
    pub source_checkpoint_digest: String,
    /// Digest of the compiled artifact recorded by certification.
    pub certification_recorded_artifact_digest: String,
}

/// The source checkpoint selected by the manifest and the optional compiled
/// runtime artifact. `None` denotes an unloaded arm, which is eligible for
/// bounded readiness only when the source checkpoint still matches.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeArtifactIdentity {
    /// Source checkpoint digest selected by the runtime manifest.
    pub manifest_source_checkpoint_digest: String,
    /// Digest of a loaded or newly-built compiled artifact, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_or_compiled_artifact_digest: Option<String>,
}

impl RuntimeArtifactIdentity {
    fn is_source_match(&self, certified: &CertifiedArtifactIdentity) -> bool {
        self.manifest_source_checkpoint_digest == certified.source_checkpoint_digest
    }

    fn is_loaded_digest_match(&self, certified: &CertifiedArtifactIdentity) -> bool {
        self.derived_or_compiled_artifact_digest
            .as_deref()
            .is_none_or(|digest| digest == certified.certification_recorded_artifact_digest)
    }

    fn is_selectable(&self, certified: &CertifiedArtifactIdentity) -> bool {
        self.is_source_match(certified) && self.is_loaded_digest_match(certified)
    }

    fn readiness_required(&self) -> bool {
        self.derived_or_compiled_artifact_digest.is_none()
    }
}

/// The three disjoint runtime certification states for an exact split arm.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitArmCertification {
    Certified {
        artifacts: CertifiedArtifactIdentity,
    },
    BucketAbsent,
    NotCertified,
}

/// Consecutive-only health state for an exact split arm. This is intentionally
/// separate from the existing worker-key crash budget, whose cumulative
/// saturating semantics must remain unchanged.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitArmHealth {
    pub consecutive_strikes: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quarantined_until_ms: Option<u64>,
    #[serde(default)]
    pub probation: bool,
}

impl SplitArmHealth {
    fn refresh_expiry(&mut self, now_ms: u64, policy: SplitHealthPolicy) {
        if self
            .quarantined_until_ms
            .is_some_and(|until_ms| until_ms <= now_ms)
        {
            self.quarantined_until_ms = None;
            self.consecutive_strikes = policy.max_strikes.saturating_sub(1);
            self.probation = true;
        }
    }

    fn is_quarantined(&mut self, now_ms: u64, policy: SplitHealthPolicy) -> bool {
        self.refresh_expiry(now_ms, policy);
        self.quarantined_until_ms.is_some()
    }

    fn record_failure(&mut self, now_ms: u64, policy: SplitHealthPolicy) {
        self.refresh_expiry(now_ms, policy);
        self.consecutive_strikes = self
            .consecutive_strikes
            .saturating_add(1)
            .min(policy.max_strikes);
        if self.consecutive_strikes >= policy.max_strikes {
            self.quarantined_until_ms = Some(now_ms.saturating_add(policy.quarantine_duration_ms));
            self.probation = false;
        }
    }

    fn record_success(&mut self) {
        self.consecutive_strikes = 0;
        self.quarantined_until_ms = None;
        self.probation = false;
    }
}

/// Health policy for the new split-arm key space. Zero strikes would make every
/// arm permanently quarantined, so construction rejects it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitHealthPolicy {
    pub max_strikes: u32,
    pub quarantine_duration_ms: u64,
}

impl SplitHealthPolicy {
    /// Construct a valid split-arm health policy.
    pub fn new(max_strikes: u32, quarantine_duration_ms: u64) -> Result<Self, &'static str> {
        if max_strikes == 0 {
            return Err("split-arm max_strikes must be greater than zero");
        }
        Ok(Self {
            max_strikes,
            quarantine_duration_ms,
        })
    }
}

impl Default for SplitHealthPolicy {
    fn default() -> Self {
        Self {
            max_strikes: 2,
            quarantine_duration_ms: 60_000,
        }
    }
}

/// State used during routing for one exact arm.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitArmState {
    pub certification: SplitArmCertification,
    pub runtime_artifacts: RuntimeArtifactIdentity,
    /// The exact decode cache capacity selected for this arm. A candidate needs
    /// at least `prompt_token_count + max_tokens` positions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_cache_bucket: Option<u32>,
    #[serde(default)]
    pub health: SplitArmHealth,
}

impl SplitArmState {
    fn has_fitting_cache(&self, required_positions: u32) -> bool {
        self.decode_cache_bucket
            .is_some_and(|capacity| capacity >= required_positions)
    }

    fn is_selectable(&mut self, now_ms: u64, policy: SplitHealthPolicy) -> bool {
        let SplitArmCertification::Certified { artifacts } = &self.certification else {
            return false;
        };
        self.runtime_artifacts.is_selectable(artifacts)
            && !self.health.is_quarantined(now_ms, policy)
    }

    fn terminal_reason(&mut self, now_ms: u64, policy: SplitHealthPolicy) -> PrefillBypassReason {
        match &self.certification {
            SplitArmCertification::BucketAbsent => PrefillBypassReason::BucketAbsent,
            SplitArmCertification::NotCertified => PrefillBypassReason::NotCertified,
            SplitArmCertification::Certified { artifacts } => {
                if !self.runtime_artifacts.is_selectable(artifacts) {
                    PrefillBypassReason::ArtifactDigestMismatch
                } else if self.health.is_quarantined(now_ms, policy) {
                    PrefillBypassReason::Quarantined
                } else {
                    // A matching, non-quarantined green arm is selectable, so this
                    // branch can only be reached if a future eligibility gate is
                    // added without updating the selection loop.
                    PrefillBypassReason::NotCertified
                }
            }
        }
    }
}

/// Calibrated timing inputs and their binding derived budgets. The same derived
/// ANE attempt budget bounds guard waiting and CoreML prediction; there is no
/// independently configured guard-wait knob.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitTimingBudgets {
    pub calibrated_prediction_p95_ms: u64,
    pub ane_attempt_budget_ms: u64,
    pub readiness_budget_ms: u64,
    pub calibrated_handoff_p95_ms: u64,
    pub handoff_budget_ms: u64,
    pub measured_gpu_prefill_p95_ms: u64,
}

impl SplitTimingBudgets {
    /// Derive all request-time bounds from calibration and loader configuration.
    #[must_use]
    pub fn from_calibration(
        calibrated_prediction_p95_ms: u64,
        readiness_budget_ms: u64,
        calibrated_handoff_p95_ms: u64,
        measured_gpu_prefill_p95_ms: u64,
    ) -> Self {
        Self {
            calibrated_prediction_p95_ms,
            ane_attempt_budget_ms: calibrated_prediction_p95_ms.saturating_mul(2),
            readiness_budget_ms,
            calibrated_handoff_p95_ms,
            handoff_budget_ms: calibrated_handoff_p95_ms.saturating_mul(2),
            measured_gpu_prefill_p95_ms,
        }
    }

    /// Upper bound after a guard has been acquired, including the GPU work that
    /// may still be needed for a transparent fallback.
    #[must_use]
    pub const fn remaining_after_guard_ms(self) -> u64 {
        self.readiness_budget_ms
            .saturating_add(self.ane_attempt_budget_ms)
            .saturating_add(self.handoff_budget_ms)
            .saturating_add(self.measured_gpu_prefill_p95_ms)
    }

    /// Full split-path ceiling before guard acquisition. The guard and
    /// prediction each use the same derived attempt budget.
    #[must_use]
    pub const fn full_split_ceiling_ms(self) -> u64 {
        self.ane_attempt_budget_ms
            .saturating_add(self.readiness_budget_ms)
            .saturating_add(self.ane_attempt_budget_ms)
            .saturating_add(self.handoff_budget_ms)
    }
}

/// Result of attempting bounded ANE guard acquisition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardAcquisition {
    pub acquired: bool,
    pub waited_ms: u64,
}

/// ANE guard seam. Implementations must use `max_wait_ms` as their queue-wait
/// bound and report the elapsed time that was actually consumed.
pub trait AneGuard {
    fn acquire_within(&mut self, max_wait_ms: u64) -> GuardAcquisition;
}

/// Caller inputs that affect split routing.
#[derive(Clone, Debug, PartialEq)]
pub struct AneSplitRequest {
    pub family: Family,
    pub decode_config: SplitDecodeConfig,
    pub prompt_token_count: u32,
    pub max_tokens: u32,
    pub sampling: SamplingMode,
    /// An existing exact processing pin, if the caller surface exposed one.
    pub required_processing_fingerprint: Option<Fingerprint>,
    /// Remaining caller deadline immediately before guard acquisition. `None`
    /// means the request has no deadline feasibility constraint.
    pub deadline_remaining_ms: Option<u64>,
}

/// The complete machine-local split-routing inputs.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AneSplitRoutingConfig {
    pub decode_prefill: DecodePrefill,
    pub platform: AnePlatform,
    pub machine_profile: String,
    pub timing: SplitTimingBudgets,
    pub health_policy: SplitHealthPolicy,
}

/// The completed prefill engine in response provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PrefillEngine {
    Gpu,
    AneW128,
    AneW256,
    AneW512,
}

impl PrefillEngine {
    fn for_bucket(bucket: PrefillBucket) -> Self {
        match bucket {
            PrefillBucket::W128 => Self::AneW128,
            PrefillBucket::W256 => Self::AneW256,
            PrefillBucket::W512 => Self::AneW512,
        }
    }
}

/// The closed pre-attempt provenance vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillBypassReason {
    Disabled,
    PlatformUnsupported,
    FamilyUnsupported,
    SamplingUncertified,
    IdentityPinnedGpu,
    PromptOverMaxBucket,
    NoFittingCacheBucket,
    BucketAbsent,
    NotCertified,
    ArtifactDigestMismatch,
    Quarantined,
    AneBusy,
    DeadlineTooTight,
}

impl PrefillBypassReason {
    /// Every bypass reason, for table-driven acceptance tests.
    pub const ALL: [Self; 13] = [
        Self::Disabled,
        Self::PlatformUnsupported,
        Self::FamilyUnsupported,
        Self::SamplingUncertified,
        Self::IdentityPinnedGpu,
        Self::PromptOverMaxBucket,
        Self::NoFittingCacheBucket,
        Self::BucketAbsent,
        Self::NotCertified,
        Self::ArtifactDigestMismatch,
        Self::Quarantined,
        Self::AneBusy,
        Self::DeadlineTooTight,
    ];

    /// Canonical response token.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::PlatformUnsupported => "platform_unsupported",
            Self::FamilyUnsupported => "family_unsupported",
            Self::SamplingUncertified => "sampling_uncertified",
            Self::IdentityPinnedGpu => "identity_pinned_gpu",
            Self::PromptOverMaxBucket => "prompt_over_max_bucket",
            Self::NoFittingCacheBucket => "no_fitting_cache_bucket",
            Self::BucketAbsent => "bucket_absent",
            Self::NotCertified => "not_certified",
            Self::ArtifactDigestMismatch => "artifact_digest_mismatch",
            Self::Quarantined => "quarantined",
            Self::AneBusy => "ane_busy",
            Self::DeadlineTooTight => "deadline_too_tight",
        }
    }
}

/// The closed in-attempt provenance vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefillFallbackReason {
    CompileFailure,
    LoadFailure,
    DispatchFailure,
    PredictionFailure,
    PredictionTimeout,
    KvConversionFailure,
    IpcHandoffFailure,
    CacheHandoffFailure,
    MetalUploadFailure,
    TransferBudgetExceeded,
    ReadinessBudgetExhausted,
    ArtifactMismatch,
    LogitsPublicationFailure,
}

impl PrefillFallbackReason {
    /// Every fallback reason, for table-driven fault-map tests.
    pub const ALL: [Self; 13] = [
        Self::CompileFailure,
        Self::LoadFailure,
        Self::DispatchFailure,
        Self::PredictionFailure,
        Self::PredictionTimeout,
        Self::KvConversionFailure,
        Self::IpcHandoffFailure,
        Self::CacheHandoffFailure,
        Self::MetalUploadFailure,
        Self::TransferBudgetExceeded,
        Self::ReadinessBudgetExhausted,
        Self::ArtifactMismatch,
        Self::LogitsPublicationFailure,
    ];

    /// Canonical response token.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CompileFailure => "compile_failure",
            Self::LoadFailure => "load_failure",
            Self::DispatchFailure => "dispatch_failure",
            Self::PredictionFailure => "prediction_failure",
            Self::PredictionTimeout => "prediction_timeout",
            Self::KvConversionFailure => "kv_conversion_failure",
            Self::IpcHandoffFailure => "ipc_handoff_failure",
            Self::CacheHandoffFailure => "cache_handoff_failure",
            Self::MetalUploadFailure => "metal_upload_failure",
            Self::TransferBudgetExceeded => "transfer_budget_exceeded",
            Self::ReadinessBudgetExhausted => "readiness_budget_exhausted",
            Self::ArtifactMismatch => "artifact_mismatch",
            Self::LogitsPublicationFailure => "logits_publication_failure",
        }
    }
}

/// Low-level fault categories from the worker boundary. Their mapping is closed
/// so routing never chooses an arbitrary fallback reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnePrefillFault {
    Compile,
    Load,
    Dispatch,
    Prediction,
    PredictionTimeout,
    KvConversion,
    IpcHandoff,
    CacheHandoff,
    MetalUpload,
    HandoffBudget,
    ReadinessBudget,
    ArtifactMismatch,
    LogitsPublication,
}

impl AnePrefillFault {
    /// Every mapped fault category.
    pub const ALL: [Self; 13] = [
        Self::Compile,
        Self::Load,
        Self::Dispatch,
        Self::Prediction,
        Self::PredictionTimeout,
        Self::KvConversion,
        Self::IpcHandoff,
        Self::CacheHandoff,
        Self::MetalUpload,
        Self::HandoffBudget,
        Self::ReadinessBudget,
        Self::ArtifactMismatch,
        Self::LogitsPublication,
    ];

    /// The sole fallback reason for this fault.
    pub const fn fallback_reason(self) -> PrefillFallbackReason {
        match self {
            Self::Compile => PrefillFallbackReason::CompileFailure,
            Self::Load => PrefillFallbackReason::LoadFailure,
            Self::Dispatch => PrefillFallbackReason::DispatchFailure,
            Self::Prediction => PrefillFallbackReason::PredictionFailure,
            Self::PredictionTimeout => PrefillFallbackReason::PredictionTimeout,
            Self::KvConversion => PrefillFallbackReason::KvConversionFailure,
            Self::IpcHandoff => PrefillFallbackReason::IpcHandoffFailure,
            Self::CacheHandoff => PrefillFallbackReason::CacheHandoffFailure,
            Self::MetalUpload => PrefillFallbackReason::MetalUploadFailure,
            Self::HandoffBudget => PrefillFallbackReason::TransferBudgetExceeded,
            Self::ReadinessBudget => PrefillFallbackReason::ReadinessBudgetExhausted,
            Self::ArtifactMismatch => PrefillFallbackReason::ArtifactMismatch,
            Self::LogitsPublication => PrefillFallbackReason::LogitsPublicationFailure,
        }
    }
}

/// Additive response provenance for prefill completion. Its constructors make a
/// bypass and a fallback mutually exclusive and always report GPU as the
/// completed engine after substitution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillProvenance {
    pub prefill_engine: PrefillEngine,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_bucket: Option<PrefillBucket>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_bypass_reason: Option<PrefillBypassReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_fallback_from: Option<PrefillEngine>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_fallback_reason: Option<PrefillFallbackReason>,
}

impl PrefillProvenance {
    fn gpu_bypass(reason: PrefillBypassReason) -> Self {
        Self {
            prefill_engine: PrefillEngine::Gpu,
            prefill_bucket: None,
            prefill_bypass_reason: Some(reason),
            prefill_fallback_from: None,
            prefill_fallback_reason: None,
        }
    }

    fn gpu_fallback(bucket: PrefillBucket, reason: PrefillFallbackReason) -> Self {
        Self {
            prefill_engine: PrefillEngine::Gpu,
            prefill_bucket: None,
            prefill_bypass_reason: None,
            prefill_fallback_from: Some(PrefillEngine::for_bucket(bucket)),
            prefill_fallback_reason: Some(reason),
        }
    }

    fn split_success(bucket: PrefillBucket) -> Self {
        Self {
            prefill_engine: PrefillEngine::for_bucket(bucket),
            prefill_bucket: Some(bucket),
            prefill_bypass_reason: None,
            prefill_fallback_from: None,
            prefill_fallback_reason: None,
        }
    }

    /// Verify the additive response fields describe exactly one completed path.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        match self.prefill_engine {
            PrefillEngine::Gpu => {
                self.prefill_bucket.is_none()
                    && matches!(
                        (
                            self.prefill_bypass_reason,
                            self.prefill_fallback_from,
                            self.prefill_fallback_reason,
                        ),
                        (None, None, None) | (Some(_), None, None) | (None, Some(_), Some(_))
                    )
            }
            PrefillEngine::AneW128 | PrefillEngine::AneW256 | PrefillEngine::AneW512 => {
                self.prefill_bucket.is_some()
                    && self.prefill_bypass_reason.is_none()
                    && self.prefill_fallback_from.is_none()
                    && self.prefill_fallback_reason.is_none()
            }
        }
    }
}

/// A failed exact pin preserves the requested identity rather than substituting
/// GPU. A pre-attempt class consumes no health; an execution class is charged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityPreservingFailure {
    FingerprintMismatch,
    PreAttempt(PrefillBypassReason),
    Execution(PrefillFallbackReason),
}

/// A selected split attempt. It is intentionally not `Clone`: completion
/// consumes it, preventing one attempt result from being charged twice.
#[derive(Debug)]
pub struct AneSplitAttempt {
    pub arm: SplitArmKey,
    pub readiness_required: bool,
    pub processing_fingerprint: Fingerprint,
    pinned_ane_split: bool,
}

/// The pre-dispatch outcome of split routing.
#[derive(Debug)]
pub enum AneSplitRouteDecision {
    GpuBypass {
        provenance: PrefillProvenance,
        processing_fingerprint: Fingerprint,
    },
    Attempt(AneSplitAttempt),
    IdentityPreservingFailure(IdentityPreservingFailure),
}

/// The worker's terminal outcome for an attempt that has already started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AneSplitAttemptResult {
    Success,
    Failure(PrefillFallbackReason),
}

/// The routed completion after health accounting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AneSplitCompletion {
    SplitSuccess {
        provenance: PrefillProvenance,
        processing_fingerprint: Fingerprint,
    },
    GpuFallback {
        provenance: PrefillProvenance,
        processing_fingerprint: Fingerprint,
    },
    IdentityPreservingFailure(IdentityPreservingFailure),
}

/// Pure split routing and exact-arm health state. Its state map can be persisted
/// as the split-arm health table without rewriting worker-key crash records.
pub struct AnePrefillRouter {
    config: AneSplitRoutingConfig,
    gpu_processing_fingerprint: Fingerprint,
    ane_split_processing_fingerprint: Fingerprint,
    arms: BTreeMap<SplitArmKey, SplitArmState>,
    platform_inert_notice_emitted: bool,
}

impl AnePrefillRouter {
    /// Construct a router over exact split-arm state.
    #[must_use]
    pub fn new(
        config: AneSplitRoutingConfig,
        gpu_processing_fingerprint: Fingerprint,
        ane_split_processing_fingerprint: Fingerprint,
        arms: BTreeMap<SplitArmKey, SplitArmState>,
    ) -> Self {
        Self {
            config,
            gpu_processing_fingerprint,
            ane_split_processing_fingerprint,
            arms,
            platform_inert_notice_emitted: false,
        }
    }

    /// Return true exactly once when an ANE-split configuration is inert on this
    /// platform. The host uses this to emit one process-local notice while every
    /// request remains a typed GPU `platform_unsupported` bypass.
    pub fn take_platform_inert_notice(&mut self) -> bool {
        if self.config.decode_prefill == DecodePrefill::AneSplit
            && self.config.platform == AnePlatform::Unsupported
            && !self.platform_inert_notice_emitted
        {
            self.platform_inert_notice_emitted = true;
            return true;
        }
        false
    }

    /// Read an exact arm's state for operations reporting.
    #[must_use]
    pub fn arm(&self, key: &SplitArmKey) -> Option<&SplitArmState> {
        self.arms.get(key)
    }

    fn pinned_engine_class(&self, pin: Option<&Fingerprint>) -> Result<Option<bool>, ()> {
        match pin {
            None => Ok(None),
            Some(pin) if pin == &self.gpu_processing_fingerprint => Ok(Some(false)),
            Some(pin) if pin == &self.ane_split_processing_fingerprint => Ok(Some(true)),
            Some(_) => Err(()),
        }
    }

    fn gpu_bypass(
        &self,
        reason: PrefillBypassReason,
        pinned_ane_split: bool,
    ) -> AneSplitRouteDecision {
        if pinned_ane_split {
            AneSplitRouteDecision::IdentityPreservingFailure(IdentityPreservingFailure::PreAttempt(
                reason,
            ))
        } else {
            AneSplitRouteDecision::GpuBypass {
                provenance: PrefillProvenance::gpu_bypass(reason),
                processing_fingerprint: self.gpu_processing_fingerprint.clone(),
            }
        }
    }

    /// Evaluate global gates before inspecting any bucket, select the first
    /// ascending eligible exact arm, then acquire the guard and check deadline
    /// feasibility. No bypass path mutates split-arm health.
    pub fn route<G: AneGuard>(
        &mut self,
        request: &AneSplitRequest,
        now_ms: u64,
        guard: &mut G,
    ) -> AneSplitRouteDecision {
        // Global-first precedence. Keep these checks before every bucket lookup.
        if self.config.decode_prefill == DecodePrefill::Gpu {
            return self.gpu_bypass(PrefillBypassReason::Disabled, false);
        }
        if self.config.platform == AnePlatform::Unsupported {
            return self.gpu_bypass(PrefillBypassReason::PlatformUnsupported, false);
        }
        if request.family != Family::Qwen3_0_6b {
            return self.gpu_bypass(PrefillBypassReason::FamilyUnsupported, false);
        }
        if !request.sampling.is_greedy_top1() {
            return self.gpu_bypass(PrefillBypassReason::SamplingUncertified, false);
        }
        let pinned_ane_split =
            match self.pinned_engine_class(request.required_processing_fingerprint.as_ref()) {
                Ok(Some(false)) => {
                    return self.gpu_bypass(PrefillBypassReason::IdentityPinnedGpu, false)
                }
                Ok(Some(true)) => true,
                Ok(None) => false,
                Err(()) => {
                    return AneSplitRouteDecision::IdentityPreservingFailure(
                        IdentityPreservingFailure::FingerprintMismatch,
                    )
                }
            };

        if request.prompt_token_count == 0 || request.max_tokens == 0 {
            return AneSplitRouteDecision::IdentityPreservingFailure(
                IdentityPreservingFailure::FingerprintMismatch,
            );
        }

        let required_positions = request
            .prompt_token_count
            .saturating_add(request.max_tokens);
        let windows: Vec<_> = PrefillBucket::ALL
            .into_iter()
            .filter(|bucket| bucket.tokens() >= request.prompt_token_count)
            .collect();
        if windows.is_empty() {
            return self.gpu_bypass(PrefillBypassReason::PromptOverMaxBucket, pinned_ane_split);
        }

        let fitting_windows: Vec<_> = windows
            .iter()
            .copied()
            .filter(|bucket| {
                let key = SplitArmKey::new(
                    self.config.machine_profile.clone(),
                    request.family,
                    *bucket,
                    request.decode_config,
                );
                self.arms
                    .get(&key)
                    .is_some_and(|arm| arm.has_fitting_cache(required_positions))
            })
            .collect();
        if fitting_windows.is_empty() {
            return self.gpu_bypass(PrefillBypassReason::NoFittingCacheBucket, pinned_ane_split);
        }

        for bucket in &fitting_windows {
            let key = SplitArmKey::new(
                self.config.machine_profile.clone(),
                request.family,
                *bucket,
                request.decode_config,
            );
            let arm = self
                .arms
                .get_mut(&key)
                .expect("fitting window was derived from an existing exact arm");
            if arm.is_selectable(now_ms, self.config.health_policy) {
                let readiness_required = arm.runtime_artifacts.readiness_required();
                let guard_result = guard.acquire_within(self.config.timing.ane_attempt_budget_ms);
                if !guard_result.acquired
                    || guard_result.waited_ms > self.config.timing.ane_attempt_budget_ms
                {
                    return self.gpu_bypass(PrefillBypassReason::AneBusy, pinned_ane_split);
                }
                if request.deadline_remaining_ms.is_some_and(|remaining_ms| {
                    remaining_ms.saturating_sub(guard_result.waited_ms)
                        < self.config.timing.remaining_after_guard_ms()
                }) {
                    return self
                        .gpu_bypass(PrefillBypassReason::DeadlineTooTight, pinned_ane_split);
                }
                return AneSplitRouteDecision::Attempt(AneSplitAttempt {
                    arm: key,
                    readiness_required,
                    processing_fingerprint: self.ane_split_processing_fingerprint.clone(),
                    pinned_ane_split,
                });
            }
        }

        // The smallest fitting window alone supplies the terminal state. A later
        // window is never scanned to find a different, higher-precedence label.
        let terminal_key = SplitArmKey::new(
            self.config.machine_profile.clone(),
            request.family,
            fitting_windows[0],
            request.decode_config,
        );
        let reason = self
            .arms
            .get_mut(&terminal_key)
            .expect("fitting window was derived from an existing exact arm")
            .terminal_reason(now_ms, self.config.health_policy);
        self.gpu_bypass(reason, pinned_ane_split)
    }

    /// Complete a started split-path attempt. Success resets only the exact
    /// split-arm's consecutive strikes. Failure charges only that arm and gives
    /// unpinned requests GPU fallback provenance with the completed GPU identity.
    pub fn complete_attempt(
        &mut self,
        attempt: AneSplitAttempt,
        result: AneSplitAttemptResult,
        now_ms: u64,
    ) -> AneSplitCompletion {
        let bucket = attempt.arm.bucket;
        let arm = self
            .arms
            .get_mut(&attempt.arm)
            .expect("attempt key must remain in the router arm map");
        match result {
            AneSplitAttemptResult::Success => {
                arm.health.record_success();
                AneSplitCompletion::SplitSuccess {
                    provenance: PrefillProvenance::split_success(bucket),
                    processing_fingerprint: self.ane_split_processing_fingerprint.clone(),
                }
            }
            AneSplitAttemptResult::Failure(reason) => {
                arm.health.record_failure(now_ms, self.config.health_policy);
                if attempt.pinned_ane_split {
                    AneSplitCompletion::IdentityPreservingFailure(
                        IdentityPreservingFailure::Execution(reason),
                    )
                } else {
                    AneSplitCompletion::GpuFallback {
                        provenance: PrefillProvenance::gpu_fallback(bucket, reason),
                        processing_fingerprint: self.gpu_processing_fingerprint.clone(),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RecordingGuard {
        result: GuardAcquisition,
        budgets: Vec<u64>,
    }

    impl AneGuard for RecordingGuard {
        fn acquire_within(&mut self, max_wait_ms: u64) -> GuardAcquisition {
            self.budgets.push(max_wait_ms);
            self.result
        }
    }

    fn artifacts(loaded_digest: Option<&str>) -> RuntimeArtifactIdentity {
        RuntimeArtifactIdentity {
            manifest_source_checkpoint_digest: "source-a".to_string(),
            derived_or_compiled_artifact_digest: loaded_digest.map(str::to_string),
        }
    }

    fn certified(loaded_digest: Option<&str>, cache_bucket: u32) -> SplitArmState {
        SplitArmState {
            certification: SplitArmCertification::Certified {
                artifacts: CertifiedArtifactIdentity {
                    source_checkpoint_digest: "source-a".to_string(),
                    certification_recorded_artifact_digest: "compiled-a".to_string(),
                },
            },
            runtime_artifacts: artifacts(loaded_digest),
            decode_cache_bucket: Some(cache_bucket),
            health: SplitArmHealth::default(),
        }
    }

    fn key(bucket: PrefillBucket) -> SplitArmKey {
        SplitArmKey::new(
            "profile-a",
            Family::Qwen3_0_6b,
            bucket,
            SplitDecodeConfig::F16Step,
        )
    }

    fn request() -> AneSplitRequest {
        AneSplitRequest {
            family: Family::Qwen3_0_6b,
            decode_config: SplitDecodeConfig::F16Step,
            prompt_token_count: 128,
            max_tokens: 64,
            sampling: SamplingMode::GreedyTop1,
            required_processing_fingerprint: None,
            deadline_remaining_ms: None,
        }
    }

    fn router(arms: BTreeMap<SplitArmKey, SplitArmState>) -> AnePrefillRouter {
        AnePrefillRouter::new(
            AneSplitRoutingConfig {
                decode_prefill: DecodePrefill::AneSplit,
                platform: AnePlatform::Supported,
                machine_profile: "profile-a".to_string(),
                timing: SplitTimingBudgets::from_calibration(10, 100, 5, 700),
                health_policy: SplitHealthPolicy::default(),
            },
            Fingerprint("gpu-processing".to_string()),
            Fingerprint("ane-split-processing".to_string()),
            arms,
        )
    }

    fn success_guard() -> RecordingGuard {
        RecordingGuard {
            result: GuardAcquisition {
                acquired: true,
                waited_ms: 1,
            },
            budgets: Vec::new(),
        }
    }

    fn one_certified_arm() -> BTreeMap<SplitArmKey, SplitArmState> {
        BTreeMap::from([(key(PrefillBucket::W128), certified(Some("compiled-a"), 512))])
    }

    #[test]
    fn timing_derives_the_single_attempt_budget_for_guard_and_prediction() {
        let timing = SplitTimingBudgets::from_calibration(11, 101, 7, 701);
        assert_eq!(timing.ane_attempt_budget_ms, 22);
        assert_eq!(timing.handoff_budget_ms, 14);
        assert_eq!(timing.full_split_ceiling_ms(), 22 + 101 + 22 + 14);
        assert_eq!(timing.remaining_after_guard_ms(), 101 + 22 + 14 + 701);
    }

    #[test]
    fn global_gates_precede_bucket_enumeration() {
        let mut disabled = router(BTreeMap::new());
        disabled.config.decode_prefill = DecodePrefill::Gpu;
        let mut oversized = request();
        oversized.prompt_token_count = 1_000;
        let outcome = disabled.route(&oversized, 0, &mut success_guard());
        assert!(matches!(
            outcome,
            AneSplitRouteDecision::GpuBypass {
                provenance: PrefillProvenance {
                    prefill_bypass_reason: Some(PrefillBypassReason::Disabled),
                    ..
                },
                ..
            }
        ));

        let mut non_mac = router(BTreeMap::new());
        non_mac.config.platform = AnePlatform::Unsupported;
        let outcome = non_mac.route(&oversized, 0, &mut success_guard());
        assert!(matches!(
            outcome,
            AneSplitRouteDecision::GpuBypass {
                provenance: PrefillProvenance {
                    prefill_bypass_reason: Some(PrefillBypassReason::PlatformUnsupported),
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn global_family_sampling_and_gpu_pin_bypasses_are_typed() {
        let mut routing = router(one_certified_arm());
        let mut unsupported_family = request();
        unsupported_family.family = Family::Lfm2_1_2b;
        assert!(matches!(
            routing.route(&unsupported_family, 0, &mut success_guard()),
            AneSplitRouteDecision::GpuBypass {
                provenance: PrefillProvenance {
                    prefill_bypass_reason: Some(PrefillBypassReason::FamilyUnsupported),
                    ..
                },
                ..
            }
        ));

        let mut sampled = request();
        sampled.sampling = SamplingMode::TopK { k: 2 };
        assert!(matches!(
            routing.route(&sampled, 0, &mut success_guard()),
            AneSplitRouteDecision::GpuBypass {
                provenance: PrefillProvenance {
                    prefill_bypass_reason: Some(PrefillBypassReason::SamplingUncertified),
                    ..
                },
                ..
            }
        ));

        let mut gpu_pinned = request();
        gpu_pinned.required_processing_fingerprint =
            Some(Fingerprint("gpu-processing".to_string()));
        assert!(matches!(
            routing.route(&gpu_pinned, 0, &mut success_guard()),
            AneSplitRouteDecision::GpuBypass {
                provenance: PrefillProvenance {
                    prefill_bypass_reason: Some(PrefillBypassReason::IdentityPinnedGpu),
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn routes_ascending_and_escalates_past_an_ineligible_smaller_arm() {
        let mut arms = one_certified_arm();
        arms.insert(key(PrefillBucket::W256), certified(Some("compiled-a"), 512));
        let first = arms.get_mut(&key(PrefillBucket::W128)).unwrap();
        first.runtime_artifacts.derived_or_compiled_artifact_digest = Some("wrong".to_string());
        let mut routing = router(arms);
        let attempt = match routing.route(&request(), 0, &mut success_guard()) {
            AneSplitRouteDecision::Attempt(attempt) => attempt,
            outcome => panic!("expected escalated attempt, got {outcome:?}"),
        };
        assert_eq!(attempt.arm.bucket, PrefillBucket::W256);
    }

    #[test]
    fn terminal_reason_comes_from_the_smallest_fitting_window_only() {
        let mut arms = BTreeMap::new();
        let mut first = certified(Some("compiled-a"), 512);
        first.certification = SplitArmCertification::NotCertified;
        arms.insert(key(PrefillBucket::W128), first);
        let mut larger = certified(Some("compiled-a"), 512);
        larger.certification = SplitArmCertification::BucketAbsent;
        arms.insert(key(PrefillBucket::W256), larger);
        let mut routing = router(arms);
        assert!(matches!(
            routing.route(&request(), 0, &mut success_guard()),
            AneSplitRouteDecision::GpuBypass {
                provenance: PrefillProvenance {
                    prefill_bypass_reason: Some(PrefillBypassReason::NotCertified),
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn unloaded_matching_arm_selects_for_bounded_readiness_but_present_mismatch_does_not() {
        let mut unloaded = router(BTreeMap::from([(
            key(PrefillBucket::W128),
            certified(None, 512),
        )]));
        let attempt = match unloaded.route(&request(), 0, &mut success_guard()) {
            AneSplitRouteDecision::Attempt(attempt) => attempt,
            outcome => panic!("expected readiness attempt, got {outcome:?}"),
        };
        assert!(attempt.readiness_required);

        let mut wrong = router(BTreeMap::from([(
            key(PrefillBucket::W128),
            certified(Some("wrong"), 512),
        )]));
        assert!(matches!(
            wrong.route(&request(), 0, &mut success_guard()),
            AneSplitRouteDecision::GpuBypass {
                provenance: PrefillProvenance {
                    prefill_bypass_reason: Some(PrefillBypassReason::ArtifactDigestMismatch),
                    ..
                },
                ..
            }
        ));
    }

    #[test]
    fn guard_wait_is_bounded_and_deadline_counts_the_consumed_guard_time_once() {
        let mut routing = router(one_certified_arm());
        let mut busy = RecordingGuard {
            result: GuardAcquisition {
                acquired: false,
                waited_ms: 20,
            },
            budgets: Vec::new(),
        };
        assert!(matches!(
            routing.route(&request(), 0, &mut busy),
            AneSplitRouteDecision::GpuBypass {
                provenance: PrefillProvenance {
                    prefill_bypass_reason: Some(PrefillBypassReason::AneBusy),
                    ..
                },
                ..
            }
        ));
        assert_eq!(busy.budgets, vec![20]);

        let mut deadline = request();
        // 100 readiness + 20 prediction + 10 handoff + 700 GPU fallback = 830;
        // the elapsed 5ms guard wait leaves only 830ms from this deadline.
        deadline.deadline_remaining_ms = Some(834);
        let mut waited = RecordingGuard {
            result: GuardAcquisition {
                acquired: true,
                waited_ms: 5,
            },
            budgets: Vec::new(),
        };
        assert!(matches!(
            routing.route(&deadline, 0, &mut waited),
            AneSplitRouteDecision::GpuBypass {
                provenance: PrefillProvenance {
                    prefill_bypass_reason: Some(PrefillBypassReason::DeadlineTooTight),
                    ..
                },
                ..
            }
        ));
        deadline.deadline_remaining_ms = Some(835);
        assert!(matches!(
            routing.route(&deadline, 0, &mut waited),
            AneSplitRouteDecision::Attempt(_)
        ));
    }

    #[test]
    fn fallback_is_gpu_provenance_and_charges_only_the_exact_arm() {
        let mut routing = router(one_certified_arm());
        let attempt = match routing.route(&request(), 0, &mut success_guard()) {
            AneSplitRouteDecision::Attempt(attempt) => attempt,
            outcome => panic!("expected split attempt, got {outcome:?}"),
        };
        let completion = routing.complete_attempt(
            attempt,
            AneSplitAttemptResult::Failure(PrefillFallbackReason::CacheHandoffFailure),
            1,
        );
        match completion {
            AneSplitCompletion::GpuFallback {
                provenance,
                processing_fingerprint,
            } => {
                assert!(provenance.is_valid());
                assert_eq!(provenance.prefill_engine, PrefillEngine::Gpu);
                assert_eq!(
                    provenance.prefill_fallback_reason,
                    Some(PrefillFallbackReason::CacheHandoffFailure)
                );
                assert_eq!(
                    processing_fingerprint,
                    Fingerprint("gpu-processing".to_string())
                );
            }
            other => panic!("expected GPU fallback, got {other:?}"),
        }
        assert_eq!(
            routing
                .arm(&key(PrefillBucket::W128))
                .unwrap()
                .health
                .consecutive_strikes,
            1
        );
    }

    #[test]
    fn successful_split_uses_class_identity_without_bucket_granularity() {
        let mut routing = router(one_certified_arm());
        let attempt = match routing.route(&request(), 0, &mut success_guard()) {
            AneSplitRouteDecision::Attempt(attempt) => attempt,
            outcome => panic!("expected split attempt, got {outcome:?}"),
        };
        let completion = routing.complete_attempt(attempt, AneSplitAttemptResult::Success, 1);
        match completion {
            AneSplitCompletion::SplitSuccess {
                provenance,
                processing_fingerprint,
            } => {
                assert!(provenance.is_valid());
                assert_eq!(provenance.prefill_bucket, Some(PrefillBucket::W128));
                assert_eq!(
                    processing_fingerprint,
                    Fingerprint("ane-split-processing".to_string())
                );
            }
            other => panic!("expected split success, got {other:?}"),
        }
    }

    #[test]
    fn ane_pin_refuses_pre_attempt_substitution_and_charges_execution_failure() {
        let mut routing = router(one_certified_arm());
        let mut pinned = request();
        pinned.required_processing_fingerprint =
            Some(Fingerprint("ane-split-processing".to_string()));
        let attempt = match routing.route(&pinned, 0, &mut success_guard()) {
            AneSplitRouteDecision::Attempt(attempt) => attempt,
            outcome => panic!("expected pinned split attempt, got {outcome:?}"),
        };
        assert!(matches!(
            routing.complete_attempt(
                attempt,
                AneSplitAttemptResult::Failure(PrefillFallbackReason::PredictionFailure),
                1,
            ),
            AneSplitCompletion::IdentityPreservingFailure(IdentityPreservingFailure::Execution(
                PrefillFallbackReason::PredictionFailure
            ))
        ));
        assert_eq!(
            routing
                .arm(&key(PrefillBucket::W128))
                .unwrap()
                .health
                .consecutive_strikes,
            1
        );

        let mut not_certified = router(BTreeMap::from([(
            key(PrefillBucket::W128),
            SplitArmState {
                certification: SplitArmCertification::NotCertified,
                runtime_artifacts: artifacts(None),
                decode_cache_bucket: Some(512),
                health: SplitArmHealth::default(),
            },
        )]));
        assert!(matches!(
            not_certified.route(&pinned, 0, &mut success_guard()),
            AneSplitRouteDecision::IdentityPreservingFailure(
                IdentityPreservingFailure::PreAttempt(PrefillBypassReason::NotCertified)
            )
        ));
    }

    #[test]
    fn split_arm_failures_are_consecutive_and_probation_requarantines_on_one_failure() {
        let mut routing = router(one_certified_arm());
        for now_ms in [1, 2] {
            let attempt = match routing.route(&request(), now_ms, &mut success_guard()) {
                AneSplitRouteDecision::Attempt(attempt) => attempt,
                outcome => panic!("expected split attempt, got {outcome:?}"),
            };
            let _ = routing.complete_attempt(
                attempt,
                AneSplitAttemptResult::Failure(PrefillFallbackReason::DispatchFailure),
                now_ms,
            );
        }
        let state = routing.arm(&key(PrefillBucket::W128)).unwrap();
        assert_eq!(state.health.consecutive_strikes, 2);
        assert_eq!(state.health.quarantined_until_ms, Some(60_002));

        assert!(matches!(
            routing.route(&request(), 3, &mut success_guard()),
            AneSplitRouteDecision::GpuBypass {
                provenance: PrefillProvenance {
                    prefill_bypass_reason: Some(PrefillBypassReason::Quarantined),
                    ..
                },
                ..
            }
        ));
        let probation_attempt = match routing.route(&request(), 60_002, &mut success_guard()) {
            AneSplitRouteDecision::Attempt(attempt) => attempt,
            outcome => panic!("expected probation attempt, got {outcome:?}"),
        };
        assert!(
            routing
                .arm(&key(PrefillBucket::W128))
                .unwrap()
                .health
                .probation
        );
        let _ = routing.complete_attempt(
            probation_attempt,
            AneSplitAttemptResult::Failure(PrefillFallbackReason::DispatchFailure),
            60_002,
        );
        assert!(routing
            .arm(&key(PrefillBucket::W128))
            .unwrap()
            .health
            .quarantined_until_ms
            .is_some());

        // A separate exact arm remains healthy, proving no worker-key sharing.
        let other = SplitArmKey::new(
            "profile-a",
            Family::Qwen3_0_6b,
            PrefillBucket::W256,
            SplitDecodeConfig::F16Step,
        );
        assert!(routing.arm(&other).is_none());
    }

    #[test]
    fn successful_attempt_clears_consecutive_strikes_and_probation() {
        let mut arms = one_certified_arm();
        let health = &mut arms.get_mut(&key(PrefillBucket::W128)).unwrap().health;
        health.consecutive_strikes = 1;
        health.probation = true;
        let mut routing = router(arms);
        let attempt = match routing.route(&request(), 0, &mut success_guard()) {
            AneSplitRouteDecision::Attempt(attempt) => attempt,
            outcome => panic!("expected split attempt, got {outcome:?}"),
        };
        let _ = routing.complete_attempt(attempt, AneSplitAttemptResult::Success, 0);
        assert_eq!(
            routing.arm(&key(PrefillBucket::W128)).unwrap().health,
            SplitArmHealth::default()
        );
    }

    #[test]
    fn fault_map_and_provenance_vocabularies_are_complete_and_disjoint() {
        for fault in AnePrefillFault::ALL {
            assert!(PrefillFallbackReason::ALL.contains(&fault.fallback_reason()));
        }
        for bypass in PrefillBypassReason::ALL {
            assert!(!PrefillFallbackReason::ALL
                .iter()
                .any(|fallback| bypass.as_str() == fallback.as_str()));
        }
    }

    #[test]
    fn provenance_rejects_mixed_bypass_and_fallback_fields() {
        let mut mixed = PrefillProvenance::gpu_bypass(PrefillBypassReason::Disabled);
        mixed.prefill_fallback_from = Some(PrefillEngine::AneW128);
        mixed.prefill_fallback_reason = Some(PrefillFallbackReason::DispatchFailure);
        assert!(!mixed.is_valid());
    }

    #[test]
    fn every_bypass_reason_is_reachable_without_charging_health() {
        let reason = |outcome: AneSplitRouteDecision| match outcome {
            AneSplitRouteDecision::GpuBypass { provenance, .. } => {
                provenance.prefill_bypass_reason.expect("typed bypass")
            }
            other => panic!("expected GPU bypass, got {other:?}"),
        };

        let mut disabled = router(one_certified_arm());
        disabled.config.decode_prefill = DecodePrefill::Gpu;
        assert_eq!(
            reason(disabled.route(&request(), 0, &mut success_guard())),
            PrefillBypassReason::Disabled
        );

        let mut unsupported = router(one_certified_arm());
        unsupported.config.platform = AnePlatform::Unsupported;
        assert_eq!(
            reason(unsupported.route(&request(), 0, &mut success_guard())),
            PrefillBypassReason::PlatformUnsupported
        );

        let mut family = request();
        family.family = Family::Lfm2_1_2b;
        assert_eq!(
            reason(router(one_certified_arm()).route(&family, 0, &mut success_guard())),
            PrefillBypassReason::FamilyUnsupported
        );

        let mut sampled = request();
        sampled.sampling = SamplingMode::TopP { p: 0.5 };
        assert_eq!(
            reason(router(one_certified_arm()).route(&sampled, 0, &mut success_guard())),
            PrefillBypassReason::SamplingUncertified
        );

        let mut gpu_pin = request();
        gpu_pin.required_processing_fingerprint = Some(Fingerprint("gpu-processing".to_string()));
        assert_eq!(
            reason(router(one_certified_arm()).route(&gpu_pin, 0, &mut success_guard())),
            PrefillBypassReason::IdentityPinnedGpu
        );

        let mut over_max = request();
        over_max.prompt_token_count = 513;
        assert_eq!(
            reason(router(BTreeMap::new()).route(&over_max, 0, &mut success_guard())),
            PrefillBypassReason::PromptOverMaxBucket
        );

        let mut no_cache_arms = one_certified_arm();
        no_cache_arms
            .get_mut(&key(PrefillBucket::W128))
            .unwrap()
            .decode_cache_bucket = Some(1);
        assert_eq!(
            reason(router(no_cache_arms).route(&request(), 0, &mut success_guard())),
            PrefillBypassReason::NoFittingCacheBucket
        );

        let mut absent_arms = one_certified_arm();
        absent_arms
            .get_mut(&key(PrefillBucket::W128))
            .unwrap()
            .certification = SplitArmCertification::BucketAbsent;
        assert_eq!(
            reason(router(absent_arms).route(&request(), 0, &mut success_guard())),
            PrefillBypassReason::BucketAbsent
        );

        let mut uncertified_arms = one_certified_arm();
        uncertified_arms
            .get_mut(&key(PrefillBucket::W128))
            .unwrap()
            .certification = SplitArmCertification::NotCertified;
        assert_eq!(
            reason(router(uncertified_arms).route(&request(), 0, &mut success_guard())),
            PrefillBypassReason::NotCertified
        );

        let mut artifact_arms = one_certified_arm();
        artifact_arms
            .get_mut(&key(PrefillBucket::W128))
            .unwrap()
            .runtime_artifacts
            .manifest_source_checkpoint_digest = "wrong-source".to_string();
        assert_eq!(
            reason(router(artifact_arms).route(&request(), 0, &mut success_guard())),
            PrefillBypassReason::ArtifactDigestMismatch
        );

        let mut quarantined_arms = one_certified_arm();
        quarantined_arms
            .get_mut(&key(PrefillBucket::W128))
            .unwrap()
            .health
            .quarantined_until_ms = Some(1_000);
        assert_eq!(
            reason(router(quarantined_arms).route(&request(), 0, &mut success_guard())),
            PrefillBypassReason::Quarantined
        );

        let mut busy_guard = RecordingGuard {
            result: GuardAcquisition {
                acquired: false,
                waited_ms: 0,
            },
            budgets: Vec::new(),
        };
        assert_eq!(
            reason(router(one_certified_arm()).route(&request(), 0, &mut busy_guard)),
            PrefillBypassReason::AneBusy
        );

        let mut too_tight = request();
        too_tight.deadline_remaining_ms = Some(0);
        assert_eq!(
            reason(router(one_certified_arm()).route(&too_tight, 0, &mut success_guard())),
            PrefillBypassReason::DeadlineTooTight
        );
    }

    #[test]
    fn every_fallback_reason_is_gpu_provenance_and_one_exact_arm_debit() {
        for fallback_reason in PrefillFallbackReason::ALL {
            let mut routing = router(one_certified_arm());
            let attempt = match routing.route(&request(), 0, &mut success_guard()) {
                AneSplitRouteDecision::Attempt(attempt) => attempt,
                outcome => panic!("expected split attempt, got {outcome:?}"),
            };
            match routing.complete_attempt(
                attempt,
                AneSplitAttemptResult::Failure(fallback_reason),
                1,
            ) {
                AneSplitCompletion::GpuFallback { provenance, .. } => {
                    assert_eq!(provenance.prefill_engine, PrefillEngine::Gpu);
                    assert_eq!(provenance.prefill_fallback_reason, Some(fallback_reason));
                    assert!(provenance.is_valid());
                }
                other => panic!("expected GPU fallback, got {other:?}"),
            }
            assert_eq!(
                routing
                    .arm(&key(PrefillBucket::W128))
                    .unwrap()
                    .health
                    .consecutive_strikes,
                1,
                "{fallback_reason:?}"
            );
        }
    }

    #[test]
    fn non_mac_config_is_inert_and_requests_one_log_notice() {
        let mut routing = router(one_certified_arm());
        routing.config.platform = AnePlatform::Unsupported;
        assert!(routing.take_platform_inert_notice());
        assert!(!routing.take_platform_inert_notice());
        assert!(matches!(
            routing.route(&request(), 0, &mut success_guard()),
            AneSplitRouteDecision::GpuBypass {
                provenance: PrefillProvenance {
                    prefill_engine: PrefillEngine::Gpu,
                    prefill_bypass_reason: Some(PrefillBypassReason::PlatformUnsupported),
                    ..
                },
                ..
            }
        ));
    }
}
