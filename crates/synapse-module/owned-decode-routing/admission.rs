//! Platform-envelope admission and resident-artifact routing for wave 1.
//!
//! This layer makes the admission promise independently testable from Metal and
//! worker execution. It reserves only metadata here; the worker owns the actual
//! weight and KV allocations. The accounting is deliberately conservative: a
//! session reserves its complete declared context budget before it exists, and
//! the embed/rerank lane remains reserved even while decode is idle.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::owned_decode_routing::request::SamplingMode;

/// The only platform-envelope schema accepted by the wave-1 admission layer.
pub const PLATFORM_ENVELOPE_VERSION: &str = "platform-envelope-v1";
/// Every supported platform must be able to admit this complete context ceiling.
pub const WAVE_ONE_CONTEXT_CEILING_TOKENS: u32 = 32_768;
/// Wave 1 does not serve a machine with less than 64 GiB of unified memory.
pub const WAVE_ONE_MIN_UNIFIED_MEMORY_BYTES: u64 = 64 * 1024 * 1024 * 1024;

/// The observed machine facts compared with a certification-time envelope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineTuple {
    pub machine_profile_hash: String,
    pub macos_build: String,
    pub unified_memory_bytes: u64,
}

impl MachineTuple {
    #[must_use]
    pub fn new(
        machine_profile_hash: impl Into<String>,
        macos_build: impl Into<String>,
        unified_memory_bytes: u64,
    ) -> Self {
        Self {
            machine_profile_hash: machine_profile_hash.into(),
            macos_build: macos_build.into(),
            unified_memory_bytes,
        }
    }
}

/// The exact bytes attributed to one complete certified artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactReservation {
    /// The complete base-plus-head-plus-gate certification-unit identity.
    pub catalog_fingerprint: String,
    /// Literal resident weight allocation measured for this artifact.
    pub weight_bytes: u64,
    /// KV allocation derived from the artifact configuration for one token.
    pub kv_bytes_per_token: u64,
}

impl ArtifactReservation {
    pub fn new(
        catalog_fingerprint: impl Into<String>,
        weight_bytes: u64,
        kv_bytes_per_token: u64,
    ) -> Result<Self, PlatformEnvelopeConfigurationError> {
        let artifact = Self {
            catalog_fingerprint: catalog_fingerprint.into(),
            weight_bytes,
            kv_bytes_per_token,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    fn validate(&self) -> Result<(), PlatformEnvelopeConfigurationError> {
        if self.catalog_fingerprint.is_empty() {
            return Err(PlatformEnvelopeConfigurationError::EmptyCatalogFingerprint);
        }
        if self.weight_bytes == 0 {
            return Err(PlatformEnvelopeConfigurationError::ZeroArtifactWeightBytes);
        }
        if self.kv_bytes_per_token == 0 {
            return Err(PlatformEnvelopeConfigurationError::ZeroKvBytesPerToken);
        }
        Ok(())
    }
}

/// Certification-time record for one artifact on one supported machine tuple.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformEnvelope {
    schema: String,
    machine_profile_hash: String,
    macos_build: String,
    minimum_unified_memory_bytes: u64,
    reserved_embed_rerank_bytes: u64,
    artifact: ArtifactReservation,
}

impl PlatformEnvelope {
    /// Creates a `platform-envelope-v1` record and proves its 32k reservation
    /// can fit at the recorded minimum memory capacity.
    pub fn new(
        machine_profile_hash: impl Into<String>,
        macos_build: impl Into<String>,
        minimum_unified_memory_bytes: u64,
        reserved_embed_rerank_bytes: u64,
        artifact: ArtifactReservation,
    ) -> Result<Self, PlatformEnvelopeConfigurationError> {
        artifact.validate()?;
        let envelope = Self {
            schema: PLATFORM_ENVELOPE_VERSION.to_string(),
            machine_profile_hash: machine_profile_hash.into(),
            macos_build: macos_build.into(),
            minimum_unified_memory_bytes,
            reserved_embed_rerank_bytes,
            artifact,
        };
        envelope.validate_configuration()?;
        Ok(envelope)
    }

    #[must_use]
    pub fn schema(&self) -> &str {
        &self.schema
    }

    #[must_use]
    pub fn machine_profile_hash(&self) -> &str {
        &self.machine_profile_hash
    }

    #[must_use]
    pub fn macos_build(&self) -> &str {
        &self.macos_build
    }

    #[must_use]
    pub const fn minimum_unified_memory_bytes(&self) -> u64 {
        self.minimum_unified_memory_bytes
    }

    #[must_use]
    pub const fn reserved_embed_rerank_bytes(&self) -> u64 {
        self.reserved_embed_rerank_bytes
    }

    #[must_use]
    pub fn artifact(&self) -> &ArtifactReservation {
        &self.artifact
    }

    fn validate_configuration(&self) -> Result<(), PlatformEnvelopeConfigurationError> {
        if self.schema != PLATFORM_ENVELOPE_VERSION {
            return Err(PlatformEnvelopeConfigurationError::WrongSchema {
                actual: self.schema.clone(),
            });
        }
        if self.machine_profile_hash.is_empty() {
            return Err(PlatformEnvelopeConfigurationError::EmptyMachineProfileHash);
        }
        if self.macos_build.is_empty() {
            return Err(PlatformEnvelopeConfigurationError::EmptyMacosBuild);
        }
        if self.minimum_unified_memory_bytes < WAVE_ONE_MIN_UNIFIED_MEMORY_BYTES {
            return Err(
                PlatformEnvelopeConfigurationError::MinimumMemoryBelowWaveOneFloor {
                    actual_bytes: self.minimum_unified_memory_bytes,
                },
            );
        }

        let required_bytes = self
            .reserved_embed_rerank_bytes
            .checked_add(self.artifact.weight_bytes)
            .and_then(|bytes| {
                u64::from(WAVE_ONE_CONTEXT_CEILING_TOKENS)
                    .checked_mul(self.artifact.kv_bytes_per_token)
                    .and_then(|kv_bytes| bytes.checked_add(kv_bytes))
            })
            .ok_or(PlatformEnvelopeConfigurationError::ReservationArithmeticOverflow)?;
        if required_bytes > self.minimum_unified_memory_bytes {
            return Err(
                PlatformEnvelopeConfigurationError::RequiredReservationExceedsMinimumMemory {
                    required_bytes,
                    minimum_unified_memory_bytes: self.minimum_unified_memory_bytes,
                },
            );
        }
        Ok(())
    }

    /// Validates the exact certified profile hash and macOS build while allowing
    /// only machines with at least the recorded (and 64 GiB minimum) memory.
    pub fn validate_machine_tuple(&self, machine: &MachineTuple) -> Result<(), AdmissionRefusal> {
        if self.machine_profile_hash == machine.machine_profile_hash
            && self.macos_build == machine.macos_build
            && machine.unified_memory_bytes >= self.minimum_unified_memory_bytes
            && machine.unified_memory_bytes >= WAVE_ONE_MIN_UNIFIED_MEMORY_BYTES
        {
            Ok(())
        } else {
            Err(AdmissionRefusal::UnsupportedPlatformTuple)
        }
    }

    /// Validates that admission uses the exact measured artifact reservation.
    pub fn validate_artifact(
        &self,
        artifact: &ArtifactReservation,
    ) -> Result<(), AdmissionRefusal> {
        if self.artifact == *artifact {
            Ok(())
        } else {
            Err(AdmissionRefusal::IncompatibleArtifact)
        }
    }
}

/// Invalid durable envelope data is rejected before it can authorize serving.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlatformEnvelopeConfigurationError {
    WrongSchema {
        actual: String,
    },
    EmptyMachineProfileHash,
    EmptyMacosBuild,
    EmptyCatalogFingerprint,
    ZeroArtifactWeightBytes,
    ZeroKvBytesPerToken,
    MinimumMemoryBelowWaveOneFloor {
        actual_bytes: u64,
    },
    ReservationArithmeticOverflow,
    RequiredReservationExceedsMinimumMemory {
        required_bytes: u64,
        minimum_unified_memory_bytes: u64,
    },
    EmptyEnvelopeSet,
}

impl std::fmt::Display for PlatformEnvelopeConfigurationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongSchema { actual } => {
                write!(formatter, "unsupported platform envelope schema {actual}")
            }
            Self::EmptyMachineProfileHash => formatter.write_str("machine profile hash is empty"),
            Self::EmptyMacosBuild => formatter.write_str("macOS build is empty"),
            Self::EmptyCatalogFingerprint => formatter.write_str("catalog fingerprint is empty"),
            Self::ZeroArtifactWeightBytes => formatter.write_str("artifact weight bytes must be nonzero"),
            Self::ZeroKvBytesPerToken => formatter.write_str("KV bytes per token must be nonzero"),
            Self::MinimumMemoryBelowWaveOneFloor { actual_bytes } => write!(
                formatter,
                "minimum unified memory {actual_bytes} is below the wave-1 64 GiB floor"
            ),
            Self::ReservationArithmeticOverflow => {
                formatter.write_str("platform-envelope reservation arithmetic overflowed")
            }
            Self::RequiredReservationExceedsMinimumMemory {
                required_bytes,
                minimum_unified_memory_bytes,
            } => write!(
                formatter,
                "platform-envelope requires {required_bytes} bytes but records only {minimum_unified_memory_bytes} bytes"
            ),
            Self::EmptyEnvelopeSet => formatter.write_str("at least one platform envelope is required"),
        }
    }
}

impl std::error::Error for PlatformEnvelopeConfigurationError {}

/// Generation parameters carried into wave-1 admission.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationConfiguration {
    pub sampling: SamplingMode,
}

impl GenerationConfiguration {
    #[must_use]
    pub const fn greedy_top1() -> Self {
        Self {
            sampling: SamplingMode::GreedyTop1,
        }
    }

    fn is_wave_one_supported(&self) -> bool {
        self.sampling.is_greedy_top1()
    }
}

/// Coarse KV settings for session reuse; wave 1 accepts only the fixed block
/// sizes and recurrent-state granularities validated by its measurement matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SessionKvConfiguration {
    pub block_size_tokens: u32,
    pub recurrent_state_grain_tokens: u32,
}

impl SessionKvConfiguration {
    pub fn new(
        block_size_tokens: u32,
        recurrent_state_grain_tokens: u32,
    ) -> Result<Self, AdmissionRefusal> {
        let configuration = Self {
            block_size_tokens,
            recurrent_state_grain_tokens,
        };
        let _ = configuration.alignment_tokens()?;
        Ok(configuration)
    }

    pub fn alignment_tokens(&self) -> Result<u32, AdmissionRefusal> {
        if !matches!(self.block_size_tokens, 256 | 512 | 1024)
            || self.recurrent_state_grain_tokens == 0
        {
            return Err(AdmissionRefusal::InvalidKvConfiguration);
        }
        let divisor = gcd(self.block_size_tokens, self.recurrent_state_grain_tokens);
        self.block_size_tokens
            .checked_div(divisor)
            .and_then(|multiple| multiple.checked_mul(self.recurrent_state_grain_tokens))
            .ok_or(AdmissionRefusal::InvalidKvConfiguration)
    }

    fn validate_boundary(&self, position_tokens: u32) -> Result<(), AdmissionRefusal> {
        let alignment_tokens = self.alignment_tokens()?;
        if position_tokens % alignment_tokens != 0 {
            return Err(AdmissionRefusal::InvalidKvAlignment {
                position_tokens,
                alignment_tokens,
            });
        }
        Ok(())
    }
}

const fn gcd(mut left: u32, mut right: u32) -> u32 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// The immutable admission inputs for one new decode session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionRequest {
    pub caller_id: String,
    pub artifact: ArtifactReservation,
    pub context_ceiling_tokens: u32,
    pub generation: GenerationConfiguration,
    pub kv_configuration: SessionKvConfiguration,
}

impl AdmissionRequest {
    #[must_use]
    pub fn new(
        caller_id: impl Into<String>,
        artifact: ArtifactReservation,
        context_ceiling_tokens: u32,
        generation: GenerationConfiguration,
        kv_configuration: SessionKvConfiguration,
    ) -> Self {
        Self {
            caller_id: caller_id.into(),
            artifact,
            context_ceiling_tokens,
            generation,
            kv_configuration,
        }
    }
}

/// Stable identity of a route-local decode session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(pub u64);

/// Every admission refusal is deliberate and caller-distinguishable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionRefusal {
    UnsupportedPlatformTuple,
    InvalidContextCeiling {
        requested_tokens: u32,
        required_tokens: u32,
    },
    InsufficientReservedLaneMemory {
        required_bytes: u64,
        available_bytes: u64,
    },
    IncompatibleArtifact,
    UnsupportedGenerationConfig,
    InvalidKvConfiguration,
    InvalidKvAlignment {
        position_tokens: u32,
        alignment_tokens: u32,
    },
    UnknownSession {
        session_id: SessionId,
    },
    ArtifactSwapActiveSessions {
        active_sessions: usize,
    },
}

impl AdmissionRefusal {
    /// Stable wire IDs used by callers to distinguish refuse-with-reason paths.
    #[must_use]
    pub const fn wire_id(&self) -> &'static str {
        match self {
            Self::UnsupportedPlatformTuple => "unsupported_platform_tuple",
            Self::InvalidContextCeiling { .. } => "invalid_context_ceiling",
            Self::InsufficientReservedLaneMemory { .. } => "insufficient_reserved_lane_memory",
            Self::IncompatibleArtifact => "incompatible_artifact",
            Self::UnsupportedGenerationConfig => "unsupported_generation_config",
            Self::InvalidKvConfiguration => "invalid_kv_configuration",
            Self::InvalidKvAlignment { .. } => "invalid_kv_alignment",
            Self::UnknownSession { .. } => "unknown_session",
            Self::ArtifactSwapActiveSessions { .. } => "artifact_swap_active_sessions",
        }
    }
}

impl std::fmt::Display for AdmissionRefusal {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.wire_id())
    }
}

impl std::error::Error for AdmissionRefusal {}

/// The hard-reservation receipt returned for an admitted session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReservationReceipt {
    pub session_id: SessionId,
    pub catalog_fingerprint: String,
    pub reserved_embed_rerank_bytes: u64,
    pub reserved_artifact_weight_bytes: u64,
    pub reserved_session_kv_bytes: u64,
    pub context_ceiling_tokens: u32,
}

/// Successful admission identity and its complete reservation receipt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionAdmission {
    pub session_id: SessionId,
    pub receipt: ReservationReceipt,
}

/// Current accounting, exposed so supervisory code can verify reservation release.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyAccounting {
    pub total_unified_memory_bytes: u64,
    pub reserved_embed_rerank_bytes: u64,
    pub resident_artifact_weight_bytes: u64,
    pub session_kv_bytes: u64,
    pub available_bytes: u64,
    pub active_session_count: usize,
}

/// A validated same-session prefix that can be reused by continuation routing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlignedReuseAdmission {
    pub session_id: SessionId,
    pub retained_prefix_tokens: u32,
    pub reused_blocks: u32,
}

/// A continuation route carries only the target session's retained prefix.
///
/// It intentionally has no source-session field or shared-KV handle. A caller
/// can continue only the same session whose reservation supplied the KV table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContinuationRoute {
    pub session_id: SessionId,
    pub retained_prefix_tokens: u32,
    pub reused_blocks: u32,
}

#[derive(Clone, Debug)]
struct ResidentArtifact {
    artifact: ArtifactReservation,
}

#[derive(Clone, Debug)]
struct SessionReservation {
    context_ceiling_tokens: u32,
    kv_configuration: SessionKvConfiguration,
    kv_budget_bytes: u64,
}

/// Routes admissions against one machine's certified platform envelopes.
///
/// State changes happen only after all validation and projected-memory checks
/// pass. This ensures a refusal cannot leave a partly resident artifact or a
/// partly created session behind.
#[derive(Debug)]
pub struct ResidencyRouter {
    machine: MachineTuple,
    envelopes: Vec<PlatformEnvelope>,
    reserved_embed_rerank_bytes: u64,
    resident_artifact: Option<ResidentArtifact>,
    sessions: BTreeMap<SessionId, SessionReservation>,
    next_session_id: u64,
}

impl ResidencyRouter {
    #[must_use]
    pub fn new(machine: MachineTuple, envelope: PlatformEnvelope) -> Self {
        Self::with_envelopes(machine, vec![envelope])
            .expect("a singleton platform-envelope set is never empty")
    }

    pub fn with_envelopes(
        machine: MachineTuple,
        envelopes: Vec<PlatformEnvelope>,
    ) -> Result<Self, PlatformEnvelopeConfigurationError> {
        if envelopes.is_empty() {
            return Err(PlatformEnvelopeConfigurationError::EmptyEnvelopeSet);
        }
        for envelope in &envelopes {
            envelope.validate_configuration()?;
        }
        let reserved_embed_rerank_bytes = envelopes
            .iter()
            .map(PlatformEnvelope::reserved_embed_rerank_bytes)
            .max()
            .expect("nonempty envelopes have a maximum reservation");
        Ok(Self {
            machine,
            envelopes,
            reserved_embed_rerank_bytes,
            resident_artifact: None,
            sessions: BTreeMap::new(),
            next_session_id: 0,
        })
    }

    #[must_use]
    pub fn machine(&self) -> &MachineTuple {
        &self.machine
    }

    #[must_use]
    pub fn resident_artifact(&self) -> Option<&ArtifactReservation> {
        self.resident_artifact
            .as_ref()
            .map(|resident| &resident.artifact)
    }

    #[must_use]
    pub fn active_session_count(&self) -> usize {
        self.sessions.len()
    }

    #[must_use]
    pub fn accounting(&self) -> ResidencyAccounting {
        let resident_artifact_weight_bytes = self
            .resident_artifact
            .as_ref()
            .map_or(0, |resident| resident.artifact.weight_bytes);
        let session_kv_bytes = self.sessions.values().fold(0_u64, |total, session| {
            total.saturating_add(session.kv_budget_bytes)
        });
        let reserved_bytes = self
            .reserved_embed_rerank_bytes
            .saturating_add(resident_artifact_weight_bytes)
            .saturating_add(session_kv_bytes);
        ResidencyAccounting {
            total_unified_memory_bytes: self.machine.unified_memory_bytes,
            reserved_embed_rerank_bytes: self.reserved_embed_rerank_bytes,
            resident_artifact_weight_bytes,
            session_kv_bytes,
            available_bytes: self
                .machine
                .unified_memory_bytes
                .saturating_sub(reserved_bytes),
            active_session_count: self.sessions.len(),
        }
    }

    /// Validates a request and atomically reserves resident weights plus the
    /// complete declared KV ceiling. No field mutates before this succeeds.
    pub fn admit_session(
        &mut self,
        request: AdmissionRequest,
    ) -> Result<SessionAdmission, AdmissionRefusal> {
        if !request.generation.is_wave_one_supported() {
            return Err(AdmissionRefusal::UnsupportedGenerationConfig);
        }
        if request.context_ceiling_tokens != WAVE_ONE_CONTEXT_CEILING_TOKENS {
            return Err(AdmissionRefusal::InvalidContextCeiling {
                requested_tokens: request.context_ceiling_tokens,
                required_tokens: WAVE_ONE_CONTEXT_CEILING_TOKENS,
            });
        }
        let certified_artifact = self.envelope_for(&request.artifact)?.artifact.clone();
        request.kv_configuration.alignment_tokens()?;
        let session_kv_bytes = kv_budget_bytes(
            request.context_ceiling_tokens,
            request.artifact.kv_bytes_per_token,
        )?;

        if self
            .resident_artifact
            .as_ref()
            .is_some_and(|resident| resident.artifact != request.artifact)
        {
            return Err(AdmissionRefusal::IncompatibleArtifact);
        }

        let projected_artifact_weight_bytes = self
            .resident_artifact
            .as_ref()
            .map_or(request.artifact.weight_bytes, |resident| {
                resident.artifact.weight_bytes
            });
        let current_session_kv_bytes = self.session_kv_bytes()?;
        let required_bytes = self.projected_total_bytes(
            projected_artifact_weight_bytes,
            current_session_kv_bytes,
            session_kv_bytes,
        )?;
        if required_bytes > self.machine.unified_memory_bytes {
            return Err(AdmissionRefusal::InsufficientReservedLaneMemory {
                required_bytes,
                available_bytes: self.machine.unified_memory_bytes,
            });
        }

        let session_id = SessionId(self.next_session_id);
        let next_session_id = self.next_session_id.checked_add(1).ok_or(
            AdmissionRefusal::InsufficientReservedLaneMemory {
                required_bytes,
                available_bytes: self.machine.unified_memory_bytes,
            },
        )?;
        let receipt = ReservationReceipt {
            session_id,
            catalog_fingerprint: request.artifact.catalog_fingerprint.clone(),
            reserved_embed_rerank_bytes: self.reserved_embed_rerank_bytes,
            reserved_artifact_weight_bytes: request.artifact.weight_bytes,
            reserved_session_kv_bytes: session_kv_bytes,
            context_ceiling_tokens: request.context_ceiling_tokens,
        };

        // All fallible checks are above this point, making these updates one
        // transaction from the caller's perspective.
        if self.resident_artifact.is_none() {
            self.resident_artifact = Some(ResidentArtifact {
                artifact: certified_artifact,
            });
        }
        self.sessions.insert(
            session_id,
            SessionReservation {
                context_ceiling_tokens: request.context_ceiling_tokens,
                kv_configuration: request.kv_configuration,
                kv_budget_bytes: session_kv_bytes,
            },
        );
        self.next_session_id = next_session_id;
        Ok(SessionAdmission {
            session_id,
            receipt,
        })
    }

    /// Changes the resident artifact only while no session can observe it.
    pub fn swap_resident_artifact(
        &mut self,
        artifact: ArtifactReservation,
    ) -> Result<(), AdmissionRefusal> {
        if self
            .resident_artifact
            .as_ref()
            .is_some_and(|resident| resident.artifact == artifact)
        {
            return Ok(());
        }
        if !self.sessions.is_empty() {
            return Err(AdmissionRefusal::ArtifactSwapActiveSessions {
                active_sessions: self.sessions.len(),
            });
        }
        let certified_artifact = self.envelope_for(&artifact)?.artifact.clone();
        let required_bytes = self.projected_total_bytes(artifact.weight_bytes, 0, 0)?;
        if required_bytes > self.machine.unified_memory_bytes {
            return Err(AdmissionRefusal::InsufficientReservedLaneMemory {
                required_bytes,
                available_bytes: self.machine.unified_memory_bytes,
            });
        }
        self.resident_artifact = Some(ResidentArtifact {
            artifact: certified_artifact,
        });
        Ok(())
    }

    /// Unloads an idle artifact. Active sessions retain their resident weights.
    pub fn unload_resident_artifact(&mut self) -> Result<(), AdmissionRefusal> {
        if !self.sessions.is_empty() {
            return Err(AdmissionRefusal::ArtifactSwapActiveSessions {
                active_sessions: self.sessions.len(),
            });
        }
        self.resident_artifact = None;
        Ok(())
    }

    /// Releases a session's complete KV reservation. An idle resident artifact
    /// remains loaded until an explicit between-session swap or unload.
    pub fn close_session(
        &mut self,
        session_id: SessionId,
    ) -> Result<ResidencyAccounting, AdmissionRefusal> {
        self.sessions
            .remove(&session_id)
            .ok_or(AdmissionRefusal::UnknownSession { session_id })?;
        Ok(self.accounting())
    }

    /// Validates aligned reuse for one existing session. No method accepts a
    /// second session or a shared cache identifier, so cross-session KV reuse
    /// cannot be represented by this API.
    pub fn admit_aligned_reuse(
        &self,
        session_id: SessionId,
        retained_prefix_tokens: u32,
    ) -> Result<AlignedReuseAdmission, AdmissionRefusal> {
        let session = self
            .sessions
            .get(&session_id)
            .ok_or(AdmissionRefusal::UnknownSession { session_id })?;
        if retained_prefix_tokens > session.context_ceiling_tokens {
            return Err(AdmissionRefusal::InvalidContextCeiling {
                requested_tokens: retained_prefix_tokens,
                required_tokens: session.context_ceiling_tokens,
            });
        }
        session
            .kv_configuration
            .validate_boundary(retained_prefix_tokens)?;
        Ok(AlignedReuseAdmission {
            session_id,
            retained_prefix_tokens,
            reused_blocks: retained_prefix_tokens / session.kv_configuration.block_size_tokens,
        })
    }

    /// Produces the worker-facing continuation route after same-session reuse
    /// admission. The returned route contains no cross-session sharing handle.
    pub fn route_continuation(
        &self,
        session_id: SessionId,
        retained_prefix_tokens: u32,
    ) -> Result<ContinuationRoute, AdmissionRefusal> {
        let reuse = self.admit_aligned_reuse(session_id, retained_prefix_tokens)?;
        Ok(ContinuationRoute {
            session_id: reuse.session_id,
            retained_prefix_tokens: reuse.retained_prefix_tokens,
            reused_blocks: reuse.reused_blocks,
        })
    }

    fn envelope_for(
        &self,
        artifact: &ArtifactReservation,
    ) -> Result<&PlatformEnvelope, AdmissionRefusal> {
        let matching_tuple: Vec<&PlatformEnvelope> = self
            .envelopes
            .iter()
            .filter(|envelope| envelope.validate_machine_tuple(&self.machine).is_ok())
            .collect();
        if matching_tuple.is_empty() {
            return Err(AdmissionRefusal::UnsupportedPlatformTuple);
        }
        matching_tuple
            .into_iter()
            .find(|envelope| envelope.validate_artifact(artifact).is_ok())
            .ok_or(AdmissionRefusal::IncompatibleArtifact)
    }

    fn session_kv_bytes(&self) -> Result<u64, AdmissionRefusal> {
        self.sessions.values().try_fold(0_u64, |total, session| {
            total.checked_add(session.kv_budget_bytes).ok_or(
                AdmissionRefusal::InsufficientReservedLaneMemory {
                    required_bytes: u64::MAX,
                    available_bytes: self.machine.unified_memory_bytes,
                },
            )
        })
    }

    fn projected_total_bytes(
        &self,
        artifact_weight_bytes: u64,
        existing_session_kv_bytes: u64,
        additional_session_kv_bytes: u64,
    ) -> Result<u64, AdmissionRefusal> {
        self.reserved_embed_rerank_bytes
            .checked_add(artifact_weight_bytes)
            .and_then(|bytes| bytes.checked_add(existing_session_kv_bytes))
            .and_then(|bytes| bytes.checked_add(additional_session_kv_bytes))
            .ok_or(AdmissionRefusal::InsufficientReservedLaneMemory {
                required_bytes: u64::MAX,
                available_bytes: self.machine.unified_memory_bytes,
            })
    }
}

fn kv_budget_bytes(
    context_ceiling_tokens: u32,
    kv_bytes_per_token: u64,
) -> Result<u64, AdmissionRefusal> {
    u64::from(context_ceiling_tokens)
        .checked_mul(kv_bytes_per_token)
        .ok_or(AdmissionRefusal::InsufficientReservedLaneMemory {
            required_bytes: u64::MAX,
            available_bytes: 0,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn primary_artifact() -> ArtifactReservation {
        ArtifactReservation::new("catalog-primary", GIB, 1024 * 1024).unwrap()
    }

    fn alternate_artifact() -> ArtifactReservation {
        ArtifactReservation::new("catalog-alternate", 2 * GIB, 1024 * 1024).unwrap()
    }

    fn envelope(artifact: ArtifactReservation) -> PlatformEnvelope {
        PlatformEnvelope::new(
            "profile-certified",
            "24A123",
            WAVE_ONE_MIN_UNIFIED_MEMORY_BYTES,
            GIB,
            artifact,
        )
        .unwrap()
    }

    fn machine() -> MachineTuple {
        MachineTuple::new(
            "profile-certified",
            "24A123",
            WAVE_ONE_MIN_UNIFIED_MEMORY_BYTES,
        )
    }

    fn request(artifact: ArtifactReservation) -> AdmissionRequest {
        AdmissionRequest::new(
            "caller-a",
            artifact,
            WAVE_ONE_CONTEXT_CEILING_TOKENS,
            GenerationConfiguration::greedy_top1(),
            SessionKvConfiguration::new(256, 1).unwrap(),
        )
    }

    #[test]
    fn platform_envelope_requires_the_recorded_tuple_and_64_gib_floor() {
        let artifact = primary_artifact();
        let envelope = envelope(artifact.clone());
        let wrong_profile = MachineTuple::new("other-profile", "24A123", 64 * GIB);
        let wrong_build = MachineTuple::new("profile-certified", "other-build", 64 * GIB);
        let low_memory = MachineTuple::new("profile-certified", "24A123", 63 * GIB);

        for unsupported in [wrong_profile, wrong_build, low_memory] {
            let mut router = ResidencyRouter::new(unsupported, envelope.clone());
            assert_eq!(
                router.admit_session(request(artifact.clone())),
                Err(AdmissionRefusal::UnsupportedPlatformTuple)
            );
        }
    }

    #[test]
    fn admission_reserves_literal_lanes_weights_and_complete_32k_kv_budget() {
        let artifact = primary_artifact();
        let mut router = ResidencyRouter::new(machine(), envelope(artifact.clone()));

        let admission = router.admit_session(request(artifact)).unwrap();
        assert_eq!(
            admission.receipt.reserved_session_kv_bytes,
            u64::from(WAVE_ONE_CONTEXT_CEILING_TOKENS) * 1024 * 1024
        );
        assert_eq!(admission.receipt.reserved_embed_rerank_bytes, GIB);
        assert_eq!(admission.receipt.reserved_artifact_weight_bytes, GIB);
        assert_eq!(router.accounting().active_session_count, 1);
        assert_eq!(
            router.accounting().session_kv_bytes,
            admission.receipt.reserved_session_kv_bytes
        );
    }

    #[test]
    fn failed_admission_leaves_no_partial_session_or_weight_reservation() {
        let artifact = primary_artifact();
        let mut router = ResidencyRouter::new(machine(), envelope(artifact.clone()));
        let first = router.admit_session(request(artifact.clone())).unwrap();
        let before = router.accounting();

        assert!(matches!(
            router.admit_session(request(artifact)),
            Err(AdmissionRefusal::InsufficientReservedLaneMemory { .. })
        ));
        assert_eq!(router.accounting(), before);
        assert_eq!(router.active_session_count(), 1);
        assert_eq!(router.resident_artifact(), Some(&primary_artifact()));

        router.close_session(first.session_id).unwrap();
    }

    #[test]
    fn ceiling_generation_and_artifact_refusals_are_typed() {
        let artifact = primary_artifact();
        let mut router = ResidencyRouter::new(machine(), envelope(artifact.clone()));

        let mut invalid_ceiling = request(artifact.clone());
        invalid_ceiling.context_ceiling_tokens = WAVE_ONE_CONTEXT_CEILING_TOKENS - 1;
        assert_eq!(
            router.admit_session(invalid_ceiling),
            Err(AdmissionRefusal::InvalidContextCeiling {
                requested_tokens: WAVE_ONE_CONTEXT_CEILING_TOKENS - 1,
                required_tokens: WAVE_ONE_CONTEXT_CEILING_TOKENS,
            })
        );

        let mut non_greedy = request(artifact.clone());
        non_greedy.generation.sampling = SamplingMode::TopK { k: 2 };
        assert_eq!(
            router.admit_session(non_greedy),
            Err(AdmissionRefusal::UnsupportedGenerationConfig)
        );

        assert_eq!(
            router.admit_session(request(alternate_artifact())),
            Err(AdmissionRefusal::IncompatibleArtifact)
        );
        assert_eq!(router.accounting().active_session_count, 0);
        assert!(router.resident_artifact().is_none());
    }

    #[test]
    fn resident_artifacts_do_not_swap_until_active_sessions_close() {
        let primary = primary_artifact();
        let alternate = alternate_artifact();
        let mut router = ResidencyRouter::with_envelopes(
            machine(),
            vec![envelope(primary.clone()), envelope(alternate.clone())],
        )
        .unwrap();
        let admission = router.admit_session(request(primary.clone())).unwrap();

        assert_eq!(
            router.swap_resident_artifact(alternate.clone()),
            Err(AdmissionRefusal::ArtifactSwapActiveSessions { active_sessions: 1 })
        );
        assert_eq!(router.resident_artifact(), Some(&primary));

        router.close_session(admission.session_id).unwrap();
        router.swap_resident_artifact(alternate.clone()).unwrap();
        assert_eq!(router.resident_artifact(), Some(&alternate));
    }

    #[test]
    fn reuse_and_continuation_are_aligned_and_session_local() {
        let artifact = primary_artifact();
        let mut router = ResidencyRouter::new(machine(), envelope(artifact.clone()));
        let admission = router.admit_session(request(artifact)).unwrap();

        assert_eq!(
            router.admit_aligned_reuse(admission.session_id, 257),
            Err(AdmissionRefusal::InvalidKvAlignment {
                position_tokens: 257,
                alignment_tokens: 256,
            })
        );
        let route = router
            .route_continuation(admission.session_id, 1024)
            .unwrap();
        assert_eq!(route.session_id, admission.session_id);
        assert_eq!(route.retained_prefix_tokens, 1024);
        assert_eq!(route.reused_blocks, 4);
        assert_eq!(
            router.route_continuation(SessionId(999), 1024),
            Err(AdmissionRefusal::UnknownSession {
                session_id: SessionId(999)
            })
        );
    }

    #[test]
    fn refusal_wire_ids_are_pinned() {
        assert_eq!(
            AdmissionRefusal::UnsupportedPlatformTuple.wire_id(),
            "unsupported_platform_tuple"
        );
        assert_eq!(
            AdmissionRefusal::InvalidContextCeiling {
                requested_tokens: 1,
                required_tokens: WAVE_ONE_CONTEXT_CEILING_TOKENS,
            }
            .wire_id(),
            "invalid_context_ceiling"
        );
        assert_eq!(
            AdmissionRefusal::InsufficientReservedLaneMemory {
                required_bytes: 1,
                available_bytes: 0,
            }
            .wire_id(),
            "insufficient_reserved_lane_memory"
        );
        assert_eq!(
            AdmissionRefusal::IncompatibleArtifact.wire_id(),
            "incompatible_artifact"
        );
        assert_eq!(
            AdmissionRefusal::UnsupportedGenerationConfig.wire_id(),
            "unsupported_generation_config"
        );
    }
}
