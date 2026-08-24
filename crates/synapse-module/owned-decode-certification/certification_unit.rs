//! Records and validates the complete wave-1 certification unit.
//!
//! The immutable artifact lineage is deliberately separate from the machine-scoped
//! evidence. A source or derived artifact can therefore be reused by another
//! machine without accidentally reusing its probe, runtime, or M5 measurement.
//! A record is usable only after every required gate validates against the exact
//! artifact, machine tuple, runtime configuration, and battery manifest.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The immutable battery identity used by every wave-1 certification result.
pub const AGENTIC_BATTERY_ID: &str = "agentic-battery-v1";
/// The pinned oracle revision used by serial fidelity certification.
pub const LLAMA_CPP_ORACLE_REVISION: &str = "b9580";
/// The only accepted deterministic repack contract for this certification unit.
pub const Q8_INGEST_DERIVATION_CONTRACT: &str = "q8-ingest-v1";
/// The platform envelope certification identity.
pub const PLATFORM_ENVELOPE_ID: &str = "platform-envelope-v1";
/// The concurrent embed-load certification identity.
pub const EMBED_LOAD_ID: &str = "embed-load-v1";
/// The required admitted context ceiling for the wave-1 platform envelope.
pub const WAVE_1_CONTEXT_CEILING_TOKENS: u32 = 32_768;
/// The minimum unified memory required by the wave-1 platform envelope.
pub const WAVE_1_MIN_UNIFIED_MEMORY_BYTES: u64 = 64 * 1024 * 1024 * 1024;

/// A deterministic ingest-time derived artifact and its verified lineage.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedArtifactLineage {
    /// Must be [`Q8_INGEST_DERIVATION_CONTRACT`] for wave-1 repacks.
    pub derivation_contract: String,
    /// Digest of the deterministic ingest inputs.
    pub deterministic_inputs_digest: String,
    /// Source artifact digest retained in the derivation record.
    pub source_digest: String,
    /// Digest produced by the deterministic derivation.
    pub derived_digest: String,
    /// Digest independently verified before the derived artifact was trusted.
    pub verified_derived_digest: String,
}

/// Immutable artifact identity and, when applicable, its certified derivation.
///
/// This type contains no machine, probe, runtime, or benchmark state. Those
/// values belong to [`MachineScopedEvidence`] and cannot be inherited merely by
/// reusing an artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactLineage {
    pub artifact_id: String,
    pub model_id: String,
    pub quantization: String,
    pub source_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived: Option<DerivedArtifactLineage>,
}

impl ArtifactLineage {
    fn validate(&self) -> Result<(), CertificationError> {
        required(&self.artifact_id, "artifact_id")?;
        required(&self.model_id, "model_id")?;
        required(&self.quantization, "quantization")?;
        required(&self.source_digest, "source_digest")?;

        if let Some(derived) = &self.derived {
            if derived.derivation_contract != Q8_INGEST_DERIVATION_CONTRACT {
                return Err(CertificationError::InvalidDerivationContract);
            }
            required(
                &derived.deterministic_inputs_digest,
                "deterministic_inputs_digest",
            )?;
            required(&derived.source_digest, "derived.source_digest")?;
            required(&derived.derived_digest, "derived_digest")?;
            required(&derived.verified_derived_digest, "verified_derived_digest")?;
            if derived.source_digest != self.source_digest {
                return Err(CertificationError::DerivationSourceMismatch);
            }
            if derived.derived_digest != derived.verified_derived_digest {
                return Err(CertificationError::DerivedDigestMismatch);
            }
        }
        Ok(())
    }
}

/// The indivisible base-plus-native-MTP-head-plus-depth-controller-gate unit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificationUnit {
    /// References [`ArtifactLineage::artifact_id`] rather than embedding mutable
    /// machine evidence in the immutable artifact identity.
    pub base_artifact_id: String,
    pub native_mtp_head_digest: String,
    pub depth_controller_gate_digest: String,
    pub catalog_fingerprint: String,
}

impl CertificationUnit {
    fn validate(&self, lineage: &ArtifactLineage) -> Result<(), CertificationError> {
        required(&self.base_artifact_id, "base_artifact_id")?;
        required(&self.native_mtp_head_digest, "native_mtp_head_digest")?;
        required(
            &self.depth_controller_gate_digest,
            "depth_controller_gate_digest",
        )?;
        required(&self.catalog_fingerprint, "catalog_fingerprint")?;
        if self.base_artifact_id != lineage.artifact_id {
            return Err(CertificationError::CertificationUnitArtifactMismatch);
        }
        Ok(())
    }
}

/// The immutable machine tuple for one machine-scoped certification record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineTuple {
    pub machine_profile_hash: String,
    pub macos_build: String,
    pub unified_memory_bytes: u64,
}

impl MachineTuple {
    fn validate(&self) -> Result<(), CertificationError> {
        required(&self.machine_profile_hash, "machine_profile_hash")?;
        required(&self.macos_build, "macos_build")?;
        if self.unified_memory_bytes == 0 {
            return Err(CertificationError::MissingEvidence("unified_memory_bytes"));
        }
        Ok(())
    }
}

/// The probe execution that produced machine-local evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeEvidence {
    pub probe_id: String,
    pub harness_revision: String,
    pub observed_at_ms: u64,
}

impl ProbeEvidence {
    fn validate(&self) -> Result<(), CertificationError> {
        required(&self.probe_id, "probe_id")?;
        required(&self.harness_revision, "harness_revision")?;
        if self.observed_at_ms == 0 {
            return Err(CertificationError::MissingEvidence("probe observed_at_ms"));
        }
        Ok(())
    }
}

/// The runtime configuration that was actually exercised by the certification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfiguration {
    pub runtime_config_digest: String,
    pub runtime_revision: String,
}

impl RuntimeConfiguration {
    fn validate(&self) -> Result<(), CertificationError> {
        required(&self.runtime_config_digest, "runtime_config_digest")?;
        required(&self.runtime_revision, "runtime_revision")
    }
}

/// The only M5 evidence that can supply production depth-controller constants.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct M5MeasurementEvidence {
    pub measurement_id: String,
    pub measurement_revision: String,
    pub machine_profile_hash: String,
    pub base_artifact_id: String,
    pub catalog_fingerprint: String,
    pub native_mtp_head_digest: String,
    pub runtime_config_digest: String,
    pub head_forward_ms: f64,
    pub backbone_step_ms: f64,
    pub controller_constants_digest: String,
    /// The constants are unavailable to approval until this registered result is
    /// present and matches the record exactly.
    pub registered: bool,
    pub depth_zero_executes_no_head_work: bool,
    pub positive_depth_chains_command_buffer: bool,
}

impl M5MeasurementEvidence {
    fn validate(
        &self,
        unit: &CertificationUnit,
        machine: &MachineTuple,
        runtime: &RuntimeConfiguration,
    ) -> Result<(), CertificationError> {
        required(&self.measurement_id, "m5 measurement_id")?;
        required(&self.measurement_revision, "m5 measurement_revision")?;
        required(
            &self.controller_constants_digest,
            "m5 controller_constants_digest",
        )?;
        if !self.registered {
            return Err(CertificationError::MissingM5Measurement);
        }
        if self.machine_profile_hash != machine.machine_profile_hash
            || self.base_artifact_id != unit.base_artifact_id
            || self.catalog_fingerprint != unit.catalog_fingerprint
            || self.native_mtp_head_digest != unit.native_mtp_head_digest
            || self.runtime_config_digest != runtime.runtime_config_digest
        {
            return Err(CertificationError::M5MeasurementMismatch);
        }
        if !positive_finite(self.head_forward_ms) || !positive_finite(self.backbone_step_ms) {
            return Err(CertificationError::InvalidM5Measurement);
        }
        if !self.depth_zero_executes_no_head_work || !self.positive_depth_chains_command_buffer {
            return Err(CertificationError::InvalidM5Measurement);
        }
        Ok(())
    }
}

/// Machine-local evidence intentionally kept outside immutable artifact lineage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MachineScopedEvidence {
    pub machine: MachineTuple,
    pub probe: ProbeEvidence,
    pub runtime: RuntimeConfiguration,
    pub m5_measurement: M5MeasurementEvidence,
}

impl MachineScopedEvidence {
    fn validate(&self, unit: &CertificationUnit) -> Result<(), CertificationError> {
        self.machine.validate()?;
        self.probe.validate()?;
        self.runtime.validate()?;
        self.m5_measurement
            .validate(unit, &self.machine, &self.runtime)
    }
}

/// The result record names required for a certifiable wave-1 unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CertificationGate {
    SerialOracleFidelity,
    SpeculativeSerialFidelity,
    MtpSpeed,
    KvMatrix,
    PlatformEnvelope,
    EmbedLoad,
    TokenTap,
}

/// Every gate that must be present exactly once before a record can certify.
pub const REQUIRED_CERTIFICATION_GATES: [CertificationGate; 7] = [
    CertificationGate::SerialOracleFidelity,
    CertificationGate::SpeculativeSerialFidelity,
    CertificationGate::MtpSpeed,
    CertificationGate::KvMatrix,
    CertificationGate::PlatformEnvelope,
    CertificationGate::EmbedLoad,
    CertificationGate::TokenTap,
];

/// Exact-comparison evidence shared by the two fidelity results.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenFidelityEvidence {
    pub generated_token_ids_match: bool,
    pub stop_position_matches: bool,
    pub finish_reason_matches: bool,
}

impl TokenFidelityEvidence {
    fn validates_exactly(&self) -> bool {
        self.generated_token_ids_match && self.stop_position_matches && self.finish_reason_matches
    }
}

/// Serial owned decode compared with the pinned llama.cpp oracle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SerialOracleFidelityResult {
    pub manifest_digest: String,
    pub battery_id: String,
    pub oracle_revision: String,
    pub greedy_only: bool,
    pub fidelity: TokenFidelityEvidence,
}

/// Speculative owned decode compared with the certified owned serial path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeSerialFidelityResult {
    pub manifest_digest: String,
    pub battery_id: String,
    pub serial_certification_id: String,
    pub leviathan_greedy_acceptance: bool,
    pub fidelity: TokenFidelityEvidence,
}

/// Per-round telemetry required for a speculative MTP certification result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeTelemetry {
    pub proposed_depth: u32,
    pub accepted_depth: u32,
    pub acceptance_rate: f64,
    pub verification_work: u64,
    pub controller_decisions_digest: String,
}

impl SpeculativeTelemetry {
    fn validate(&self, gate: CertificationGate) -> Result<(), CertificationError> {
        if self.accepted_depth > self.proposed_depth
            || !self.acceptance_rate.is_finite()
            || !(0.0..=1.0).contains(&self.acceptance_rate)
        {
            return Err(CertificationError::InvalidGateEvidence {
                gate,
                reason: "speculative telemetry is invalid",
            });
        }
        required(
            &self.controller_decisions_digest,
            "controller_decisions_digest",
        )
    }
}

/// Timing evidence for one arm of one same-session MTP speed repetition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimingArmEvidence {
    pub loaded_session_id: String,
    pub machine_profile_hash: String,
    pub macos_build: String,
    pub ac_power_connected: bool,
    pub one_minute_load_average: f64,
    /// Mean tok/s across the battery prompts for this repetition.
    pub mean_tokens_per_second: f64,
}

impl TimingArmEvidence {
    fn validate(
        &self,
        machine: &MachineTuple,
        gate: CertificationGate,
    ) -> Result<(), CertificationError> {
        required(&self.loaded_session_id, "loaded_session_id")?;
        if self.machine_profile_hash != machine.machine_profile_hash
            || self.macos_build != machine.macos_build
        {
            return Err(CertificationError::MachineTupleMismatch);
        }
        if !self.ac_power_connected
            || !nonnegative_finite(self.one_minute_load_average)
            || self.one_minute_load_average >= 6.0
            || !positive_finite(self.mean_tokens_per_second)
        {
            return Err(CertificationError::InvalidGateEvidence {
                gate,
                reason: "speed arm does not meet the power, load, or timing requirements",
            });
        }
        Ok(())
    }
}

/// One serial/MTP pair collected consecutively in the same loaded session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MtpSpeedRepetition {
    pub serial: TimingArmEvidence,
    pub mtp: TimingArmEvidence,
}

/// The complete MTP speed certification result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MtpSpeedResult {
    pub manifest_digest: String,
    pub battery_id: String,
    pub generated_token_window: u32,
    pub serial_warmup_last_three_tok_s: [f64; 3],
    pub mtp_warmup_last_three_tok_s: [f64; 3],
    pub repetitions: Vec<MtpSpeedRepetition>,
    pub telemetry: SpeculativeTelemetry,
}

impl MtpSpeedResult {
    /// Median MTP mean tok/s divided by median serial mean tok/s.
    pub fn speedup(&self) -> Option<f64> {
        if self.repetitions.len() != 3 {
            return None;
        }
        let mut serial = self
            .repetitions
            .iter()
            .map(|repetition| repetition.serial.mean_tokens_per_second)
            .collect::<Vec<_>>();
        let mut mtp = self
            .repetitions
            .iter()
            .map(|repetition| repetition.mtp.mean_tokens_per_second)
            .collect::<Vec<_>>();
        if !serial.iter().copied().all(positive_finite) || !mtp.iter().copied().all(positive_finite)
        {
            return None;
        }
        serial.sort_by(f64::total_cmp);
        mtp.sort_by(f64::total_cmp);
        Some(mtp[1] / serial[1])
    }

    fn validate(&self, machine: &MachineTuple) -> Result<(), CertificationError> {
        let gate = CertificationGate::MtpSpeed;
        if self.battery_id != AGENTIC_BATTERY_ID || self.generated_token_window != 256 {
            return Err(CertificationError::InvalidGateEvidence {
                gate,
                reason: "speed evidence has the wrong battery or token window",
            });
        }
        if !warmup_converged(&self.serial_warmup_last_three_tok_s)
            || !warmup_converged(&self.mtp_warmup_last_three_tok_s)
            || self.repetitions.len() != 3
        {
            return Err(CertificationError::InvalidGateEvidence {
                gate,
                reason: "speed evidence requires three converged repetitions",
            });
        }
        for repetition in &self.repetitions {
            repetition.serial.validate(machine, gate)?;
            repetition.mtp.validate(machine, gate)?;
            if repetition.serial.loaded_session_id != repetition.mtp.loaded_session_id {
                return Err(CertificationError::InvalidGateEvidence {
                    gate,
                    reason: "serial and MTP timing arms did not share a loaded session",
                });
            }
        }
        if self.speedup().map_or(true, |speedup| speedup < 1.5) {
            return Err(CertificationError::InvalidGateEvidence {
                gate,
                reason: "MTP throughput is below the required 1.5x speedup",
            });
        }
        self.telemetry.validate(gate)
    }
}

/// One measured cell in the pre-registered KV selection matrix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvMatrixCandidate {
    pub block_size_tokens: u32,
    pub reused_prefix_bucket_tokens: u32,
    pub alignment_valid: bool,
    pub retained_memory_overhead_percent: f64,
    pub warm_ttft_ms: f64,
}

impl KvMatrixCandidate {
    fn eligible(&self) -> bool {
        self.alignment_valid && self.retained_memory_overhead_percent <= 10.0
    }
}

/// Result facts verified at the selected KV matrix candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvSelectionEvidence {
    pub selected: KvMatrixCandidate,
    pub continuation_token_ids_identical: bool,
    pub reused_token_count: u32,
    pub reused_block_count: u32,
    pub prefill_dispatches_over_reused_range: u64,
    pub cold_ttft_ms: f64,
    pub warm_ttft_ms: f64,
    pub close_restored_allocator_accounting: bool,
}

/// Full measurement and selection evidence for the wave-1 KV matrix.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvMatrixResult {
    pub manifest_digest: String,
    pub machine_profile_hash: String,
    pub candidates: Vec<KvMatrixCandidate>,
    pub selection: KvSelectionEvidence,
}

impl KvMatrixResult {
    fn validate(&self, machine: &MachineTuple) -> Result<(), CertificationError> {
        let gate = CertificationGate::KvMatrix;
        if self.machine_profile_hash != machine.machine_profile_hash {
            return Err(CertificationError::MachineTupleMismatch);
        }
        let expected: BTreeSet<(u32, u32)> = [256, 512, 1024]
            .into_iter()
            .flat_map(|block| {
                [4096, 8192, 16384]
                    .into_iter()
                    .map(move |bucket| (block, bucket))
            })
            .collect();
        let actual: BTreeSet<(u32, u32)> = self
            .candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.block_size_tokens,
                    candidate.reused_prefix_bucket_tokens,
                )
            })
            .collect();
        if self.candidates.len() != expected.len() || actual != expected {
            return Err(CertificationError::InvalidGateEvidence {
                gate,
                reason: "KV evidence does not execute the complete pre-registered matrix",
            });
        }
        if self.candidates.iter().any(|candidate| {
            !candidate.retained_memory_overhead_percent.is_finite()
                || candidate.retained_memory_overhead_percent < 0.0
                || !positive_finite(candidate.warm_ttft_ms)
        }) {
            return Err(CertificationError::InvalidGateEvidence {
                gate,
                reason: "KV matrix contains invalid measurement values",
            });
        }
        let selected = &self.selection.selected;
        if !selected.eligible() || !self.candidates.contains(selected) {
            return Err(CertificationError::InvalidGateEvidence {
                gate,
                reason: "selected KV candidate is absent or ineligible",
            });
        }
        let expected_selected = self
            .candidates
            .iter()
            .filter(|candidate| candidate.eligible())
            .min_by(|left, right| {
                left.warm_ttft_ms
                    .total_cmp(&right.warm_ttft_ms)
                    .then_with(|| right.block_size_tokens.cmp(&left.block_size_tokens))
                    .then_with(|| {
                        right
                            .reused_prefix_bucket_tokens
                            .cmp(&left.reused_prefix_bucket_tokens)
                    })
            });
        if expected_selected != Some(selected) {
            return Err(CertificationError::InvalidGateEvidence {
                gate,
                reason: "KV selection does not apply the registered deterministic rule",
            });
        }
        if !self.selection.continuation_token_ids_identical
            || self.selection.reused_token_count < selected.reused_prefix_bucket_tokens
            || self.selection.reused_block_count == 0
            || self.selection.prefill_dispatches_over_reused_range != 0
            || !positive_finite(self.selection.cold_ttft_ms)
            || !positive_finite(self.selection.warm_ttft_ms)
            || self.selection.warm_ttft_ms > self.selection.cold_ttft_ms * 0.20
            || !self.selection.close_restored_allocator_accounting
        {
            return Err(CertificationError::InvalidGateEvidence {
                gate,
                reason: "selected KV evidence does not meet the reuse requirements",
            });
        }
        Ok(())
    }
}

/// The measured single-machine platform envelope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlatformEnvelopeResult {
    pub manifest_digest: String,
    pub envelope_id: String,
    pub machine_profile_hash: String,
    pub macos_build: String,
    pub unified_memory_bytes: u64,
    pub reserved_embed_rerank_bytes: u64,
    pub artifact_weight_bytes: u64,
    pub kv_bytes_per_token: u64,
    pub mandatory_context_ceiling_tokens: u32,
    pub admitted_and_exercised_32k_session: bool,
    pub exercised_reservation_accounting: bool,
    pub exercised_kv_reuse: bool,
    pub exercised_streaming: bool,
    pub exercised_scheduler_interleaving: bool,
}

impl PlatformEnvelopeResult {
    fn validate(&self, machine: &MachineTuple) -> Result<(), CertificationError> {
        let gate = CertificationGate::PlatformEnvelope;
        if self.envelope_id != PLATFORM_ENVELOPE_ID
            || self.machine_profile_hash != machine.machine_profile_hash
            || self.macos_build != machine.macos_build
            || self.unified_memory_bytes != machine.unified_memory_bytes
            || self.unified_memory_bytes < WAVE_1_MIN_UNIFIED_MEMORY_BYTES
            || self.reserved_embed_rerank_bytes == 0
            || self.artifact_weight_bytes == 0
            || self.kv_bytes_per_token == 0
            || self.mandatory_context_ceiling_tokens != WAVE_1_CONTEXT_CEILING_TOKENS
            || !self.admitted_and_exercised_32k_session
            || !self.exercised_reservation_accounting
            || !self.exercised_kv_reuse
            || !self.exercised_streaming
            || !self.exercised_scheduler_interleaving
        {
            return Err(CertificationError::InvalidGateEvidence {
                gate,
                reason: "platform envelope does not match the certified tuple or required exercise",
            });
        }
        Ok(())
    }
}

/// The concurrent embed-load certification result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbedLoadResult {
    pub manifest_digest: String,
    pub workload_id: String,
    pub runtime_config_digest: String,
    pub concurrent_clients: u32,
    pub poisson_aggregate_rate_per_second: f64,
    pub duration_seconds: u32,
    pub warmup_seconds: u32,
    pub completed_samples: u32,
    pub failed_embeddings: u32,
    pub timed_out_embeddings: u32,
    pub nearest_rank_p95_ms: f64,
    pub active_decode_context_ceiling_tokens: u32,
    pub used_shipped_scheduler_configuration: bool,
}

impl EmbedLoadResult {
    fn validate(&self, runtime: &RuntimeConfiguration) -> Result<(), CertificationError> {
        let gate = CertificationGate::EmbedLoad;
        if self.workload_id != EMBED_LOAD_ID
            || self.runtime_config_digest != runtime.runtime_config_digest
            || self.concurrent_clients != 8
            || self.poisson_aggregate_rate_per_second != 5.0
            || self.duration_seconds != 120
            || self.warmup_seconds != 10
            || self.completed_samples < 500
            || self.failed_embeddings != 0
            || self.timed_out_embeddings != 0
            || !positive_finite(self.nearest_rank_p95_ms)
            || self.nearest_rank_p95_ms > 150.0
            || self.active_decode_context_ceiling_tokens != WAVE_1_CONTEXT_CEILING_TOKENS
            || !self.used_shipped_scheduler_configuration
        {
            return Err(CertificationError::InvalidGateEvidence {
                gate,
                reason: "embed-load evidence does not meet the required workload or SLO",
            });
        }
        Ok(())
    }
}

/// Certification evidence for the read-only pre-commit token tap.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenTapResult {
    pub manifest_digest: String,
    pub observed_after_acceptance_before_emission: bool,
    pub read_only: bool,
    pub token_ids_identical_when_enabled: bool,
    pub stop_position_identical_when_enabled: bool,
    pub finish_reason_identical_when_enabled: bool,
    pub emitted_bytes_identical_when_enabled: bool,
}

impl TokenTapResult {
    fn validate(&self) -> Result<(), CertificationError> {
        if !self.observed_after_acceptance_before_emission
            || !self.read_only
            || !self.token_ids_identical_when_enabled
            || !self.stop_position_identical_when_enabled
            || !self.finish_reason_identical_when_enabled
            || !self.emitted_bytes_identical_when_enabled
        {
            return Err(CertificationError::InvalidGateEvidence {
                gate: CertificationGate::TokenTap,
                reason: "token tap is not read-only and byte-identical",
            });
        }
        Ok(())
    }
}

/// One machine-local certification result, tagged for durable serialization.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "result", rename_all = "kebab-case")]
pub enum CertificationGateResult {
    SerialOracleFidelity(SerialOracleFidelityResult),
    SpeculativeSerialFidelity(SpeculativeSerialFidelityResult),
    MtpSpeed(MtpSpeedResult),
    KvMatrix(KvMatrixResult),
    PlatformEnvelope(PlatformEnvelopeResult),
    EmbedLoad(EmbedLoadResult),
    TokenTap(TokenTapResult),
}

impl CertificationGateResult {
    #[must_use]
    pub const fn gate(&self) -> CertificationGate {
        match self {
            Self::SerialOracleFidelity(_) => CertificationGate::SerialOracleFidelity,
            Self::SpeculativeSerialFidelity(_) => CertificationGate::SpeculativeSerialFidelity,
            Self::MtpSpeed(_) => CertificationGate::MtpSpeed,
            Self::KvMatrix(_) => CertificationGate::KvMatrix,
            Self::PlatformEnvelope(_) => CertificationGate::PlatformEnvelope,
            Self::EmbedLoad(_) => CertificationGate::EmbedLoad,
            Self::TokenTap(_) => CertificationGate::TokenTap,
        }
    }

    #[must_use]
    pub fn manifest_digest(&self) -> &str {
        match self {
            Self::SerialOracleFidelity(result) => &result.manifest_digest,
            Self::SpeculativeSerialFidelity(result) => &result.manifest_digest,
            Self::MtpSpeed(result) => &result.manifest_digest,
            Self::KvMatrix(result) => &result.manifest_digest,
            Self::PlatformEnvelope(result) => &result.manifest_digest,
            Self::EmbedLoad(result) => &result.manifest_digest,
            Self::TokenTap(result) => &result.manifest_digest,
        }
    }

    fn validate(
        &self,
        record_manifest_digest: &str,
        machine: &MachineTuple,
        runtime: &RuntimeConfiguration,
    ) -> Result<(), CertificationError> {
        let gate = self.gate();
        required(self.manifest_digest(), "gate manifest_digest")?;
        if self.manifest_digest() != record_manifest_digest {
            return Err(CertificationError::ManifestDigestMismatch { gate });
        }
        match self {
            Self::SerialOracleFidelity(result) => {
                if result.battery_id != AGENTIC_BATTERY_ID
                    || result.oracle_revision != LLAMA_CPP_ORACLE_REVISION
                    || !result.greedy_only
                    || !result.fidelity.validates_exactly()
                {
                    return Err(CertificationError::InvalidGateEvidence {
                        gate,
                        reason: "serial oracle fidelity is not exact and pinned",
                    });
                }
                Ok(())
            }
            Self::SpeculativeSerialFidelity(result) => {
                required(&result.serial_certification_id, "serial_certification_id")?;
                if result.battery_id != AGENTIC_BATTERY_ID
                    || !result.leviathan_greedy_acceptance
                    || !result.fidelity.validates_exactly()
                {
                    return Err(CertificationError::InvalidGateEvidence {
                        gate,
                        reason:
                            "speculative fidelity is not exact against the certified serial path",
                    });
                }
                Ok(())
            }
            Self::MtpSpeed(result) => result.validate(machine),
            Self::KvMatrix(result) => result.validate(machine),
            Self::PlatformEnvelope(result) => result.validate(machine),
            Self::EmbedLoad(result) => result.validate(runtime),
            Self::TokenTap(result) => result.validate(),
        }
    }
}

/// One immutable, complete certification record. The artifact lineage is distinct
/// from `machine_evidence`, which prevents certification from crossing a machine
/// profile, probe, runtime configuration, or M5 measurement boundary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificationRecord {
    pub record_id: String,
    pub manifest_digest: String,
    pub artifact_lineage: ArtifactLineage,
    pub unit: CertificationUnit,
    pub machine_evidence: MachineScopedEvidence,
    pub gate_results: Vec<CertificationGateResult>,
}

impl CertificationRecord {
    /// Validate all immutable lineage, machine-local evidence, and mandatory gate
    /// records. A missing or mismatched value is an error rather than a partial
    /// certification, so callers fail closed by construction.
    pub fn validate(&self) -> Result<(), CertificationError> {
        required(&self.record_id, "record_id")?;
        required(&self.manifest_digest, "manifest_digest")?;
        self.artifact_lineage.validate()?;
        self.unit.validate(&self.artifact_lineage)?;
        self.machine_evidence.validate(&self.unit)?;

        let mut seen = BTreeSet::new();
        for result in &self.gate_results {
            let gate = result.gate();
            if !seen.insert(gate) {
                return Err(CertificationError::DuplicateGateResult { gate });
            }
            result.validate(
                &self.manifest_digest,
                &self.machine_evidence.machine,
                &self.machine_evidence.runtime,
            )?;
        }
        for gate in REQUIRED_CERTIFICATION_GATES {
            if !seen.contains(&gate) {
                return Err(CertificationError::MissingGateResult { gate });
            }
        }
        Ok(())
    }

    /// Check that this already-valid record applies to the exact serving tuple.
    pub fn validate_for(&self, request: &CertificationRequest) -> Result<(), CertificationError> {
        self.validate()?;
        if self.artifact_lineage != request.artifact_lineage {
            return Err(CertificationError::ArtifactLineageMismatch);
        }
        if self.unit != request.unit {
            return Err(CertificationError::CertificationUnitMismatch);
        }
        if self.machine_evidence.machine != request.machine {
            return Err(CertificationError::MachineTupleMismatch);
        }
        if self.machine_evidence.runtime != request.runtime {
            return Err(CertificationError::RuntimeConfigurationMismatch);
        }
        Ok(())
    }
}

/// The exact serving tuple seeking authorization from a certification record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CertificationRequest {
    pub artifact_lineage: ArtifactLineage,
    pub unit: CertificationUnit,
    pub machine: MachineTuple,
    pub runtime: RuntimeConfiguration,
}

/// Immutable record storage with no replacement operation. Records are validated
/// at registration and exposed only by shared reference.
#[derive(Clone, Debug, Default)]
pub struct CertificationRegistry {
    records: BTreeMap<String, CertificationRecord>,
}

impl CertificationRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one validated immutable certification record.
    pub fn register(&mut self, record: CertificationRecord) -> Result<(), CertificationError> {
        record.validate()?;
        if self.records.contains_key(&record.record_id) {
            return Err(CertificationError::RecordAlreadyRegistered);
        }
        self.records.insert(record.record_id.clone(), record);
        Ok(())
    }

    /// Return an immutable record by ID.
    #[must_use]
    pub fn record(&self, record_id: &str) -> Option<&CertificationRecord> {
        self.records.get(record_id)
    }

    /// Resolve a record for the supplied unit and revalidate every matching
    /// boundary before allowing serving to continue.
    pub fn certify_request(
        &self,
        request: &CertificationRequest,
    ) -> Result<&CertificationRecord, CertificationError> {
        let mut first_mismatch = None;
        for record in self
            .records
            .values()
            .filter(|record| record.unit.catalog_fingerprint == request.unit.catalog_fingerprint)
        {
            match record.validate_for(request) {
                Ok(()) => return Ok(record),
                Err(error) if first_mismatch.is_none() => first_mismatch = Some(error),
                Err(_) => {}
            }
        }
        Err(first_mismatch.unwrap_or(CertificationError::NoMatchingCertification))
    }
}

/// A fail-closed certification validation result.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum CertificationError {
    #[error("required certification evidence is missing: {0}")]
    MissingEvidence(&'static str),
    #[error("artifact derivation does not use the q8 ingest contract")]
    InvalidDerivationContract,
    #[error("derived artifact lineage points at a different source digest")]
    DerivationSourceMismatch,
    #[error("derived artifact digest was not independently verified")]
    DerivedDigestMismatch,
    #[error("certification unit does not reference its immutable base artifact")]
    CertificationUnitArtifactMismatch,
    #[error("registered M5 measurement evidence is absent")]
    MissingM5Measurement,
    #[error("M5 measurement does not match the machine, artifact, unit, or runtime")]
    M5MeasurementMismatch,
    #[error("M5 measurement evidence is invalid")]
    InvalidM5Measurement,
    #[error("required certification gate result is absent: {gate:?}")]
    MissingGateResult { gate: CertificationGate },
    #[error("duplicate certification gate result: {gate:?}")]
    DuplicateGateResult { gate: CertificationGate },
    #[error("certification gate manifest digest is missing or mismatched: {gate:?}")]
    ManifestDigestMismatch { gate: CertificationGate },
    #[error("invalid certification gate evidence for {gate:?}: {reason}")]
    InvalidGateEvidence {
        gate: CertificationGate,
        reason: &'static str,
    },
    #[error("artifact lineage does not match the requested serving artifact")]
    ArtifactLineageMismatch,
    #[error("certification unit does not match the requested serving unit")]
    CertificationUnitMismatch,
    #[error("machine tuple does not match the certified tuple")]
    MachineTupleMismatch,
    #[error("runtime configuration does not match the certified runtime")]
    RuntimeConfigurationMismatch,
    #[error("certification record ID is already registered")]
    RecordAlreadyRegistered,
    #[error("no certification record matches the requested catalog fingerprint")]
    NoMatchingCertification,
}

fn required(value: &str, name: &'static str) -> Result<(), CertificationError> {
    if value.trim().is_empty() {
        Err(CertificationError::MissingEvidence(name))
    } else {
        Ok(())
    }
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn nonnegative_finite(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn warmup_converged(samples: &[f64; 3]) -> bool {
    if !samples.iter().copied().all(positive_finite) {
        return false;
    }
    let min = samples
        .iter()
        .copied()
        .reduce(f64::min)
        .expect("three samples");
    let max = samples
        .iter()
        .copied()
        .reduce(f64::max)
        .expect("three samples");
    max <= min * 1.15
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lineage() -> ArtifactLineage {
        ArtifactLineage {
            artifact_id: "qwen3-27b-q4km".to_string(),
            model_id: "qwen3.8-27b".to_string(),
            quantization: "gguf-q4-k-m-compatible".to_string(),
            source_digest: "source-digest".to_string(),
            derived: Some(DerivedArtifactLineage {
                derivation_contract: Q8_INGEST_DERIVATION_CONTRACT.to_string(),
                deterministic_inputs_digest: "ingest-inputs".to_string(),
                source_digest: "source-digest".to_string(),
                derived_digest: "derived-digest".to_string(),
                verified_derived_digest: "derived-digest".to_string(),
            }),
        }
    }

    fn unit() -> CertificationUnit {
        CertificationUnit {
            base_artifact_id: "qwen3-27b-q4km".to_string(),
            native_mtp_head_digest: "native-mtp-head".to_string(),
            depth_controller_gate_digest: "depth-controller-gate".to_string(),
            catalog_fingerprint: "catalog-fingerprint".to_string(),
        }
    }

    fn machine() -> MachineTuple {
        MachineTuple {
            machine_profile_hash: "machine-profile".to_string(),
            macos_build: "25F84".to_string(),
            unified_memory_bytes: 128 * 1024 * 1024 * 1024,
        }
    }

    fn runtime() -> RuntimeConfiguration {
        RuntimeConfiguration {
            runtime_config_digest: "runtime-config".to_string(),
            runtime_revision: "runtime-v1".to_string(),
        }
    }

    fn timing_arm(session: &str, tok_s: f64) -> TimingArmEvidence {
        TimingArmEvidence {
            loaded_session_id: session.to_string(),
            machine_profile_hash: machine().machine_profile_hash,
            macos_build: machine().macos_build,
            ac_power_connected: true,
            one_minute_load_average: 1.0,
            mean_tokens_per_second: tok_s,
        }
    }

    fn kv_candidates() -> Vec<KvMatrixCandidate> {
        let mut candidates = Vec::new();
        for block in [256, 512, 1024] {
            for bucket in [4096, 8192, 16384] {
                candidates.push(KvMatrixCandidate {
                    block_size_tokens: block,
                    reused_prefix_bucket_tokens: bucket,
                    alignment_valid: true,
                    retained_memory_overhead_percent: 10.0,
                    warm_ttft_ms: if (block, bucket) == (1024, 4096) {
                        1.0
                    } else {
                        2.0
                    },
                });
            }
        }
        candidates
    }

    fn complete_record() -> CertificationRecord {
        let selected = KvMatrixCandidate {
            block_size_tokens: 1024,
            reused_prefix_bucket_tokens: 4096,
            alignment_valid: true,
            retained_memory_overhead_percent: 10.0,
            warm_ttft_ms: 1.0,
        };
        CertificationRecord {
            record_id: "certification-record-1".to_string(),
            manifest_digest: "agentic-manifest-digest".to_string(),
            artifact_lineage: lineage(),
            unit: unit(),
            machine_evidence: MachineScopedEvidence {
                machine: machine(),
                probe: ProbeEvidence {
                    probe_id: "certification-probe".to_string(),
                    harness_revision: "harness-r1".to_string(),
                    observed_at_ms: 1,
                },
                runtime: runtime(),
                m5_measurement: M5MeasurementEvidence {
                    measurement_id: "m5-native-head-cost".to_string(),
                    measurement_revision: "m5-r1".to_string(),
                    machine_profile_hash: machine().machine_profile_hash,
                    base_artifact_id: unit().base_artifact_id,
                    catalog_fingerprint: unit().catalog_fingerprint,
                    native_mtp_head_digest: unit().native_mtp_head_digest,
                    runtime_config_digest: runtime().runtime_config_digest,
                    head_forward_ms: 1.0,
                    backbone_step_ms: 4.0,
                    controller_constants_digest: "controller-constants".to_string(),
                    registered: true,
                    depth_zero_executes_no_head_work: true,
                    positive_depth_chains_command_buffer: true,
                },
            },
            gate_results: vec![
                CertificationGateResult::SerialOracleFidelity(SerialOracleFidelityResult {
                    manifest_digest: "agentic-manifest-digest".to_string(),
                    battery_id: AGENTIC_BATTERY_ID.to_string(),
                    oracle_revision: LLAMA_CPP_ORACLE_REVISION.to_string(),
                    greedy_only: true,
                    fidelity: TokenFidelityEvidence {
                        generated_token_ids_match: true,
                        stop_position_matches: true,
                        finish_reason_matches: true,
                    },
                }),
                CertificationGateResult::SpeculativeSerialFidelity(
                    SpeculativeSerialFidelityResult {
                        manifest_digest: "agentic-manifest-digest".to_string(),
                        battery_id: AGENTIC_BATTERY_ID.to_string(),
                        serial_certification_id: "serial-certification".to_string(),
                        leviathan_greedy_acceptance: true,
                        fidelity: TokenFidelityEvidence {
                            generated_token_ids_match: true,
                            stop_position_matches: true,
                            finish_reason_matches: true,
                        },
                    },
                ),
                CertificationGateResult::MtpSpeed(MtpSpeedResult {
                    manifest_digest: "agentic-manifest-digest".to_string(),
                    battery_id: AGENTIC_BATTERY_ID.to_string(),
                    generated_token_window: 256,
                    serial_warmup_last_three_tok_s: [10.0, 10.5, 10.2],
                    mtp_warmup_last_three_tok_s: [16.0, 16.5, 16.2],
                    repetitions: vec![
                        MtpSpeedRepetition {
                            serial: timing_arm("session-1", 10.0),
                            mtp: timing_arm("session-1", 16.0),
                        },
                        MtpSpeedRepetition {
                            serial: timing_arm("session-2", 10.5),
                            mtp: timing_arm("session-2", 16.5),
                        },
                        MtpSpeedRepetition {
                            serial: timing_arm("session-3", 10.2),
                            mtp: timing_arm("session-3", 16.2),
                        },
                    ],
                    telemetry: SpeculativeTelemetry {
                        proposed_depth: 4,
                        accepted_depth: 3,
                        acceptance_rate: 0.75,
                        verification_work: 42,
                        controller_decisions_digest: "controller-decisions".to_string(),
                    },
                }),
                CertificationGateResult::KvMatrix(KvMatrixResult {
                    manifest_digest: "agentic-manifest-digest".to_string(),
                    machine_profile_hash: machine().machine_profile_hash,
                    candidates: kv_candidates(),
                    selection: KvSelectionEvidence {
                        selected,
                        continuation_token_ids_identical: true,
                        reused_token_count: 4096,
                        reused_block_count: 4,
                        prefill_dispatches_over_reused_range: 0,
                        cold_ttft_ms: 10.0,
                        warm_ttft_ms: 1.0,
                        close_restored_allocator_accounting: true,
                    },
                }),
                CertificationGateResult::PlatformEnvelope(PlatformEnvelopeResult {
                    manifest_digest: "agentic-manifest-digest".to_string(),
                    envelope_id: PLATFORM_ENVELOPE_ID.to_string(),
                    machine_profile_hash: machine().machine_profile_hash,
                    macos_build: machine().macos_build,
                    unified_memory_bytes: machine().unified_memory_bytes,
                    reserved_embed_rerank_bytes: 1,
                    artifact_weight_bytes: 1,
                    kv_bytes_per_token: 1,
                    mandatory_context_ceiling_tokens: WAVE_1_CONTEXT_CEILING_TOKENS,
                    admitted_and_exercised_32k_session: true,
                    exercised_reservation_accounting: true,
                    exercised_kv_reuse: true,
                    exercised_streaming: true,
                    exercised_scheduler_interleaving: true,
                }),
                CertificationGateResult::EmbedLoad(EmbedLoadResult {
                    manifest_digest: "agentic-manifest-digest".to_string(),
                    workload_id: EMBED_LOAD_ID.to_string(),
                    runtime_config_digest: runtime().runtime_config_digest,
                    concurrent_clients: 8,
                    poisson_aggregate_rate_per_second: 5.0,
                    duration_seconds: 120,
                    warmup_seconds: 10,
                    completed_samples: 500,
                    failed_embeddings: 0,
                    timed_out_embeddings: 0,
                    nearest_rank_p95_ms: 150.0,
                    active_decode_context_ceiling_tokens: WAVE_1_CONTEXT_CEILING_TOKENS,
                    used_shipped_scheduler_configuration: true,
                }),
                CertificationGateResult::TokenTap(TokenTapResult {
                    manifest_digest: "agentic-manifest-digest".to_string(),
                    observed_after_acceptance_before_emission: true,
                    read_only: true,
                    token_ids_identical_when_enabled: true,
                    stop_position_identical_when_enabled: true,
                    finish_reason_identical_when_enabled: true,
                    emitted_bytes_identical_when_enabled: true,
                }),
            ],
        }
    }

    fn request() -> CertificationRequest {
        CertificationRequest {
            artifact_lineage: lineage(),
            unit: unit(),
            machine: machine(),
            runtime: runtime(),
        }
    }

    #[test]
    fn complete_record_validates_and_registry_resolves_exact_request() {
        let record = complete_record();
        record.validate().expect("complete record certifies");

        let mut registry = CertificationRegistry::new();
        registry
            .register(record)
            .expect("validated immutable record registers");
        assert!(registry
            .certify_request(&request())
            .expect("exact tuple is certified")
            .record_id
            .starts_with("certification-record"));
    }

    #[test]
    fn missing_gate_or_manifest_mismatch_fails_closed() {
        let mut missing = complete_record();
        missing.gate_results.pop();
        assert_eq!(
            missing.validate(),
            Err(CertificationError::MissingGateResult {
                gate: CertificationGate::TokenTap,
            })
        );

        let mut mismatched = complete_record();
        let CertificationGateResult::TokenTap(tap) = mismatched
            .gate_results
            .last_mut()
            .expect("token tap is the final gate")
        else {
            panic!("expected token tap");
        };
        tap.manifest_digest = "other-manifest".to_string();
        assert_eq!(
            mismatched.validate(),
            Err(CertificationError::ManifestDigestMismatch {
                gate: CertificationGate::TokenTap,
            })
        );
    }

    #[test]
    fn derivation_m5_machine_and_runtime_mismatches_are_refused() {
        let mut bad_derivation = complete_record();
        bad_derivation
            .artifact_lineage
            .derived
            .as_mut()
            .expect("derived record")
            .verified_derived_digest = "wrong".to_string();
        assert_eq!(
            bad_derivation.validate(),
            Err(CertificationError::DerivedDigestMismatch)
        );

        let mut missing_m5 = complete_record();
        missing_m5.machine_evidence.m5_measurement.registered = false;
        assert_eq!(
            missing_m5.validate(),
            Err(CertificationError::MissingM5Measurement)
        );

        let record = complete_record();
        let mut mismatched_request = request();
        mismatched_request.machine.macos_build = "other-build".to_string();
        assert_eq!(
            record.validate_for(&mismatched_request),
            Err(CertificationError::MachineTupleMismatch)
        );
        let mut mismatched_request = request();
        mismatched_request.runtime.runtime_config_digest = "other-runtime".to_string();
        assert_eq!(
            record.validate_for(&mismatched_request),
            Err(CertificationError::RuntimeConfigurationMismatch)
        );
    }

    #[test]
    fn speed_kv_and_embed_gates_reject_invalid_measurement_claims() {
        let mut too_slow = complete_record();
        let CertificationGateResult::MtpSpeed(speed) = &mut too_slow.gate_results[2] else {
            panic!("expected MTP speed");
        };
        speed.repetitions[0].mtp.mean_tokens_per_second = 10.0;
        speed.repetitions[1].mtp.mean_tokens_per_second = 10.0;
        assert!(matches!(
            too_slow.validate(),
            Err(CertificationError::InvalidGateEvidence {
                gate: CertificationGate::MtpSpeed,
                ..
            })
        ));

        let mut invalid_kv = complete_record();
        let CertificationGateResult::KvMatrix(kv) = &mut invalid_kv.gate_results[3] else {
            panic!("expected KV matrix");
        };
        kv.selection.prefill_dispatches_over_reused_range = 1;
        assert!(matches!(
            invalid_kv.validate(),
            Err(CertificationError::InvalidGateEvidence {
                gate: CertificationGate::KvMatrix,
                ..
            })
        ));

        let mut invalid_embed = complete_record();
        let CertificationGateResult::EmbedLoad(embed) = &mut invalid_embed.gate_results[5] else {
            panic!("expected embed load");
        };
        embed.failed_embeddings = 1;
        assert!(matches!(
            invalid_embed.validate(),
            Err(CertificationError::InvalidGateEvidence {
                gate: CertificationGate::EmbedLoad,
                ..
            })
        ));
    }

    #[test]
    fn registry_rejects_replacement_of_an_immutable_record() {
        let record = complete_record();
        let mut registry = CertificationRegistry::new();
        registry
            .register(record.clone())
            .expect("first record registers");
        assert_eq!(
            registry.register(record),
            Err(CertificationError::RecordAlreadyRegistered)
        );
    }
}
