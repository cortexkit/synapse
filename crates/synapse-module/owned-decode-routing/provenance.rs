//! Selected-lane response provenance.
//!
//! `interfaces` fixes the provenance a selected-lane response must carry. The
//! required fields identify the lane and its identities; additive fields appear
//! only when applicable (fallback, lane-native finish reason, crash retry,
//! constraint identities, and the mapped-refusal underlying ID). A fallback
//! response carries the llama lane's actual fingerprint and provenance, never
//! the refused owned fingerprint.

use serde::{Deserialize, Serialize};
use synapse_core::{worker_engine_names::DECODE_WORKER_ENGINE, Fingerprint};

use crate::owned_decode_routing::family::FamilyRegistration;

/// Canonical identity strings for the production owned-decode lane.
pub const OWNED_ENGINE: &str = DECODE_WORKER_ENGINE;
pub const OWNED_LANE: &str = "decode";
pub const OWNED_WORKER: &str = "supervised";
pub const OWNED_RISK_CLASS: &str = "abort_capable";

/// External finish reasons. Exactly these four are caller-visible; llama-native
/// reasons are normalized to one of these and preserved additively as
/// `lane_finish_reason`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    StopToken,
    MaxTokens,
    GrammarComplete,
    Cancelled,
}

impl FinishReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StopToken => "stop_token",
            Self::MaxTokens => "max_tokens",
            Self::GrammarComplete => "grammar_complete",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Provenance attached to a selected-lane response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LaneProvenance {
    // -- required identity fields --
    pub engine: String,
    pub lane: String,
    pub worker: String,
    pub risk_class: String,
    pub decode_fingerprint: Fingerprint,
    pub processing_fingerprint: Fingerprint,
    pub tokenizer_sanitized_digest: String,
    pub prompt_template_revision: String,
    pub special_token_policy_revision: String,
    pub stop_token_policy_revision: String,
    pub detokenizer_revision: String,
    pub arithmetic_identity_revision: String,
    pub metallib_revision: String,
    pub worker_generation: u64,
    pub last_completed_quantum_sequence: u32,

    // -- additive fields (present only when applicable) --
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lane_finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "u32_is_zero")]
    pub crash_retry_count: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failure_classifications: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint_runtime_identity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constraint_fingerprint: Option<Fingerprint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar_compiler_revision: Option<String>,
    /// Effective GPU chain span. Additive response provenance; llama and legacy
    /// responses omit it because no owned chain policy executed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub underlying_owned_decode_refusal_id: Option<String>,
}

fn u32_is_zero(value: &u32) -> bool {
    *value == 0
}

/// Inputs needed to build owned-lane provenance beyond the family registration
/// and computed fingerprints.
pub struct OwnedProvenanceInputs {
    pub decode_fingerprint: Fingerprint,
    pub processing_fingerprint: Fingerprint,
    pub arithmetic_identity_revision: String,
    pub metallib_revision: String,
    pub worker_generation: u64,
    pub last_completed_quantum_sequence: u32,
    pub chain_k: u32,
}

impl LaneProvenance {
    /// Build provenance for a successful owned-metal-decode response.
    pub fn owned(registration: &FamilyRegistration, inputs: OwnedProvenanceInputs) -> Self {
        Self {
            engine: OWNED_ENGINE.to_string(),
            lane: OWNED_LANE.to_string(),
            worker: OWNED_WORKER.to_string(),
            risk_class: OWNED_RISK_CLASS.to_string(),
            decode_fingerprint: inputs.decode_fingerprint,
            processing_fingerprint: inputs.processing_fingerprint,
            tokenizer_sanitized_digest: registration.tokenizer_sanitized_digest.clone(),
            prompt_template_revision: registration.prompt_template_revision.clone(),
            special_token_policy_revision: registration.special_token_policy_revision.clone(),
            stop_token_policy_revision: registration.stop_token_policy_revision.clone(),
            detokenizer_revision: registration.detokenizer_revision.clone(),
            arithmetic_identity_revision: inputs.arithmetic_identity_revision,
            metallib_revision: inputs.metallib_revision,
            worker_generation: inputs.worker_generation,
            last_completed_quantum_sequence: inputs.last_completed_quantum_sequence,
            fallback_reason: None,
            lane_finish_reason: None,
            crash_retry_count: 0,
            failure_classifications: Vec::new(),
            constraint_runtime_identity: None,
            constraint_fingerprint: None,
            grammar_compiler_revision: None,
            chain_k: Some(inputs.chain_k),
            underlying_owned_decode_refusal_id: None,
        }
    }

    /// Build provenance for a llama fallback response. Carries the llama lane's
    /// actual fingerprints, never the refused owned fingerprints.
    pub fn llama(
        llama_decode_fingerprint: Fingerprint,
        llama_processing_fingerprint: Fingerprint,
    ) -> Self {
        Self {
            engine: "llama.cpp".to_string(),
            lane: "llama".to_string(),
            worker: "supervised".to_string(),
            risk_class: "abort_capable".to_string(),
            decode_fingerprint: llama_decode_fingerprint,
            processing_fingerprint: llama_processing_fingerprint,
            tokenizer_sanitized_digest: String::new(),
            prompt_template_revision: String::new(),
            special_token_policy_revision: String::new(),
            stop_token_policy_revision: String::new(),
            detokenizer_revision: String::new(),
            arithmetic_identity_revision: String::new(),
            metallib_revision: String::new(),
            worker_generation: 0,
            last_completed_quantum_sequence: 0,
            fallback_reason: None,
            lane_finish_reason: None,
            crash_retry_count: 0,
            failure_classifications: Vec::new(),
            constraint_runtime_identity: None,
            constraint_fingerprint: None,
            grammar_compiler_revision: None,
            chain_k: None,
            underlying_owned_decode_refusal_id: None,
        }
    }

    /// Record the fallback reason (additive provenance; the owned refusal ID is
    /// also surfaced for telemetry/lookup).
    pub fn with_fallback_reason(mut self, reason: &str) -> Self {
        self.fallback_reason = Some(reason.to_string());
        self
    }

    /// Record the underlying owned-decode refusal ID for a mapped
    /// `grammar_disabled` error.
    pub fn with_underlying_refusal(mut self, refusal_id: &str) -> Self {
        self.underlying_owned_decode_refusal_id = Some(refusal_id.to_string());
        self
    }

    /// Record crash-retry metadata (ordered classifications and count).
    pub fn with_crash_retry(mut self, count: u32, classifications: Vec<String>) -> Self {
        self.crash_retry_count = count;
        self.failure_classifications = classifications;
        self
    }

    /// Record constraint identities for a constrained response.
    pub fn with_constraint(
        mut self,
        runtime_identity: String,
        fingerprint: Fingerprint,
        grammar_compiler_revision: String,
    ) -> Self {
        self.constraint_runtime_identity = Some(runtime_identity);
        self.constraint_fingerprint = Some(fingerprint);
        self.grammar_compiler_revision = Some(grammar_compiler_revision);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_decode_routing::family::{Family, FamilyRegistry};

    fn fp(s: &str) -> Fingerprint {
        Fingerprint(s.to_string())
    }

    #[test]
    fn owned_provenance_carries_required_identity_fields() {
        let registry = FamilyRegistry::production();
        let registration = registry.get(Family::Qwen3_0_6b).unwrap();
        let provenance = LaneProvenance::owned(
            registration,
            OwnedProvenanceInputs {
                decode_fingerprint: fp("decode-1"),
                processing_fingerprint: fp("proc-1"),
                arithmetic_identity_revision: "arith-v1".to_string(),
                metallib_revision: "metallib-v1".to_string(),
                worker_generation: 7,
                last_completed_quantum_sequence: 3,
                chain_k: 1,
            },
        );

        assert_eq!(provenance.engine, DECODE_WORKER_ENGINE);
        assert_eq!(provenance.lane, "decode");
        assert_eq!(provenance.worker, "supervised");
        assert_eq!(provenance.risk_class, "abort_capable");
        assert_eq!(provenance.decode_fingerprint, fp("decode-1"));
        assert_eq!(provenance.processing_fingerprint, fp("proc-1"));
        assert_eq!(
            provenance.tokenizer_sanitized_digest,
            registration.tokenizer_sanitized_digest
        );
        assert_eq!(provenance.worker_generation, 7);
        assert_eq!(provenance.last_completed_quantum_sequence, 3);
        assert_eq!(provenance.chain_k, Some(1));
        // Additive fields absent by default.
        assert!(provenance.fallback_reason.is_none());
        assert_eq!(provenance.crash_retry_count, 0);
        assert!(provenance.constraint_fingerprint.is_none());
    }

    #[test]
    fn llama_fallback_provenance_uses_llama_fingerprints_not_owned() {
        let provenance = LaneProvenance::llama(fp("llama-decode"), fp("llama-proc"))
            .with_fallback_reason("owned_decode_quarantined");
        assert_eq!(provenance.engine, "llama.cpp");
        assert_eq!(provenance.decode_fingerprint, fp("llama-decode"));
        assert_eq!(provenance.processing_fingerprint, fp("llama-proc"));
        assert_eq!(
            provenance.fallback_reason.as_deref(),
            Some("owned_decode_quarantined")
        );
    }

    #[test]
    fn mapped_refusal_records_underlying_id() {
        let provenance = LaneProvenance::llama(fp("d"), fp("p"))
            .with_underlying_refusal("owned_decode_not_certified");
        assert_eq!(
            provenance.underlying_owned_decode_refusal_id.as_deref(),
            Some("owned_decode_not_certified")
        );
    }

    #[test]
    fn finish_reasons_serialize_to_external_literals() {
        assert_eq!(FinishReason::StopToken.as_str(), "stop_token");
        assert_eq!(FinishReason::MaxTokens.as_str(), "max_tokens");
        assert_eq!(FinishReason::GrammarComplete.as_str(), "grammar_complete");
        assert_eq!(FinishReason::Cancelled.as_str(), "cancelled");
    }

    #[test]
    fn additive_fields_omit_when_unset() {
        let provenance = LaneProvenance::llama(fp("d"), fp("p"));
        let json = serde_json::to_value(&provenance).unwrap();
        assert!(json.get("fallback_reason").is_none());
        assert!(json.get("crash_retry_count").is_none());
        assert!(json.get("constraint_fingerprint").is_none());
        assert!(json.get("underlying_owned_decode_refusal_id").is_none());
    }
}
