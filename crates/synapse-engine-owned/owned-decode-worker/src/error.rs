//! Stable owned-decode error contract.
//!
//! Every error ID here is a stable wire literal: callers, fixtures, and
//! telemetry match on the exact string returned by [`DecodeError::as_str`].
//! The IDs are grouped by the contract section that owns them:
//!
//! - `error_contract`: the eleven stable owned-decode refusals and the grammar
//!   errors.
//! - `wire_error_bindings`: the literal deadline and cancellation IDs (see
//!   [`crate::wire_error_bindings`]). Symbolic placeholder names such as
//!   `existing_deadline_error` never appear as emitted IDs.
//!
//! The three worker-start mismatch IDs are dedicated and non-overlapping
//! (resolution r2 #7): protocol/frame-structure faults map to
//! `owned_decode_protocol_mismatch`, loaded-model / decode-fingerprint /
//! runtime-manifest identity faults map to `owned_decode_runtime_config_mismatch`,
//! and constraint-identity faults map to `owned_decode_constraint_version_mismatch`.

use serde::{Deserialize, Serialize};

/// A stable owned-decode error with a fixed wire literal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecodeError {
    // ---- error_contract: stable owned-decode refusals ----
    /// No certification row authorizes this lane on this machine profile.
    NotCertified,
    /// Certification ran and failed (e.g. structural-band fork on Q8).
    CertificationFailed,
    /// The quarantine key exhausted its crash budget and is quarantined.
    Quarantined,
    /// A Q8 artifact failed digest verification or was corrupted.
    ArtifactPoisoned,
    /// Capacity/reservation/startup refusal: nothing was dispatched.
    Unavailable,
    /// Platform or context bucket is not supported.
    Unsupported,
    /// Protocol-ID, frame-structure, framing, sequence, generation, session,
    /// continuation, or malformed-frame violation.
    ProtocolMismatch,
    /// Worker-start loaded-model, decode-fingerprint, or runtime-manifest
    /// identity mismatch.
    RuntimeConfigMismatch,
    /// Constraint representation, compiler, vocabulary, limits, runtime,
    /// schema, initial-state, automaton, or request-fingerprint mismatch.
    ConstraintVersionMismatch,
    /// A sampling mode other than greedy-top-1 was requested.
    SamplingUnsupported,
    /// `prompt_token_count + max_tokens > max_context_tokens`.
    ContextCapacityExceeded,

    // ---- grammar errors ----
    /// Grammar is disabled, or a constrained request hit a pre-dispatch owned
    /// refusal that maps to `grammar_disabled`.
    GrammarDisabled,
    /// Malformed JSON or malformed schema structure.
    GrammarParseFailed,
    /// Schema outside the accepted subset or its checked-in limits.
    GrammarFeatureUnsupported,
    /// No content token and no stop control candidate is selectable.
    GrammarUnsatisfiable,
    /// A stop token won while the automaton was incomplete.
    GrammarStopBeforeCompletion,
    /// Generation reached `max_tokens` without completing a value.
    GrammarMaxTokensExhausted,

    // ---- wire_error_bindings: literal deadline and cancellation ----
    /// The bound deadline error (literal wire ID `deadline_exceeded`).
    DeadlineExceeded,
    /// The bound cancellation error (literal wire ID `cancelled`).
    Cancelled,
}

impl DecodeError {
    /// The stable wire literal for this error.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotCertified => "owned_decode_not_certified",
            Self::CertificationFailed => "owned_decode_certification_failed",
            Self::Quarantined => "owned_decode_quarantined",
            Self::ArtifactPoisoned => "artifact_poisoned",
            Self::Unavailable => "owned_decode_unavailable",
            Self::Unsupported => "owned_decode_unsupported",
            Self::ProtocolMismatch => "owned_decode_protocol_mismatch",
            Self::RuntimeConfigMismatch => "owned_decode_runtime_config_mismatch",
            Self::ConstraintVersionMismatch => "owned_decode_constraint_version_mismatch",
            Self::SamplingUnsupported => "owned_decode_sampling_unsupported",
            Self::ContextCapacityExceeded => "context_capacity_exceeded",
            Self::GrammarDisabled => "grammar_disabled",
            Self::GrammarParseFailed => "grammar_parse_failed",
            Self::GrammarFeatureUnsupported => "grammar_feature_unsupported",
            Self::GrammarUnsatisfiable => "grammar_unsatisfiable",
            Self::GrammarStopBeforeCompletion => "grammar_stop_before_completion",
            Self::GrammarMaxTokensExhausted => "grammar_max_tokens_exhausted",
            Self::DeadlineExceeded => crate::wire_error_bindings::DEADLINE_ERROR_ID,
            Self::Cancelled => crate::wire_error_bindings::CANCELLATION_ERROR_ID,
        }
    }

    /// Whether this error is eligible for llama fallback during pre-dispatch
    /// lane selection. Exactly the six IDs listed by `lane_selection_and_fallback`
    /// are eligible; every other error returns directly.
    #[must_use]
    pub const fn is_fallback_eligible(self) -> bool {
        matches!(
            self,
            Self::NotCertified
                | Self::CertificationFailed
                | Self::Quarantined
                | Self::ArtifactPoisoned
                | Self::Unavailable
                | Self::Unsupported
        )
    }

    /// Whether a constrained request maps this pre-dispatch owned refusal to
    /// `grammar_disabled`. The same six fallback-eligible IDs are the mappable
    /// set; protocol, runtime, constraint, sampling, context, fingerprint, and
    /// grammar errors return directly.
    #[must_use]
    pub const fn maps_to_grammar_disabled_for_constrained(self) -> bool {
        self.is_fallback_eligible()
    }

    /// Parse a stable wire literal back into its error. Returns `None` for an
    /// unknown ID. Used to interpret a worker's typed-error frame.
    #[must_use]
    pub fn from_id(id: &str) -> Option<Self> {
        let error = match id {
            "owned_decode_not_certified" => Self::NotCertified,
            "owned_decode_certification_failed" => Self::CertificationFailed,
            "owned_decode_quarantined" => Self::Quarantined,
            "artifact_poisoned" => Self::ArtifactPoisoned,
            "owned_decode_unavailable" => Self::Unavailable,
            "owned_decode_unsupported" => Self::Unsupported,
            "owned_decode_protocol_mismatch" => Self::ProtocolMismatch,
            "owned_decode_runtime_config_mismatch" => Self::RuntimeConfigMismatch,
            "owned_decode_constraint_version_mismatch" => Self::ConstraintVersionMismatch,
            "owned_decode_sampling_unsupported" => Self::SamplingUnsupported,
            "context_capacity_exceeded" => Self::ContextCapacityExceeded,
            "grammar_disabled" => Self::GrammarDisabled,
            "grammar_parse_failed" => Self::GrammarParseFailed,
            "grammar_feature_unsupported" => Self::GrammarFeatureUnsupported,
            "grammar_unsatisfiable" => Self::GrammarUnsatisfiable,
            "grammar_stop_before_completion" => Self::GrammarStopBeforeCompletion,
            "grammar_max_tokens_exhausted" => Self::GrammarMaxTokensExhausted,
            _ => return None,
        };
        Some(error)
    }

    /// Whether this error is a grammar error. Grammar errors are clean typed
    /// errors: they do not crash or quarantine the worker and consume no budget.
    #[must_use]
    pub const fn is_grammar_error(self) -> bool {
        matches!(
            self,
            Self::GrammarDisabled
                | Self::GrammarParseFailed
                | Self::GrammarFeatureUnsupported
                | Self::GrammarUnsatisfiable
                | Self::GrammarStopBeforeCompletion
                | Self::GrammarMaxTokensExhausted
        )
    }
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for DecodeError {}

/// A chargeable failure classification for crash-budget accounting.
///
/// Every crash, protocol-fatal response, startup failure, timeout, or failed
/// cancellation charges exactly one unit to the affected quarantine key
/// (resolution r2 #8). Acknowledged cancellation and acknowledged deadline
/// cleanup before timeout charge nothing. Coincident timeout/crash/startup/
/// protocol-fatal failure is charged once, under the single classification the
/// supervisor records for that failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureClassification {
    /// The worker process died unexpectedly.
    Crash,
    /// The worker returned a protocol-fatal frame.
    ProtocolFatal,
    /// Replacement startup or model load failed.
    StartupFailure,
    /// The worker exceeded its deadline without a terminal frame.
    Timeout,
    /// The worker failed to acknowledge cancellation within the cancel timeout,
    /// escalating to a kill. An unresponsive worker is a worker fault.
    FailedCancellation,
}

impl FailureClassification {
    /// The stable wire literal for this classification, recorded in ordered
    /// `failure_classifications` provenance and telemetry.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Crash => "crash",
            Self::ProtocolFatal => "protocol_fatal",
            Self::StartupFailure => "startup_failure",
            Self::Timeout => "timeout",
            Self::FailedCancellation => "failed_cancellation",
        }
    }
}

impl std::fmt::Display for FailureClassification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_ids_are_the_documented_wire_literals() {
        assert_eq!(
            DecodeError::NotCertified.as_str(),
            "owned_decode_not_certified"
        );
        assert_eq!(
            DecodeError::CertificationFailed.as_str(),
            "owned_decode_certification_failed"
        );
        assert_eq!(
            DecodeError::Quarantined.as_str(),
            "owned_decode_quarantined"
        );
        assert_eq!(DecodeError::ArtifactPoisoned.as_str(), "artifact_poisoned");
        assert_eq!(
            DecodeError::Unavailable.as_str(),
            "owned_decode_unavailable"
        );
        assert_eq!(
            DecodeError::Unsupported.as_str(),
            "owned_decode_unsupported"
        );
        assert_eq!(
            DecodeError::ProtocolMismatch.as_str(),
            "owned_decode_protocol_mismatch"
        );
        assert_eq!(
            DecodeError::RuntimeConfigMismatch.as_str(),
            "owned_decode_runtime_config_mismatch"
        );
        assert_eq!(
            DecodeError::ConstraintVersionMismatch.as_str(),
            "owned_decode_constraint_version_mismatch"
        );
        assert_eq!(
            DecodeError::SamplingUnsupported.as_str(),
            "owned_decode_sampling_unsupported"
        );
        assert_eq!(
            DecodeError::ContextCapacityExceeded.as_str(),
            "context_capacity_exceeded"
        );
    }

    #[test]
    fn deadline_and_cancel_use_literal_wire_binding_ids() {
        // The symbolic placeholders must never be the emitted IDs.
        assert_eq!(DecodeError::DeadlineExceeded.as_str(), "deadline_exceeded");
        assert_eq!(DecodeError::Cancelled.as_str(), "cancelled");
        assert_ne!(
            DecodeError::DeadlineExceeded.as_str(),
            "existing_deadline_error"
        );
        assert_ne!(
            DecodeError::Cancelled.as_str(),
            "existing_cancellation_error"
        );
    }

    #[test]
    fn fallback_eligibility_is_exactly_the_six_pre_dispatch_refusals() {
        let eligible = [
            DecodeError::NotCertified,
            DecodeError::CertificationFailed,
            DecodeError::Quarantined,
            DecodeError::ArtifactPoisoned,
            DecodeError::Unavailable,
            DecodeError::Unsupported,
        ];
        for error in eligible {
            assert!(error.is_fallback_eligible(), "{error:?} should be eligible");
        }
        let ineligible = [
            DecodeError::ProtocolMismatch,
            DecodeError::RuntimeConfigMismatch,
            DecodeError::ConstraintVersionMismatch,
            DecodeError::SamplingUnsupported,
            DecodeError::ContextCapacityExceeded,
            DecodeError::GrammarParseFailed,
            DecodeError::DeadlineExceeded,
            DecodeError::Cancelled,
        ];
        for error in ineligible {
            assert!(
                !error.is_fallback_eligible(),
                "{error:?} must return directly"
            );
        }
    }
}
