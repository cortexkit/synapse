//! Stable owned-decode and grammar error identities for the production
//! owned-metal-decode lane.
//!
//! The specification's `error_contract` section fixes a closed set of stable
//! wire error IDs. Each variant here serializes to exactly one of those literal
//! strings through [`OwnedDecodeError::as_str`]; the strings are the wire
//! contract, so they never change without a new binding-manifest revision.
//!
//! Two classifications drive routing behavior and are encoded here so every
//! caller agrees:
//!
//! - [`OwnedDecodeError::is_predispatch_fallback_eligible`] marks the exactly
//!   six pre-dispatch owned-lane refusals that may select a configured llama
//!   lane for substitutable unconstrained requests (`lane_selection_and_fallback`).
//! - [`OwnedDecodeError::is_execution_phase`] marks failures that can only
//!   surface after a worker dispatch; these always return directly and never
//!   re-enter lane selection.

use serde::{Deserialize, Serialize};

/// The closed set of stable owned-decode and grammar error IDs.
///
/// Variants are grouped by the contract section that owns them: the first
/// eleven are the stable owned-decode errors; the remainder are the grammar
/// errors. `as_str` returns the canonical wire literal for each.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnedDecodeError {
    // -- stable owned-decode errors (error_contract) --
    /// No certification row authorizes this fingerprint on this machine
    /// profile, or a Q8 artifact lacks a registered expected digest.
    NotCertified,
    /// A certification attempt ran and failed (e.g. structural-band fork
    /// signature violated).
    CertificationFailed,
    /// The quarantine key exhausted its crash budget and is quarantined.
    Quarantined,
    /// A Q8 artifact failed digest verification or was corrupted after
    /// publication.
    ArtifactPoisoned,
    /// Reservation failure or a terminal non-exhausting execution charge; the
    /// owned lane cannot serve this request right now.
    Unavailable,
    /// The requested family, format, or context bucket is not in the shippable
    /// production set.
    Unsupported,
    /// Framing, sequence, generation, session, continuation, or malformed-frame
    /// violation on the worker protocol.
    ProtocolMismatch,
    /// Loaded-model, decode-fingerprint, or runtime-manifest mismatch at worker
    /// start.
    RuntimeConfigMismatch,
    /// Constraint representation, compiler, vocabulary, limits, runtime,
    /// schema, initial-state, automaton, or request-fingerprint mismatch.
    ConstraintVersionMismatch,
    /// A sampling mode other than greedy-top-1 was requested.
    SamplingUnsupported,
    /// `prompt_token_count + max_tokens` exceeds the selected context bucket.
    ContextCapacityExceeded,

    // -- grammar errors (error_contract) --
    /// Grammar is disabled, or a constrained request hit a pre-dispatch owned
    /// refusal that the error contract maps to `grammar_disabled`.
    GrammarDisabled,
    /// Malformed JSON or malformed schema structure.
    GrammarParseFailed,
    /// A schema outside the accepted subset or its checked-in limits.
    GrammarFeatureUnsupported,
    /// No content token and no stop control candidate is selectable.
    GrammarUnsatisfiable,
    /// A stop token won while the automaton was incomplete.
    GrammarStopBeforeCompletion,
    /// Generation reached `max_tokens` without completing a value.
    GrammarMaxTokensExhausted,
}

impl OwnedDecodeError {
    /// The canonical stable wire literal for this error.
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
        }
    }

    /// Whether this is one of the exactly six pre-dispatch owned-lane refusals
    /// eligible to fall back to a configured llama lane for substitutable
    /// unconstrained requests. `lane_selection_and_fallback` fixes this list
    /// exhaustively; every other error returns directly.
    pub const fn is_predispatch_fallback_eligible(self) -> bool {
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

    /// Whether this error can only arise after a worker dispatch. Execution-phase
    /// failures never re-enter lane selection and never convert into a llama
    /// dispatch, even when the same ID also appears in the pre-dispatch
    /// eligibility list.
    pub const fn is_execution_phase(self) -> bool {
        matches!(
            self,
            Self::ProtocolMismatch
                | Self::RuntimeConfigMismatch
                | Self::ConstraintVersionMismatch
                | Self::SamplingUnsupported
        )
    }

    /// Whether this is a grammar error (as opposed to an owned-decode error).
    pub const fn is_grammar(self) -> bool {
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

    /// For a constrained request, the six pre-dispatch owned refusals are
    /// caller-mapped to `grammar_disabled` by the error contract, with the
    /// original ID recorded as `underlying_owned_decode_refusal_id`. Returns
    /// `Some(GrammarDisabled)` for those six and `None` for every other error
    /// (which returns directly, unmapped).
    pub const fn constrained_predispatch_mapping(self) -> Option<OwnedDecodeError> {
        if self.is_predispatch_fallback_eligible() {
            Some(Self::GrammarDisabled)
        } else {
            None
        }
    }
}

impl std::fmt::Display for OwnedDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::error::Error for OwnedDecodeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_literals_match_error_contract() {
        // The literal strings ARE the wire contract; pin them explicitly so a
        // rename fails loudly rather than silently changing emitted responses.
        assert_eq!(
            OwnedDecodeError::NotCertified.as_str(),
            "owned_decode_not_certified"
        );
        assert_eq!(
            OwnedDecodeError::CertificationFailed.as_str(),
            "owned_decode_certification_failed"
        );
        assert_eq!(
            OwnedDecodeError::Quarantined.as_str(),
            "owned_decode_quarantined"
        );
        assert_eq!(
            OwnedDecodeError::ArtifactPoisoned.as_str(),
            "artifact_poisoned"
        );
        assert_eq!(
            OwnedDecodeError::Unavailable.as_str(),
            "owned_decode_unavailable"
        );
        assert_eq!(
            OwnedDecodeError::Unsupported.as_str(),
            "owned_decode_unsupported"
        );
        assert_eq!(
            OwnedDecodeError::ProtocolMismatch.as_str(),
            "owned_decode_protocol_mismatch"
        );
        assert_eq!(
            OwnedDecodeError::RuntimeConfigMismatch.as_str(),
            "owned_decode_runtime_config_mismatch"
        );
        assert_eq!(
            OwnedDecodeError::ConstraintVersionMismatch.as_str(),
            "owned_decode_constraint_version_mismatch"
        );
        assert_eq!(
            OwnedDecodeError::SamplingUnsupported.as_str(),
            "owned_decode_sampling_unsupported"
        );
        assert_eq!(
            OwnedDecodeError::ContextCapacityExceeded.as_str(),
            "context_capacity_exceeded"
        );
        assert_eq!(
            OwnedDecodeError::GrammarDisabled.as_str(),
            "grammar_disabled"
        );
    }

    #[test]
    fn fallback_eligible_set_is_exactly_the_six_predispatch_refusals() {
        let eligible = [
            OwnedDecodeError::NotCertified,
            OwnedDecodeError::CertificationFailed,
            OwnedDecodeError::Quarantined,
            OwnedDecodeError::ArtifactPoisoned,
            OwnedDecodeError::Unavailable,
            OwnedDecodeError::Unsupported,
        ];
        for error in eligible {
            assert!(error.is_predispatch_fallback_eligible(), "{error:?}");
        }
        // Everything else is NOT fallback-eligible.
        for error in [
            OwnedDecodeError::ProtocolMismatch,
            OwnedDecodeError::RuntimeConfigMismatch,
            OwnedDecodeError::ConstraintVersionMismatch,
            OwnedDecodeError::SamplingUnsupported,
            OwnedDecodeError::ContextCapacityExceeded,
            OwnedDecodeError::GrammarDisabled,
            OwnedDecodeError::GrammarParseFailed,
            OwnedDecodeError::GrammarFeatureUnsupported,
            OwnedDecodeError::GrammarUnsatisfiable,
            OwnedDecodeError::GrammarStopBeforeCompletion,
            OwnedDecodeError::GrammarMaxTokensExhausted,
        ] {
            assert!(!error.is_predispatch_fallback_eligible(), "{error:?}");
        }
    }

    #[test]
    fn execution_phase_errors_are_distinct_from_predispatch_refusals() {
        for error in [
            OwnedDecodeError::ProtocolMismatch,
            OwnedDecodeError::RuntimeConfigMismatch,
            OwnedDecodeError::ConstraintVersionMismatch,
            OwnedDecodeError::SamplingUnsupported,
        ] {
            assert!(error.is_execution_phase(), "{error:?}");
            assert!(!error.is_predispatch_fallback_eligible(), "{error:?}");
        }
    }

    #[test]
    fn constrained_mapping_covers_exactly_the_six_refusals() {
        for error in [
            OwnedDecodeError::NotCertified,
            OwnedDecodeError::CertificationFailed,
            OwnedDecodeError::Quarantined,
            OwnedDecodeError::ArtifactPoisoned,
            OwnedDecodeError::Unavailable,
            OwnedDecodeError::Unsupported,
        ] {
            assert_eq!(
                error.constrained_predispatch_mapping(),
                Some(OwnedDecodeError::GrammarDisabled),
                "{error:?}"
            );
        }
        // Non-eligible errors return directly, unmapped.
        assert_eq!(
            OwnedDecodeError::ProtocolMismatch.constrained_predispatch_mapping(),
            None
        );
        assert_eq!(
            OwnedDecodeError::ContextCapacityExceeded.constrained_predispatch_mapping(),
            None
        );
        assert_eq!(
            OwnedDecodeError::GrammarParseFailed.constrained_predispatch_mapping(),
            None
        );
    }
}
