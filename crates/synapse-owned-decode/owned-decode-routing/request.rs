//! Request-domain model and validation for `microllm.oneshot` on the owned lane.
//!
//! Validation is split into two classes the specification treats differently:
//!
//! - Invalid request boundaries (empty prompt, zero `max_tokens`, unsupported
//!   sampling) are caller errors that return directly and are never
//!   fallback-eligible.
//! - Context boundaries (`prompt_token_count + max_tokens` exceeding the
//!   selected bucket) return `context_capacity_exceeded` before dispatch and
//!   consume no crash budget.
//!
//! A constrained request carries a `grammar` schema; constrained requests are
//! owned-only and never fall back to llama.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use synapse_core::Fingerprint;

use crate::owned_decode_routing::error::OwnedDecodeError;
use crate::owned_decode_routing::family::Family;
use crate::owned_decode_routing::identity::WeightQuant;

/// Sampling mode for a oneshot request. Version 1 accepts only `greedy_top1`;
/// every other mode returns `owned_decode_sampling_unsupported` before the
/// first token commit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum SamplingMode {
    GreedyTop1,
    TopK { k: u32 },
    TopP { p: f32 },
    Temperature { temperature: f32 },
}

impl SamplingMode {
    pub const fn is_greedy_top1(&self) -> bool {
        matches!(self, Self::GreedyTop1)
    }
}

/// A `microllm.oneshot` generation request as seen by the routing layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OneshotRequest {
    /// Requested model family.
    pub family: Family,
    /// Requested weight format.
    pub weight_quant: WeightQuant,
    /// Canonical prompt length in tokens (after module tokenization/templating).
    pub prompt_token_count: u32,
    /// Maximum content tokens to generate.
    pub max_tokens: u32,
    #[serde(default = "default_sampling")]
    pub sampling: SamplingMode,
    /// Optional JSON schema (project subset `synapse-json-schema-v1`). Presence
    /// makes the request constrained and owned-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<Value>,
    /// If present, execution may use only this exact fingerprint unless
    /// `allow_equivalent` authorizes an alias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_fingerprint: Option<Fingerprint>,
    /// Whether an equivalent alias of `required_fingerprint` is acceptable.
    #[serde(default)]
    pub allow_equivalent: bool,
    /// Advisory fingerprint; fallback is allowed only when the request contract
    /// permits substitution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_fingerprint: Option<Fingerprint>,
    /// If present, requires the complete processing identity exactly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_processing_fingerprint: Option<Fingerprint>,
    /// Explicit owned-only model selection. When true, routing never substitutes
    /// a different model (no llama fallback) even if the owned lane refuses.
    #[serde(default)]
    pub owned_only: bool,
}

fn default_sampling() -> SamplingMode {
    SamplingMode::GreedyTop1
}

impl OneshotRequest {
    /// Whether this request carries a grammar and is therefore constrained and
    /// owned-only.
    pub fn is_constrained(&self) -> bool {
        self.grammar.is_some()
    }

    /// Whether the request pins an exact fingerprint (no substitution).
    pub fn requires_exact_fingerprint(&self) -> bool {
        self.required_fingerprint.is_some() && !self.allow_equivalent
    }

    /// Whether the request is substitutable: unconstrained, no exact fingerprint
    /// pin, and not an explicit owned-only selection. Only substitutable
    /// unconstrained requests may fall back to llama.
    pub fn is_substitutable(&self) -> bool {
        !self.is_constrained() && !self.requires_exact_fingerprint() && !self.owned_only
    }
}

/// Error returned by request-domain validation. Invalid boundaries and context
/// boundaries are distinct so callers can map them to the correct wire ID.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestValidationError {
    /// A caller-supplied boundary is invalid (empty prompt, zero max_tokens).
    InvalidRequest(String),
    /// A non-greedy sampling mode was requested.
    SamplingUnsupported,
    /// `prompt_token_count + max_tokens` exceeds the selected context bucket.
    ContextCapacityExceeded {
        prompt_token_count: u32,
        max_tokens: u32,
        max_context_tokens: u32,
    },
}

impl RequestValidationError {
    /// The stable wire ID for this validation failure.
    pub fn wire_id(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => "invalid_request",
            Self::SamplingUnsupported => OwnedDecodeError::SamplingUnsupported.as_str(),
            Self::ContextCapacityExceeded { .. } => {
                OwnedDecodeError::ContextCapacityExceeded.as_str()
            }
        }
    }
}

impl OneshotRequest {
    /// Validate request-domain boundaries against the selected context bucket.
    ///
    /// Order matters: invalid caller boundaries and unsupported sampling are
    /// reported before the context capacity check, matching the contract's
    /// pre-dispatch refusal ordering.
    pub fn validate(&self, max_context_tokens: u32) -> Result<(), RequestValidationError> {
        if self.prompt_token_count == 0 {
            return Err(RequestValidationError::InvalidRequest(
                "prompt_token_count must be greater than zero".to_string(),
            ));
        }
        if self.max_tokens == 0 {
            return Err(RequestValidationError::InvalidRequest(
                "max_tokens must be greater than zero".to_string(),
            ));
        }
        if !self.sampling.is_greedy_top1() {
            return Err(RequestValidationError::SamplingUnsupported);
        }
        let total = self.prompt_token_count.saturating_add(self.max_tokens);
        if total > max_context_tokens {
            return Err(RequestValidationError::ContextCapacityExceeded {
                prompt_token_count: self.prompt_token_count,
                max_tokens: self.max_tokens,
                max_context_tokens,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_decode_routing::identity::WeightQuant;

    fn base_request() -> OneshotRequest {
        OneshotRequest {
            family: Family::Qwen3_0_6b,
            weight_quant: WeightQuant::F16,
            prompt_token_count: 100,
            max_tokens: 64,
            sampling: SamplingMode::GreedyTop1,
            grammar: None,
            required_fingerprint: None,
            allow_equivalent: false,
            target_fingerprint: None,
            required_processing_fingerprint: None,
            owned_only: false,
        }
    }

    #[test]
    fn valid_request_passes_within_context() {
        assert_eq!(base_request().validate(512), Ok(()));
        // Exactly at the boundary is allowed.
        let mut edge = base_request();
        edge.prompt_token_count = 448;
        edge.max_tokens = 64;
        assert_eq!(edge.validate(512), Ok(()));
    }

    #[test]
    fn empty_prompt_is_invalid() {
        let mut request = base_request();
        request.prompt_token_count = 0;
        assert_eq!(
            request.validate(512),
            Err(RequestValidationError::InvalidRequest(
                "prompt_token_count must be greater than zero".to_string()
            ))
        );
        assert_eq!(
            request.validate(512).unwrap_err().wire_id(),
            "invalid_request"
        );
    }

    #[test]
    fn zero_max_tokens_is_invalid() {
        let mut request = base_request();
        request.max_tokens = 0;
        assert!(matches!(
            request.validate(512),
            Err(RequestValidationError::InvalidRequest(_))
        ));
    }

    #[test]
    fn context_overflow_is_context_capacity_exceeded() {
        let mut request = base_request();
        request.prompt_token_count = 500;
        request.max_tokens = 64;
        assert_eq!(
            request.validate(512),
            Err(RequestValidationError::ContextCapacityExceeded {
                prompt_token_count: 500,
                max_tokens: 64,
                max_context_tokens: 512,
            })
        );
        assert_eq!(
            request.validate(512).unwrap_err().wire_id(),
            "context_capacity_exceeded"
        );
    }

    #[test]
    fn non_greedy_sampling_is_unsupported() {
        let mut request = base_request();
        request.sampling = SamplingMode::TopK { k: 5 };
        assert_eq!(
            request.validate(512),
            Err(RequestValidationError::SamplingUnsupported)
        );
        assert_eq!(
            request.validate(512).unwrap_err().wire_id(),
            "owned_decode_sampling_unsupported"
        );
    }

    #[test]
    fn invalid_boundary_is_checked_before_context() {
        // Both an invalid boundary and an overflow apply; the invalid boundary
        // wins because caller errors precede capacity checks.
        let mut request = base_request();
        request.max_tokens = 0;
        request.prompt_token_count = 10_000;
        assert!(matches!(
            request.validate(512),
            Err(RequestValidationError::InvalidRequest(_))
        ));
    }

    #[test]
    fn constrained_and_substitutable_flags_follow_grammar_and_pins() {
        let plain = base_request();
        assert!(!plain.is_constrained());
        assert!(plain.is_substitutable());

        let mut constrained = base_request();
        constrained.grammar = Some(serde_json::json!({"type": "object"}));
        assert!(constrained.is_constrained());
        assert!(!constrained.is_substitutable());

        let mut pinned = base_request();
        pinned.required_fingerprint = Some(Fingerprint("fp".to_string()));
        assert!(pinned.requires_exact_fingerprint());
        assert!(!pinned.is_substitutable());

        // allow_equivalent relaxes the pin back to substitutable.
        pinned.allow_equivalent = true;
        assert!(!pinned.requires_exact_fingerprint());
        assert!(pinned.is_substitutable());
    }

    #[test]
    fn oneshot_request_rejects_unknown_field() {
        // fail-closed posture: a typo or forward-incompatible field addition
        // is rejected at parse time rather than silently dropped.
        let json = serde_json::json!({
            "family": "qwen3-0.6b",
            "weight_quant": "f16",
            "prompt_token_count": 100,
            "max_tokens": 64,
            "sampling": {"mode": "greedy_top1"},
            "unknown_field": "should be rejected",
        });
        assert!(serde_json::from_value::<OneshotRequest>(json).is_err());
    }
}
