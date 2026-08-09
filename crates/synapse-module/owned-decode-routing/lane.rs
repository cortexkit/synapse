//! Lane selection, fallback, and the D-009 cutover predicate.
//!
//! `lane_selection_and_fallback` is the sole fallback authority. Selection runs
//! before admission and evaluates fingerprint constraints against the lane
//! actually selected:
//!
//! - A `required_fingerprint` may use only that exact identity (or an authorized
//!   alias via `allow_equivalent`); mismatch returns the existing
//!   fingerprint-mismatch error and never silently falls back.
//! - Constrained requests are owned-only: a pre-dispatch owned refusal from the
//!   six fallback-eligible IDs is caller-mapped to `grammar_disabled` (original
//!   recorded as `underlying_owned_decode_refusal_id`); no llama, no
//!   unconstrained retry.
//! - Substitutable unconstrained requests fall back to a configured llama lane
//!   only for one of the six pre-dispatch owned refusals; otherwise the original
//!   refusal is returned without claiming fallback.
//!
//! Fallback eligibility is evaluated only during pre-dispatch lane selection.
//! Once dispatched, execution-phase failures return directly and never re-enter
//! selection; that behavior is enforced by the orchestrator, not here.

use std::{collections::BTreeSet, fmt::Debug};

use serde::{Deserialize, Serialize};
use synapse_core::Fingerprint;

use crate::owned_decode_routing::error::OwnedDecodeError;
use crate::owned_decode_routing::request::OneshotRequest;

/// The two routable lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaneKind {
    OwnedDecode,
    Llama,
}

/// Why a substitutable request landed on the llama lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallbackReason {
    /// A pre-dispatch owned-lane refusal from the fallback-eligible six.
    OwnedRefusal(OwnedDecodeError),
    /// The D-009 cutover is not enabled for this profile, so substitutable
    /// unconstrained requests default to the configured llama lane.
    CutoverDisabled,
}

/// A refusal that routing can emit. Either an owned-decode/grammar error, the
/// existing request-contract fingerprint-mismatch error (`substitution_rejected`),
/// or an invalid-request caller error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingRefusal {
    Owned(OwnedDecodeError),
    /// The existing fingerprint-mismatch error; not fallback-eligible.
    FingerprintMismatch,
    /// An invalid caller-supplied request boundary; not fallback-eligible.
    InvalidRequest,
}

impl RoutingRefusal {
    /// The stable wire ID for this refusal.
    pub fn wire_id(&self) -> &'static str {
        match self {
            Self::Owned(error) => error.as_str(),
            Self::FingerprintMismatch => "substitution_rejected",
            Self::InvalidRequest => "invalid_request",
        }
    }
}

/// Result of evaluating the owned lane before dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OwnedEvaluation {
    /// Cutover enabled and certified; the owned lane is selectable.
    Selectable,
    /// The owned lane refused with a specific pre-dispatch (or other) error.
    Refused(OwnedDecodeError),
    /// The D-009 cutover is not enabled for this profile; owned is not the
    /// preferred lane.
    NotPreferred,
}

/// A configured llama fallback lane with its own identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LlamaLane {
    pub decode_fingerprint: Fingerprint,
    pub processing_fingerprint: Fingerprint,
}

/// Inputs to lane selection.
pub struct LaneSelectionContext<'a> {
    pub request: &'a OneshotRequest,
    /// The owned lane's decode fingerprint (the lane actually selected).
    pub owned_decode_fingerprint: Fingerprint,
    /// The owned lane's processing fingerprint.
    pub owned_processing_fingerprint: Fingerprint,
    /// Owned-lane evaluation result.
    pub owned: OwnedEvaluation,
    /// Configured llama lane, if any.
    pub llama: Option<LlamaLane>,
    /// Fingerprints authorized as equivalent aliases of `required_fingerprint`.
    pub equivalent_fingerprints: BTreeSet<Fingerprint>,
}

/// The routing decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LaneOutcome {
    /// Select the owned-metal-decode lane.
    Owned,
    /// Select the llama lane with a recorded fallback reason.
    Llama { fallback_reason: FallbackReason },
    /// Refuse the request. `underlying_owned_decode_refusal_id` is set when a
    /// constrained pre-dispatch refusal was mapped to `grammar_disabled`.
    Refused {
        error: RoutingRefusal,
        underlying_owned_decode_refusal_id: Option<OwnedDecodeError>,
    },
}

/// Select a lane for the request. This is pure pre-dispatch logic; it does not
/// dispatch and never observes execution-phase outcomes.
pub fn select_lane(ctx: &LaneSelectionContext<'_>) -> LaneOutcome {
    let request = ctx.request;

    // 1. Exact fingerprint constraints, evaluated against the lane actually
    //    selected. A mismatch is the existing fingerprint-mismatch error and
    //    never silently falls back.
    if let Some(required) = &request.required_fingerprint {
        let alias_ok = request.allow_equivalent && ctx.equivalent_fingerprints.contains(required);
        if required != &ctx.owned_decode_fingerprint && !alias_ok {
            return LaneOutcome::Refused {
                error: RoutingRefusal::FingerprintMismatch,
                underlying_owned_decode_refusal_id: None,
            };
        }
    }

    // 2. Exact processing-fingerprint constraint requires the complete
    //    processing identity.
    if let Some(required_processing) = &request.required_processing_fingerprint {
        if required_processing != &ctx.owned_processing_fingerprint {
            return LaneOutcome::Refused {
                error: RoutingRefusal::FingerprintMismatch,
                underlying_owned_decode_refusal_id: None,
            };
        }
    }

    // 3. Advisory target fingerprint. `target_fingerprint` is advisory: for a
    //    substitutable request, a lane whose fingerprint differs from the
    //    advisory target is still eligible and fallback may proceed (the
    //    response provenance already carries the actual fingerprint). For a
    //    non-substitutable request — one whose contract does not permit
    //    substitution (constrained, exact-pinned, or explicit owned-only) —
    //    the owned lane is the only allowed lane, so an advisory target that
    //    does not match it must refuse with the existing fingerprint-mismatch
    //    error rather than silently serving a different fingerprint.
    if let Some(target) = &request.target_fingerprint {
        if target != &ctx.owned_decode_fingerprint && !request.is_substitutable() {
            return LaneOutcome::Refused {
                error: RoutingRefusal::FingerprintMismatch,
                underlying_owned_decode_refusal_id: None,
            };
        }
    }

    // 4. Constrained requests are owned-only.
    if request.is_constrained() {
        return match &ctx.owned {
            OwnedEvaluation::Selectable => LaneOutcome::Owned,
            OwnedEvaluation::Refused(error) => match error.constrained_predispatch_mapping() {
                Some(mapped) => LaneOutcome::Refused {
                    error: RoutingRefusal::Owned(mapped),
                    underlying_owned_decode_refusal_id: Some(*error),
                },
                None => LaneOutcome::Refused {
                    error: RoutingRefusal::Owned(*error),
                    underlying_owned_decode_refusal_id: None,
                },
            },
            // Cutover disabled means no certified owned lane; for a constrained
            // request that is the fallback-eligible NotCertified refusal, mapped
            // to grammar_disabled.
            OwnedEvaluation::NotPreferred => LaneOutcome::Refused {
                error: RoutingRefusal::Owned(OwnedDecodeError::GrammarDisabled),
                underlying_owned_decode_refusal_id: Some(OwnedDecodeError::NotCertified),
            },
        };
    }

    // 5. Unconstrained, non-substitutable requests (exact pin or explicit
    //    owned-only selection) never substitute a different model.
    if !request.is_substitutable() {
        return match &ctx.owned {
            OwnedEvaluation::Selectable => LaneOutcome::Owned,
            OwnedEvaluation::Refused(error) => LaneOutcome::Refused {
                error: RoutingRefusal::Owned(*error),
                underlying_owned_decode_refusal_id: None,
            },
            OwnedEvaluation::NotPreferred => LaneOutcome::Refused {
                error: RoutingRefusal::Owned(OwnedDecodeError::NotCertified),
                underlying_owned_decode_refusal_id: None,
            },
        };
    }

    // 6. Substitutable unconstrained requests.
    match &ctx.owned {
        OwnedEvaluation::Selectable => LaneOutcome::Owned,
        OwnedEvaluation::NotPreferred => match &ctx.llama {
            Some(_) => LaneOutcome::Llama {
                fallback_reason: FallbackReason::CutoverDisabled,
            },
            None => LaneOutcome::Refused {
                error: RoutingRefusal::Owned(OwnedDecodeError::NotCertified),
                underlying_owned_decode_refusal_id: None,
            },
        },
        OwnedEvaluation::Refused(error) => {
            if error.is_predispatch_fallback_eligible() {
                match &ctx.llama {
                    Some(_) => LaneOutcome::Llama {
                        fallback_reason: FallbackReason::OwnedRefusal(*error),
                    },
                    // No compatible llama: return the original refusal and do not
                    // claim fallback occurred.
                    None => LaneOutcome::Refused {
                        error: RoutingRefusal::Owned(*error),
                        underlying_owned_decode_refusal_id: None,
                    },
                }
            } else {
                // Non-fallback-eligible refusals return directly.
                LaneOutcome::Refused {
                    error: RoutingRefusal::Owned(*error),
                    underlying_owned_decode_refusal_id: None,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// D-009 cutover predicate
// ---------------------------------------------------------------------------

/// The checked-in D-009 cutover record for one machine profile.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CutoverRecord {
    pub machine_profile_hash: String,
    pub enabled_catalog_entry_ids: Vec<String>,
    pub decode_fingerprints: Vec<Fingerprint>,
    pub processing_fingerprints: Vec<Fingerprint>,
    /// Constrained runtime identities, present when grammar is enabled.
    #[serde(default)]
    pub constrained_runtime_identities: Vec<String>,
    pub runtime_config_digest: String,
    pub fixture_registry_revision: String,
    pub context_bucket_manifest_revision: String,
    pub scheduler_manifest_revision: String,
    pub certification_evidence_ids: Vec<String>,
    pub wire_error_binding_revision: String,
    /// Completed acceptance-gate evidence set (G-DEC-01..12).
    pub acceptance_gate_evidence: Vec<String>,
    /// Whether grammar (constrained decoding) is enabled for this profile.
    #[serde(default)]
    pub grammar_enabled: bool,
}

/// Runtime inputs to the cutover predicate, evaluated immediately before and
/// after applying the record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CutoverInputs {
    /// All listed artifacts are trusted.
    pub artifacts_trusted: bool,
    /// The exact runtime and processing identities are installed.
    pub identities_installed: bool,
    /// Unconstrained certification rows are valid for this profile.
    pub unconstrained_certified: bool,
    /// Applicable constrained certification rows are valid for this profile.
    pub constrained_certified: bool,
    /// The quarantine key is currently quarantined.
    pub quarantined: bool,
    /// The wire error bindings contain literal IDs (not symbolic placeholders).
    pub wire_bindings_literal: bool,
    /// Every applicable G-DEC-01..12 consequence has passed.
    pub gates_passed: bool,
    /// The final scheduler numeric manifest is committed and executed
    /// (G-DEC-11 and the scheduler-dependent portion of G-DEC-12).
    pub scheduler_evidence_committed: bool,
}

/// All inputs that are independent of SQLite row loading but still belong to
/// the owned-decode serving predicate. Keeping these checks in one pure
/// function prevents a caller from treating a partial match as admission.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ServingPredicateInputs {
    pub approval_enabled: bool,
    pub approval_identity_matches: bool,
    pub current_profile_matches: bool,
    pub current_epoch_valid: bool,
    pub certification_matches: bool,
    pub evidence_revisions_compatible: bool,
    pub gates_complete: bool,
    pub processing_fingerprint_matches: bool,
    pub runtime_config_digest_matches: bool,
    pub worker_path_matches: bool,
    pub constrained_identities_match: bool,
    pub artifacts_trusted: bool,
    pub identities_installed: bool,
    pub quarantined: bool,
    pub wire_bindings_literal: bool,
    pub scheduler_evidence_committed: bool,
}

/// Evaluate the complete fail-closed owned-decode serving predicate.
#[must_use]
pub fn serving_predicate(inputs: &ServingPredicateInputs) -> bool {
    inputs.approval_enabled
        && inputs.approval_identity_matches
        && inputs.current_profile_matches
        && inputs.current_epoch_valid
        && inputs.certification_matches
        && inputs.evidence_revisions_compatible
        && inputs.gates_complete
        && inputs.processing_fingerprint_matches
        && inputs.runtime_config_digest_matches
        && inputs.worker_path_matches
        && inputs.constrained_identities_match
        && inputs.artifacts_trusted
        && inputs.identities_installed
        && !inputs.quarantined
        && inputs.wire_bindings_literal
        && inputs.scheduler_evidence_committed
}

/// The epoch reader used at the last owned-dispatch boundary. Implementations
/// must read the persisted value, not a request-local cache.
pub trait AdmissionEpochReader: Send + Sync {
    fn current_profile_activation_epoch(&self) -> Result<Option<u64>, String>;
}

/// Effective grammar enablement is the conjunction of the runtime switch and
/// the approval's per-artifact switch.
#[must_use]
pub fn effective_grammar_enabled(runtime_enabled: bool, approval_enabled: bool) -> bool {
    runtime_enabled && approval_enabled
}

/// Evaluate the fail-closed owned-decode cutover predicate. Owned decode may
/// become the preferred lane only when every required condition holds; if
/// scheduler evidence or another required certification condition is blocked or
/// incomplete, owned decode remains disabled.
pub fn cutover_enabled(record: &CutoverRecord, inputs: &CutoverInputs) -> bool {
    let constrained_ok = !record.grammar_enabled || inputs.constrained_certified;
    inputs.artifacts_trusted
        && inputs.identities_installed
        && inputs.unconstrained_certified
        && constrained_ok
        && !inputs.quarantined
        && inputs.wire_bindings_literal
        && inputs.gates_passed
        && inputs.scheduler_evidence_committed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owned_decode_routing::family::Family;
    use crate::owned_decode_routing::identity::WeightQuant;
    use crate::owned_decode_routing::request::SamplingMode;

    fn fp(s: &str) -> Fingerprint {
        Fingerprint(s.to_string())
    }

    fn unconstrained_request() -> OneshotRequest {
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

    fn llama() -> LlamaLane {
        LlamaLane {
            decode_fingerprint: fp("llama-decode"),
            processing_fingerprint: fp("llama-proc"),
        }
    }

    fn ctx<'a>(
        request: &'a OneshotRequest,
        owned: OwnedEvaluation,
        llama: Option<LlamaLane>,
    ) -> LaneSelectionContext<'a> {
        LaneSelectionContext {
            request,
            owned_decode_fingerprint: fp("owned-decode"),
            owned_processing_fingerprint: fp("owned-proc"),
            owned,
            llama,
            equivalent_fingerprints: BTreeSet::new(),
        }
    }

    #[test]
    fn serving_predicate_is_fail_closed_for_every_gate() {
        let mut inputs = ServingPredicateInputs {
            approval_enabled: true,
            approval_identity_matches: true,
            current_profile_matches: true,
            current_epoch_valid: true,
            certification_matches: true,
            evidence_revisions_compatible: true,
            gates_complete: true,
            processing_fingerprint_matches: true,
            runtime_config_digest_matches: true,
            worker_path_matches: true,
            constrained_identities_match: true,
            artifacts_trusted: true,
            identities_installed: true,
            quarantined: false,
            wire_bindings_literal: true,
            scheduler_evidence_committed: true,
        };
        assert!(serving_predicate(&inputs));
        macro_rules! rejects_when_false {
            ($field:ident) => {
                inputs.$field = false;
                assert!(!serving_predicate(&inputs));
                inputs.$field = true;
            };
        }
        rejects_when_false!(approval_enabled);
        rejects_when_false!(approval_identity_matches);
        rejects_when_false!(current_profile_matches);
        rejects_when_false!(current_epoch_valid);
        rejects_when_false!(certification_matches);
        rejects_when_false!(evidence_revisions_compatible);
        rejects_when_false!(gates_complete);
        rejects_when_false!(processing_fingerprint_matches);
        rejects_when_false!(runtime_config_digest_matches);
        rejects_when_false!(worker_path_matches);
        rejects_when_false!(constrained_identities_match);
        rejects_when_false!(artifacts_trusted);
        rejects_when_false!(identities_installed);
        rejects_when_false!(wire_bindings_literal);
        rejects_when_false!(scheduler_evidence_committed);
        inputs.quarantined = true;
        assert!(!serving_predicate(&inputs));
    }

    #[test]
    fn selectable_owned_lane_is_chosen() {
        let request = unconstrained_request();
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::Selectable, Some(llama())));
        assert_eq!(outcome, LaneOutcome::Owned);
    }

    #[test]
    fn substitutable_unconstrained_falls_back_on_eligible_refusal() {
        let request = unconstrained_request();
        for refusal in [
            OwnedDecodeError::NotCertified,
            OwnedDecodeError::CertificationFailed,
            OwnedDecodeError::Quarantined,
            OwnedDecodeError::ArtifactPoisoned,
            OwnedDecodeError::Unavailable,
            OwnedDecodeError::Unsupported,
        ] {
            let outcome = select_lane(&ctx(
                &request,
                OwnedEvaluation::Refused(refusal),
                Some(llama()),
            ));
            assert_eq!(
                outcome,
                LaneOutcome::Llama {
                    fallback_reason: FallbackReason::OwnedRefusal(refusal)
                },
                "{refusal:?}"
            );
        }
    }

    #[test]
    fn substitutable_refusal_without_llama_returns_original_and_no_fallback_claim() {
        let request = unconstrained_request();
        let outcome = select_lane(&ctx(
            &request,
            OwnedEvaluation::Refused(OwnedDecodeError::Quarantined),
            None,
        ));
        assert_eq!(
            outcome,
            LaneOutcome::Refused {
                error: RoutingRefusal::Owned(OwnedDecodeError::Quarantined),
                underlying_owned_decode_refusal_id: None,
            }
        );
    }

    #[test]
    fn non_eligible_refusal_returns_directly_even_with_llama() {
        let request = unconstrained_request();
        // context_capacity_exceeded is not fallback-eligible.
        let outcome = select_lane(&ctx(
            &request,
            OwnedEvaluation::Refused(OwnedDecodeError::ContextCapacityExceeded),
            Some(llama()),
        ));
        assert_eq!(
            outcome,
            LaneOutcome::Refused {
                error: RoutingRefusal::Owned(OwnedDecodeError::ContextCapacityExceeded),
                underlying_owned_decode_refusal_id: None,
            }
        );
    }

    #[test]
    fn constrained_predispatch_refusal_maps_to_grammar_disabled_with_underlying() {
        let mut request = unconstrained_request();
        request.grammar = Some(serde_json::json!({"type": "object"}));
        let outcome = select_lane(&ctx(
            &request,
            OwnedEvaluation::Refused(OwnedDecodeError::NotCertified),
            Some(llama()),
        ));
        assert_eq!(
            outcome,
            LaneOutcome::Refused {
                error: RoutingRefusal::Owned(OwnedDecodeError::GrammarDisabled),
                underlying_owned_decode_refusal_id: Some(OwnedDecodeError::NotCertified),
            }
        );
    }

    #[test]
    fn constrained_non_eligible_refusal_returns_directly() {
        let mut request = unconstrained_request();
        request.grammar = Some(serde_json::json!({"type": "object"}));
        let outcome = select_lane(&ctx(
            &request,
            OwnedEvaluation::Refused(OwnedDecodeError::ProtocolMismatch),
            Some(llama()),
        ));
        assert_eq!(
            outcome,
            LaneOutcome::Refused {
                error: RoutingRefusal::Owned(OwnedDecodeError::ProtocolMismatch),
                underlying_owned_decode_refusal_id: None,
            }
        );
    }

    #[test]
    fn constrained_selectable_owned_is_chosen_and_never_llama() {
        let mut request = unconstrained_request();
        request.grammar = Some(serde_json::json!({"type": "object"}));
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::Selectable, Some(llama())));
        assert_eq!(outcome, LaneOutcome::Owned);
    }

    #[test]
    fn exact_fingerprint_mismatch_never_falls_back() {
        let mut request = unconstrained_request();
        request.required_fingerprint = Some(fp("some-other-fp"));
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::Selectable, Some(llama())));
        assert_eq!(
            outcome,
            LaneOutcome::Refused {
                error: RoutingRefusal::FingerprintMismatch,
                underlying_owned_decode_refusal_id: None,
            }
        );
        assert_eq!(outcome_wire(&outcome), "substitution_rejected");
    }

    #[test]
    fn exact_fingerprint_match_selects_owned() {
        let mut request = unconstrained_request();
        request.required_fingerprint = Some(fp("owned-decode"));
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::Selectable, Some(llama())));
        assert_eq!(outcome, LaneOutcome::Owned);
    }

    #[test]
    fn allow_equivalent_alias_selects_owned() {
        let mut request = unconstrained_request();
        request.required_fingerprint = Some(fp("alias-of-owned"));
        request.allow_equivalent = true;
        let mut context = ctx(&request, OwnedEvaluation::Selectable, Some(llama()));
        context.equivalent_fingerprints = BTreeSet::from([fp("alias-of-owned")]);
        assert_eq!(select_lane(&context), LaneOutcome::Owned);
    }

    #[test]
    fn processing_fingerprint_mismatch_refuses() {
        let mut request = unconstrained_request();
        request.required_processing_fingerprint = Some(fp("wrong-proc"));
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::Selectable, Some(llama())));
        assert_eq!(
            outcome,
            LaneOutcome::Refused {
                error: RoutingRefusal::FingerprintMismatch,
                underlying_owned_decode_refusal_id: None,
            }
        );
    }

    #[test]
    fn advisory_target_fingerprint_match_selects_owned() {
        // An advisory target that matches the owned lane's decode fingerprint
        // is satisfied; the owned lane is selected.
        let mut request = unconstrained_request();
        request.target_fingerprint = Some(fp("owned-decode"));
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::Selectable, Some(llama())));
        assert_eq!(outcome, LaneOutcome::Owned);
    }

    #[test]
    fn advisory_target_fingerprint_substituted_when_substitution_permitted() {
        // A substitutable request with an advisory target that does NOT match
        // the owned lane: the advisory target allows substitution, so the
        // owned lane is still eligible. When the owned lane is selectable, it
        // is chosen; the response provenance carries the actual fingerprint
        // (verified end-to-end elsewhere). The advisory mismatch never
        // refuses for a substitutable request.
        let mut request = unconstrained_request();
        request.target_fingerprint = Some(fp("some-other-fp"));
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::Selectable, Some(llama())));
        assert_eq!(outcome, LaneOutcome::Owned);

        // When the owned lane refuses with a fallback-eligible error, a
        // substitutable request still falls back to llama despite the advisory
        // target mismatch.
        let outcome = select_lane(&ctx(
            &request,
            OwnedEvaluation::Refused(OwnedDecodeError::NotCertified),
            Some(llama()),
        ));
        assert_eq!(
            outcome,
            LaneOutcome::Llama {
                fallback_reason: FallbackReason::OwnedRefusal(OwnedDecodeError::NotCertified)
            }
        );
    }

    #[test]
    fn advisory_target_fingerprint_mismatch_refuses_when_substitution_not_permitted() {
        // A non-substitutable request (explicit owned-only) with an advisory
        // target that does not match the owned lane: substitution is not
        // permitted, so the mismatch refuses with the existing
        // fingerprint-mismatch error rather than serving silently.
        let mut request = unconstrained_request();
        request.target_fingerprint = Some(fp("some-other-fp"));
        request.owned_only = true;
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::Selectable, Some(llama())));
        assert_eq!(
            outcome,
            LaneOutcome::Refused {
                error: RoutingRefusal::FingerprintMismatch,
                underlying_owned_decode_refusal_id: None,
            }
        );
        assert_eq!(outcome_wire(&outcome), "substitution_rejected");
    }

    #[test]
    fn advisory_target_fingerprint_mismatch_refuses_for_constrained_request() {
        // A constrained request is owned-only (substitution not permitted), so
        // an advisory target mismatch refuses.
        let mut request = unconstrained_request();
        request.grammar = Some(serde_json::json!({"type": "object"}));
        request.target_fingerprint = Some(fp("some-other-fp"));
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::Selectable, Some(llama())));
        assert_eq!(
            outcome,
            LaneOutcome::Refused {
                error: RoutingRefusal::FingerprintMismatch,
                underlying_owned_decode_refusal_id: None,
            }
        );
    }

    #[test]
    fn advisory_target_fingerprint_does_not_weaken_exact_required_pin() {
        // required_fingerprint (exact) is unchanged: an exact pin that
        // mismatches still refuses regardless of target_fingerprint.
        let mut request = unconstrained_request();
        request.required_fingerprint = Some(fp("not-the-owned-fp"));
        request.target_fingerprint = Some(fp("owned-decode"));
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::Selectable, Some(llama())));
        assert_eq!(
            outcome,
            LaneOutcome::Refused {
                error: RoutingRefusal::FingerprintMismatch,
                underlying_owned_decode_refusal_id: None,
            }
        );

        // And when the exact pin matches, an advisory target mismatch on a
        // non-substitutable request still refuses (exact pin makes it
        // non-substitutable).
        let mut request = unconstrained_request();
        request.required_fingerprint = Some(fp("owned-decode"));
        request.target_fingerprint = Some(fp("some-other-fp"));
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::Selectable, Some(llama())));
        assert_eq!(
            outcome,
            LaneOutcome::Refused {
                error: RoutingRefusal::FingerprintMismatch,
                underlying_owned_decode_refusal_id: None,
            }
        );
    }

    #[test]
    fn owned_only_selection_never_substitutes_llama() {
        let mut request = unconstrained_request();
        request.owned_only = true;
        let outcome = select_lane(&ctx(
            &request,
            OwnedEvaluation::Refused(OwnedDecodeError::Unavailable),
            Some(llama()),
        ));
        assert_eq!(
            outcome,
            LaneOutcome::Refused {
                error: RoutingRefusal::Owned(OwnedDecodeError::Unavailable),
                underlying_owned_decode_refusal_id: None,
            }
        );
    }

    #[test]
    fn cutover_disabled_routes_substitutable_to_llama() {
        let request = unconstrained_request();
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::NotPreferred, Some(llama())));
        assert_eq!(
            outcome,
            LaneOutcome::Llama {
                fallback_reason: FallbackReason::CutoverDisabled
            }
        );
    }

    #[test]
    fn cutover_disabled_without_llama_refuses_not_certified() {
        let request = unconstrained_request();
        let outcome = select_lane(&ctx(&request, OwnedEvaluation::NotPreferred, None));
        assert_eq!(
            outcome,
            LaneOutcome::Refused {
                error: RoutingRefusal::Owned(OwnedDecodeError::NotCertified),
                underlying_owned_decode_refusal_id: None,
            }
        );
    }

    fn outcome_wire(outcome: &LaneOutcome) -> &'static str {
        match outcome {
            LaneOutcome::Refused { error, .. } => error.wire_id(),
            _ => panic!("not a refusal"),
        }
    }

    fn cutover_record() -> CutoverRecord {
        CutoverRecord {
            machine_profile_hash: "profile-a".to_string(),
            enabled_catalog_entry_ids: vec!["qwen3-f16".to_string()],
            decode_fingerprints: vec![fp("owned-decode")],
            processing_fingerprints: vec![fp("owned-proc")],
            constrained_runtime_identities: vec![],
            runtime_config_digest: "rcd".to_string(),
            fixture_registry_revision: "decode-fixture-registry-v1".to_string(),
            context_bucket_manifest_revision: "decode-context-buckets-v1".to_string(),
            scheduler_manifest_revision: "decode-sched-manifest-v1".to_string(),
            certification_evidence_ids: vec!["cert-1".to_string()],
            wire_error_binding_revision: "owned-decode-wire-error-bindings-v1".to_string(),
            acceptance_gate_evidence: vec!["G-DEC-01".to_string()],
            grammar_enabled: false,
        }
    }

    fn all_true_inputs() -> CutoverInputs {
        CutoverInputs {
            artifacts_trusted: true,
            identities_installed: true,
            unconstrained_certified: true,
            constrained_certified: true,
            quarantined: false,
            wire_bindings_literal: true,
            gates_passed: true,
            scheduler_evidence_committed: true,
        }
    }

    #[test]
    fn cutover_enabled_requires_every_condition() {
        let record = cutover_record();
        assert!(cutover_enabled(&record, &all_true_inputs()));

        // Each false condition independently disables cutover.
        let mut inputs = all_true_inputs();
        inputs.artifacts_trusted = false;
        assert!(!cutover_enabled(&record, &inputs));

        let mut inputs = all_true_inputs();
        inputs.quarantined = true;
        assert!(!cutover_enabled(&record, &inputs));

        let mut inputs = all_true_inputs();
        inputs.wire_bindings_literal = false;
        assert!(!cutover_enabled(&record, &inputs));

        let mut inputs = all_true_inputs();
        inputs.scheduler_evidence_committed = false;
        assert!(
            !cutover_enabled(&record, &inputs),
            "scheduler gate blocks enablement"
        );

        let mut inputs = all_true_inputs();
        inputs.unconstrained_certified = false;
        assert!(!cutover_enabled(&record, &inputs));
    }

    #[test]
    fn cutover_grammar_enabled_requires_constrained_certification() {
        let mut record = cutover_record();
        record.grammar_enabled = true;

        let mut inputs = all_true_inputs();
        inputs.constrained_certified = false;
        assert!(!cutover_enabled(&record, &inputs));

        inputs.constrained_certified = true;
        assert!(cutover_enabled(&record, &inputs));
    }

    #[test]
    fn cutover_record_rejects_unknown_field() {
        // fail-closed posture: an unknown field in a cutover record is
        // rejected at parse time rather than silently dropped.
        let json = serde_json::json!({
            "machine_profile_hash": "profile-a",
            "enabled_catalog_entry_ids": ["qwen3-f16"],
            "decode_fingerprints": ["owned-decode"],
            "processing_fingerprints": ["owned-proc"],
            "constrained_runtime_identities": [],
            "runtime_config_digest": "rcd",
            "fixture_registry_revision": "decode-fixture-registry-v1",
            "context_bucket_manifest_revision": "decode-context-buckets-v1",
            "scheduler_manifest_revision": "decode-sched-manifest-v1",
            "certification_evidence_ids": ["cert-1"],
            "wire_error_binding_revision": "owned-decode-wire-error-bindings-v1",
            "acceptance_gate_evidence": ["G-DEC-01"],
            "grammar_enabled": false,
            "unknown_field": "should be rejected",
        });
        assert!(serde_json::from_value::<CutoverRecord>(json).is_err());
    }
}
