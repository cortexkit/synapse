//! Worker-start validation with dedicated, non-overlapping error mappings.
//!
//! Resolution r2 #7 fixes three dedicated IDs at worker start:
//! - protocol-ID / frame-structure incompatibility → `owned_decode_protocol_mismatch`
//! - loaded-model / decode-fingerprint / runtime-manifest identity mismatch
//!   → `owned_decode_runtime_config_mismatch`
//! - constraint-identity mismatch → `owned_decode_constraint_version_mismatch`
//!
//! In addition, version 1 accepts only greedy-top-1; any other sampling mode
//! returns `owned_decode_sampling_unsupported`. The checks run in a fixed order
//! so the mapping is deterministic: protocol structure, then sampling, then
//! runtime identity, then constraint identity.

use crate::error::DecodeError;
use crate::protocol::{GenerateStart, TokenIdJsonConstraint, GREEDY_TOP1};

/// The worker-side view of what is currently loaded and certified. The worker
/// compares the start frame against this before committing the first token.
#[derive(Clone, Debug)]
pub struct WorkerStartContext {
    pub loaded_model_ref: String,
    pub decode_fingerprint: String,
    pub runtime_config_digest: String,
    /// The constraint the worker was built/certified with, if constrained.
    pub expected_constraint: Option<TokenIdJsonConstraint>,
}

/// The result of a successful start validation: the authorized first-quantum
/// token budget, `min(production_n, max_tokens)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StartAuthorization {
    pub first_quantum_budget: u32,
}

/// Validate a start frame against the worker context.
///
/// `production_n` is the single committed production N. The authorized budget is
/// `min(production_n, max_tokens)`; a zero `max_tokens` is a protocol violation
/// because no quantum could be authorized.
pub fn validate_start(
    start: &GenerateStart,
    context: &WorkerStartContext,
    production_n: u32,
) -> Result<StartAuthorization, DecodeError> {
    // 1. Frame-structure invariants. A start with no generation id or no prompt
    //    is structurally invalid: dedicated protocol mismatch.
    if start.generation_id.is_empty() || start.prompt_ids.is_empty() || start.max_tokens == 0 {
        return Err(DecodeError::ProtocolMismatch);
    }

    // 2. Sampling mode. Version 1 accepts only greedy-top-1.
    if start.sampling.mode != GREEDY_TOP1 {
        return Err(DecodeError::SamplingUnsupported);
    }

    // 3. Runtime identity: loaded-model reference, decode fingerprint, and
    //    runtime-config digest must all agree. Any mismatch is the dedicated
    //    runtime-config mismatch (never protocol or constraint).
    if start.loaded_model_ref != context.loaded_model_ref
        || start.decode_fingerprint != context.decode_fingerprint
        || start.runtime_config_digest != context.runtime_config_digest
    {
        return Err(DecodeError::RuntimeConfigMismatch);
    }

    // 4. Constraint identity. Presence must agree, and every field must match.
    match (&start.constraint, &context.expected_constraint) {
        (None, None) => {}
        (Some(_), None) | (None, Some(_)) => {
            return Err(DecodeError::ConstraintVersionMismatch);
        }
        (Some(actual), Some(expected)) => {
            if actual.first_mismatched_field(expected).is_some() {
                return Err(DecodeError::ConstraintVersionMismatch);
            }
        }
    }

    Ok(StartAuthorization {
        first_quantum_budget: production_n.min(start.max_tokens),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{sample_constraint, Sampling};

    fn context() -> WorkerStartContext {
        WorkerStartContext {
            loaded_model_ref: "model-qwen3-f16".into(),
            decode_fingerprint: "dfp-1".into(),
            runtime_config_digest: "rt-1".into(),
            expected_constraint: None,
        }
    }

    fn start() -> GenerateStart {
        GenerateStart {
            generation_id: "g1".into(),
            loaded_model_ref: "model-qwen3-f16".into(),
            decode_fingerprint: "dfp-1".into(),
            runtime_config_digest: "rt-1".into(),
            prompt_ids: vec![10, 11, 12],
            stop_ids: vec![2],
            max_tokens: 64,
            sampling: Sampling::greedy_top1(),
            constraint: None,
        }
    }

    #[test]
    fn clean_start_authorizes_min_n_max_tokens() {
        let auth = validate_start(&start(), &context(), 16).expect("ok");
        assert_eq!(auth.first_quantum_budget, 16);
        // max_tokens below N truncates the first quantum.
        let mut small = start();
        small.max_tokens = 5;
        let auth = validate_start(&small, &context(), 16).expect("ok");
        assert_eq!(auth.first_quantum_budget, 5);
    }

    #[test]
    fn runtime_identity_mismatches_map_to_runtime_config_mismatch() {
        for perturb in [
            |s: &mut GenerateStart| s.loaded_model_ref = "other".into(),
            |s: &mut GenerateStart| s.decode_fingerprint = "other".into(),
            |s: &mut GenerateStart| s.runtime_config_digest = "other".into(),
        ] {
            let mut s = start();
            perturb(&mut s);
            assert_eq!(
                validate_start(&s, &context(), 16),
                Err(DecodeError::RuntimeConfigMismatch)
            );
        }
    }

    #[test]
    fn non_greedy_sampling_is_unsupported() {
        let mut s = start();
        s.sampling.mode = "top_p".into();
        assert_eq!(
            validate_start(&s, &context(), 16),
            Err(DecodeError::SamplingUnsupported)
        );
    }

    #[test]
    fn structural_faults_map_to_protocol_mismatch() {
        let mut s = start();
        s.generation_id = String::new();
        assert_eq!(
            validate_start(&s, &context(), 16),
            Err(DecodeError::ProtocolMismatch)
        );
        let mut s = start();
        s.prompt_ids.clear();
        assert_eq!(
            validate_start(&s, &context(), 16),
            Err(DecodeError::ProtocolMismatch)
        );
        let mut s = start();
        s.max_tokens = 0;
        assert_eq!(
            validate_start(&s, &context(), 16),
            Err(DecodeError::ProtocolMismatch)
        );
    }

    #[test]
    fn constraint_mismatches_map_to_constraint_version_mismatch() {
        let mut ctx = context();
        ctx.expected_constraint = Some(sample_constraint());

        // Missing constraint where one is expected.
        assert_eq!(
            validate_start(&start(), &ctx, 16),
            Err(DecodeError::ConstraintVersionMismatch)
        );

        // A present constraint that matches is accepted.
        let mut ok = start();
        ok.constraint = Some(sample_constraint());
        assert!(validate_start(&ok, &ctx, 16).is_ok());

        // Each perturbed field maps to the constraint mismatch ID.
        for perturb in [
            |c: &mut TokenIdJsonConstraint| c.grammar_compiler_revision = "x".into(),
            |c: &mut TokenIdJsonConstraint| c.tokenizer_vocabulary_digest = "x".into(),
            |c: &mut TokenIdJsonConstraint| c.limits_manifest_id = "x".into(),
            |c: &mut TokenIdJsonConstraint| c.canonical_schema_digest = "x".into(),
            |c: &mut TokenIdJsonConstraint| c.initial_state_digest = "x".into(),
            |c: &mut TokenIdJsonConstraint| c.compiled_automaton_digest = "x".into(),
            |c: &mut TokenIdJsonConstraint| c.automaton_bytes = vec![9, 9],
        ] {
            let mut bad = sample_constraint();
            perturb(&mut bad);
            let mut s = start();
            s.constraint = Some(bad);
            assert_eq!(
                validate_start(&s, &ctx, 16),
                Err(DecodeError::ConstraintVersionMismatch)
            );
        }
    }
}
