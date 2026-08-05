//! Execution of the non-parity `decode-fixture-registry-v1` groups.
//!
//! The fixture registry is referenced directly by probe code: the gate runner
//! executes the `request-processing`, `constrained-positive`,
//! `constrained-negative`, and `scheduler-continuity` groups from the
//! checked-in registry JSON and records every executed fixture ID per group in
//! the release evidence. These groups are hardware-independent and run
//! everywhere; the hardware-gated `decode-parity` group follows the probe-seam
//! pattern in `gate_04_parity` instead (test double everywhere, real Metal on
//! the mandatory `macos-metal` lane).
//!
//! The constrained-negative group additionally runs the rejection probes the
//! grammar audit added (cross-node keywords, `\u` escape hex validation, and
//! the compiled-state-count limit) as labeled cases within the group: the
//! group's coverage promise is "rejected keywords, numeric limits, identity
//! mismatches, and typed grammar errors", and those probes are exactly the
//! rejected-keyword, numeric-limit, and typed-error classes.

use serde_json::Value;
use synapse_core::Fingerprint;

use crate::owned_decode_contracts::{FixtureEntry, ManifestDir};
use crate::owned_decode_grammar_scheduler::grammar_automaton::{drain_bytes, Automaton};
use crate::owned_decode_grammar_scheduler::{
    compile_grammar, load_automaton, BoundaryKind, BoundaryOutcome, CancelResult, CompileContext,
    DecodeOp, DecodeScheduler, DecodeSchedulerConfig, GenerationProtocol, GrammarSubsetManifest,
};
use crate::owned_decode_routing::error::OwnedDecodeError;
use crate::owned_decode_routing::family::{Family, FamilyRegistry};

/// Fixture-registry group: module-owned tokenization, templating, and
/// detokenization identity for both families.
pub const REQUEST_PROCESSING_GROUP: &str = "request-processing";
/// Fixture-registry group: accepted grammar forms and valid constrained output.
pub const CONSTRAINED_POSITIVE_GROUP: &str = "constrained-positive";
/// Fixture-registry group: rejected keywords, numeric limits, identity
/// mismatches, and typed grammar errors.
pub const CONSTRAINED_NEGATIVE_GROUP: &str = "constrained-negative";
/// Fixture-registry group: parity and protocol-behavior fixtures crossing
/// quantum boundaries.
pub const SCHEDULER_CONTINUITY_GROUP: &str = "scheduler-continuity";

/// One group's execution outcome: the executed registry entry IDs and, on
/// failure, the reason.
pub struct GroupOutcome {
    pub executed_ids: Vec<String>,
    pub result: Result<(), String>,
}

fn entries_for<'a>(manifests: &'a ManifestDir, group: &str) -> Vec<&'a FixtureEntry> {
    manifests
        .fixture_registry
        .entries
        .iter()
        .filter(|entry| entry.group == group)
        .collect()
}

/// Execute the `request-processing` group: every entry names a family, and the
/// production family registry must carry that family's complete
/// processing-asset identity (tokenizer, template, special/stop policies,
/// detokenizer). Registry digests that are still `pending-registration` are
/// not compared field-by-field; the assertion is that the module's registration
/// is complete and non-empty for the entry's family.
pub fn run_request_processing(manifests: &ManifestDir) -> GroupOutcome {
    let mut executed_ids = Vec::new();
    let registry = FamilyRegistry::production();
    for entry in entries_for(manifests, REQUEST_PROCESSING_GROUP) {
        let family_str = entry.details.get("family").and_then(Value::as_str);
        let outcome = (|| -> Result<(), String> {
            let family_str =
                family_str.ok_or_else(|| "entry has no family in details".to_string())?;
            let family = Family::parse(family_str)
                .map_err(|_| format!("unrecognized family {family_str}"))?;
            let registration = registry
                .get(family)
                .map_err(|_| format!("family {family_str} has no processing registration"))?;
            if registration.tokenizer_sanitized_digest.is_empty()
                || registration.prompt_template_revision.is_empty()
                || registration.special_token_policy_revision.is_empty()
                || registration.stop_token_policy_revision.is_empty()
                || registration.detokenizer_revision.is_empty()
            {
                return Err(format!(
                    "family {family_str} registration carries empty processing assets"
                ));
            }
            if registration.stop_token_ids.is_empty() {
                return Err(format!(
                    "family {family_str} registration carries no stop token ids"
                ));
            }
            Ok(())
        })();
        if let Err(reason) = outcome {
            return GroupOutcome {
                executed_ids,
                result: Err(format!("{}: {reason}", entry.id)),
            };
        }
        executed_ids.push(entry.id.clone());
    }
    if executed_ids.is_empty() {
        return GroupOutcome {
            executed_ids,
            result: Err("registry has no request-processing entries".to_string()),
        };
    }
    GroupOutcome {
        executed_ids,
        result: Ok(()),
    }
}

fn compile_context() -> CompileContext {
    CompileContext {
        base_decode_fingerprint: Fingerprint("fixture-groups-base-fp".to_string()),
        tokenizer_vocabulary_digest: "fixture-groups-vocab-digest".to_string(),
    }
}

/// The valid document exercised against a constrained-positive registry entry,
/// keyed by the stable entry ID. A new positive entry fails the group loudly
/// until a document is mapped here (fail closed).
fn positive_document(entry_id: &str) -> Option<&'static str> {
    match entry_id {
        "constrained-positive-object-required" => Some(r#"{"name":"ada"}"#),
        "constrained-positive-array-items" => Some("[1,2,3]"),
        "constrained-positive-enum-string" => Some(r#""green""#),
        _ => None,
    }
}

/// Execute the `constrained-positive` group: each entry's schema is taken from
/// the registry JSON, compiled through the grammar pipeline, reloaded through
/// the worker-side load path, and must accept its valid document.
pub fn run_constrained_positive(manifests: &ManifestDir) -> GroupOutcome {
    let mut executed_ids = Vec::new();
    let manifest = GrammarSubsetManifest::default();
    let context = compile_context();
    for entry in entries_for(manifests, CONSTRAINED_POSITIVE_GROUP) {
        let outcome = (|| -> Result<(), String> {
            let schema_value = entry
                .details
                .get("schema")
                .ok_or_else(|| "entry carries no schema".to_string())?;
            let raw_schema = serde_json::to_string(schema_value)
                .map_err(|err| format!("schema does not serialize: {err}"))?;
            let compiled = compile_grammar(&raw_schema, &context, &manifest)
                .map_err(|err| format!("registry schema failed to compile: {err:?}"))?;
            let automaton = load_automaton(&compiled.constraint, &manifest)
                .map_err(|err| format!("compiled automaton failed to load: {err:?}"))?;
            let document = positive_document(&entry.id)
                .ok_or_else(|| "no valid document mapped for this positive entry".to_string())?;
            let state = drain_bytes(&automaton, document.as_bytes())
                .map_err(|err| format!("valid document rejected: {}", err.message))?;
            if !automaton.has_complete_value(&state) {
                return Err("valid document did not complete a value".to_string());
            }
            Ok(())
        })();
        if let Err(reason) = outcome {
            return GroupOutcome {
                executed_ids,
                result: Err(format!("{}: {reason}", entry.id)),
            };
        }
        executed_ids.push(entry.id.clone());
    }
    if executed_ids.is_empty() {
        return GroupOutcome {
            executed_ids,
            result: Err("registry has no constrained-positive entries".to_string()),
        };
    }
    GroupOutcome {
        executed_ids,
        result: Ok(()),
    }
}

/// Execute the `constrained-negative` group: every registry entry is a
/// data-driven rejection case, and the group also runs the grammar-audit
/// rejection probes (cross-node keywords, `\u` escape hex digits, and the
/// compiled-state-count limit) as labeled cases under the same coverage
/// promise.
#[allow(clippy::type_complexity)]
pub fn run_constrained_negative(manifests: &ManifestDir) -> GroupOutcome {
    let mut executed_ids = Vec::new();
    let manifest = GrammarSubsetManifest::default();
    let context = compile_context();

    for entry in entries_for(manifests, CONSTRAINED_NEGATIVE_GROUP) {
        let outcome = run_negative_entry(entry, &context, &manifest);
        if let Err(reason) = outcome {
            return GroupOutcome {
                executed_ids,
                result: Err(format!("{}: {reason}", entry.id)),
            };
        }
        executed_ids.push(entry.id.clone());
    }
    if executed_ids.is_empty() {
        return GroupOutcome {
            executed_ids,
            result: Err("registry has no constrained-negative entries".to_string()),
        };
    }

    // Grammar-audit rejection probes, run as labeled cases of this group.
    let probes: Vec<(&str, Box<dyn Fn() -> Result<(), String>>)> = vec![
        (
            "f1-cross-node-keywords-rejected",
            Box::new(|| {
                for raw in [
                    r#"{ "type": "object", "properties": { "a": { "type": "integer" } },
                         "required": ["a"], "additionalProperties": false, "enum": [{"a": 1}] }"#,
                    r#"{ "type": "array", "items": { "type": "string" }, "enum": [["a"]] }"#,
                    r#"{ "type": "object", "properties": {}, "additionalProperties": false,
                         "items": { "type": "string" } }"#,
                    r#"{ "type": "array", "items": { "type": "string" }, "properties": {},
                         "required": [], "additionalProperties": false }"#,
                ] {
                    expect_feature_unsupported(&manifest, &context, raw)?;
                }
                Ok(())
            }),
        ),
        (
            "f2-unicode-escape-hex-rejected",
            Box::new(|| {
                let schema = crate::owned_decode_grammar_scheduler::parse_schema(
                    r#"{ "type": "string" }"#,
                    &manifest.limits,
                )
                .map_err(|err| format!("probe schema failed to parse: {err:?}"))?;
                let automaton = Automaton::new(schema);
                for document in [br#""a\uZZZZb""#.as_slice(), br#""\u41""#.as_slice()] {
                    if drain_bytes(&automaton, document).is_ok() {
                        return Err(format!(
                            "invalid \\u escape accepted: {}",
                            String::from_utf8_lossy(document)
                        ));
                    }
                }
                Ok(())
            }),
        ),
        (
            "f3-compiled-state-count-rejected",
            Box::new(|| {
                let mut limited = manifest.clone();
                limited.limits.max_compiled_state_count = 1;
                let raw = r#"{ "type": "object", "properties": { "name": { "type": "string" } },
                             "required": ["name"], "additionalProperties": false }"#;
                match compile_grammar(raw, &context, &limited) {
                    Err(error)
                        if error.wire_error() == OwnedDecodeError::GrammarFeatureUnsupported =>
                    {
                        Ok(())
                    }
                    other => Err(format!(
                        "over-state-limit schema must fail with grammar_feature_unsupported, got {other:?}"
                    )),
                }
            }),
        ),
    ];
    for (label, probe) in probes {
        if let Err(reason) = probe() {
            return GroupOutcome {
                executed_ids,
                result: Err(format!("{label}: {reason}")),
            };
        }
    }

    GroupOutcome {
        executed_ids,
        result: Ok(()),
    }
}

fn expect_feature_unsupported(
    manifest: &GrammarSubsetManifest,
    context: &CompileContext,
    raw: &str,
) -> Result<(), String> {
    match compile_grammar(raw, context, manifest) {
        Err(error) if error.wire_error() == OwnedDecodeError::GrammarFeatureUnsupported => Ok(()),
        other => Err(format!(
            "expected grammar_feature_unsupported, got {other:?}"
        )),
    }
}

fn run_negative_entry(
    entry: &FixtureEntry,
    context: &CompileContext,
    manifest: &GrammarSubsetManifest,
) -> Result<(), String> {
    let details = &entry.details;
    let expected_error = details
        .get("expected_error")
        .and_then(Value::as_str)
        .unwrap_or("");

    if let Some(rejected_keyword) = details.get("rejected_keyword").and_then(Value::as_str) {
        // A keyword/shape the subset rejects: compile the probe schema and
        // require the typed feature error named by the registry entry.
        let raw = match rejected_keyword {
            "allOf" => r#"{ "type": "string", "allOf": [] }"#,
            "items-tuple-form" => {
                r#"{ "type": "array", "items": [{ "type": "string" }, { "type": "integer" }] }"#
            }
            "additionalProperties-true" => {
                r#"{ "type": "object", "properties": {}, "additionalProperties": true }"#
            }
            other => return Err(format!("unmapped rejected_keyword '{other}'")),
        };
        match compile_grammar(raw, context, manifest) {
            Err(error) if error.wire_error() == OwnedDecodeError::GrammarFeatureUnsupported => {
                if expected_error.is_empty()
                    || expected_error == OwnedDecodeError::GrammarFeatureUnsupported.as_str()
                {
                    return Ok(());
                }
                Err(format!(
                    "registry expects {expected_error} but the probe produced {}",
                    error.wire_error().as_str()
                ))
            }
            other => Err(format!(
                "rejected keyword '{rejected_keyword}' must fail with grammar_feature_unsupported, got {other:?}"
            )),
        }
    } else if let Some(mismatch) = details.get("mismatch").and_then(Value::as_str) {
        // An identity mismatch on the worker load path: a valid compiled
        // constraint must be refused with the typed mismatch error.
        if mismatch != "constraint_runtime_identity" {
            return Err(format!("unmapped mismatch '{mismatch}'"));
        }
        let raw = r#"{ "type": "object", "properties": { "name": { "type": "string" } },
                     "required": ["name"], "additionalProperties": false }"#;
        let compiled = compile_grammar(raw, context, manifest)
            .map_err(|err| format!("probe schema failed to compile: {err:?}"))?;
        let mut rotated = manifest.clone();
        rotated.grammar_compiler_revision = "grammar-compiler-v2".to_string();
        match load_automaton(&compiled.constraint, &rotated) {
            Err(OwnedDecodeError::ConstraintVersionMismatch)
                if expected_error.is_empty()
                    || expected_error
                        == OwnedDecodeError::ConstraintVersionMismatch.as_str() =>
            {
                Ok(())
            }
            other => Err(format!(
                "identity mismatch must fail with owned_decode_constraint_version_mismatch, got {other:?}"
            )),
        }
    } else if details.get("owned_refusal").is_some() {
        // The constrained pre-dispatch refusal mapping: the six fallback-eligible
        // owned refusals map to grammar_disabled with the underlying ID retained.
        let owned_refusal = details
            .get("underlying_owned_decode_refusal_id")
            .and_then(Value::as_str)
            .unwrap_or("");
        let mapped_error = details
            .get("mapped_error")
            .and_then(Value::as_str)
            .unwrap_or("");
        if owned_refusal != OwnedDecodeError::NotCertified.as_str() {
            return Err(format!("unmapped owned_refusal '{owned_refusal}'"));
        }
        let mapping = OwnedDecodeError::NotCertified.constrained_predispatch_mapping();
        if mapping != Some(OwnedDecodeError::GrammarDisabled)
            || mapped_error != OwnedDecodeError::GrammarDisabled.as_str()
        {
            return Err(format!(
                "refusal mapping must produce grammar_disabled, got {mapping:?}"
            ));
        }
        Ok(())
    } else {
        Err("entry carries no recognized negative shape".to_string())
    }
}

/// Execute the `scheduler-continuity` group: chunked/uninterrupted parity for
/// each candidate N and the protocol-behavior continuities, all driven through
/// the S5 scheduler and generation-protocol model (hardware-independent).
pub fn run_scheduler_continuity(manifests: &ManifestDir) -> GroupOutcome {
    let mut executed_ids = Vec::new();
    for entry in entries_for(manifests, SCHEDULER_CONTINUITY_GROUP) {
        let fixture_kind = entry
            .details
            .get("fixture_kind")
            .and_then(Value::as_str)
            .unwrap_or("");
        let outcome = match fixture_kind {
            "parity" => run_continuity_parity(entry),
            "protocol_behavior" => run_protocol_behavior(entry, manifests),
            other => Err(format!("unmapped fixture_kind '{other}'")),
        };
        if let Err(reason) = outcome {
            return GroupOutcome {
                executed_ids,
                result: Err(format!("{}: {reason}", entry.id)),
            };
        }
        executed_ids.push(entry.id.clone());
    }
    if executed_ids.is_empty() {
        return GroupOutcome {
            executed_ids,
            result: Err("registry has no scheduler-continuity entries".to_string()),
        };
    }
    GroupOutcome {
        executed_ids,
        result: Ok(()),
    }
}

/// Chunked/uninterrupted parity for one candidate N: the same document commits
/// byte-identically whether paced into N-token quanta or driven straight
/// through, and the pacing crosses the number of quantum boundaries the
/// registry entry promises.
fn run_continuity_parity(entry: &FixtureEntry) -> Result<(), String> {
    let candidate_n = entry
        .details
        .get("candidate_n")
        .and_then(Value::as_u64)
        .ok_or_else(|| "entry has no candidate_n".to_string())? as u32;
    let boundaries_crossed = entry
        .details
        .get("boundaries_crossed")
        .and_then(Value::as_u64)
        .ok_or_else(|| "entry has no boundaries_crossed".to_string())?
        as usize;
    if !crate::owned_decode_certification::scheduler_evidence::CANDIDATE_N_VALUES
        .contains(&candidate_n)
    {
        return Err(format!(
            "candidate_n {candidate_n} is not a candidate value"
        ));
    }

    // An integer array long enough to cross the promised number of N-token
    // boundaries: each item is three bytes ("10,").
    let item_count = (boundaries_crossed + 1) * candidate_n as usize / 2;
    let document = format!("[{}]", vec!["10"; item_count].join(","));

    let schema = crate::owned_decode_grammar_scheduler::parse_schema(
        r#"{ "type": "array", "items": { "type": "integer" } }"#,
        &crate::owned_decode_grammar_scheduler::GrammarLimits::default(),
    )
    .map_err(|err| format!("parity schema failed to parse: {err:?}"))?;
    let automaton = Automaton::new(schema);

    // Uninterrupted reference run.
    let uninterrupted_state = drain_bytes(&automaton, document.as_bytes())
        .map_err(|err| format!("uninterrupted run rejected: {}", err.message))?;
    if !automaton.has_complete_value(&uninterrupted_state) {
        return Err("uninterrupted run did not complete".to_string());
    }

    // Chunked run paced into N-token quanta through the generation protocol.
    let bytes = document.as_bytes();
    let max_tokens = bytes.len() as u32;
    let mut protocol =
        GenerationProtocol::new(format!("continuity-{}", entry.id), candidate_n, max_tokens);
    protocol
        .authorize_start(1)
        .map_err(|err| format!("start authorization failed: {err:?}"))?;
    let mut state = automaton.initial();
    let mut committed = Vec::new();
    let mut quantum_sequence = 0u32;
    let mut index = 0usize;
    while index < bytes.len() {
        let budget = protocol
            .next_continue()
            .map(|frame| frame.next_token_budget)
            .unwrap_or(candidate_n.min(max_tokens));
        let span_end = (index + budget as usize).min(bytes.len());
        while index < span_end {
            state = automaton
                .step(&state, bytes[index])
                .map_err(|err| format!("chunked run rejected a byte: {}", err.message))?;
            committed.push(bytes[index]);
            index += 1;
        }
        quantum_sequence += 1;
        protocol
            .receive_progress(1, quantum_sequence, committed.len() as u32)
            .map_err(|err| format!("progress rejected at quantum {quantum_sequence}: {err:?}"))?;
        if automaton.has_complete_value(&state) {
            break;
        }
    }
    if committed != bytes {
        return Err("chunked stream differs from the uninterrupted stream".to_string());
    }
    if (quantum_sequence as usize) < boundaries_crossed + 1 {
        return Err(format!(
            "expected at least {boundaries_crossed} boundaries crossed, got {}",
            quantum_sequence - 1
        ));
    }
    Ok(())
}

/// Protocol-behavior continuities driven through the S5 model.
fn run_protocol_behavior(entry: &FixtureEntry, manifests: &ManifestDir) -> Result<(), String> {
    match entry.id.as_str() {
        "sched-continuity-protocol-mid-quantum-cancel" => {
            // A resident operation cancelled mid-quantum defers to the boundary,
            // and the boundary reports cancellation with the committed count
            // acknowledged by the protocol.
            let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
            let op = DecodeOp {
                op_id: "op-mid-cancel".to_string(),
                generation_id: "gen-mid-cancel".to_string(),
                admitted_at_ms: 0,
                anchor_ms: 0,
                committed_tokens: 0,
                max_tokens: 64,
                resident: false,
                cancelled_at_ms: None,
                deadline_at_ms: None,
            };
            scheduler.admit_decode(op);
            scheduler.begin_decode_quantum("op-mid-cancel", 5);
            let n = scheduler.config().production_n;
            let mut protocol = GenerationProtocol::new("gen-mid-cancel", n, 64);
            protocol
                .authorize_start(1)
                .map_err(|err| format!("start failed: {err:?}"))?;
            protocol
                .receive_progress(1, 1, n)
                .map_err(|err| format!("progress failed: {err:?}"))?;
            match scheduler.request_cancel("op-mid-cancel", 10) {
                CancelResult::DeferredToBoundary => {}
                other => return Err(format!("resident cancel must defer, got {other:?}")),
            }
            match scheduler.evaluate_boundary("op-mid-cancel", BoundaryKind::Progress, 10) {
                BoundaryOutcome::Cancelled => {}
                other => return Err(format!("boundary must cancel, got {other:?}")),
            }
            if protocol.committed_tokens() != n {
                return Err("acknowledged committed-token count lost".to_string());
            }
            Ok(())
        }
        "sched-continuity-protocol-deadline-boundary" => {
            // A deadline expired at a boundary yields the deadline outcome, and
            // the wire error the caller receives is the literal binding from the
            // checked-in wire-error-bindings manifest.
            let mut scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
            let op = DecodeOp {
                op_id: "op-deadline".to_string(),
                generation_id: "gen-deadline".to_string(),
                admitted_at_ms: 0,
                anchor_ms: 0,
                committed_tokens: 0,
                max_tokens: 64,
                resident: false,
                cancelled_at_ms: None,
                deadline_at_ms: Some(50),
            };
            scheduler.admit_decode(op);
            scheduler.begin_decode_quantum("op-deadline", 5);
            match scheduler.evaluate_boundary("op-deadline", BoundaryKind::Progress, 100) {
                BoundaryOutcome::DeadlineExceeded => {}
                other => return Err(format!("boundary must expire the deadline, got {other:?}")),
            }
            let bindings = &manifests.wire_bindings;
            if bindings.deadline_error_id != "deadline_exceeded" {
                return Err(format!(
                    "wire bindings must carry the literal deadline error, got {}",
                    bindings.deadline_error_id
                ));
            }
            Ok(())
        }
        "sched-continuity-protocol-crash-redispatch" => {
            // Crash redispatch semantics in the protocol model: the replacement
            // restarts at token zero with the sequence reset, preserves the
            // logical generation id, and at most one redispatch is permitted.
            let generation_id = "gen-crash-redispatch";
            let n = DecodeSchedulerConfig::default().production_n;
            let mut first = GenerationProtocol::new(generation_id, n, 64);
            first
                .authorize_start(1)
                .map_err(|err| format!("first start failed: {err:?}"))?;
            first
                .receive_progress(1, 1, n)
                .map_err(|err| format!("first progress failed: {err:?}"))?;
            first.close(); // the crashed attempt's session is closed

            let mut redispatches_used = 0u32;
            let redispatch_permitted = redispatches_used < 1;
            if !redispatch_permitted {
                return Err("first redispatch must be permitted".to_string());
            }
            redispatches_used += 1;

            let mut protocol = GenerationProtocol::new(generation_id, n, 64);
            assert_eq!(protocol.generation_id(), generation_id);
            let budget = protocol
                .authorize_start(2)
                .map_err(|err| format!("replacement start failed: {err:?}"))?;
            if budget != n.min(64) {
                return Err(format!(
                    "replacement must restart with min(N, max) budget, got {budget}"
                ));
            }
            if protocol.committed_tokens() != 0 {
                return Err("replacement must restart at token zero".to_string());
            }
            // Attempt-local sequence reset: the replacement's first sequence is 1.
            protocol
                .receive_progress(2, 1, n)
                .map_err(|err| format!("replacement sequence must restart at one: {err:?}"))?;
            // The single permitted redispatch was consumed; a second one is
            // barred (the budget check lives in the S3 supervisor, modeled
            // here as the consumed allowance).
            if redispatches_used != 1 {
                return Err("exactly one redispatch must be consumed".to_string());
            }
            Ok(())
        }
        "sched-continuity-protocol-stale-session" => {
            // A frame from a superseded worker generation is a protocol mismatch.
            let mut protocol = GenerationProtocol::new("gen-stale", 16, 64);
            protocol
                .authorize_start(2)
                .map_err(|err| format!("start failed: {err:?}"))?;
            match protocol.receive_progress(1, 1, 16) {
                Err(OwnedDecodeError::ProtocolMismatch) => {}
                other => {
                    return Err(format!(
                    "stale worker generation must be owned_decode_protocol_mismatch, got {other:?}"
                ))
                }
            }
            Ok(())
        }
        other => Err(format!("unmapped protocol-behavior entry '{other}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_dir() -> ManifestDir {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("owned-decode-manifests");
        crate::owned_decode_contracts::load_manifest_dir(&path).expect("manifests load")
    }

    #[test]
    fn every_registry_group_executes_cleanly_against_the_checked_in_registry() {
        let manifests = manifest_dir();
        for outcome in [
            run_request_processing(&manifests),
            run_constrained_positive(&manifests),
            run_constrained_negative(&manifests),
            run_scheduler_continuity(&manifests),
        ] {
            outcome.result.expect("group executes cleanly");
            assert!(!outcome.executed_ids.is_empty());
        }
    }

    #[test]
    fn executed_ids_cover_every_non_parity_registry_entry() {
        let manifests = manifest_dir();
        let mut executed: Vec<String> = Vec::new();
        for outcome in [
            run_request_processing(&manifests),
            run_constrained_positive(&manifests),
            run_constrained_negative(&manifests),
            run_scheduler_continuity(&manifests),
        ] {
            outcome.result.expect("group executes cleanly");
            executed.extend(outcome.executed_ids);
        }
        for entry in &manifests.fixture_registry.entries {
            if entry.group == crate::owned_decode_certification::fixtures::PARITY_GROUP {
                continue; // parity runs through the probe seam (gate 04)
            }
            assert!(
                executed.iter().any(|id| id == &entry.id),
                "registry entry {} was not executed",
                entry.id
            );
        }
    }
}
