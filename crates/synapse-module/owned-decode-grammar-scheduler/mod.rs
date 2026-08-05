//! Grammar compilation and the dedicated DECODE scheduler for the production
//! owned-metal-decode lane.
//!
//! This module is the hardware-independent mechanism layer for two contracts:
//!
//! - **Grammar** (`grammar_contract`): the module exclusively owns grammar
//!   compilation. A request's `grammar` field is a JSON schema in the
//!   `synapse-json-schema-v1` subset; it is parsed and validated
//!   ([`grammar_schema`]), enforced against checked-in limits
//!   ([`grammar_limits`]), compiled into a byte-level JSON automaton
//!   ([`grammar_automaton`]), and shipped to the worker only as the versioned
//!   [`grammar_compile::TokenIdJsonConstraintV1`] representation. Raw schema or
//!   grammar never crosses the worker boundary.
//! - **Scheduler** (`scheduler_contract`): owned generation uses a dedicated
//!   [`scheduler::QueueClass::Decode`] with Control precedence, weighted boundary
//!   arbitration, oldest-anchor aging, a module-held execution permit with
//!   yield-on-contention release and reacquisition, FIFO continuation, queued
//!   cancellation and deadline removal, and N-token quantum sequencing
//!   ([`scheduler`]).
//!
//! The source lives under `crates/synapse-module/owned-decode-grammar-scheduler/`;
//! a `#[path]` attribute in the crate root wires that directory in as this module,
//! matching the way `owned-decode-routing` is wired.
//!
//! Everything here is pure Rust with no Metal dependency, so the mechanism and its
//! measurements can be exercised before the numeric scheduler values are committed
//! (the specification permits hardware-independent mechanism work to proceed).

pub mod grammar_automaton;
pub mod grammar_compile;
pub mod grammar_limits;
pub mod grammar_schema;
pub mod scheduler;

pub use grammar_compile::{
    compile_grammar, load_automaton, vocabulary_digest, CompileContext, CompiledConstraint,
    TokenIdJsonConstraintV1,
};
pub use grammar_limits::{GrammarLimits, GrammarSubsetManifest};
pub use grammar_schema::{parse_schema, Schema, SchemaError};
pub use scheduler::{
    Arbitration, BoundaryKind, BoundaryOutcome, CancelResult, ContinueFrame, DecodeOp,
    DecodeScheduler, DecodeSchedulerConfig, FinishReason, GenerationProtocol, Measurements,
    PermitEvent, QueueClass,
};

#[cfg(test)]
mod integration_tests {
    //! Cross-cutting tests that tie the grammar automaton and the quantum
    //! sequencer together: candidate-N chunked/uninterrupted parity, the four
    //! grammar-performance lanes, and end-to-end grammar correctness outcomes.

    use super::*;
    use crate::owned_decode_grammar_scheduler::grammar_automaton::{drain_bytes, mask_tokens};
    use crate::owned_decode_routing::error::OwnedDecodeError;
    use sha2::{Digest, Sha256};
    use std::time::Instant;

    fn automaton(raw: &str) -> grammar_automaton::Automaton {
        let schema = parse_schema(raw, &GrammarLimits::default()).expect("schema parses");
        grammar_automaton::Automaton::new(schema)
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex::encode(hasher.finalize())
    }

    /// The four grammar-performance lanes fixed by `grammar-cost-corpus-v1`.
    const FOUR_LANES: &[(&str, &str, &str)] = &[
        (
            "grammar-cost-object-small",
            r#"{
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "age": { "type": "integer" }
                },
                "required": ["name", "age"],
                "additionalProperties": false
            }"#,
            r#"{"name":"ada","age":36}"#,
        ),
        (
            "grammar-cost-array-strings",
            r#"{ "type": "array", "items": { "type": "string" } }"#,
            r#"["a","b","c"]"#,
        ),
        (
            "grammar-cost-nested-object",
            r#"{
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "address": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"],
                        "additionalProperties": false
                    }
                },
                "required": ["name", "address"],
                "additionalProperties": false
            }"#,
            r#"{"name":"ada","address":{"city":"paris"}}"#,
        ),
        (
            "grammar-cost-enum",
            r#"{ "type": "string", "enum": ["red", "green", "blue"] }"#,
            r#""green""#,
        ),
    ];

    /// Commit a fixed document one byte at a time, returning the committed byte
    /// stream. This is the "uninterrupted" reference run.
    fn commit_document(automaton: &grammar_automaton::Automaton, document: &str) -> Vec<u8> {
        let mut state = automaton.initial();
        let mut committed = Vec::new();
        for &byte in document.as_bytes() {
            state = automaton
                .step(&state, byte)
                .unwrap_or_else(|err| panic!("byte {} rejected: {}", byte as char, err.message));
            committed.push(byte);
        }
        assert!(
            automaton.has_complete_value(&state),
            "document did not complete a value"
        );
        committed
    }

    /// Commit the same document paced into N-token quanta, driving the quantum
    /// sequencer at each boundary. Returns the committed byte stream and the
    /// sequence-number trace. The committed bytes must be byte-identical to the
    /// uninterrupted run: host pacing does not change token output because the
    /// automaton state and selection are unchanged across quanta.
    fn commit_chunked(
        automaton: &grammar_automaton::Automaton,
        document: &str,
        n: u32,
    ) -> (Vec<u8>, Vec<u32>) {
        let bytes = document.as_bytes();
        let max_tokens = bytes.len() as u32;
        let mut protocol = GenerationProtocol::new("gen-parity", n, max_tokens);
        protocol.authorize_start(1).expect("starts");

        let mut state = automaton.initial();
        let mut committed = Vec::new();
        let mut sequence_trace = Vec::new();
        let mut quantum_sequence = 0u32;
        let mut index = 0usize;

        while index < bytes.len() {
            // Authorize one span of at most N bytes (tokens modeled as bytes).
            let budget = protocol
                .next_continue()
                .map(|frame| frame.next_token_budget)
                .unwrap_or(n.min(max_tokens));
            let span_end = (index + budget as usize).min(bytes.len());
            while index < span_end {
                let byte = bytes[index];
                state = automaton
                    .step(&state, byte)
                    .unwrap_or_else(|err| panic!("byte rejected: {}", err.message));
                committed.push(byte);
                index += 1;
            }
            quantum_sequence += 1;
            // Emit progress and validate the sequence at the boundary.
            protocol
                .receive_progress(1, quantum_sequence, committed.len() as u32)
                .expect("progress accepted");
            sequence_trace.push(quantum_sequence);
            if automaton.has_complete_value(&state) {
                break;
            }
        }
        (committed, sequence_trace)
    }

    #[test]
    fn chunked_uninterrupted_parity_for_all_candidate_n() {
        // Parity fixtures cross at least two quantum boundaries and prove
        // byte-identical chunked and uninterrupted streams for each candidate N
        // in {8,16,32} and for a committed N.
        for &(fixture_id, schema, document) in FOUR_LANES {
            let automaton = automaton(schema);
            let uninterrupted = commit_document(&automaton, document);
            let uninterrupted_digest = sha256_hex(&uninterrupted);
            for &n in crate::owned_decode_contracts::CANDIDATE_PRODUCTION_N {
                // Only exercise N values that cross at least two boundaries for this
                // document; shorter documents still prove single-span parity.
                let (chunked, trace) = commit_chunked(&automaton, document, n);
                let chunked_digest = sha256_hex(&chunked);
                assert_eq!(
                    uninterrupted_digest, chunked_digest,
                    "fixture {fixture_id} N={n}: stream digests differ"
                );
                assert_eq!(chunked, uninterrupted, "fixture {fixture_id} N={n}");
                // The sequence trace starts at one and increments by one.
                for (position, sequence) in trace.iter().enumerate() {
                    assert_eq!(
                        *sequence as usize,
                        position + 1,
                        "sequence trace contiguous"
                    );
                }
            }
        }
    }

    #[test]
    fn parity_crosses_at_least_two_boundaries_for_small_n() {
        // A document longer than 2N must cross at least two quantum boundaries at
        // the smallest candidate N, satisfying the fixture-registry requirement.
        let schema = r#"{ "type": "array", "items": { "type": "integer" } }"#;
        let document = "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]";
        let automaton = automaton(schema);
        let n = 8;
        let (chunked, trace) = commit_chunked(&automaton, document, n);
        assert_eq!(chunked, document.as_bytes());
        assert!(
            trace.len() >= 3,
            "expected at least two boundaries (three quanta), got {}",
            trace.len()
        );
    }

    #[test]
    fn four_grammar_performance_lanes_compile_and_mask_under_bound() {
        // Each grammar-cost lane compiles, generates its document with per-step
        // masking, and measures its masking latency. Every committed byte must
        // be permitted by the mask at its step.
        //
        // The constrained-throughput ratio is deliberately NOT asserted here:
        // both arms of this test commit the same fixed document, so any ratio
        // computed between them is 1.0 by construction and could never fail.
        // The real ratio gate is G-DEC-09: the `grammar-cost-corpus-v1`
        // manifest fixes fixtures, warmup, repetitions, sampling, and
        // percentile calculation, and a machine run records constrained vs
        // unconstrained owned-worker throughput there (see certification
        // `gate_09_grammar_cost`).
        //
        // The latency half is likewise a pre-check, not the gate: the
        // authoritative 0.50 ms/token p95 ship bound lives in the G-DEC-09
        // grammar-cost-corpus measurement protocol. A wall-clock p95 asserted
        // in a unit test on a shared runner conflates machine load with
        // regression, so the strict ship bound is asserted only when this
        // test runs explicitly as the measurement gate
        // (SYNAPSE_GRAMMAR_LATENCY_GATE=1); the default CI run asserts a
        // loose sanity ceiling (10x the ship bound) that catches algorithmic
        // blowups but not scheduler noise.
        let vocabulary: Vec<String> = (0x20u8..=0x7e)
            .map(|byte| (byte as char).to_string())
            .collect();
        let ship_bound_ms = 0.50_f64;
        let latency_gate = std::env::var("SYNAPSE_GRAMMAR_LATENCY_GATE").as_deref() == Ok("1");

        for &(fixture_id, schema, document) in FOUR_LANES {
            let automaton = automaton(schema);

            // Constrained run: mask the vocabulary at each step before committing.
            let mut state = automaton.initial();
            let mut masking_samples_ms: Vec<f64> = Vec::new();
            for &byte in document.as_bytes() {
                let start = Instant::now();
                let permitted = mask_tokens(&automaton, &state, &vocabulary);
                masking_samples_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                assert!(
                    permitted
                        .iter()
                        .any(|&index| vocabulary[index].as_bytes()[0] == byte),
                    "fixture {fixture_id}: committed byte must be permitted by masking"
                );
                state = automaton.step(&state, byte).expect("commit permitted byte");
            }
            assert!(
                automaton.has_complete_value(&state),
                "fixture {fixture_id} completes"
            );

            // Nearest-rank p95 masking latency, reported for every run.
            masking_samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let rank = ((masking_samples_ms.len() as f64) * 0.95).ceil() as usize;
            let p95 = masking_samples_ms[(rank.max(1) - 1).min(masking_samples_ms.len() - 1)];
            println!(
                "fixture {fixture_id}: p95 masking latency {p95:.4} ms/token (ship bound {ship_bound_ms} ms)"
            );
            if latency_gate {
                assert!(
                    p95 < ship_bound_ms,
                    "fixture {fixture_id}: p95 masking {p95}ms exceeded the {ship_bound_ms}ms ship bound"
                );
            } else {
                let sanity_ceiling_ms = ship_bound_ms * 10.0;
                assert!(
                    p95 < sanity_ceiling_ms,
                    "fixture {fixture_id}: p95 masking {p95}ms suggests an algorithmic blowup \
                     (sanity ceiling {sanity_ceiling_ms}ms; the ship bound is enforced by the \
                     G-DEC-09 grammar-cost-corpus gate)"
                );
            }
        }
    }

    // -- end-to-end grammar correctness outcomes --

    /// A tiny deterministic generator modeling the worker's greedy loop over the
    /// union of permitted content tokens and stop-token control candidates.
    #[derive(Debug)]
    enum GenOutcome {
        /// The automaton completed a value: `finish_reason=grammar_complete`.
        Complete(Vec<u8>),
        /// A stop candidate won greedy selection while the automaton was
        /// incomplete: `grammar_stop_before_completion`. The stop token is a
        /// non-committed control candidate, so it never appears in the
        /// committed bytes.
        StopBeforeCompletion(Vec<u8>),
        /// No content token and no stop candidate selectable.
        Unsatisfiable,
        MaxTokensExhausted(Vec<u8>),
    }

    /// Generate by greedily committing the lowest-index permitted content token,
    /// modeling the worker's greedy selection over the union of permitted content
    /// tokens and stop-token control candidates. `stop_bytes` are the configured
    /// stop IDs as non-committed control candidates: they are selectable at any
    /// step but never committed. When a stop candidate wins a step, the outcome
    /// is `grammar_complete` if the automaton already completed a value and
    /// `grammar_stop_before_completion` otherwise. Content tokens win the steps
    /// whose plan byte is not a stop candidate.
    fn greedy_generate(
        automaton: &grammar_automaton::Automaton,
        vocabulary: &[String],
        plan: &[u8],
        stop_bytes: &[u8],
        max_tokens: usize,
    ) -> GenOutcome {
        let mut state = automaton.initial();
        let mut committed = Vec::new();
        for &byte in plan {
            if committed.len() >= max_tokens {
                return GenOutcome::MaxTokensExhausted(committed);
            }
            // Stop candidates are control candidates in the greedy union: when
            // one wins, it is never committed.
            if stop_bytes.contains(&byte) {
                return if automaton.has_complete_value(&state) {
                    // Stop winning after completion is a clean grammar_complete.
                    GenOutcome::Complete(committed)
                } else {
                    GenOutcome::StopBeforeCompletion(committed)
                };
            }
            let permitted = mask_tokens(automaton, &state, vocabulary);
            if permitted.is_empty() {
                return GenOutcome::Unsatisfiable;
            }
            if !permitted
                .iter()
                .any(|&index| vocabulary[index].as_bytes() == [byte])
            {
                // The planned byte is not selectable; treat as unsatisfiable for the
                // constrained path.
                return GenOutcome::Unsatisfiable;
            }
            state = automaton
                .step(&state, byte)
                .expect("planned byte permitted");
            committed.push(byte);
        }
        if automaton.has_complete_value(&state) {
            GenOutcome::Complete(committed)
        } else if committed.len() >= max_tokens {
            GenOutcome::MaxTokensExhausted(committed)
        } else {
            GenOutcome::Unsatisfiable
        }
    }

    #[test]
    fn grammar_complete_on_valid_object() {
        let schema = r#"{
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
            "additionalProperties": false
        }"#;
        let automaton = automaton(schema);
        let vocabulary: Vec<String> = (0x20u8..=0x7e)
            .map(|byte| (byte as char).to_string())
            .collect();
        let plan = br#"{"name":"ada"}"#.to_vec();
        match greedy_generate(&automaton, &vocabulary, &plan, &[], 64) {
            GenOutcome::Complete(committed) => assert_eq!(committed, plan),
            other => panic!("expected grammar_complete, got {other:?}"),
        }
    }

    #[test]
    fn grammar_max_tokens_exhausted_before_completion() {
        let schema = r#"{
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
            "additionalProperties": false
        }"#;
        let automaton = automaton(schema);
        let vocabulary: Vec<String> = (0x20u8..=0x7e)
            .map(|byte| (byte as char).to_string())
            .collect();
        let plan = br#"{"name":"adelaide"}"#.to_vec();
        // A budget shorter than the document exhausts before the value completes.
        match greedy_generate(&automaton, &vocabulary, &plan, &[], 5) {
            GenOutcome::MaxTokensExhausted(committed) => assert_eq!(committed.len(), 5),
            other => panic!("expected grammar_max_tokens_exhausted, got {other:?}"),
        }
    }

    #[test]
    fn grammar_unsatisfiable_when_no_content_token_selectable() {
        let schema = r#"{ "type": "string" }"#;
        let automaton = automaton(schema);
        // A vocabulary with no string-opening quote has no selectable content token
        // at the start, so the constrained path is unsatisfiable.
        let vocabulary = vec!["{".to_string(), "[".to_string(), "1".to_string()];
        let plan = br#""hello""#.to_vec();
        match greedy_generate(&automaton, &vocabulary, &plan, &[], 64) {
            GenOutcome::Unsatisfiable => {}
            other => panic!("expected grammar_unsatisfiable, got {other:?}"),
        }
    }

    #[test]
    fn stop_before_completion_is_typed() {
        // A stop token winning while the automaton is incomplete must surface as
        // grammar_stop_before_completion, and the stop token is not committed.
        let schema = r#"{ "type": "string" }"#;
        let automaton = automaton(schema);
        let state = automaton.initial();
        // After committing only the opening quote, the value is incomplete; a stop
        // token selected here is a premature stop.
        let after_quote = automaton.step(&state, b'"').expect("open quote");
        assert!(!automaton.has_complete_value(&after_quote));
        // The worker models "stop wins while incomplete" as this typed error.
        let outcome = OwnedDecodeError::GrammarStopBeforeCompletion;
        assert_eq!(outcome.as_str(), "grammar_stop_before_completion");
        assert!(outcome.is_grammar());
    }

    #[test]
    fn enum_lane_only_permits_members() {
        let schema = r#"{ "type": "string", "enum": ["red", "green", "blue"] }"#;
        let automaton = automaton(schema);
        let vocabulary: Vec<String> = (0x20u8..=0x7e)
            .map(|byte| (byte as char).to_string())
            .collect();
        // A plan spelling a non-member is rejected by masking (unsatisfiable path).
        let bad_plan = br#""yellow""#.to_vec();
        match greedy_generate(&automaton, &vocabulary, &bad_plan, &[], 64) {
            GenOutcome::Unsatisfiable => {}
            other => panic!("expected non-member to be masked out, got {other:?}"),
        }
        // A member plan completes.
        let good_plan = br#""blue""#.to_vec();
        match greedy_generate(&automaton, &vocabulary, &good_plan, &[], 64) {
            GenOutcome::Complete(committed) => assert_eq!(committed, good_plan),
            other => panic!("expected member to complete, got {other:?}"),
        }
    }

    #[test]
    fn stop_winning_while_incomplete_is_stop_before_completion() {
        // A stop candidate winning greedy selection while the automaton is
        // incomplete surfaces grammar_stop_before_completion, and the stop
        // candidate is never committed.
        let schema = r#"{ "type": "string" }"#;
        let automaton = automaton(schema);
        let vocabulary: Vec<String> = (0x20u8..=0x7e)
            .map(|byte| (byte as char).to_string())
            .collect();
        // Commit only the opening quote (incomplete value), then the stop
        // candidate wins the next selection.
        let plan = vec![b'"', b'~'];
        match greedy_generate(&automaton, &vocabulary, &plan, b"~", 64) {
            GenOutcome::StopBeforeCompletion(committed) => {
                assert_eq!(committed, vec![b'"'], "stop token is not committed");
                assert!(!committed.contains(&b'~'));
            }
            other => panic!("expected grammar_stop_before_completion, got {other:?}"),
        }
    }

    #[test]
    fn stop_winning_after_completion_is_clean_grammar_complete() {
        // A stop candidate winning after the value completed is a clean
        // grammar_complete, not grammar_stop_before_completion; the stop
        // candidate is still never committed.
        let schema = r#"{ "type": "string" }"#;
        let automaton = automaton(schema);
        let vocabulary: Vec<String> = (0x20u8..=0x7e)
            .map(|byte| (byte as char).to_string())
            .collect();
        let document = br#""hi""#;
        let mut plan = document.to_vec();
        plan.push(b'~'); // stop candidate wins the selection after completion
        match greedy_generate(&automaton, &vocabulary, &plan, b"~", 64) {
            GenOutcome::Complete(committed) => {
                assert_eq!(committed, document, "only content is committed");
            }
            other => panic!("expected clean grammar_complete, got {other:?}"),
        }
    }

    #[test]
    fn content_wins_while_stop_candidate_is_selectable() {
        // With a stop candidate configured, content tokens keep winning the
        // steps before the stop is selected: the document completes normally.
        let schema = r#"{
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
            "additionalProperties": false
        }"#;
        let automaton = automaton(schema);
        let vocabulary: Vec<String> = (0x20u8..=0x7e)
            .map(|byte| (byte as char).to_string())
            .collect();
        let plan = br#"{"name":"ada"}"#.to_vec();
        // The stop byte `~` never appears in the plan, so content wins every
        // step even though the stop candidate is selectable at each one.
        match greedy_generate(&automaton, &vocabulary, &plan, b"~", 64) {
            GenOutcome::Complete(committed) => assert_eq!(committed, plan),
            other => panic!("expected content to win and complete, got {other:?}"),
        }
    }

    #[test]
    fn compiled_constraint_drives_generation_end_to_end() {
        // Compile a grammar to the wire representation, reload the automaton from
        // the shipped bytes (the worker load path), and generate a valid document.
        let manifest = GrammarSubsetManifest::default();
        let context = CompileContext {
            base_decode_fingerprint: synapse_core::Fingerprint("base-fp".to_string()),
            tokenizer_vocabulary_digest: vocabulary_digest(&["a".to_string(), "\"".to_string()]),
        };
        let schema = r#"{
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
            "additionalProperties": false
        }"#;
        let compiled = compile_grammar(schema, &context, &manifest).expect("compiles");
        let automaton = load_automaton(&compiled.constraint, &manifest).expect("loads");
        let state = drain_bytes(&automaton, br#"{"name":"ada"}"#).expect("accepted");
        assert!(automaton.has_complete_value(&state));
    }
}
