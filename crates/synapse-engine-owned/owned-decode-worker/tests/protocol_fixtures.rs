//! Protocol-behavior fixture battery for `owned-metal-decode-worker-v1`.
//!
//! These fixtures are the hardware-independent half of the `protocol_behavior`
//! fixture group. They prove the acceptance criteria that do not require a Metal
//! GPU:
//! - dedicated, non-overlapping mismatch mappings (runtime, constraint, protocol,
//!   sampling);
//! - timeout classifications and their crash-budget consequences;
//! - terminal-control precedence (completion > cancellation > deadline);
//! - literal cleanup-timeout and cancellation wire errors (no symbolic
//!   placeholders);
//! - sequence and session violations, malformed frames, and stop controls;
//! - the single permitted worker-crash redispatch: token-zero restart,
//!   attempt-local sequence reset, rejection of delayed prior-session frames,
//!   preservation of the logical `generation_id`, and at most one redispatch.
//!
//! The Metal-dependent parity and throughput fixtures run on the mandatory
//! `macos-metal` lane; these run everywhere.

use owned_decode_worker::{
    evaluate_boundary, validate_start, BoundaryDecision, BoundaryInputs, BudgetPolicy, CrashBudget,
    DecodeError, FailureClassification, FinishReason, FrameEnvelope, GenerateStart,
    GenerationRequest, InMemoryBudgetStore, ManualClock, QuarantineKey, Sampling, ScriptedEvent,
    ScriptedWorkerFactory, Supervisor, TerminalControl, WorkerStartContext,
};

const N: u32 = 16;

fn context() -> WorkerStartContext {
    WorkerStartContext {
        loaded_model_ref: "model-qwen3-f16".into(),
        decode_fingerprint: "dfp-qwen3-f16".into(),
        runtime_config_digest: "rt-digest".into(),
        expected_constraint: None,
    }
}

fn start() -> GenerateStart {
    GenerateStart {
        generation_id: "gen-1".into(),
        loaded_model_ref: "model-qwen3-f16".into(),
        decode_fingerprint: "dfp-qwen3-f16".into(),
        runtime_config_digest: "rt-digest".into(),
        prompt_ids: vec![10, 11, 12, 13],
        stop_ids: vec![2],
        max_tokens: 64,
        sampling: Sampling::greedy_top1(),
        constraint: None,
    }
}

fn request() -> GenerationRequest {
    GenerationRequest {
        key: QuarantineKey::new("machine-profile", "dfp-qwen3-f16", "rt-digest"),
        start: start(),
    }
}

fn supervisor() -> Supervisor<InMemoryBudgetStore> {
    Supervisor::new(
        CrashBudget::new(InMemoryBudgetStore::default(), BudgetPolicy::default()),
        N,
    )
}

fn final_event(finish: FinishReason, ids: Vec<u32>) -> ScriptedEvent {
    ScriptedEvent::Final {
        finish,
        ids,
        constraint_complete: false,
    }
}

// ---------------------------------------------------------------------------
// Dedicated, non-overlapping mismatch mappings (resolution r2 #7).
// ---------------------------------------------------------------------------

#[test]
fn fixture_runtime_config_mismatch_is_dedicated() {
    // A loaded-model / decode-fingerprint / runtime-digest mismatch at worker
    // start maps to owned_decode_runtime_config_mismatch, never protocol or
    // constraint.
    let mut bad = start();
    bad.runtime_config_digest = "stale-digest".into();
    let error = validate_start(&bad, &context(), N).unwrap_err();
    assert_eq!(error, DecodeError::RuntimeConfigMismatch);
    assert_eq!(error.as_str(), "owned_decode_runtime_config_mismatch");
}

#[test]
fn fixture_sampling_mismatch_is_dedicated() {
    let mut bad = start();
    bad.sampling.mode = "nucleus".into();
    let error = validate_start(&bad, &context(), N).unwrap_err();
    assert_eq!(error, DecodeError::SamplingUnsupported);
    assert_eq!(error.as_str(), "owned_decode_sampling_unsupported");
}

#[test]
fn fixture_protocol_mismatch_is_dedicated_for_frame_structure() {
    // A structurally invalid start (empty generation id) is a protocol mismatch,
    // not a runtime or constraint mismatch.
    let mut bad = start();
    bad.generation_id = String::new();
    assert_eq!(
        validate_start(&bad, &context(), N),
        Err(DecodeError::ProtocolMismatch)
    );

    // A foreign protocol ID on the wire is a protocol mismatch.
    let foreign = "{\"protocol\":\"llama-generate-v0\",\"kind\":\"progress\",\"generation_id\":\"g\",\"quantum_sequence\":1,\"committed_token_count\":1}";
    assert_eq!(
        FrameEnvelope::from_wire(foreign),
        Err(DecodeError::ProtocolMismatch)
    );
}

#[test]
fn fixture_constraint_mismatch_is_dedicated_for_every_field() {
    use owned_decode_worker::TokenIdJsonConstraint;
    let mut ctx = context();
    let base = owned_decode_worker::protocol::sample_constraint();
    ctx.expected_constraint = Some(base.clone());

    // Perturb each field in turn; every one maps to the constraint mismatch ID.
    type Perturb = fn(&mut TokenIdJsonConstraint);
    let perturbations: Vec<(&str, Perturb)> = vec![
        ("encoding_id", |c| c.encoding_id = "x".into()),
        ("constraint_runtime_identity", |c| {
            c.constraint_runtime_identity = "x".into()
        }),
        ("constraint_fingerprint", |c| {
            c.constraint_fingerprint = "x".into()
        }),
        ("grammar_subset_revision", |c| {
            c.grammar_subset_revision = "x".into()
        }),
        ("grammar_compiler_revision", |c| {
            c.grammar_compiler_revision = "x".into()
        }),
        ("tokenizer_vocabulary_digest", |c| {
            c.tokenizer_vocabulary_digest = "x".into()
        }),
        ("limits_manifest_id", |c| c.limits_manifest_id = "x".into()),
        ("worker_constraint_runtime_revision", |c| {
            c.worker_constraint_runtime_revision = "x".into()
        }),
        ("canonical_schema_digest", |c| {
            c.canonical_schema_digest = "x".into()
        }),
        ("initial_state_encoding", |c| {
            c.initial_state_encoding = "x".into()
        }),
        ("initial_state_digest", |c| {
            c.initial_state_digest = "x".into()
        }),
        ("compiled_automaton_digest", |c| {
            c.compiled_automaton_digest = "x".into()
        }),
        ("automaton_bytes", |c| c.automaton_bytes = vec![0]),
    ];
    for (field, perturb) in perturbations {
        let mut bad = base.clone();
        perturb(&mut bad);
        let mut s = start();
        s.constraint = Some(bad);
        assert_eq!(
            validate_start(&s, &ctx, N),
            Err(DecodeError::ConstraintVersionMismatch),
            "field {field} should map to constraint mismatch"
        );
    }
}

// ---------------------------------------------------------------------------
// Timeout classification and crash-budget consequences.
// ---------------------------------------------------------------------------

#[test]
fn fixture_timeout_charges_once_and_is_terminal() {
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![
            vec![ScriptedEvent::Timeout],
            vec![final_event(FinishReason::StopToken, vec![1])],
        ],
        context(),
    );
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &request(),
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    // Timeout is terminal: no redispatch even though a second script exists.
    assert_eq!(outcome.result, Err(DecodeError::Unavailable));
    assert_eq!(outcome.provenance.crash_retry_count, 0);
    assert_eq!(
        outcome.provenance.failure_classifications,
        vec![FailureClassification::Timeout]
    );
    assert_eq!(factory.spawn_count(), 1);
    // Exactly one unit charged.
    assert_eq!(sup.budget().remaining(&request().key), 1);
}

#[test]
fn fixture_timeout_with_single_strike_policy_quarantines() {
    let mut sup = Supervisor::new(
        CrashBudget::new(InMemoryBudgetStore::default(), BudgetPolicy::new(1, 60_000)),
        N,
    );
    let mut factory = ScriptedWorkerFactory::new(vec![vec![ScriptedEvent::Timeout]], context());
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &request(),
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    assert_eq!(outcome.result, Err(DecodeError::Quarantined));
    assert!(sup.budget().is_quarantined(&request().key, 0));
}

// ---------------------------------------------------------------------------
// Terminal-control precedence and literal wire errors.
// ---------------------------------------------------------------------------

#[test]
fn fixture_terminal_completion_beats_deadline_and_cancellation() {
    // A natural completion during a quantum whose deadline has expired still
    // succeeds; the deadline does not retroactively fail it.
    let decision = evaluate_boundary(BoundaryInputs {
        completion: Some(FinishReason::GrammarComplete),
        cancel_recorded_at: Some(10),
        deadline_at: Some(20),
        observed_at: 100,
    });
    assert_eq!(
        decision,
        BoundaryDecision::AcceptCompletion(FinishReason::GrammarComplete)
    );
}

#[test]
fn fixture_cancellation_beats_deadline_at_non_terminal_boundary() {
    // Binding precedence (spec resolutions round 2, #4): terminal completion >
    // cancellation > deadline. A caller who abandoned the operation receives
    // `cancelled` even though the deadline also expired during the quantum.
    let decision = evaluate_boundary(BoundaryInputs {
        completion: None,
        cancel_recorded_at: Some(10),
        deadline_at: Some(20),
        observed_at: 100,
    });
    assert_eq!(decision, BoundaryDecision::Cancelled);
}

#[test]
fn fixture_deadline_cleanup_uses_literal_wire_error_and_no_budget() {
    let mut sup = supervisor();
    // The worker would emit progress; the deadline has already expired at the
    // boundary, so the supervisor suppresses the payload and returns the literal
    // deadline error.
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![ScriptedEvent::Progress { committed: 16 }]],
        context(),
    );
    let clock = ManualClock::new(100);
    let control = TerminalControl {
        deadline_at: Some(50), // expired before the boundary at 100
        cancel_at: None,
    };
    let outcome = sup.run_generation(&request(), &mut factory, &context(), &control, &clock);
    assert_eq!(outcome.result, Err(DecodeError::DeadlineExceeded));
    // The literal wire ID, never a symbolic placeholder.
    assert_eq!(outcome.result.unwrap_err().as_str(), "deadline_exceeded");
    // Deadline cleanup before timeout consumes no crash budget.
    assert_eq!(sup.budget().remaining(&request().key), 2);
    assert_eq!(outcome.provenance.failure_classifications, vec![]);
    // The supervisor cancelled the worker during cleanup.
    assert_eq!(factory.log().cancels, 1);
}

#[test]
fn fixture_cancellation_uses_literal_wire_error_and_no_budget() {
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![ScriptedEvent::Progress { committed: 16 }]],
        context(),
    );
    let clock = ManualClock::new(100);
    let control = TerminalControl {
        deadline_at: Some(500), // not expired
        cancel_at: Some(50),    // recorded before the boundary
    };
    let outcome = sup.run_generation(&request(), &mut factory, &context(), &control, &clock);
    assert_eq!(outcome.result, Err(DecodeError::Cancelled));
    assert_eq!(outcome.result.unwrap_err().as_str(), "cancelled");
    // Acknowledged cancellation consumes no crash budget.
    assert_eq!(sup.budget().remaining(&request().key), 2);
    assert_eq!(factory.log().cancels, 1);
}

// ---------------------------------------------------------------------------
// Sequence and session violations.
// ---------------------------------------------------------------------------

#[test]
fn fixture_repeated_sequence_is_protocol_fatal() {
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![
            ScriptedEvent::Progress { committed: 16 },
            // Repeats sequence 1 (the worker assigns 2 here, so force it).
            ScriptedEvent::ProgressWithSequence {
                sequence: 1,
                committed: 32,
            },
        ]],
        context(),
    );
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &request(),
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    // A repeated sequence is protocol-fatal: charged, terminal, unavailable.
    assert_eq!(outcome.result, Err(DecodeError::Unavailable));
    assert_eq!(
        outcome.provenance.failure_classifications,
        vec![FailureClassification::ProtocolFatal]
    );
}

#[test]
fn fixture_unknown_generation_id_is_protocol_fatal() {
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![ScriptedEvent::ProgressForeignGeneration {
            committed: 16,
        }]],
        context(),
    );
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &request(),
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    assert_eq!(outcome.result, Err(DecodeError::Unavailable));
    assert_eq!(
        outcome.provenance.failure_classifications,
        vec![FailureClassification::ProtocolFatal]
    );
}

#[test]
fn fixture_non_advancing_committed_count_is_protocol_fatal() {
    // The cumulative committed count must advance (parity with the S5
    // GenerationProtocol validator): a repeated count is a protocol fault,
    // charged as protocol-fatal.
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![
            ScriptedEvent::Progress { committed: 16 },
            ScriptedEvent::Progress { committed: 16 }, // does not advance
        ]],
        context(),
    );
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &request(),
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    assert_eq!(outcome.result, Err(DecodeError::Unavailable));
    assert_eq!(
        outcome.provenance.failure_classifications,
        vec![FailureClassification::ProtocolFatal]
    );
}

#[test]
fn fixture_skipped_sequence_is_protocol_fatal() {
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![ScriptedEvent::ProgressWithSequence {
            sequence: 5, // supervisor expects 1
            committed: 16,
        }]],
        context(),
    );
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &request(),
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    assert_eq!(outcome.result, Err(DecodeError::Unavailable));
    assert_eq!(
        outcome.provenance.failure_classifications,
        vec![FailureClassification::ProtocolFatal]
    );
}

#[test]
fn fixture_stale_session_frame_is_rejected() {
    // After a crash redispatch, a delayed frame tagged with the old worker
    // generation must be rejected as protocol-fatal.
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![
            vec![ScriptedEvent::Crash],
            vec![ScriptedEvent::StaleFinal {
                generation: 1, // the crashed worker's generation, not the current (2)
                finish: FinishReason::StopToken,
            }],
        ],
        context(),
    );
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &request(),
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    // First crash charges one and redispatches; the stale frame on the second
    // attempt is protocol-fatal, charges one more, exhausts, and quarantines.
    assert_eq!(outcome.result, Err(DecodeError::Quarantined));
    assert_eq!(outcome.provenance.crash_retry_count, 1);
    assert_eq!(
        outcome.provenance.failure_classifications,
        vec![
            FailureClassification::Crash,
            FailureClassification::ProtocolFatal
        ]
    );
}

#[test]
fn fixture_malformed_wire_frame_is_protocol_mismatch() {
    assert_eq!(
        FrameEnvelope::from_wire("{ this is not json"),
        Err(DecodeError::ProtocolMismatch)
    );
    assert_eq!(
        FrameEnvelope::from_wire("{\"protocol\":\"owned-metal-decode-worker-v1\"}"),
        Err(DecodeError::ProtocolMismatch)
    );
}

// ---------------------------------------------------------------------------
// Stop controls and finish reasons.
// ---------------------------------------------------------------------------

#[test]
fn fixture_every_finish_reason_is_accepted() {
    for (finish, ids) in [
        (FinishReason::StopToken, vec![100, 101]),
        (FinishReason::MaxTokens, vec![100, 101, 102]),
        (FinishReason::GrammarComplete, vec![100]),
    ] {
        let mut sup = supervisor();
        let mut factory =
            ScriptedWorkerFactory::new(vec![vec![final_event(finish, ids.clone())]], context());
        let clock = ManualClock::new(0);
        let outcome = sup.run_generation(
            &request(),
            &mut factory,
            &context(),
            &TerminalControl::default(),
            &clock,
        );
        let success = outcome.result.expect("success");
        assert_eq!(success.finish_reason, finish);
        assert_eq!(success.generated_ids, ids);
    }
}

#[test]
fn fixture_stop_controls_are_omitted_from_generated_ids() {
    // Stop-token omission is worker-side selection behavior, not pass-through:
    // the scripted double models the greedy union of content tokens and stop
    // control candidates (reference semantics: the S5 grammar-scheduler
    // `greedy_generate` stop union; production selection is owned by the real
    // Metal worker). The stop candidate wins the final selection here, so it
    // must be omitted from generated_ids and from the committed count. A
    // double that committed the winning stop would fail this fixture.
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![ScriptedEvent::StopSelectionWins {
            content_ids: vec![100, 101],
            stop_id: 2,
        }]],
        context(),
    );
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &request(),
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    let success = outcome.result.expect("success");
    assert_eq!(success.finish_reason, FinishReason::StopToken);
    assert!(
        !success.generated_ids.contains(&2),
        "stop control must be omitted from generated ids"
    );
    assert_eq!(
        success.generated_ids,
        vec![100, 101],
        "only content tokens are committed"
    );
    assert_eq!(
        success.committed_token_count, 2,
        "the winning stop token is not counted as committed"
    );
}

// ---------------------------------------------------------------------------
// Progress / continuation framing and remaining-budget truncation.
// ---------------------------------------------------------------------------

#[test]
fn fixture_multi_quantum_progress_sequence_trace() {
    // Cross two quantum boundaries: progress seq 1, continue, progress seq 2,
    // continue, final. The sequence trace is exactly 1, 2 and continuations carry
    // the expected next sequences.
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![
            ScriptedEvent::Progress { committed: 16 },
            ScriptedEvent::Progress { committed: 32 },
            final_event(FinishReason::StopToken, vec![1; 40]),
        ]],
        context(),
    );
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &request(),
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    let success = outcome.result.expect("success");
    assert_eq!(success.committed_token_count, 40);
    let log = factory.log();
    assert_eq!(log.continue_sequences, vec![2, 3]);
    assert_eq!(log.continue_budgets, vec![16, 16]);
}

#[test]
fn fixture_remaining_budget_truncates_continuation() {
    // max_tokens=20, N=16: the first quantum authorizes 16, leaving 4. The
    // continuation budget must truncate to the remaining request budget (4).
    let mut sup = supervisor();
    let mut req = request();
    req.start.max_tokens = 20;
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![
            ScriptedEvent::Progress { committed: 16 },
            final_event(FinishReason::MaxTokens, vec![1; 20]),
        ]],
        context(),
    );
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &req,
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    outcome.result.expect("success");
    let log = factory.log();
    assert_eq!(
        log.continue_budgets,
        vec![4],
        "continuation truncates to remaining budget"
    );
}

#[test]
fn fixture_first_quantum_authorizes_min_n_max_tokens() {
    // With max_tokens < N, the first quantum is truncated to max_tokens.
    let auth = validate_start(
        &{
            let mut s = start();
            s.max_tokens = 5;
            s
        },
        &context(),
        N,
    )
    .expect("ok");
    assert_eq!(auth.first_quantum_budget, 5);
}

// ---------------------------------------------------------------------------
// Crash redispatch semantics.
// ---------------------------------------------------------------------------

#[test]
fn fixture_crash_redispatch_restarts_at_token_zero_and_resets_sequence() {
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![
            // First attempt commits some tokens, then crashes.
            vec![
                ScriptedEvent::Progress { committed: 16 },
                ScriptedEvent::Crash,
            ],
            // Replacement restarts from the original prompt: sequence resets to 1
            // and committed count restarts from zero.
            vec![
                ScriptedEvent::Progress { committed: 16 },
                final_event(FinishReason::StopToken, vec![9; 24]),
            ],
        ],
        context(),
    );
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &request(),
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    let success = outcome.result.expect("redispatch succeeds");
    assert_eq!(outcome.provenance.crash_retry_count, 1);
    // The replacement ran on a new worker generation.
    assert_eq!(success.worker_generation, 2);
    // The continuation sequence on the replacement restarted at 2 (after seq 1).
    let log = factory.log();
    // First attempt sent continue seq 2; replacement sent continue seq 2 again
    // (attempt-local reset), then completed.
    assert_eq!(log.continue_sequences, vec![2, 2]);
}

#[test]
fn fixture_at_most_one_redispatch() {
    // Crash, redispatch, crash again: the second crash is terminal. No third
    // spawn occurs.
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![
            vec![ScriptedEvent::Crash],
            vec![ScriptedEvent::Crash],
            vec![final_event(FinishReason::StopToken, vec![1])],
        ],
        context(),
    );
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &request(),
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    assert_eq!(outcome.result, Err(DecodeError::Quarantined));
    assert_eq!(outcome.provenance.crash_retry_count, 1);
    assert_eq!(factory.spawn_count(), 2, "no third worker is spawned");
}

#[test]
fn fixture_crash_redispatch_barred_by_deadline_returns_deadline() {
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![
            vec![ScriptedEvent::Crash],
            vec![final_event(FinishReason::StopToken, vec![1])],
        ],
        context(),
    );
    // The deadline is already invalid at the crash boundary, so redispatch is
    // barred and the bound deadline error is returned.
    let clock = ManualClock::new(100);
    let control = TerminalControl {
        deadline_at: Some(50),
        cancel_at: None,
    };
    let outcome = sup.run_generation(&request(), &mut factory, &context(), &control, &clock);
    assert_eq!(outcome.result, Err(DecodeError::DeadlineExceeded));
    assert_eq!(outcome.provenance.crash_retry_count, 0);
    assert_eq!(factory.spawn_count(), 1);
}

#[test]
fn fixture_crash_redispatch_barred_by_cancellation_returns_cancelled() {
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![
            vec![ScriptedEvent::Crash],
            vec![final_event(FinishReason::StopToken, vec![1])],
        ],
        context(),
    );
    let clock = ManualClock::new(100);
    let control = TerminalControl {
        deadline_at: Some(500),
        cancel_at: Some(50), // cancelled before the crash boundary
    };
    let outcome = sup.run_generation(&request(), &mut factory, &context(), &control, &clock);
    assert_eq!(outcome.result, Err(DecodeError::Cancelled));
    assert_eq!(outcome.provenance.crash_retry_count, 0);
}

// ---------------------------------------------------------------------------
// Startup failure and failed cancellation.
// ---------------------------------------------------------------------------

#[test]
fn fixture_replacement_startup_failure_is_terminal() {
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![final_event(FinishReason::StopToken, vec![1])]],
        context(),
    );
    factory.fail_spawn_at = Some(0); // the initial spawn fails to start
    let clock = ManualClock::new(0);
    let outcome = sup.run_generation(
        &request(),
        &mut factory,
        &context(),
        &TerminalControl::default(),
        &clock,
    );
    // Startup failure charges one and is terminal; budget not exhausted -> unavailable.
    assert_eq!(outcome.result, Err(DecodeError::Unavailable));
    assert_eq!(
        outcome.provenance.failure_classifications,
        vec![FailureClassification::StartupFailure]
    );
    assert_eq!(outcome.provenance.crash_retry_count, 0);
}

#[test]
fn fixture_failed_cancellation_at_deadline_boundary_charges_and_kills() {
    // The deadline expires at the boundary; the supervisor cancels during
    // cleanup, but the worker never acknowledges (CancelFailure). Per the
    // error contract, a cancellation the worker fails to acknowledge is a
    // worker fault: it escalates to a worker kill and charges exactly one
    // FailedCancellation strike (resolution r2 #8).
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![
            ScriptedEvent::Progress { committed: 16 },
            ScriptedEvent::CancelFailure,
        ]],
        context(),
    );
    let clock = ManualClock::new(100);
    let control = TerminalControl {
        deadline_at: Some(50), // expired before the boundary at 100
        cancel_at: None,
    };
    let outcome = sup.run_generation(&request(), &mut factory, &context(), &control, &clock);
    // Failed cancellation is terminal (no redispatch) and surfaces per the
    // error contract: budget not exhausted -> owned_decode_unavailable.
    assert_eq!(outcome.result, Err(DecodeError::Unavailable));
    assert_eq!(
        outcome.provenance.failure_classifications,
        vec![FailureClassification::FailedCancellation]
    );
    assert_eq!(outcome.provenance.crash_retry_count, 0);
    // The worker was killed and exactly one strike charged.
    assert_eq!(factory.log().kills, 1);
    assert_eq!(sup.budget().remaining(&request().key), 1);
    assert_eq!(factory.spawn_count(), 1);
}

#[test]
fn fixture_failed_cancellation_at_cancel_boundary_charges_and_kills() {
    // Cancellation was recorded before the boundary; the worker fails to
    // acknowledge the supervisor's cancel. Same escalation and charge as the
    // deadline variant, even though an acknowledged cancellation would have
    // charged nothing.
    let mut sup = supervisor();
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![
            ScriptedEvent::Progress { committed: 16 },
            ScriptedEvent::CancelFailure,
        ]],
        context(),
    );
    let clock = ManualClock::new(100);
    let control = TerminalControl {
        deadline_at: Some(500), // not expired
        cancel_at: Some(50),    // recorded before the boundary
    };
    let outcome = sup.run_generation(&request(), &mut factory, &context(), &control, &clock);
    assert_eq!(outcome.result, Err(DecodeError::Unavailable));
    assert_eq!(
        outcome.provenance.failure_classifications,
        vec![FailureClassification::FailedCancellation]
    );
    assert_eq!(factory.log().kills, 1);
    assert_eq!(sup.budget().remaining(&request().key), 1);
}

#[test]
fn fixture_failed_cancellation_exhausting_budget_quarantines() {
    // With a single-unit budget, the one FailedCancellation strike exhausts
    // the budget and quarantines the key.
    let mut sup = Supervisor::new(
        CrashBudget::new(InMemoryBudgetStore::default(), BudgetPolicy::new(1, 60_000)),
        N,
    );
    let mut factory = ScriptedWorkerFactory::new(
        vec![vec![
            ScriptedEvent::Progress { committed: 16 },
            ScriptedEvent::CancelFailure,
        ]],
        context(),
    );
    let clock = ManualClock::new(100);
    let control = TerminalControl {
        deadline_at: Some(50),
        cancel_at: None,
    };
    let outcome = sup.run_generation(&request(), &mut factory, &context(), &control, &clock);
    // The single strike exhausts the budget: quarantined.
    assert_eq!(outcome.result, Err(DecodeError::Quarantined));
    assert!(sup.budget().is_quarantined(&request().key, 100));
    assert_eq!(factory.log().kills, 1);
}

// ---------------------------------------------------------------------------
// Wire-error binding literal guard.
// ---------------------------------------------------------------------------

#[test]
fn fixture_wire_bindings_are_literal_not_symbolic() {
    owned_decode_worker::wire_error_bindings::assert_no_symbolic_placeholders();
    assert_eq!(DecodeError::DeadlineExceeded.as_str(), "deadline_exceeded");
    assert_eq!(DecodeError::Cancelled.as_str(), "cancelled");
}
