//! The module-side supervisor for the owned-decode worker.
//!
//! The supervisor drives one logical generation against a supervised worker:
//! start validation, the progress/continuation loop, terminal-control boundary
//! evaluation, crash-budget accounting, quarantine, and the single permitted
//! worker-crash redispatch. It owns the execution permit logic's accounting
//! surface (retry count and ordered failure classifications) but not tokenizer
//! assets, schemas, or response text.
//!
//! Redispatch semantics (worker_isolation_contract): only a worker-crash
//! classification permits exactly one from-scratch redispatch, and only when,
//! after charging the first failure, at least one budget unit remains, the
//! request is not cancelled, and the original deadline remains valid. Redispatch
//! preserves the logical `generation_id` and deadline but uses a new worker
//! generation and session, restarts at token zero, and resets attempt-local
//! sequence and committed count. Timeout, protocol-fatal, and startup failures
//! are terminal after one charge.

use serde::{Deserialize, Serialize};

use crate::boundary::{evaluate_boundary, BoundaryDecision, BoundaryInputs, Timestamp};
use crate::budget::{ChargeOutcome, CrashBudget, CrashBudgetStore};
use crate::error::{DecodeError, FailureClassification};
use crate::identity::QuarantineKey;
use crate::protocol::{
    FinalResponse, FinishReason, GenerateCancel, GenerateContinue, GenerateStart, WorkerFrame,
};
use crate::validation::{validate_start, WorkerStartContext};
use crate::worker::{CancelAck, DecodeWorker, WorkerFactory, WorkerFault};

/// A monotonic clock the supervisor reads at each boundary.
pub trait Clock {
    fn now(&self) -> Timestamp;
}

/// A manually driven clock for deterministic fixtures.
#[derive(Default)]
pub struct ManualClock {
    now: std::cell::Cell<Timestamp>,
}

impl ManualClock {
    #[must_use]
    pub fn new(start: Timestamp) -> Self {
        Self {
            now: std::cell::Cell::new(start),
        }
    }

    pub fn set(&self, now: Timestamp) {
        self.now.set(now);
    }

    pub fn advance(&self, delta: Timestamp) {
        self.now.set(self.now.get() + delta);
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Timestamp {
        self.now.get()
    }
}

/// Terminal-control inputs for a generation: the request deadline and the time
/// cancellation was recorded, if any.
#[derive(Clone, Copy, Debug, Default)]
pub struct TerminalControl {
    pub deadline_at: Option<Timestamp>,
    pub cancel_at: Option<Timestamp>,
}

impl TerminalControl {
    /// Whether cancellation was recorded at or before `now`.
    #[must_use]
    pub fn is_cancelled(&self, now: Timestamp) -> bool {
        self.cancel_at.is_some_and(|at| at <= now)
    }

    /// Whether the original deadline remains valid (strictly after `now`).
    #[must_use]
    pub fn deadline_valid(&self, now: Timestamp) -> bool {
        !self.deadline_at.is_some_and(|at| at <= now)
    }
}

/// A generation request: the start frame plus the quarantine key it runs under.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub key: QuarantineKey,
    pub start: GenerateStart,
}

/// The successful output of a generation. Carries complete generated IDs and
/// accounting but no authoritative text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SuccessOutput {
    pub generated_ids: Vec<u32>,
    pub committed_token_count: u32,
    pub finish_reason: FinishReason,
    pub worker_generation: u64,
    pub last_completed_sequence: u32,
    pub constraint_identity: Option<String>,
    pub constraint_complete: bool,
}

/// Additive provenance for the operation. Retry metadata is operation-level.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Provenance {
    pub crash_retry_count: u32,
    pub failure_classifications: Vec<FailureClassification>,
    pub worker_generation: u64,
    pub last_completed_quantum_sequence: u32,
}

/// The outcome of a generation: a typed result plus provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationOutcome {
    pub result: Result<SuccessOutput, DecodeError>,
    pub provenance: Provenance,
}

/// The result of a single worker attempt.
enum AttemptResult {
    /// A clean terminal: either success or a typed error that is not chargeable
    /// (deadline, cancellation, grammar, mismatch).
    Clean(Result<SuccessOutput, DecodeError>, AttemptAccounting),
    /// A chargeable failure: crash, timeout, protocol-fatal, startup, or failed
    /// cancellation.
    Chargeable(FailureClassification, AttemptAccounting),
}

/// Per-attempt accounting surfaced into provenance.
#[derive(Clone, Copy, Debug, Default)]
struct AttemptAccounting {
    worker_generation: u64,
    last_completed_sequence: u32,
}

/// The supervisor.
pub struct Supervisor<S: CrashBudgetStore> {
    budget: CrashBudget<S>,
    production_n: u32,
}

impl<S: CrashBudgetStore> Supervisor<S> {
    /// Create a supervisor. `production_n` is the single committed production N.
    ///
    /// # Panics
    /// Panics if `production_n` is not one of `{8, 16, 32}`.
    #[must_use]
    pub fn new(budget: CrashBudget<S>, production_n: u32) -> Self {
        assert!(
            matches!(production_n, 8 | 16 | 32),
            "production_n must be one of {{8, 16, 32}}"
        );
        Self {
            budget,
            production_n,
        }
    }

    #[must_use]
    pub fn budget(&self) -> &CrashBudget<S> {
        &self.budget
    }

    /// Refuse a quarantined key before dispatch. Dispatches nothing and consumes
    /// no crash budget.
    pub fn refuse_if_quarantined(
        &self,
        request: &GenerationRequest,
        now: Timestamp,
    ) -> Option<DecodeError> {
        if self.budget.is_quarantined(&request.key, now) {
            Some(DecodeError::Quarantined)
        } else {
            None
        }
    }

    /// Run one logical generation, including the single permitted crash
    /// redispatch.
    pub fn run_generation(
        &mut self,
        request: &GenerationRequest,
        factory: &mut dyn WorkerFactory,
        context: &WorkerStartContext,
        control: &TerminalControl,
        clock: &dyn Clock,
    ) -> GenerationOutcome {
        let mut provenance = Provenance::default();

        // Pre-dispatch quarantine refusal: dispatch nothing, charge nothing.
        if let Some(error) = self.refuse_if_quarantined(request, clock.now()) {
            return GenerationOutcome {
                result: Err(error),
                provenance,
            };
        }

        // Fail-closed persistence: a key whose latest crash-budget record
        // could not be persisted refuses dispatch (typed unavailable) until a
        // save succeeds, so a disk failure can never silently reset
        // quarantine state across restarts.
        if self.budget.persistence_failed(&request.key) {
            return GenerationOutcome {
                result: Err(DecodeError::Unavailable),
                provenance,
            };
        }

        // Module-side start validation. Clean typed errors here consume no budget
        // and never redispatch.
        if let Err(error) = validate_start(&request.start, context, self.production_n) {
            return GenerationOutcome {
                result: Err(error),
                provenance,
            };
        }

        let first = self.run_attempt(request, factory, context, control, clock);
        match first {
            AttemptResult::Clean(result, accounting) => {
                provenance.worker_generation = accounting.worker_generation;
                provenance.last_completed_quantum_sequence = accounting.last_completed_sequence;
                GenerationOutcome { result, provenance }
            }
            AttemptResult::Chargeable(classification, accounting) => {
                provenance.failure_classifications.push(classification);
                provenance.worker_generation = accounting.worker_generation;
                provenance.last_completed_quantum_sequence = accounting.last_completed_sequence;

                let now = clock.now();
                let outcome = match self.budget.charge(&request.key, classification, now) {
                    Ok(outcome) => outcome,
                    Err(_) => {
                        // The strike could not be persisted: fail closed. The
                        // key refuses further dispatch until a save succeeds.
                        return GenerationOutcome {
                            result: Err(DecodeError::Unavailable),
                            provenance,
                        };
                    }
                };

                // Only a worker-crash classification permits exactly one redispatch.
                let can_redispatch = classification == FailureClassification::Crash
                    && self.budget.redispatch_permitted(&request.key, now)
                    && !control.is_cancelled(now)
                    && control.deadline_valid(now);

                if !can_redispatch {
                    let error = barred_error(classification, outcome, control, now);
                    return GenerationOutcome {
                        result: Err(error),
                        provenance,
                    };
                }

                // Redispatch from the original prompt and initial constraint state
                // on a fresh worker generation and session.
                provenance.crash_retry_count = 1;
                let second = self.run_attempt(request, factory, context, control, clock);
                match second {
                    AttemptResult::Clean(result, accounting2) => {
                        provenance.worker_generation = accounting2.worker_generation;
                        provenance.last_completed_quantum_sequence =
                            accounting2.last_completed_sequence;
                        GenerationOutcome { result, provenance }
                    }
                    AttemptResult::Chargeable(classification2, accounting2) => {
                        // A second crash or replacement failure is terminal.
                        provenance.failure_classifications.push(classification2);
                        provenance.worker_generation = accounting2.worker_generation;
                        provenance.last_completed_quantum_sequence =
                            accounting2.last_completed_sequence;
                        let outcome2 =
                            match self
                                .budget
                                .charge(&request.key, classification2, clock.now())
                            {
                                Ok(outcome) => outcome,
                                Err(_) => {
                                    // Unpersisted strike: fail closed as above.
                                    return GenerationOutcome {
                                        result: Err(DecodeError::Unavailable),
                                        provenance,
                                    };
                                }
                            };
                        let error = if outcome2.exhausted {
                            DecodeError::Quarantined
                        } else {
                            DecodeError::Unavailable
                        };
                        GenerationOutcome {
                            result: Err(error),
                            provenance,
                        }
                    }
                }
            }
        }
    }

    /// Drive a single worker attempt through the progress/continuation loop.
    fn run_attempt(
        &self,
        request: &GenerationRequest,
        factory: &mut dyn WorkerFactory,
        context: &WorkerStartContext,
        control: &TerminalControl,
        clock: &dyn Clock,
    ) -> AttemptResult {
        // Spawn a fresh worker process. A spawn failure is a startup failure.
        let mut worker = match factory.spawn() {
            Ok(worker) => worker,
            Err(_) => {
                return AttemptResult::Chargeable(
                    FailureClassification::StartupFailure,
                    AttemptAccounting::default(),
                );
            }
        };
        let generation = worker.worker_generation();
        let mut accounting = AttemptAccounting {
            worker_generation: generation,
            last_completed_sequence: 0,
        };

        // Worker-side start validation mirrors the module-side check.
        if let Err(error) = worker.start(&request.start, context, self.production_n) {
            return AttemptResult::Clean(Err(error), accounting);
        }

        let generation_id = request.start.generation_id.clone();
        let max_tokens = request.start.max_tokens;
        let mut expected_sequence = 1u32;
        // Attempt-local cumulative committed tokens; progress frames must
        // advance this (a non-advancing count is a protocol fault).
        let mut last_committed = 0u32;

        loop {
            let stepped = match worker.step() {
                Ok(stepped) => stepped,
                Err(WorkerFault::Crash) => {
                    return AttemptResult::Chargeable(FailureClassification::Crash, accounting);
                }
                Err(WorkerFault::Timeout) => {
                    return AttemptResult::Chargeable(FailureClassification::Timeout, accounting);
                }
                Err(WorkerFault::FailedCancellation) => {
                    return AttemptResult::Chargeable(
                        FailureClassification::FailedCancellation,
                        accounting,
                    );
                }
                Err(WorkerFault::StartupFailure) => {
                    return AttemptResult::Chargeable(
                        FailureClassification::StartupFailure,
                        accounting,
                    );
                }
            };

            // Reject delayed frames from a closed or superseded session. A frame
            // whose generation does not match the current session is protocol-fatal.
            if stepped.worker_generation != generation {
                return AttemptResult::Chargeable(FailureClassification::ProtocolFatal, accounting);
            }

            match stepped.frame {
                WorkerFrame::Error { id } => {
                    // A clean typed-error frame (grammar or mismatch) returns
                    // directly and consumes no budget. An unrecognized id is a
                    // protocol violation.
                    let error = DecodeError::from_id(&id).unwrap_or(DecodeError::ProtocolMismatch);
                    return AttemptResult::Clean(Err(error), accounting);
                }
                WorkerFrame::Progress(progress) => {
                    // Generation identity: a frame carrying an unknown generation
                    // id is protocol-fatal.
                    if progress.generation_id != generation_id {
                        return AttemptResult::Chargeable(
                            FailureClassification::ProtocolFatal,
                            accounting,
                        );
                    }
                    // Sequence continuity: the first sequence is one and later
                    // sequences increment by one. A repeated or skipped sequence
                    // is protocol-fatal.
                    if progress.quantum_sequence != expected_sequence {
                        return AttemptResult::Chargeable(
                            FailureClassification::ProtocolFatal,
                            accounting,
                        );
                    }
                    expected_sequence += 1;
                    accounting.last_completed_sequence = progress.quantum_sequence;

                    // Committed-count strictness: the cumulative count must
                    // advance. A non-advancing count is a protocol fault, the
                    // same reading the S5 module-side validator
                    // (GenerationProtocol::receive_progress) applies.
                    if progress.committed_token_count <= last_committed {
                        return AttemptResult::Chargeable(
                            FailureClassification::ProtocolFatal,
                            accounting,
                        );
                    }
                    last_committed = progress.committed_token_count;

                    let decision = evaluate_boundary(BoundaryInputs {
                        completion: None,
                        cancel_recorded_at: control.cancel_at,
                        deadline_at: control.deadline_at,
                        observed_at: clock.now(),
                    });
                    match decision {
                        BoundaryDecision::AcceptProgress => {
                            let remaining_request =
                                max_tokens.saturating_sub(progress.committed_token_count);
                            if remaining_request == 0 {
                                // Nothing left to authorize but no final arrived:
                                // the worker must have emitted a final. Treat as a
                                // protocol violation.
                                return AttemptResult::Chargeable(
                                    FailureClassification::ProtocolFatal,
                                    accounting,
                                );
                            }
                            let next_budget = self.production_n.min(remaining_request);
                            if worker
                                .send_continue(&GenerateContinue {
                                    generation_id: generation_id.clone(),
                                    next_expected_sequence: expected_sequence,
                                    next_token_budget: next_budget,
                                })
                                .is_err()
                            {
                                return AttemptResult::Chargeable(
                                    FailureClassification::ProtocolFatal,
                                    accounting,
                                );
                            }
                        }
                        BoundaryDecision::DeadlineExceeded => {
                            // Deadline cleanup before timeout: cancel and charge
                            // none — unless the worker fails to acknowledge.
                            return cancel_at_boundary(
                                &mut worker,
                                &generation_id,
                                DecodeError::DeadlineExceeded,
                                accounting,
                            );
                        }
                        BoundaryDecision::Cancelled => {
                            // Acknowledged cancellation charges none; an
                            // unacknowledged one escalates and charges.
                            return cancel_at_boundary(
                                &mut worker,
                                &generation_id,
                                DecodeError::Cancelled,
                                accounting,
                            );
                        }
                        BoundaryDecision::AcceptCompletion(_) => {
                            // A progress frame cannot be a terminal completion.
                            return AttemptResult::Chargeable(
                                FailureClassification::ProtocolFatal,
                                accounting,
                            );
                        }
                    }
                }
                WorkerFrame::Final(final_response) => {
                    // Generation identity: a final frame carrying an unknown
                    // generation id is protocol-fatal.
                    if final_response.generation_id != generation_id {
                        return AttemptResult::Chargeable(
                            FailureClassification::ProtocolFatal,
                            accounting,
                        );
                    }
                    let decision = evaluate_boundary(BoundaryInputs {
                        completion: Some(final_response.finish_reason),
                        cancel_recorded_at: control.cancel_at,
                        deadline_at: control.deadline_at,
                        observed_at: clock.now(),
                    });
                    match decision {
                        BoundaryDecision::AcceptCompletion(reason) => {
                            accounting.last_completed_sequence =
                                final_response.last_completed_sequence;
                            return AttemptResult::Clean(
                                Ok(SuccessOutput {
                                    generated_ids: final_response.generated_ids.clone(),
                                    committed_token_count: final_response.committed_token_count,
                                    finish_reason: reason,
                                    worker_generation: final_response.worker_generation,
                                    last_completed_sequence: final_response.last_completed_sequence,
                                    constraint_identity: final_response.constraint_identity.clone(),
                                    constraint_complete: final_response.constraint_complete,
                                }),
                                accounting,
                            );
                        }
                        BoundaryDecision::DeadlineExceeded => {
                            return cancel_at_boundary(
                                &mut worker,
                                &generation_id,
                                DecodeError::DeadlineExceeded,
                                accounting,
                            );
                        }
                        BoundaryDecision::Cancelled => {
                            return cancel_at_boundary(
                                &mut worker,
                                &generation_id,
                                DecodeError::Cancelled,
                                accounting,
                            );
                        }
                        BoundaryDecision::AcceptProgress => {
                            // A final response is always a terminal boundary.
                            return AttemptResult::Chargeable(
                                FailureClassification::ProtocolFatal,
                                accounting,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Cancel the worker at a boundary and interpret the acknowledgment.
///
/// An acknowledged cancellation is clean: the boundary's own error is returned
/// and nothing is charged. A cancellation the worker fails to acknowledge is a
/// worker fault (error_contract, resolution r2 #8): it escalates to a worker
/// kill and charges one crash-budget strike under `FailedCancellation`.
fn cancel_at_boundary(
    worker: &mut Box<dyn DecodeWorker>,
    generation_id: &str,
    boundary_error: DecodeError,
    accounting: AttemptAccounting,
) -> AttemptResult {
    match worker.send_cancel(&GenerateCancel {
        generation_id: generation_id.to_string(),
    }) {
        Ok(_ack) => AttemptResult::Clean(Err(boundary_error), accounting),
        Err(_) => {
            // Unacknowledged cancellation: kill the worker. Resident state is
            // destroyed with the process; the fault is charged by the caller.
            worker.kill();
            AttemptResult::Chargeable(FailureClassification::FailedCancellation, accounting)
        }
    }
}

/// Compute the error returned when a redispatch is barred.
///
/// A retry barred by exhausted budget or quarantine returns
/// `owned_decode_quarantined`; one barred by deadline returns the bound deadline
/// error. A non-crash terminal failure (timeout, protocol-fatal, startup,
/// failed-cancellation) returns `owned_decode_quarantined` when the charge
/// exhausted the budget and `owned_decode_unavailable` otherwise.
fn barred_error(
    classification: FailureClassification,
    outcome: ChargeOutcome,
    control: &TerminalControl,
    now: Timestamp,
) -> DecodeError {
    if classification == FailureClassification::Crash {
        // A crash that cannot redispatch.
        if outcome.exhausted || outcome.quarantined {
            return DecodeError::Quarantined;
        }
        if !control.deadline_valid(now) {
            return DecodeError::DeadlineExceeded;
        }
        if control.is_cancelled(now) {
            return DecodeError::Cancelled;
        }
        return DecodeError::Quarantined;
    }
    // Non-crash terminal chargeable failure.
    if outcome.exhausted {
        DecodeError::Quarantined
    } else {
        DecodeError::Unavailable
    }
}

/// Interpret a worker final response's finish reason for callers that need the
/// normalized external reason. Stop controls are already omitted by the worker.
#[must_use]
pub const fn external_finish_reason(final_response: &FinalResponse) -> FinishReason {
    final_response.finish_reason
}

/// A convenience for acknowledging a cancellation result in telemetry. Returns
/// the non-authoritative committed-token count.
#[must_use]
pub const fn acknowledged_committed_count(ack: CancelAck) -> u32 {
    match ack {
        CancelAck::Acknowledged {
            committed_token_count,
        } => committed_token_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::{BudgetPolicy, BudgetRecord, BudgetStoreError, InMemoryBudgetStore};
    use crate::protocol::Sampling;
    use crate::worker::ScriptedEvent;
    use crate::worker::ScriptedWorkerFactory;

    fn context() -> WorkerStartContext {
        WorkerStartContext {
            loaded_model_ref: "model".into(),
            decode_fingerprint: "dfp".into(),
            runtime_config_digest: "rt".into(),
            expected_constraint: None,
        }
    }

    fn request() -> GenerationRequest {
        GenerationRequest {
            key: QuarantineKey::new("profile", "dfp", "rt"),
            start: GenerateStart {
                generation_id: "g1".into(),
                loaded_model_ref: "model".into(),
                decode_fingerprint: "dfp".into(),
                runtime_config_digest: "rt".into(),
                prompt_ids: vec![1, 2, 3],
                stop_ids: vec![2],
                max_tokens: 64,
                sampling: Sampling::greedy_top1(),
                constraint: None,
            },
        }
    }

    fn supervisor() -> Supervisor<InMemoryBudgetStore> {
        Supervisor::new(
            CrashBudget::new(InMemoryBudgetStore::default(), BudgetPolicy::default()),
            16,
        )
    }

    #[test]
    fn clean_single_quantum_completion_succeeds() {
        let mut sup = supervisor();
        let mut factory = ScriptedWorkerFactory::new(
            vec![vec![ScriptedEvent::Final {
                finish: FinishReason::StopToken,
                ids: vec![100, 101, 2],
                constraint_complete: false,
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
        assert_eq!(success.generated_ids, vec![100, 101, 2]);
        assert_eq!(outcome.provenance.crash_retry_count, 0);
    }

    #[test]
    fn crash_then_clean_redispatch_preserves_generation_id() {
        let mut sup = supervisor();
        let mut factory = ScriptedWorkerFactory::new(
            vec![
                vec![ScriptedEvent::Crash],
                vec![ScriptedEvent::Final {
                    finish: FinishReason::MaxTokens,
                    ids: vec![7, 8, 9],
                    constraint_complete: false,
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
        let success = outcome.result.expect("redispatch succeeds");
        assert_eq!(success.finish_reason, FinishReason::MaxTokens);
        assert_eq!(outcome.provenance.crash_retry_count, 1);
        assert_eq!(
            outcome.provenance.failure_classifications,
            vec![FailureClassification::Crash]
        );
        // Two worker processes were spawned: the crashed one and the replacement.
        assert_eq!(factory.spawn_count(), 2);
        // The replacement ran on a new generation and restarted at token zero.
        assert_eq!(success.worker_generation, 2);
    }

    #[test]
    fn second_crash_is_terminal_and_quarantines() {
        let mut sup = supervisor();
        let mut factory = ScriptedWorkerFactory::new(
            vec![vec![ScriptedEvent::Crash], vec![ScriptedEvent::Crash]],
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
        assert_eq!(
            outcome.provenance.failure_classifications,
            vec![FailureClassification::Crash, FailureClassification::Crash]
        );
        assert!(sup.budget().is_quarantined(&request().key, 0));
    }

    #[test]
    fn timeout_is_terminal_and_not_redispatched() {
        let mut sup = supervisor();
        let mut factory = ScriptedWorkerFactory::new(
            vec![
                vec![ScriptedEvent::Timeout],
                vec![ScriptedEvent::Final {
                    finish: FinishReason::StopToken,
                    ids: vec![1],
                    constraint_complete: false,
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
        // Timeout charges one unit but does not permit redispatch; with the
        // default two-strike policy the budget is not exhausted, so unavailable.
        assert_eq!(outcome.result, Err(DecodeError::Unavailable));
        assert_eq!(outcome.provenance.crash_retry_count, 0);
        assert_eq!(factory.spawn_count(), 1);
    }

    #[test]
    fn quarantined_key_is_refused_before_dispatch() {
        let mut sup = supervisor();
        // Exhaust and quarantine the key directly.
        let key = request().key.clone();
        sup.budget
            .charge(&key, FailureClassification::Crash, 0)
            .expect("in-memory save succeeds");
        sup.budget
            .charge(&key, FailureClassification::Crash, 0)
            .expect("in-memory save succeeds");
        let mut factory = ScriptedWorkerFactory::new(vec![], context());
        let clock = ManualClock::new(0);
        let outcome = sup.run_generation(
            &request(),
            &mut factory,
            &context(),
            &TerminalControl::default(),
            &clock,
        );
        assert_eq!(outcome.result, Err(DecodeError::Quarantined));
        // Nothing was dispatched and nothing further was charged.
        assert_eq!(factory.spawn_count(), 0);
    }

    #[test]
    fn failed_cancellation_at_deadline_boundary_charges_and_kills() {
        // The deadline expires at the boundary; the supervisor cancels during
        // cleanup, but the worker never acknowledges. The fault escalates to a
        // kill and charges one FailedCancellation strike.
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
    fn failed_cancellation_at_cancel_boundary_charges_and_kills() {
        // Cancellation was recorded before the boundary; the worker fails to
        // acknowledge the supervisor's cancel. Same escalation and charge as
        // the deadline variant, even though an acknowledged cancellation would
        // have charged nothing.
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
    fn failed_cancellation_exhausting_budget_quarantines() {
        let mut sup = Supervisor::new(
            CrashBudget::new(InMemoryBudgetStore::default(), BudgetPolicy::new(1, 60_000)),
            16,
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

    /// A store whose save fails while `fail` is set.
    #[derive(Default)]
    struct FailingStore {
        records: std::collections::BTreeMap<String, BudgetRecord>,
        fail: bool,
    }

    impl crate::budget::CrashBudgetStore for FailingStore {
        fn load(&self, key: &QuarantineKey) -> Option<BudgetRecord> {
            self.records.get(&key.storage_id()).cloned()
        }

        fn save(
            &mut self,
            key: &QuarantineKey,
            record: &BudgetRecord,
        ) -> Result<(), BudgetStoreError> {
            // Models FileBudgetStore: the in-memory record updates first;
            // the flush to disk is what fails.
            self.records.insert(key.storage_id(), record.clone());
            if self.fail {
                return Err(BudgetStoreError {
                    message: "disk failure".to_string(),
                });
            }
            Ok(())
        }
    }

    #[test]
    fn budget_persistence_failure_fails_closed() {
        // A crash whose strike cannot be persisted surfaces the typed
        // unavailable error, and the key refuses further dispatch until a
        // save succeeds.
        let store = FailingStore {
            records: std::collections::BTreeMap::new(),
            fail: true,
        };
        let mut sup = Supervisor::new(CrashBudget::new(store, BudgetPolicy::default()), 16);
        let mut factory = ScriptedWorkerFactory::new(vec![vec![ScriptedEvent::Crash]], context());
        let clock = ManualClock::new(0);
        let outcome = sup.run_generation(
            &request(),
            &mut factory,
            &context(),
            &TerminalControl::default(),
            &clock,
        );
        assert_eq!(outcome.result, Err(DecodeError::Unavailable));
        assert_eq!(factory.spawn_count(), 1);

        // Next generation for the same key: refused pre-dispatch, no spawn,
        // no charge.
        let mut factory2 = ScriptedWorkerFactory::new(vec![vec![ScriptedEvent::Crash]], context());
        let outcome2 = sup.run_generation(
            &request(),
            &mut factory2,
            &context(),
            &TerminalControl::default(),
            &clock,
        );
        assert_eq!(outcome2.result, Err(DecodeError::Unavailable));
        assert_eq!(factory2.spawn_count(), 0, "refused before dispatch");
    }
}
