//! Worker transport abstraction and fault-injection test double.
//!
//! The real Metal worker satisfies [`DecodeWorker`]: one transport session bound
//! to one immutable `worker_generation`. [`WorkerFactory`] models process
//! supervision — each [`WorkerFactory::spawn`] starts a fresh worker process
//! with a new generation and session, and can fail as a startup failure. The
//! supervisor uses the factory to perform the single permitted crash redispatch
//! on a brand-new worker.
//!
//! [`ScriptedWorkerFactory`] is the fault-injection double the fixtures drive:
//! each attempt gets its own event script, so a fixture can crash the first
//! worker and run the second cleanly, time out, emit a bad sequence, or emit a
//! delayed prior-session frame.

use crate::error::DecodeError;
use crate::protocol::{
    FinalResponse, FinishReason, GenerateCancel, GenerateContinue, GenerateProgress, GenerateStart,
    WorkerFrame,
};
use crate::validation::{validate_start, StartAuthorization, WorkerStartContext};
use std::cell::RefCell;
use std::rc::Rc;

/// A shared log the scripted workers write to so fixtures can assert what the
/// supervisor sent: continuation budgets (for remaining-budget truncation),
/// continuation sequences, cancellation count, and kill count.
#[derive(Default, Clone, Debug)]
pub struct ScriptedLog {
    pub continue_budgets: Vec<u32>,
    pub continue_sequences: Vec<u32>,
    pub cancels: u32,
    /// Forced worker kills (unacknowledged-cancellation escalation).
    pub kills: u32,
}

/// A transport-level worker fault. These are the chargeable failure modes that
/// are not clean typed-error frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerFault {
    /// The worker process died unexpectedly.
    Crash,
    /// The worker exceeded its deadline without a terminal frame.
    Timeout,
    /// Spawning or loading the replacement worker failed.
    StartupFailure,
    /// The worker failed to acknowledge cancellation within the cancel timeout.
    FailedCancellation,
}

/// Failure to start a resident generation. Typed validation refusals are clean;
/// transport/process faults are chargeable by the crash-budget supervisor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerStartFailure {
    Typed(DecodeError),
    Fault(WorkerFault),
}

impl From<DecodeError> for WorkerStartFailure {
    fn from(error: DecodeError) -> Self {
        Self::Typed(error)
    }
}

impl From<WorkerFault> for WorkerStartFailure {
    fn from(fault: WorkerFault) -> Self {
        Self::Fault(fault)
    }
}

/// The result of sending a cancellation to the worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancelAck {
    /// The worker acknowledged and destroyed resident state. Carries the
    /// attempt-local committed-token count for non-authoritative telemetry.
    Acknowledged { committed_token_count: u32 },
}

/// A frame emitted by a worker, tagged with the generation of the session that
/// produced it. The supervisor discards frames whose generation does not match
/// the current session (delayed prior-session frames after a redispatch).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SteppedFrame {
    pub worker_generation: u64,
    pub frame: WorkerFrame,
}

/// One transport session bound to one immutable worker generation.
pub trait DecodeWorker {
    /// The immutable generation this session is bound to.
    fn worker_generation(&self) -> u64;

    /// Worker-side start validation. Mirrors the module-side check so a mismatch
    /// is caught where the model is actually loaded.
    fn start(
        &mut self,
        start: &GenerateStart,
        context: &WorkerStartContext,
        production_n: u32,
    ) -> Result<StartAuthorization, WorkerStartFailure>;

    /// Drive the current quantum and return the frame the worker emits.
    fn step(&mut self) -> Result<SteppedFrame, WorkerFault>;

    /// Authorize the next quantum.
    fn send_continue(&mut self, cont: &GenerateContinue) -> Result<(), WorkerFault>;

    /// Request cancellation. The worker destroys resident state and acknowledges.
    fn send_cancel(&mut self, cancel: &GenerateCancel) -> Result<CancelAck, WorkerFault>;

    /// Forcibly terminate the worker process. This is the escalation for a
    /// cancellation the worker fails to acknowledge; resident state is
    /// destroyed with the process.
    fn kill(&mut self);
}

/// Process supervision: spawn a fresh worker process for an attempt.
pub trait WorkerFactory {
    /// Spawn a new worker with a fresh generation and session. A spawn failure
    /// is a startup failure (replacement startup or load failure).
    fn spawn(&mut self) -> Result<Box<dyn DecodeWorker>, WorkerFault>;
}

/// A single scripted event for one quantum of a worker attempt.
#[derive(Clone, Debug)]
pub enum ScriptedEvent {
    /// Crash while accepting the start frame, before the first quantum exists.
    StartCrash,
    /// Emit a well-formed progress frame; the worker assigns the next sequence.
    Progress { committed: u32 },
    /// Emit a progress frame with an explicit (typically wrong) sequence, to
    /// provoke a protocol-fatal classification at the supervisor.
    ProgressWithSequence { sequence: u32, committed: u32 },
    /// Emit a progress frame carrying an unknown generation id, to provoke the
    /// supervisor's generation-identity check.
    ProgressForeignGeneration { committed: u32 },
    /// Emit a successful final response.
    Final {
        finish: FinishReason,
        ids: Vec<u32>,
        constraint_complete: bool,
    },
    /// Emit a typed-error frame (e.g. a grammar error). Not protocol-fatal.
    ErrorFrame(DecodeError),
    /// Emit a final response tagged with a stale worker generation, simulating a
    /// delayed prior-session frame the supervisor must reject.
    StaleFinal {
        generation: u64,
        finish: FinishReason,
    },
    /// Crash before emitting any frame this quantum.
    Crash,
    /// Stall past the deadline without emitting.
    Timeout,
    /// The next cancellation request is never acknowledged (the worker hangs
    /// past the cancel timeout). Consumed by `send_cancel`, not by `step`, so
    /// a script places it where the supervisor's boundary cancel will hit it.
    CancelFailure,
    /// Model the worker's greedy union selection (reference semantics: the S5
    /// grammar-scheduler `greedy_generate` stop union; production selection is
    /// owned by the real Metal worker): the content IDs are committed, then a
    /// stop candidate wins the selection. The winning stop token is a
    /// non-committed control candidate: it never enters generated IDs or the
    /// committed count, and the final finishes with `stop_token`.
    StopSelectionWins { content_ids: Vec<u32>, stop_id: u32 },
}

/// A scripted worker for one attempt.
pub struct ScriptedWorker {
    generation: u64,
    events: Vec<ScriptedEvent>,
    cursor: usize,
    next_sequence: u32,
    committed: u32,
    context: WorkerStartContext,
    started: bool,
    /// The logical generation id bound at start; echoed on every emitted frame so
    /// the supervisor can reject frames carrying an unknown generation id.
    generation_id: String,
    log: Rc<RefCell<ScriptedLog>>,
}

impl DecodeWorker for ScriptedWorker {
    fn worker_generation(&self) -> u64 {
        self.generation
    }

    fn start(
        &mut self,
        start: &GenerateStart,
        context: &WorkerStartContext,
        production_n: u32,
    ) -> Result<StartAuthorization, WorkerStartFailure> {
        // The scripted worker validates against its own loaded context, exactly
        // as the real worker validates the loaded-model reference, decode
        // fingerprint, and runtime digest before the first commit.
        let _ = context;
        if matches!(
            self.events.get(self.cursor),
            Some(ScriptedEvent::StartCrash)
        ) {
            self.cursor += 1;
            return Err(WorkerStartFailure::Fault(WorkerFault::Crash));
        }
        let auth =
            validate_start(start, &self.context, production_n).map_err(WorkerStartFailure::from)?;
        self.started = true;
        self.next_sequence = 1;
        self.committed = 0;
        self.generation_id = start.generation_id.clone();
        Ok(auth)
    }

    fn step(&mut self) -> Result<SteppedFrame, WorkerFault> {
        debug_assert!(self.started, "step before start");
        let event = self
            .events
            .get(self.cursor)
            .cloned()
            .unwrap_or(ScriptedEvent::Crash);
        self.cursor += 1;
        let generation = self.generation;
        match event {
            ScriptedEvent::StartCrash => Err(WorkerFault::Crash),
            ScriptedEvent::Progress { committed } => {
                self.committed = committed;
                let sequence = self.next_sequence;
                self.next_sequence += 1;
                Ok(SteppedFrame {
                    worker_generation: generation,
                    frame: WorkerFrame::Progress(GenerateProgress {
                        generation_id: self.generation_id.clone(),
                        quantum_sequence: sequence,
                        committed_token_count: committed,
                    }),
                })
            }
            ScriptedEvent::ProgressWithSequence {
                sequence,
                committed,
            } => {
                self.committed = committed;
                Ok(SteppedFrame {
                    worker_generation: generation,
                    frame: WorkerFrame::Progress(GenerateProgress {
                        generation_id: self.generation_id.clone(),
                        quantum_sequence: sequence,
                        committed_token_count: committed,
                    }),
                })
            }
            ScriptedEvent::ProgressForeignGeneration { committed } => {
                self.committed = committed;
                let sequence = self.next_sequence;
                self.next_sequence += 1;
                Ok(SteppedFrame {
                    worker_generation: generation,
                    frame: WorkerFrame::Progress(GenerateProgress {
                        generation_id: "foreign-generation".into(),
                        quantum_sequence: sequence,
                        committed_token_count: committed,
                    }),
                })
            }
            ScriptedEvent::Final {
                finish,
                ids,
                constraint_complete,
            } => {
                let committed = ids.len() as u32;
                self.committed = committed;
                let last_sequence = self.next_sequence.saturating_sub(1).max(1);
                Ok(SteppedFrame {
                    worker_generation: generation,
                    frame: WorkerFrame::Final(FinalResponse {
                        generation_id: self.generation_id.clone(),
                        generated_ids: ids,
                        committed_token_count: committed,
                        decode_fingerprint: self.context.decode_fingerprint.clone(),
                        runtime_config_digest: self.context.runtime_config_digest.clone(),
                        worker_generation: generation,
                        finish_reason: finish,
                        constraint_identity: self
                            .context
                            .expected_constraint
                            .as_ref()
                            .map(|c| c.constraint_runtime_identity.clone()),
                        constraint_complete,
                        last_completed_sequence: last_sequence,
                    }),
                })
            }
            ScriptedEvent::ErrorFrame(error) => Ok(SteppedFrame {
                worker_generation: generation,
                frame: WorkerFrame::Error {
                    id: error.as_str().to_string(),
                },
            }),
            ScriptedEvent::StaleFinal { generation, finish } => Ok(SteppedFrame {
                worker_generation: generation,
                frame: WorkerFrame::Final(FinalResponse {
                    generation_id: String::new(),
                    generated_ids: vec![],
                    committed_token_count: 0,
                    decode_fingerprint: self.context.decode_fingerprint.clone(),
                    runtime_config_digest: self.context.runtime_config_digest.clone(),
                    worker_generation: generation,
                    finish_reason: finish,
                    constraint_identity: None,
                    constraint_complete: false,
                    last_completed_sequence: 0,
                }),
            }),
            ScriptedEvent::Crash => Err(WorkerFault::Crash),
            ScriptedEvent::Timeout => Err(WorkerFault::Timeout),
            ScriptedEvent::CancelFailure => {
                // Consumed by send_cancel, never by step; if the script reaches
                // it through step the worker is unresponsive: a crash.
                Err(WorkerFault::Crash)
            }
            ScriptedEvent::StopSelectionWins {
                content_ids,
                stop_id: _,
            } => {
                // The stop candidate wins the final selection. It is a
                // non-committed control candidate, so the final carries only
                // the content IDs and counts, exactly as the reference
                // selection semantics require.
                let committed = content_ids.len() as u32;
                self.committed = committed;
                let last_sequence = self.next_sequence.saturating_sub(1).max(1);
                Ok(SteppedFrame {
                    worker_generation: generation,
                    frame: WorkerFrame::Final(FinalResponse {
                        generation_id: self.generation_id.clone(),
                        generated_ids: content_ids,
                        committed_token_count: committed,
                        decode_fingerprint: self.context.decode_fingerprint.clone(),
                        runtime_config_digest: self.context.runtime_config_digest.clone(),
                        worker_generation: generation,
                        finish_reason: FinishReason::StopToken,
                        constraint_identity: self
                            .context
                            .expected_constraint
                            .as_ref()
                            .map(|c| c.constraint_runtime_identity.clone()),
                        constraint_complete: false,
                        last_completed_sequence: last_sequence,
                    }),
                })
            }
        }
    }

    fn send_continue(&mut self, cont: &GenerateContinue) -> Result<(), WorkerFault> {
        // A continuation whose expected sequence does not match the worker's next
        // sequence is a protocol violation; the scripted worker treats it as a
        // crash-equivalent protocol fault so the supervisor charges it.
        if cont.next_expected_sequence != self.next_sequence {
            return Err(WorkerFault::Crash);
        }
        if cont.next_token_budget == 0 {
            return Err(WorkerFault::Crash);
        }
        let mut log = self.log.borrow_mut();
        log.continue_budgets.push(cont.next_token_budget);
        log.continue_sequences.push(cont.next_expected_sequence);
        Ok(())
    }

    fn send_cancel(&mut self, _cancel: &GenerateCancel) -> Result<CancelAck, WorkerFault> {
        self.log.borrow_mut().cancels += 1;
        // A scripted CancelFailure event makes this cancellation
        // unacknowledged: the worker hangs past the cancel timeout.
        if matches!(
            self.events.get(self.cursor),
            Some(ScriptedEvent::CancelFailure)
        ) {
            self.cursor += 1;
            return Err(WorkerFault::FailedCancellation);
        }
        Ok(CancelAck::Acknowledged {
            committed_token_count: self.committed,
        })
    }

    fn kill(&mut self) {
        self.log.borrow_mut().kills += 1;
    }
}

/// A factory that hands out scripted workers, one per attempt. The script for
/// attempt `i` is `scripts[i]`; attempts beyond the script list crash on the
/// first quantum (a safe default that surfaces as a chargeable crash).
pub struct ScriptedWorkerFactory {
    scripts: Vec<Vec<ScriptedEvent>>,
    context: WorkerStartContext,
    next_generation: u64,
    /// When set, the Nth spawn (0-based) fails as a startup failure.
    pub fail_spawn_at: Option<usize>,
    spawn_count: usize,
    log: Rc<RefCell<ScriptedLog>>,
}

impl ScriptedWorkerFactory {
    #[must_use]
    pub fn new(scripts: Vec<Vec<ScriptedEvent>>, context: WorkerStartContext) -> Self {
        Self {
            scripts,
            context,
            next_generation: 1,
            fail_spawn_at: None,
            spawn_count: 0,
            log: Rc::new(RefCell::new(ScriptedLog::default())),
        }
    }

    #[must_use]
    pub const fn spawn_count(&self) -> usize {
        self.spawn_count
    }

    /// A snapshot of what the supervisor sent to the workers.
    #[must_use]
    pub fn log(&self) -> ScriptedLog {
        self.log.borrow().clone()
    }
}

impl WorkerFactory for ScriptedWorkerFactory {
    fn spawn(&mut self) -> Result<Box<dyn DecodeWorker>, WorkerFault> {
        let index = self.spawn_count;
        self.spawn_count += 1;
        if self.fail_spawn_at == Some(index) {
            return Err(WorkerFault::StartupFailure);
        }
        let generation = self.next_generation;
        self.next_generation += 1;
        let events = self
            .scripts
            .get(index)
            .cloned()
            .unwrap_or_else(|| vec![ScriptedEvent::Crash]);
        Ok(Box::new(ScriptedWorker {
            generation,
            events,
            cursor: 0,
            next_sequence: 1,
            committed: 0,
            context: self.context.clone(),
            started: false,
            generation_id: String::new(),
            log: Rc::clone(&self.log),
        }))
    }
}
