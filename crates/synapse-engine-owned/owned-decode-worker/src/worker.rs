//! Worker transport abstraction.
//!
//! The real Metal worker satisfies [`DecodeWorker`]: one transport session bound
//! to one immutable `worker_generation`. [`WorkerFactory`] models process
//! supervision — each [`WorkerFactory::spawn`] starts a fresh worker process
//! with a new generation and session, and can fail as a startup failure. The
//! supervisor uses the factory to perform the single permitted crash redispatch
//! on a brand-new worker.
//!
//! The scripted fault-injection double lives in the `scripted` submodule, which
//! is compiled only for tests or with the `test-support` feature.

use crate::error::DecodeError;
use crate::protocol::{
    GenerateCancel, GenerateContinue, GenerateInstallHintBank, GenerateStart, WorkerFrame,
};
use crate::validation::{StartAuthorization, WorkerStartContext};
use std::sync::mpsc::{Receiver, TryRecvError};
use synapse_core::SidecarHintBank;

#[cfg(any(test, feature = "test-support"))]
mod scripted;
#[cfg(any(test, feature = "test-support"))]
pub use scripted::{ScriptedEvent, ScriptedLog, ScriptedWorker, ScriptedWorkerFactory};

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
    /// A complete transport frame violated the worker protocol without a process failure.
    Protocol,
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

    /// Install a complete sidecar bank after progress and before the next
    /// continuation. The default makes this additive command safe for worker
    /// implementations that intentionally ignore sidecar work.
    fn install_hint_bank(
        &mut self,
        _installation: &GenerateInstallHintBank,
    ) -> Result<(), WorkerFault> {
        Ok(())
    }

    /// Authorize the next quantum.
    fn send_continue(&mut self, cont: &GenerateContinue) -> Result<(), WorkerFault>;

    /// Request cancellation. The worker destroys resident state and acknowledges.
    fn send_cancel(&mut self, cancel: &GenerateCancel) -> Result<CancelAck, WorkerFault>;

    /// Forcibly terminate the worker process. This is the escalation for a
    /// cancellation the worker fails to acknowledge; resident state is
    /// destroyed with the process.
    fn kill(&mut self);
}

/// A bounded, non-blocking source of a request-scoped sidecar bank.
///
/// `poll` must only inspect in-memory readiness. It may not wait for sidecar
/// work, allocate an unbounded buffer, or drive target decoding itself.
pub trait HintBankSource {
    fn poll(&mut self) -> Option<SidecarHintBank>;
}

/// The default source for the dormant feature.
#[derive(Default)]
pub struct NoHintBankSource;

impl HintBankSource for NoHintBankSource {
    fn poll(&mut self) -> Option<SidecarHintBank> {
        None
    }
}

/// A one-shot sidecar completion source backed by the module's bounded channel.
///
/// Once a bank arrives it remains available for a crash redispatch, but a pending
/// channel is queried only with `try_recv`, so it cannot delay target progress.
pub struct SidecarHintBankPickup {
    receiver: Option<Receiver<SidecarHintBank>>,
    ready: Option<SidecarHintBank>,
}

impl SidecarHintBankPickup {
    #[must_use]
    pub fn pending(receiver: Receiver<SidecarHintBank>) -> Self {
        Self {
            receiver: Some(receiver),
            ready: None,
        }
    }

    #[must_use]
    pub fn ready(bank: SidecarHintBank) -> Self {
        Self {
            receiver: None,
            ready: Some(bank),
        }
    }
}

impl HintBankSource for SidecarHintBankPickup {
    fn poll(&mut self) -> Option<SidecarHintBank> {
        if self.ready.is_none() {
            let receiver = self.receiver.as_ref()?;
            match receiver.try_recv() {
                Ok(bank) => {
                    self.ready = Some(bank);
                    self.receiver = None;
                }
                Err(TryRecvError::Empty) => return None,
                Err(TryRecvError::Disconnected) => self.receiver = None,
            }
        }
        self.ready.clone()
    }
}

/// Process supervision: spawn a fresh worker process for an attempt.
pub trait WorkerFactory {
    /// Spawn a new worker with a fresh generation and session. A spawn failure
    /// is a startup failure (replacement startup or load failure).
    fn spawn(&mut self) -> Result<Box<dyn DecodeWorker>, WorkerFault>;
}
