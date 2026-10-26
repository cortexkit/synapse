//! Supervised owned-metal-decode worker protocol and process supervision.
//!
//! This crate is the module-side supervisor for the `owned-metal-decode-worker-v1`
//! protocol, part of the production decode port. It is a pure-Rust state machine:
//! it does not link Metal and does not depend on the decode engine crate, so its
//! fixtures run on any host. The real Metal worker satisfies the
//! [`worker::DecodeWorker`] transport trait defined here.
//!
//! ## Scope
//! - Start validation with dedicated, non-overlapping mismatch mappings
//!   ([`validation`]): protocol/frame → `owned_decode_protocol_mismatch`,
//!   runtime identity → `owned_decode_runtime_config_mismatch`, constraint
//!   identity → `owned_decode_constraint_version_mismatch`, sampling →
//!   `owned_decode_sampling_unsupported`.
//! - One-generation residency and progress/continuation framing with sequence
//!   and session validation ([`protocol`], [`supervisor`]).
//! - Terminal-control boundary precedence: completion > cancellation > deadline
//!   at a non-terminal boundary ([`boundary`]).
//! - Crash-budget persistence, quarantine, and the single permitted worker-crash
//!   redispatch ([`budget`], [`supervisor`]).
//! - Process supervision ([`worker`]). The scripted fault-injection worker and
//!   the manual clock that the fixtures drive are compiled only for this crate's
//!   own tests or with the `test-support` feature, so they never ship in a
//!   production binary.
//!
//! The wire deadline and cancellation literals are mirrored from the module-owned
//! `owned-decode-wire-error-bindings-v1` manifest in [`wire_error_bindings`].

#![forbid(unsafe_code)]

pub mod boundary;
pub mod budget;
pub mod error;
pub mod identity;
pub mod protocol;
pub mod streaming;
pub mod supervisor;
pub mod validation;
pub mod wire_error_bindings;
pub mod worker;

pub use boundary::{evaluate_boundary, BoundaryDecision, BoundaryInputs, Timestamp};
pub use budget::{
    BudgetPolicy, BudgetRecord, CrashBudget, CrashBudgetStore, FileBudgetStore, InMemoryBudgetStore,
};
pub use error::{DecodeError, FailureClassification};
pub use identity::{
    DecodeIdentity, QuarantineKey, RuntimeManifest, SchedulerRuntimeRecord, CONSTRAINT_ENCODING_ID,
    WORKER_PROTOCOL_ID,
};
pub use protocol::{
    FinalResponse, FinishReason, FrameEnvelope, GenerateCancel, GenerateContinue,
    GenerateInstallHintBank, GenerateProgress, GenerateStart, HintBankInstalled, Sampling,
    TokenIdJsonConstraint, WorkerFrame, GREEDY_TOP1,
};
pub use streaming::{
    AbortOutcome, RetainedPrefix, RetentionPreflight, StreamRequest, StreamingSupervisor,
    StreamingSupervisorError, SupervisionCycle, WorkerDeathOutcome,
};
#[cfg(any(test, feature = "test-support"))]
pub use supervisor::ManualClock;
pub use supervisor::{
    Clock, GenerationOutcome, GenerationRequest, Provenance, SuccessOutput, Supervisor,
    TerminalControl,
};
pub use validation::{validate_start, StartAuthorization, WorkerStartContext};
pub use worker::{
    CancelAck, DecodeWorker, HintBankSource, NoHintBankSource, SidecarHintBankPickup, SteppedFrame,
    WorkerFactory, WorkerFault,
};
#[cfg(any(test, feature = "test-support"))]
pub use worker::{ScriptedEvent, ScriptedLog, ScriptedWorkerFactory};
