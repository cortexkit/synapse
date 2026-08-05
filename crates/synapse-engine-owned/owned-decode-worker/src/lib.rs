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
//! - Process supervision and fault injection ([`worker`]).
//! - The `decode-ownership-manifest-v1` fault sites and an ownership ledger the
//!   fixtures use to prove no double free, invalid free, use-after-free, or leak
//!   ([`ownership`]).
//!
//! The wire deadline and cancellation literals are mirrored from the module-owned
//! `owned-decode-wire-error-bindings-v1` manifest in [`wire_error_bindings`].

#![forbid(unsafe_code)]

pub mod boundary;
pub mod budget;
pub mod error;
pub mod identity;
pub mod ownership;
pub mod protocol;
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
pub use ownership::{
    OwnershipFaultSite, OwnershipLedger, OwnershipViolation, ResidentStateKind,
    OWNERSHIP_MANIFEST_REVISION,
};
pub use protocol::{
    FinalResponse, FinishReason, FrameEnvelope, GenerateCancel, GenerateContinue, GenerateProgress,
    GenerateStart, Sampling, TokenIdJsonConstraint, WorkerFrame, GREEDY_TOP1,
};
pub use supervisor::{
    Clock, GenerationOutcome, GenerationRequest, ManualClock, Provenance, SuccessOutput,
    Supervisor, TerminalControl,
};
pub use validation::{validate_start, StartAuthorization, WorkerStartContext};
pub use worker::{
    CancelAck, DecodeWorker, ScriptedEvent, ScriptedLog, ScriptedWorkerFactory, SteppedFrame,
    WorkerFactory, WorkerFault,
};
