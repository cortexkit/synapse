//! Certification, serving-predicate, and acceptance-gate machinery for the production
//! owned-metal-decode lane.
//!
//! This module is the evidence layer above routing and the scheduler mechanism:
//!
//! - [`fixtures`]: the immutable `decode-parity` battery (both families, both
//!   formats, 20 prompts by 64 tokens) and the independent oracle store whose
//!   bytes change only through explicit oracle review and are never produced
//!   from production-worker output.
//! - [`probe`]: certification probes that run the battery through the
//!   [`probe::DecodeProbe`] seam, compare against the oracle, apply the
//!   structural-band fork rules, and record machine-profile-local certification
//!   rows (fail closed).
//! - [`scheduler_evidence`]: ingestion of the `decode-sched-manifest-v1`
//!   evidence record. Until the numeric scheduler commitment lands
//!   (OQ-DEC-SCHED-01), the status is blocked and owned serving stays disabled.
//! - [`migration`]: the one-shot migration seed boundary and retained wire
//!   binding validation helper. The seed is not part of serving or certification.
//! - [`gates`]: the G-DEC-01 through G-DEC-12 gate runner and release evidence.
//!   Every applicable gate executes with zero skips; G-DEC-11 and the
//!   scheduler-dependent portion of G-DEC-12 report blocked (not skipped) until
//!   the numeric manifest is committed and executed.
//!
//! Test-only tooling (compiled under `cfg(test)` only, never linked into the
//! serving path):
//!
//! - [`spike_harness`]: spike-vs-production parity and throughput runners with
//!   deterministic synthetic timings.
//! - [`metal_probe`]: the checkpoint-gated certification probe that runs the
//!   real Metal step engines on the mandatory `macos-metal` lane.
//!
//! The source lives under `crates/synapse-module/owned-decode-certification/`;
//! a `#[path]` attribute in the crate root wires that directory in as this
//! module, matching the `owned-decode-routing` and
//! `owned-decode-grammar-scheduler` precedent.

pub mod fixture_groups;
pub mod fixtures;
pub mod gates;
pub mod migration;
pub mod probe;
pub mod scheduler_evidence;

#[cfg(test)]
mod spike_harness;

#[cfg(all(test, target_os = "macos"))]
mod metal_probe;

pub use fixture_groups::{
    run_constrained_negative, run_constrained_positive, run_request_processing,
    run_scheduler_continuity, GroupOutcome, CONSTRAINED_NEGATIVE_GROUP, CONSTRAINED_POSITIVE_GROUP,
    REQUEST_PROCESSING_GROUP, SCHEDULER_CONTINUITY_GROUP,
};
pub use fixtures::{
    battery_digest, parity_battery, spike_reference_stream, token_stream_digest, FixtureError,
    OracleProvenance, OracleStore, ParityFixture,
};
pub use gates::{
    applicable_skips, release_ready, GateId, GateRunner, GateStatus, GrammarCostEvidence,
    ReleaseEvidence, ThroughputEvidence, ALL_GATES,
};
pub use migration::wire_bindings_are_literal;
pub use probe::{
    compare_streams, fork_summary, CertificationEvidence, CertificationProbe, DecodeProbe,
    ForkDivergence, ForkSummary, OracleReproducingProbe,
};
pub use scheduler_evidence::{
    ingest_scheduler_evidence, scheduler_evidence_committed, SchedulerEvidenceStatus,
    CANDIDATE_N_VALUES,
};
