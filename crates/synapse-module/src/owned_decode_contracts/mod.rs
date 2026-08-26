//! Module-owned schemas and checked-in records for the production owned-decode
//! lane (`engine=owned-metal-decode`).
//!
//! This module is the single source of truth for the manifest shapes that gate
//! the owned-decode serving and certification contracts. It defines the serde schemas for every checked-in
//! artifact under `crates/synapse-module/owned-decode-manifests/` and the
//! validation rules the specification attaches to them. Production code and CI
//! probes load manifests through [`load_manifest_dir`], which parses each
//! artifact against its schema and runs the cross-manifest invariants the spec
//! makes binding:
//!
//! - Production manifests live only under `crates/synapse-module/`; a manifest
//!   that resolves outside that tree is rejected so a spike-only artifact can
//!   never pose as a deployed record.
//! - Wire-error bindings carry literal deadline and cancellation IDs, never the
//!   symbolic spec placeholders `existing_deadline_error` /
//!   `existing_cancellation_error`.
//! - Catalog identity fields match the canonical dedicated values and the
//!   mirrored aliases agree.
//! - Concrete writable arrays declared by the ownership manifest do not
//!   overlap, so two fault sites never claim the same buffer region.
//! - No production manifest references a path under `bench/spikes/`; the spike
//!   tree is read-only reference material for port slices.
//!
//! The schemas are intentionally narrow: every field the specification names is
//! represented, and `deny_unknown_fields` is used throughout so a typo or a
//! forward-incompatible addition fails closed at load time rather than being
//! silently dropped.

use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Revision of the owned-decode contract schema set. Bumped when any manifest
/// shape in this module changes in a way that invalidates prior checked-in
/// records.
pub const CONTRACT_SCHEMA_REVISION: &str = "owned-decode-contracts-v1";

/// Canonical root that production owned-decode manifests must live under. The
/// spec (resolutions r2 #10) requires all new production manifests to live
/// under `crates/synapse-module/`, never under `bench/spikes/`.
pub const PRODUCTION_MANIFEST_ROOT: &str = "crates/synapse-module";

/// Read-only reference tree for port slices. Production manifests may not
/// reference paths under it.
pub const SPIKE_REFERENCE_ROOT: &str = "bench/spikes/";

/// Symbolic placeholder names the spec forbids in emitted evidence. The wire
/// error binding manifest must carry literal IDs instead.
const FORBIDDEN_WIRE_SYMBOLS: &[&str] = &["existing_deadline_error", "existing_cancellation_error"];

/// Supported production model families. The spec recognizes exactly these two.
pub const SUPPORTED_FAMILIES: &[&str] = &["qwen3-0.6b", "lfm2-1.2b"];

/// Supported activation dtypes.
pub const SUPPORTED_ACTIVATION_DTYPES: &[&str] = &["f16"];

/// Supported weight quantizations.
pub const SUPPORTED_WEIGHT_QUANTS: &[&str] = &["f16", "q8_0"];

/// Candidate production N values. Exactly one is selected by the G-DEC-11
/// measurement. N=1 is prohibited.
pub const CANDIDATE_PRODUCTION_N: &[u32] = &[8, 16, 32];

/// Canonical worker protocol id.
pub const WORKER_PROTOCOL_ID: &str = "owned-metal-decode-worker-v1";

/// Canonical constraint encoding carried over the worker boundary.
pub const CONSTRAINT_ENCODING_ID: &str = "token-id-json-constraint-v1";

/// JSON schema subset accepted by the grammar compiler.
pub const GRAMMAR_SUBSET_ID: &str = "synapse-json-schema-v1";

/// Error returned when a manifest fails schema or cross-manifest validation.
#[derive(Debug, Error)]
pub enum ManifestError {
    #[error("owned-decode manifest I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("owned-decode manifest {manifest} at {path} failed to parse: {source}")]
    Parse {
        manifest: &'static str,
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("owned-decode manifest {manifest} at {path} is outside the production root `{expected_root}`")]
    OutsideProductionRoot {
        manifest: &'static str,
        path: PathBuf,
        expected_root: &'static str,
    },
    #[error("owned-decode manifest {manifest} at {path} references spike tree `{spike_root}` via field `{field}`")]
    SpikeReference {
        manifest: &'static str,
        path: PathBuf,
        field: String,
        spike_root: &'static str,
    },
    #[error("owned-decode wire error binding at {path} carries unresolved symbolic literal `{literal}`; only concrete wire IDs are permitted")]
    UnresolvedWireLiteral { path: PathBuf, literal: String },
    #[error(
        "owned-decode manifest {manifest} at {path} has invalid identity field `{field}`: {reason}"
    )]
    InvalidIdentity {
        manifest: &'static str,
        path: PathBuf,
        field: &'static str,
        reason: String,
    },
    #[error("owned-decode ownership manifest at {path} declares overlapping concrete writable arrays: {first} overlaps {second}")]
    OverlappingWritableArrays {
        path: PathBuf,
        first: String,
        second: String,
    },
    #[error("owned-decode manifest {manifest} at {path}: {reason}")]
    Invalid {
        manifest: &'static str,
        path: PathBuf,
        reason: String,
    },
}

/// Result of loading and validating the full manifest directory.
#[derive(Debug, Clone)]
pub struct ManifestDir {
    pub root: PathBuf,
    pub context_buckets: ContextBucketsManifest,
    pub scheduler: SchedulerManifest,
    pub ownership: OwnershipManifest,
    pub fixture_registry: FixtureRegistryManifest,
    pub structural_band: StructuralBandManifest,
    pub wire_bindings: WireErrorBindingsManifest,
    pub grammar_cost: GrammarCostCorpusManifest,
    pub ci_lane: CiLaneManifest,
    pub slice_plan: SlicePlanManifest,
}

// ---------------------------------------------------------------------------
// Context buckets manifest — decode-context-buckets-v1
// ---------------------------------------------------------------------------

/// `decode-context-buckets-v1`: the shippable context manifest referenced by
/// production catalog validation. It begins with `{512,1024,2048}` per family,
/// removes failed buckets before shipment, and retains at least one verified
/// bucket per family. Its revision enters `runtime_config_digest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextBucketsManifest {
    pub manifest_revision: String,
    pub schema_revision: String,
    /// Per-family verified bucket list. Each family must retain at least one
    /// verified bucket after pre-ship removal of failed buckets.
    pub families: Vec<ContextBucketFamily>,
    /// Buckets removed before shipment because they failed attention-KV or
    /// applicable LFM2 convolution-cache capacity verification.
    #[serde(default)]
    pub removed_buckets: Vec<RemovedBucket>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextBucketFamily {
    pub family: String,
    /// Verified, shippable bucket sizes in ascending order.
    pub verified_buckets: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemovedBucket {
    pub family: String,
    pub max_context_tokens: u32,
    pub reason: String,
}

// ---------------------------------------------------------------------------
// Scheduler manifest — decode-sched-manifest-v1
// ---------------------------------------------------------------------------

/// `decode-sched-manifest-v1`: separate `runtime` and `workload` records.
/// Runtime contains exactly one N from `{8,16,32}`, yield-policy revision,
/// DECODE weight, aging window, and progress-protocol revision. Exactly those
/// scheduler fields enter `runtime_config_digest`; none enters
/// `decode_fingerprint`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerManifest {
    pub manifest_revision: String,
    pub schema_revision: String,
    pub runtime: SchedulerRuntimeRecord,
    pub workload: SchedulerWorkloadRecord,
    #[serde(default)]
    pub evidence: SchedulerEvidenceRecord,
}

/// The five runtime-effective scheduler fields that enter
/// `runtime_config_digest`. The cancellation-latency bound is a derived
/// quantity and is NOT an independent digest input.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerRuntimeRecord {
    /// Exactly one production N selected from `{8,16,32}`. N=1 is prohibited.
    pub production_n: u32,
    pub yield_policy_revision: String,
    pub decode_weight: u32,
    pub decode_aging_window_ms: u64,
    pub progress_protocol_revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerWorkloadRecord {
    pub family: String,
    pub format: String,
    pub context_bucket: u32,
    pub prompt_length: u32,
    pub output_length: u32,
    pub embedding_concurrency: u32,
    pub arrival_process: String,
    pub warmup: u32,
    pub duration_ms: u64,
    pub embed_query_p95_slo_ms: f64,
    pub baseline_calculation: String,
    pub regression_calculation: String,
    /// Derived cancellation-latency bound (N x per-token time bound). Lives in
    /// the workload record, never in the runtime record's identity fields.
    pub cancellation_latency_bound_ms: u64,
    #[serde(default)]
    pub cancellation_observations: Vec<f64>,
    #[serde(default)]
    pub deadline_observations: Vec<f64>,
    pub timing_boundaries: String,
    pub percentile_method: String,
}

/// One measured candidate quantum from the OQ-DEC-SCHED-01 mixed-load run.
/// The evidence table must cover every candidate N in `{8,16,32}`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerCandidateEvidence {
    /// The candidate quantum measured by this cell.
    pub n: u32,
    /// Embed.query latency percentiles (nearest-rank) under mixed load.
    pub embed_p50_ms: f64,
    pub embed_p95_ms: f64,
    pub embed_p99_ms: f64,
    /// Effective decode throughput under mixed load, including yield time and
    /// generation restarts.
    pub decode_tokens_per_sec: f64,
    /// Quantum boundary count observed in the measured window.
    pub quantum_boundaries: u64,
    /// Continuation count (non-final boundaries followed by another quantum).
    pub continuations: u64,
    /// Longest single quantum observed in the cell (bounds cancellation
    /// deferral, which is evaluated at quantum boundaries).
    pub max_quantum_ms: f64,
    /// 1/5/15-minute loadavg recorded before and after the cell.
    pub loadavg_before: [f64; 3],
    pub loadavg_after: [f64; 3],
    /// True when the cell started with a 1-minute loadavg above 4 on the
    /// shared measurement machine.
    pub ran_above_load4: bool,
    /// Whether this candidate's embed.query p95 met the committed SLO.
    pub meets_slo: bool,
    /// Measured window length and completed embed queries.
    pub window_ms: u64,
    pub embed_queries: u64,
    /// Same-session embed.query p95 regression versus the embed-only
    /// baseline, in percent.
    pub embed_regression_pct: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerEvidenceRecord {
    #[serde(default)]
    pub committed_n: Option<u32>,
    #[serde(default)]
    pub max_uninterruptible_gpu_time_ms: Option<f64>,
    #[serde(default)]
    pub observed_n: Option<u32>,
    #[serde(default)]
    pub continuation_count: Option<u32>,
    #[serde(default)]
    pub sequence_traces: Vec<String>,
    #[serde(default)]
    pub permit_events: Vec<String>,
    #[serde(default)]
    pub queue_depth: Vec<u32>,
    #[serde(default)]
    pub per_operation_waiting_ms: Vec<f64>,
    #[serde(default)]
    pub cancellation_latency_ms: Vec<f64>,
    #[serde(default)]
    pub deadline_latency_ms: Vec<f64>,
    /// Per-candidate mixed-load measurement table covering `{8,16,32}`.
    #[serde(default)]
    pub candidates: Vec<SchedulerCandidateEvidence>,
    /// The embed.query p95 SLO the candidates were judged against (mirrors
    /// the workload record's committed SLO).
    #[serde(default)]
    pub embed_query_p95_slo_ms: Option<f64>,
    /// Same-session embed-only baseline measured before decode admission.
    #[serde(default)]
    pub baseline_embed_only_p50_ms: Option<f64>,
    #[serde(default)]
    pub baseline_embed_only_p95_ms: Option<f64>,
    /// Uninterrupted decode throughput baseline (no embed load), tok/s.
    #[serde(default)]
    pub decode_only_tokens_per_sec: Option<f64>,
    /// The machine the measurement ran on (chip, model, RAM, OS).
    #[serde(default)]
    pub machine_profile_note: Option<String>,
    /// UTC timestamp of the measurement run.
    #[serde(default)]
    pub measured_at_utc: Option<String>,
    /// The measurement protocol revision executed.
    #[serde(default)]
    pub protocol_id: Option<String>,
    /// Boundary-crossing bit-exactness spot check lines: chunked N streams
    /// versus the uninterrupted greedy stream, and (from protocol v2)
    /// chunked-prefill KV/first-token comparisons versus the uninterrupted
    /// prefill.
    #[serde(default)]
    pub parity_spot_check: Vec<String>,
    /// Prior evidence records for this workload, oldest first. A protocol
    /// re-run appends the record it replaces instead of overwriting it, so an
    /// honest negative is never lost: the v1 record (protocol
    /// oq-dec-sched-01-mixed-load-v1, no candidate met the SLO) stays in the
    /// tree behind the current record. Empty for records that have never
    /// been superseded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<SchedulerEvidenceRecord>,
}

// ---------------------------------------------------------------------------
// Ownership manifest — decode-ownership-manifest-v1
// ---------------------------------------------------------------------------

/// `decode-ownership-manifest-v1`: covers allocation, ownership transfer,
/// partial initialization, generation, cancellation, timeout cleanup, unload,
/// shutdown, and LFM2 convolution-cache ownership across Objective-C/C and
/// Rust FFI boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OwnershipManifest {
    pub manifest_revision: String,
    pub schema_revision: String,
    pub fault_sites: Vec<FaultSite>,
    /// Concrete writable arrays declared across FFI boundaries. Validation
    /// rejects overlapping arrays so two fault sites never claim the same
    /// buffer region.
    pub concrete_writable_arrays: Vec<ConcreteWritableArray>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FaultSite {
    pub id: String,
    pub group: String,
    pub description: String,
    pub ownership_rule: String,
    /// Test name exercising this fault site under AddressSanitizer.
    pub asan_test_name: String,
    /// Run record reference for the mandatory macos-metal lane.
    pub run_record: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConcreteWritableArray {
    pub id: String,
    pub owner: String,
    /// Inclusive byte offset range. Two arrays overlap if their ranges
    /// intersect; validation rejects any overlap.
    pub start_offset: u64,
    pub end_offset: u64,
    pub element_type: String,
    pub length: u64,
}

// ---------------------------------------------------------------------------
// Fixture registry — decode-fixture-registry-v1
// ---------------------------------------------------------------------------

/// `decode-fixture-registry-v1`: referenced directly by probe code and
/// mandatory `macos-metal` CI. Every entry has a stable ID and exactly one
/// group.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureRegistryManifest {
    pub manifest_revision: String,
    pub schema_revision: String,
    pub groups: Vec<FixtureGroup>,
    pub entries: Vec<FixtureEntry>,
    /// Reference to the grammar-cost corpus revision.
    pub grammar_cost_corpus_revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureGroup {
    pub id: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureEntry {
    pub id: String,
    pub group: String,
    pub kind: FixtureKind,
    #[serde(default)]
    pub details: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureKind {
    DecodeParity,
    RequestProcessing,
    ConstrainedPositive,
    ConstrainedNegative,
    SchedulerContinuity,
}

// ---------------------------------------------------------------------------
// Structural-band manifest — structural-band-v1
// ---------------------------------------------------------------------------

/// `structural-band-v1`: records the permitted structural-band fork signature
/// rules. First f16 certification records a fork signature with at most two
/// top-2 swaps; Q8 requires zero forks. Recertification requires the stored
/// signature exactly. No cross-profile f16 equality is promised.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralBandManifest {
    pub manifest_revision: String,
    pub schema_revision: String,
    pub rules: Vec<StructuralBandRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralBandRule {
    pub family: String,
    pub weight_quant: String,
    /// Maximum permitted top-2 swaps. f16 allows at most two; q8_0 allows
    /// zero.
    pub max_top2_swaps: u32,
    pub fork_signature_recertification: String,
}

// ---------------------------------------------------------------------------
// Wire error bindings — owned-decode-wire-error-bindings-v1
// ---------------------------------------------------------------------------

/// `owned-decode-wire-error-bindings-v1`: binds `deadline_error_id` and
/// `cancellation_error_id` to the exact existing stable wire literals. The
/// symbolic names `existing_deadline_error` and `existing_cancellation_error`
/// are spec placeholders only and must not appear in emitted responses or
/// passing evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireErrorBindingsManifest {
    pub manifest_revision: String,
    pub schema_revision: String,
    /// Authoritative request-contract revision the bindings are bound to.
    pub request_contract_revision: String,
    pub deadline_error_id: String,
    pub cancellation_error_id: String,
    /// Changelog of retired wire errors (e.g. `grammar_unavailable_in_build`).
    #[serde(default)]
    pub wire_changelog: Vec<WireChangelogEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WireChangelogEntry {
    pub retired_id: String,
    pub replacement_id: String,
    pub affected_consumers: Vec<String>,
    pub notes: String,
}

// ---------------------------------------------------------------------------
// Grammar cost corpus — grammar-cost-corpus-v1
// ---------------------------------------------------------------------------

/// `grammar-cost-corpus-v1`: fixes fixture IDs, model identity, prompts,
/// schemas, output count, warmup, sampling, repetitions, timing boundaries,
/// and p50/p95 calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrammarCostCorpusManifest {
    pub manifest_revision: String,
    pub schema_revision: String,
    pub model_identity: GrammarCostModelIdentity,
    pub fixtures: Vec<GrammarCostFixture>,
    pub warmup: u32,
    pub sampling: String,
    pub repetitions: u32,
    pub timing_boundaries: String,
    pub percentile_calculation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrammarCostModelIdentity {
    pub family: String,
    pub activation_dtype: String,
    pub weight_quant: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrammarCostFixture {
    pub id: String,
    pub prompt: String,
    pub schema: Value,
    pub output_count: u32,
}

// ---------------------------------------------------------------------------
// CI lane manifest — ci-lane-manifest-v1
// ---------------------------------------------------------------------------

/// `ci-lane-manifest-v1`: names normal targets and mandatory `macos-metal`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CiLaneManifest {
    pub manifest_revision: String,
    pub schema_revision: String,
    pub normal_targets: Vec<String>,
    pub mandatory_targets: Vec<String>,
    /// Gate IDs that must pass with zero skips on the mandatory lane.
    pub mandatory_lane_gates: Vec<String>,
}

// ---------------------------------------------------------------------------
// Slice plan — slice-plan-v1
// ---------------------------------------------------------------------------

/// `slice-plan-v1`: metadata for this contract-manifests slice. The file fence
/// restricts where production manifests and their schemas may live.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SlicePlanManifest {
    pub manifest_revision: String,
    pub schema_revision: String,
    pub slice_id: String,
    pub work_item: String,
    pub file_fence: Vec<String>,
    pub artifacts: Vec<String>,
}

// ---------------------------------------------------------------------------
// Loading and validation
// ---------------------------------------------------------------------------

/// Load and validate every checked-in manifest under `manifest_dir`. The
/// directory is expected to be `crates/synapse-module/owned-decode-manifests`.
/// Returns the parsed and cross-validated manifests, or the first validation
/// error.
pub fn load_manifest_dir(manifest_dir: &Path) -> Result<ManifestDir, ManifestError> {
    // Resolve the manifest directory. When the path exists, canonicalize it
    // and verify the canonicalized path sits under the production root (this
    // catches symlink escapes into the spike tree). When the path does NOT
    // exist, canonicalization would fail with an I/O error; instead, check
    // the raw path with component-wise prefix semantics so a non-existent
    // spike-only directory is rejected with the typed `OutsideProductionRoot`
    // error rather than an opaque I/O error.
    let root = match manifest_dir.canonicalize() {
        Ok(resolved) => {
            ensure_inside_production_root(&resolved)?;
            resolved
        }
        Err(source) => {
            // The path does not exist (or is unreadable). If the raw path is
            // outside the production root, fail with the typed
            // `OutsideProductionRoot` error; otherwise surface the original
            // I/O error so a genuine filesystem failure is not mislabeled.
            ensure_raw_path_inside_production_root(manifest_dir)?;
            return Err(ManifestError::Io {
                path: manifest_dir.to_path_buf(),
                source,
            });
        }
    };

    let context_buckets = load_one_with_spike_scan::<ContextBucketsManifest>(
        &root,
        "decode-context-buckets-v1.json",
        "context_buckets",
    )?;
    let scheduler = load_one_with_spike_scan::<SchedulerManifest>(
        &root,
        "decode-sched-manifest-v1.json",
        "scheduler",
    )?;
    let ownership = load_one_with_spike_scan::<OwnershipManifest>(
        &root,
        "decode-ownership-manifest-v1.json",
        "ownership",
    )?;
    let fixture_registry = load_one_with_spike_scan::<FixtureRegistryManifest>(
        &root,
        "decode-fixture-registry-v1.json",
        "fixture_registry",
    )?;
    let structural_band = load_one_with_spike_scan::<StructuralBandManifest>(
        &root,
        "structural-band-v1.json",
        "structural_band",
    )?;
    let wire_bindings = load_one_with_spike_scan::<WireErrorBindingsManifest>(
        &root,
        "owned-decode-wire-error-bindings-v1.json",
        "wire_bindings",
    )?;
    let grammar_cost = load_one_with_spike_scan::<GrammarCostCorpusManifest>(
        &root,
        "grammar-cost-corpus-v1.json",
        "grammar_cost",
    )?;
    let ci_lane =
        load_one_with_spike_scan::<CiLaneManifest>(&root, "ci-lane-manifest-v1.json", "ci_lane")?;
    let slice_plan =
        load_one_with_spike_scan::<SlicePlanManifest>(&root, "slice-plan-v1.json", "slice_plan")?;

    validate_context_buckets(
        &context_buckets,
        &root.join("decode-context-buckets-v1.json"),
    )?;
    validate_scheduler(&scheduler, &root.join("decode-sched-manifest-v1.json"))?;
    validate_ownership(&ownership, &root.join("decode-ownership-manifest-v1.json"))?;
    validate_fixture_registry(
        &fixture_registry,
        &root.join("decode-fixture-registry-v1.json"),
    )?;
    validate_structural_band(&structural_band, &root.join("structural-band-v1.json"))?;
    validate_wire_bindings(
        &wire_bindings,
        &root.join("owned-decode-wire-error-bindings-v1.json"),
    )?;
    validate_grammar_cost(&grammar_cost, &root.join("grammar-cost-corpus-v1.json"))?;
    validate_ci_lane(&ci_lane, &root.join("ci-lane-manifest-v1.json"))?;
    validate_slice_plan(&slice_plan, &root.join("slice-plan-v1.json"))?;

    Ok(ManifestDir {
        root,
        context_buckets,
        scheduler,
        ownership,
        fixture_registry,
        structural_band,
        wire_bindings,
        grammar_cost,
        ci_lane,
        slice_plan,
    })
}

/// Load and parse one manifest, then scan its raw JSON value for any string
/// field that references the spike tree (`bench/spikes/`). The spike-reference
/// invariant is binding (spec resolutions r2 #10): no production manifest may
/// reference a path under `bench/spikes/`. Scanning at load time — rather than
/// only in a test — means a checked-in manifest that gained a
/// `"run_record": "bench/spikes/..."` field fails at the production load
/// entrypoint with the existing `SpikeReference` error, not just in CI.
///
/// The typed parse and the spike scan run on the same bytes, so a field that
/// serde drops (e.g. an unknown field under `deny_unknown_fields` would already
/// have failed the parse) is still scanned in the raw value before the typed
/// value is trusted.
fn load_one_with_spike_scan<T: for<'de> Deserialize<'de>>(
    root: &Path,
    file: &str,
    manifest: &'static str,
) -> Result<T, ManifestError> {
    let path = root.join(file);
    let bytes = std::fs::read(&path).map_err(|source| ManifestError::Io {
        path: path.clone(),
        source,
    })?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|source| ManifestError::Parse {
        manifest,
        path: path.clone(),
        source,
    })?;
    let hits = find_spike_references(&value);
    if let Some(hit) = hits.first() {
        // The field name is not recoverable from the raw-value walk without
        // threading path state; the offending string value is reported as the
        // `field` so the operator can locate it.
        return Err(ManifestError::SpikeReference {
            manifest,
            path: path.clone(),
            field: hit.clone(),
            spike_root: SPIKE_REFERENCE_ROOT,
        });
    }
    serde_json::from_slice::<T>(&bytes).map_err(|source| ManifestError::Parse {
        manifest,
        path: path.clone(),
        source,
    })
}

/// Reject a raw (pre-canonicalization) manifest directory path that does not
/// sit under the production root. This catches a spike-only path before the
/// canonicalization I/O error and keeps the rejection typed as
/// `OutsideProductionRoot`.
///
/// The production root (`crates/synapse-module`) must be a *leading* component
/// sequence of the path, not merely a subsequence of its components. A path
/// like `bench/spikes/crates/synapse-module/x` contains the root segments in
/// order but not as a prefix, so it is rejected. Non-existent paths (the
/// common case for a typo or a spike-only directory) are handled here by
/// component-wise comparison, since `canonicalize` would fail on them.
fn ensure_raw_path_inside_production_root(raw: &Path) -> Result<(), ManifestError> {
    if !path_leads_with_production_root(raw) {
        return Err(ManifestError::OutsideProductionRoot {
            manifest: "manifest_dir",
            path: raw.to_path_buf(),
            expected_root: PRODUCTION_MANIFEST_ROOT,
        });
    }
    Ok(())
}

/// Reject a canonicalized manifest directory that resolves outside the
/// production root. This is the backstop for symlink escapes: the raw check
/// confirms the requested path leads with `crates/synapse-module`, but a
/// symlink could redirect it into the spike tree. The canonicalized path is
/// absolute, so the worktree root precedes the production-root segments; we require
/// `crates/synapse-module` to appear as a *contiguous* component sequence and
/// reject any canonicalized path that passes through `bench/spikes` before
/// reaching it.
fn ensure_inside_production_root(resolved: &Path) -> Result<(), ManifestError> {
    if !path_contains_production_root_not_via_spikes(resolved) {
        return Err(ManifestError::OutsideProductionRoot {
            manifest: "manifest_dir",
            path: resolved.to_path_buf(),
            expected_root: PRODUCTION_MANIFEST_ROOT,
        });
    }
    Ok(())
}

/// Whether `path` begins with the production-root component sequence
/// `crates/synapse-module`. The root segments must appear as a *contiguous
/// leading* run of normal components, after skipping any `CurDir`/`RootDir`/
/// prefix components that do not name a path element. A path that contains
/// the segments in order but not as a prefix (e.g.
/// `bench/spikes/crates/synapse-module/x`) returns false.
///
/// `ParentDir` components are treated as ordinary non-matching components:
/// they break any in-progress prefix match and can never satisfy the root
/// prefix. This rejects `../crates/synapse-module/y` even though it textually
/// contains the prefix, because the `..` component precedes it.
fn path_leads_with_production_root(path: &Path) -> bool {
    let expected: Vec<&str> = PRODUCTION_MANIFEST_ROOT.split('/').collect();
    let mut matched = 0usize;
    // Whether we have seen any normal/parent component before completing the
    // prefix. A leading prefix must match from the first path element, so any
    // non-matching normal component before the prefix completes means the
    // root is not leading (even if it appears contiguously later).
    let mut seen_non_prefix_element = false;
    for component in path.components() {
        // Once the full prefix is matched, later components do not matter.
        if matched == expected.len() {
            return true;
        }
        match component {
            // Skip directory separators and platform prefixes (e.g. `C:` on
            // Windows, `/` on POSIX) without advancing or resetting the match.
            Component::Prefix(_) | Component::RootDir | Component::CurDir => continue,
            // A parent component is a real path element that is not the next
            // expected root segment, so it breaks an in-progress match.
            Component::ParentDir => {
                seen_non_prefix_element = true;
                matched = 0;
                continue;
            }
            Component::Normal(s) => {
                if !seen_non_prefix_element && s == expected[matched] {
                    matched += 1;
                } else {
                    // A non-matching normal component means the root is not a
                    // leading prefix. Once this happens, the prefix can never
                    // be satisfied even if the sequence appears later.
                    seen_non_prefix_element = true;
                    matched = 0;
                }
            }
        }
    }
    matched == expected.len()
}

/// Whether `path` contains the production-root component sequence
/// `crates/synapse-module` as a *contiguous* run of normal components and does
/// not pass through the spike tree (`bench/spikes`) before reaching it. Used
/// for canonicalized absolute paths, where the worktree root legitimately
/// precedes the production root.
fn path_contains_production_root_not_via_spikes(path: &Path) -> bool {
    let expected: Vec<&str> = PRODUCTION_MANIFEST_ROOT.split('/').collect();
    let spike: Vec<&str> = SPIKE_REFERENCE_ROOT
        .trim_end_matches('/')
        .split('/')
        .collect();
    let components: Vec<String> = path
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    // Reject if the spike tree appears as a contiguous component sequence
    // before the production root. A symlink that redirects into
    // `bench/spikes/crates/synapse-module/...` would canonicalize to a path
    // whose components include `bench`, `spikes`, `crates`, `synapse-module`
    // in order; the contiguous production-root check below would still pass,
    // so this spike-prefix scan is what catches the escape.
    let mut spike_matched = 0usize;
    let mut root_matched = 0usize;
    let mut root_found = false;
    let mut spike_seen_before_root = false;
    for component in &components {
        // Track the spike-tree prefix only until the production root is found.
        if !root_found {
            if spike_matched < spike.len() && component.as_str() == spike[spike_matched] {
                spike_matched += 1;
                if spike_matched == spike.len() {
                    spike_seen_before_root = true;
                }
            } else {
                spike_matched = 0;
            }
        }
        // Track the contiguous production-root sequence. Once found, stop
        // scanning for it.
        if !root_found {
            if root_matched < expected.len() && component.as_str() == expected[root_matched] {
                root_matched += 1;
                if root_matched == expected.len() {
                    root_found = true;
                }
            } else {
                root_matched = 0;
            }
        }
    }
    root_found && !spike_seen_before_root
}

fn validate_context_buckets(
    manifest: &ContextBucketsManifest,
    path: &Path,
) -> Result<(), ManifestError> {
    if manifest.families.len() != SUPPORTED_FAMILIES.len() {
        return Err(ManifestError::Invalid {
            manifest: "context_buckets",
            path: path.to_path_buf(),
            reason: format!(
                "expected {} families, found {}",
                SUPPORTED_FAMILIES.len(),
                manifest.families.len()
            ),
        });
    }
    let mut seen = BTreeMap::new();
    for family in &manifest.families {
        if !SUPPORTED_FAMILIES.contains(&family.family.as_str()) {
            return Err(ManifestError::InvalidIdentity {
                manifest: "context_buckets",
                path: path.to_path_buf(),
                field: "family",
                reason: format!("unsupported family `{}`", family.family),
            });
        }
        if family.verified_buckets.is_empty() {
            return Err(ManifestError::InvalidIdentity {
                manifest: "context_buckets",
                path: path.to_path_buf(),
                field: "verified_buckets",
                reason: format!(
                    "family `{}` must retain at least one verified bucket",
                    family.family
                ),
            });
        }
        for bucket in &family.verified_buckets {
            if *bucket == 0 {
                return Err(ManifestError::InvalidIdentity {
                    manifest: "context_buckets",
                    path: path.to_path_buf(),
                    field: "verified_buckets",
                    reason: format!("family `{}` has a zero bucket", family.family),
                });
            }
        }
        seen.insert(family.family.clone(), family.verified_buckets.clone());
    }
    for required in SUPPORTED_FAMILIES {
        if !seen.contains_key(*required) {
            return Err(ManifestError::InvalidIdentity {
                manifest: "context_buckets",
                path: path.to_path_buf(),
                field: "family",
                reason: format!("missing required family `{}`", required),
            });
        }
    }
    Ok(())
}

fn validate_scheduler(manifest: &SchedulerManifest, path: &Path) -> Result<(), ManifestError> {
    if !CANDIDATE_PRODUCTION_N.contains(&manifest.runtime.production_n) {
        return Err(ManifestError::InvalidIdentity {
            manifest: "scheduler",
            path: path.to_path_buf(),
            field: "production_n",
            reason: format!(
                "production_n must be one of {:?}; N=1 is prohibited",
                CANDIDATE_PRODUCTION_N
            ),
        });
    }
    if !SUPPORTED_FAMILIES.contains(&manifest.workload.family.as_str()) {
        return Err(ManifestError::InvalidIdentity {
            manifest: "scheduler",
            path: path.to_path_buf(),
            field: "family",
            reason: format!("unsupported family `{}`", manifest.workload.family),
        });
    }
    if !SUPPORTED_WEIGHT_QUANTS.contains(&manifest.workload.format.as_str()) {
        return Err(ManifestError::InvalidIdentity {
            manifest: "scheduler",
            path: path.to_path_buf(),
            field: "format",
            reason: format!("unsupported weight format `{}`", manifest.workload.format),
        });
    }
    Ok(())
}

fn validate_ownership(manifest: &OwnershipManifest, path: &Path) -> Result<(), ManifestError> {
    // Reject overlapping concrete writable arrays: two fault sites must never
    // claim the same buffer region.
    let mut arrays = manifest.concrete_writable_arrays.clone();
    arrays.sort_by_key(|a| (a.start_offset, a.end_offset));
    for window in arrays.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        if a.end_offset >= b.start_offset {
            return Err(ManifestError::OverlappingWritableArrays {
                path: path.to_path_buf(),
                first: format!("{}[{}..{}]", a.id, a.start_offset, a.end_offset),
                second: format!("{}[{}..{}]", b.id, b.start_offset, b.end_offset),
            });
        }
    }
    for site in &manifest.fault_sites {
        if site.id.is_empty() {
            return Err(ManifestError::InvalidIdentity {
                manifest: "ownership",
                path: path.to_path_buf(),
                field: "id",
                reason: "fault site id is empty".to_string(),
            });
        }
        if site.asan_test_name.is_empty() {
            return Err(ManifestError::InvalidIdentity {
                manifest: "ownership",
                path: path.to_path_buf(),
                field: "asan_test_name",
                reason: format!("fault site `{}` has no asan test name", site.id),
            });
        }
    }
    Ok(())
}

fn validate_fixture_registry(
    manifest: &FixtureRegistryManifest,
    path: &Path,
) -> Result<(), ManifestError> {
    let group_ids: BTreeMap<String, ()> =
        manifest.groups.iter().map(|g| (g.id.clone(), ())).collect();
    let mut seen_ids = BTreeMap::new();
    for entry in &manifest.entries {
        if entry.id.is_empty() {
            return Err(ManifestError::InvalidIdentity {
                manifest: "fixture_registry",
                path: path.to_path_buf(),
                field: "id",
                reason: "fixture entry id is empty".to_string(),
            });
        }
        if seen_ids.contains_key(&entry.id) {
            return Err(ManifestError::InvalidIdentity {
                manifest: "fixture_registry",
                path: path.to_path_buf(),
                field: "id",
                reason: format!("duplicate fixture id `{}`", entry.id),
            });
        }
        seen_ids.insert(entry.id.clone(), ());
        if !group_ids.contains_key(&entry.group) {
            return Err(ManifestError::InvalidIdentity {
                manifest: "fixture_registry",
                path: path.to_path_buf(),
                field: "group",
                reason: format!(
                    "fixture `{}` references unknown group `{}`",
                    entry.id, entry.group
                ),
            });
        }
    }
    Ok(())
}

fn validate_structural_band(
    manifest: &StructuralBandManifest,
    path: &Path,
) -> Result<(), ManifestError> {
    for rule in &manifest.rules {
        if !SUPPORTED_FAMILIES.contains(&rule.family.as_str()) {
            return Err(ManifestError::InvalidIdentity {
                manifest: "structural_band",
                path: path.to_path_buf(),
                field: "family",
                reason: format!("unsupported family `{}`", rule.family),
            });
        }
        if !SUPPORTED_WEIGHT_QUANTS.contains(&rule.weight_quant.as_str()) {
            return Err(ManifestError::InvalidIdentity {
                manifest: "structural_band",
                path: path.to_path_buf(),
                field: "weight_quant",
                reason: format!("unsupported weight quant `{}`", rule.weight_quant),
            });
        }
        if rule.weight_quant == "q8_0" && rule.max_top2_swaps != 0 {
            return Err(ManifestError::InvalidIdentity {
                manifest: "structural_band",
                path: path.to_path_buf(),
                field: "max_top2_swaps",
                reason: "q8_0 requires zero forks (max_top2_swaps=0)".to_string(),
            });
        }
        if rule.weight_quant == "f16" && rule.max_top2_swaps > 2 {
            return Err(ManifestError::InvalidIdentity {
                manifest: "structural_band",
                path: path.to_path_buf(),
                field: "max_top2_swaps",
                reason: "f16 allows at most two top-2 swaps".to_string(),
            });
        }
    }
    Ok(())
}

fn validate_wire_bindings(
    manifest: &WireErrorBindingsManifest,
    path: &Path,
) -> Result<(), ManifestError> {
    // Reject unresolved symbolic placeholders. The spec forbids
    // `existing_deadline_error` and `existing_cancellation_error` in emitted
    // responses or passing evidence.
    for literal in [&manifest.deadline_error_id, &manifest.cancellation_error_id] {
        if literal.is_empty() {
            return Err(ManifestError::UnresolvedWireLiteral {
                path: path.to_path_buf(),
                literal: "<empty>".to_string(),
            });
        }
        if FORBIDDEN_WIRE_SYMBOLS.contains(&literal.as_str()) {
            return Err(ManifestError::UnresolvedWireLiteral {
                path: path.to_path_buf(),
                literal: literal.clone(),
            });
        }
    }
    if manifest.request_contract_revision.is_empty() {
        return Err(ManifestError::InvalidIdentity {
            manifest: "wire_bindings",
            path: path.to_path_buf(),
            field: "request_contract_revision",
            reason: "request contract revision is empty".to_string(),
        });
    }
    Ok(())
}

fn validate_grammar_cost(
    manifest: &GrammarCostCorpusManifest,
    path: &Path,
) -> Result<(), ManifestError> {
    if !SUPPORTED_FAMILIES.contains(&manifest.model_identity.family.as_str()) {
        return Err(ManifestError::InvalidIdentity {
            manifest: "grammar_cost",
            path: path.to_path_buf(),
            field: "family",
            reason: format!("unsupported family `{}`", manifest.model_identity.family),
        });
    }
    if !SUPPORTED_ACTIVATION_DTYPES.contains(&manifest.model_identity.activation_dtype.as_str()) {
        return Err(ManifestError::InvalidIdentity {
            manifest: "grammar_cost",
            path: path.to_path_buf(),
            field: "activation_dtype",
            reason: format!(
                "unsupported activation dtype `{}`",
                manifest.model_identity.activation_dtype
            ),
        });
    }
    if !SUPPORTED_WEIGHT_QUANTS.contains(&manifest.model_identity.weight_quant.as_str()) {
        return Err(ManifestError::InvalidIdentity {
            manifest: "grammar_cost",
            path: path.to_path_buf(),
            field: "weight_quant",
            reason: format!(
                "unsupported weight quant `{}`",
                manifest.model_identity.weight_quant
            ),
        });
    }
    Ok(())
}

fn validate_ci_lane(manifest: &CiLaneManifest, path: &Path) -> Result<(), ManifestError> {
    if !manifest
        .mandatory_targets
        .contains(&"macos-metal".to_string())
    {
        return Err(ManifestError::InvalidIdentity {
            manifest: "ci_lane",
            path: path.to_path_buf(),
            field: "mandatory_targets",
            reason: "macos-metal must be a mandatory target".to_string(),
        });
    }
    Ok(())
}

fn validate_slice_plan(manifest: &SlicePlanManifest, path: &Path) -> Result<(), ManifestError> {
    // The file fence must restrict production manifests to the module-owned
    // tree and must not allow the spike tree.
    let has_module_fence = manifest
        .file_fence
        .iter()
        .any(|f| f.starts_with(PRODUCTION_MANIFEST_ROOT));
    if !has_module_fence {
        return Err(ManifestError::InvalidIdentity {
            manifest: "slice_plan",
            path: path.to_path_buf(),
            field: "file_fence",
            reason: format!(
                "file_fence must include a path under `{}`",
                PRODUCTION_MANIFEST_ROOT
            ),
        });
    }
    for fence in &manifest.file_fence {
        if fence.starts_with(SPIKE_REFERENCE_ROOT) {
            return Err(ManifestError::SpikeReference {
                manifest: "slice_plan",
                path: path.to_path_buf(),
                field: "file_fence".to_string(),
                spike_root: SPIKE_REFERENCE_ROOT,
            });
        }
    }
    Ok(())
}

/// Scan a manifest's JSON value for any string field that references the spike
/// tree. Used by callers that want to reject production manifests carrying
/// `bench/spikes/` paths in any field.
pub fn find_spike_references(value: &Value) -> Vec<String> {
    let mut hits = Vec::new();
    walk_value(value, &mut hits);
    hits
}

fn walk_value(value: &Value, hits: &mut Vec<String>) {
    match value {
        Value::String(s) => {
            if s.contains(SPIKE_REFERENCE_ROOT) {
                hits.push(s.clone());
            }
        }
        Value::Array(items) => {
            for item in items {
                walk_value(item, hits);
            }
        }
        Value::Object(map) => {
            for (_, v) in map {
                walk_value(v, hits);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("owned-decode-manifests")
    }

    #[test]
    fn checked_in_manifests_load_and_validate() {
        // The checked-in artifacts must parse and pass every cross-manifest
        // invariant. This is the primary S1 gate: if any manifest is malformed
        // or carries an unresolved symbolic literal, this test fails.
        let dir = load_manifest_dir(&manifest_dir()).expect("checked-in manifests must validate");
        assert_eq!(
            dir.context_buckets.manifest_revision,
            "decode-context-buckets-v1"
        );
        assert_eq!(dir.scheduler.manifest_revision, "decode-sched-manifest-v1");
        assert_eq!(
            dir.ownership.manifest_revision,
            "decode-ownership-manifest-v1"
        );
        assert_eq!(
            dir.fixture_registry.manifest_revision,
            "decode-fixture-registry-v1"
        );
        assert_eq!(dir.structural_band.manifest_revision, "structural-band-v1");
        assert_eq!(
            dir.wire_bindings.manifest_revision,
            "owned-decode-wire-error-bindings-v1"
        );
        assert_eq!(dir.grammar_cost.manifest_revision, "grammar-cost-corpus-v1");
        assert_eq!(dir.ci_lane.manifest_revision, "ci-lane-manifest-v1");
        assert_eq!(dir.slice_plan.manifest_revision, "slice-plan-v1");
    }

    #[test]
    fn wire_bindings_carry_literal_ids_not_placeholders() {
        let dir = load_manifest_dir(&manifest_dir()).expect("checked-in manifests must validate");
        assert!(!FORBIDDEN_WIRE_SYMBOLS.contains(&dir.wire_bindings.deadline_error_id.as_str()));
        assert!(!FORBIDDEN_WIRE_SYMBOLS.contains(&dir.wire_bindings.cancellation_error_id.as_str()));
        assert!(!dir.wire_bindings.deadline_error_id.is_empty());
        assert!(!dir.wire_bindings.cancellation_error_id.is_empty());
    }

    #[test]
    fn context_buckets_cover_both_families() {
        let dir = load_manifest_dir(&manifest_dir()).expect("checked-in manifests must validate");
        let families: BTreeMap<String, ()> = dir
            .context_buckets
            .families
            .iter()
            .map(|f| (f.family.clone(), ()))
            .collect();
        for required in SUPPORTED_FAMILIES {
            assert!(
                families.contains_key(*required),
                "missing family {}",
                required
            );
        }
    }

    #[test]
    fn scheduler_runtime_n_is_candidate() {
        let dir = load_manifest_dir(&manifest_dir()).expect("checked-in manifests must validate");
        assert!(CANDIDATE_PRODUCTION_N.contains(&dir.scheduler.runtime.production_n));
    }

    #[test]
    fn scheduler_evidence_record_is_complete_and_consistent() {
        // The OQ-DEC-SCHED-01 mixed-load measurement has executed on the
        // validation machine: the checked-in evidence record must carry the
        // complete factual record — a per-candidate table covering every
        // candidate exactly once, the committed SLO, loadavg records, machine
        // profile, measurement date, and protocol identity — and the
        // commitment must be consistent with the selection rule: the
        // committed N is the LARGEST candidate meeting the SLO, or null when
        // no candidate met it (facts recorded, commitment pending review).
        let dir = load_manifest_dir(&manifest_dir()).expect("checked-in manifests must validate");
        let scheduler = &dir.scheduler;
        let evidence = &scheduler.evidence;
        let workload = &scheduler.workload;

        // SLO recorded and mirrored between workload and evidence.
        assert!(
            workload.embed_query_p95_slo_ms > 0.0,
            "workload SLO must be positive"
        );
        assert_eq!(
            evidence.embed_query_p95_slo_ms,
            Some(workload.embed_query_p95_slo_ms),
            "evidence SLO must mirror the workload SLO"
        );
        let slo = workload.embed_query_p95_slo_ms;

        // The per-candidate table covers every candidate exactly once, with
        // finite measurements and honest loadavg records.
        let mut covered: Vec<u32> = evidence.candidates.iter().map(|c| c.n).collect();
        covered.sort_unstable();
        assert_eq!(
            covered,
            vec![8, 16, 32],
            "evidence candidates must cover {{8,16,32}} exactly once each"
        );
        for candidate in &evidence.candidates {
            assert!(
                candidate.embed_p50_ms.is_finite()
                    && candidate.embed_p95_ms.is_finite()
                    && candidate.embed_p99_ms.is_finite(),
                "candidate {} must record finite embed percentiles",
                candidate.n
            );
            assert!(
                candidate.window_ms >= workload.duration_ms,
                "candidate {} window {} ms must cover the workload duration {} ms",
                candidate.n,
                candidate.window_ms,
                workload.duration_ms
            );
            assert!(
                candidate.embed_queries > 0,
                "candidate {} must record completed embed queries",
                candidate.n
            );
            assert!(
                candidate.loadavg_before.iter().all(|v| v.is_finite())
                    && candidate.loadavg_after.iter().all(|v| v.is_finite()),
                "candidate {} must record loadavg before and after",
                candidate.n
            );
            // The meets_slo flag must agree with the recorded p95 and SLO.
            assert_eq!(
                candidate.meets_slo,
                candidate.embed_p95_ms <= slo,
                "candidate {} meets_slo flag must match its p95 vs the SLO",
                candidate.n
            );
        }

        // Selection rule: committed_n is the largest candidate meeting the
        // SLO, or null when none met it. A committed N additionally equals
        // the runtime production_n and the observed N.
        let meeting: Vec<u32> = evidence
            .candidates
            .iter()
            .filter(|c| c.meets_slo)
            .map(|c| c.n)
            .collect();
        let expected_commit = meeting.iter().copied().max();
        assert_eq!(
            evidence.committed_n, expected_commit,
            "committed_n must be the largest SLO-meeting candidate, or null when none met it"
        );
        if let Some(committed_n) = evidence.committed_n {
            assert!(CANDIDATE_PRODUCTION_N.contains(&committed_n));
            assert_eq!(
                committed_n, scheduler.runtime.production_n,
                "committed_n must equal the runtime production_n"
            );
            assert_eq!(
                evidence.observed_n,
                Some(committed_n),
                "observed_n must match committed_n"
            );
            assert!(evidence.continuation_count.is_some());
        }

        // Baselines and machine context recorded.
        assert!(
            evidence.baseline_embed_only_p95_ms.is_some(),
            "same-session embed-only baseline p95 must be recorded"
        );
        assert!(
            evidence.decode_only_tokens_per_sec.is_some(),
            "uninterrupted decode throughput baseline must be recorded"
        );
        assert!(
            evidence
                .machine_profile_note
                .as_deref()
                .is_some_and(|note| !note.is_empty()),
            "machine profile note must be recorded"
        );
        assert!(
            evidence
                .measured_at_utc
                .as_deref()
                .is_some_and(|stamp| !stamp.is_empty()),
            "measurement date must be recorded"
        );
        assert!(
            evidence
                .protocol_id
                .as_deref()
                .is_some_and(|id| !id.is_empty()),
            "measurement protocol id must be recorded"
        );

        // Evidence history preserves superseded records instead of
        // overwriting them: every history entry carries a protocol id
        // distinct from the current record's, and the checked-in v1 honest
        // negative (no candidate met the SLO, committed_n null) must still
        // be present behind the current record.
        let current_protocol = evidence.protocol_id.clone().unwrap_or_default();
        for entry in &evidence.history {
            let entry_protocol = entry.protocol_id.clone().unwrap_or_default();
            assert!(
                !entry_protocol.is_empty() && entry_protocol != current_protocol,
                "history entries must carry their own protocol id, got {entry_protocol:?}"
            );
            assert!(
                !entry.candidates.is_empty(),
                "history entries must keep their per-candidate table"
            );
        }
        if let Some(committed) = evidence.committed_n {
            assert!(
                evidence
                    .history
                    .iter()
                    .any(|entry| entry.committed_n.is_none() && !entry.candidates.is_empty()),
                "the superseded honest negative (committed_n null) must remain in the evidence history behind committed N={committed}"
            );
        }

        // Executed-evidence fields: maximum uninterruptible GPU time,
        // sequence traces, permit events, queue depth, per-operation
        // waiting, cancellation observations, and the boundary-crossing
        // bit-exactness spot check must all be present.
        assert!(evidence.max_uninterruptible_gpu_time_ms.is_some());
        assert!(!evidence.sequence_traces.is_empty());
        assert!(!evidence.permit_events.is_empty());
        assert!(!evidence.queue_depth.is_empty());
        assert!(!evidence.per_operation_waiting_ms.is_empty());
        assert!(!evidence.cancellation_latency_ms.is_empty());
        assert!(!evidence.parity_spot_check.is_empty());
        assert!(!workload.cancellation_observations.is_empty());

        // Recorded cancellation observations (quantum-deferral samples)
        // stay inside the workload bound.
        let bound = workload.cancellation_latency_bound_ms as f64;
        for observation in workload
            .cancellation_observations
            .iter()
            .chain(evidence.cancellation_latency_ms.iter())
        {
            assert!(
                *observation <= bound,
                "cancellation observation {observation} ms exceeds the {bound} ms bound"
            );
        }
    }

    #[test]
    fn ownership_arrays_do_not_overlap() {
        let dir = load_manifest_dir(&manifest_dir()).expect("checked-in manifests must validate");
        let mut arrays = dir.ownership.concrete_writable_arrays.clone();
        arrays.sort_by_key(|a| (a.start_offset, a.end_offset));
        for window in arrays.windows(2) {
            assert!(
                window[0].end_offset < window[1].start_offset,
                "overlapping writable arrays: {} and {}",
                window[0].id,
                window[1].id
            );
        }
    }

    #[test]
    fn ci_lane_marks_macos_metal_mandatory() {
        let dir = load_manifest_dir(&manifest_dir()).expect("checked-in manifests must validate");
        assert!(dir
            .ci_lane
            .mandatory_targets
            .contains(&"macos-metal".to_string()));
    }

    #[test]
    fn slice_plan_fence_stays_in_module_tree() {
        let dir = load_manifest_dir(&manifest_dir()).expect("checked-in manifests must validate");
        for fence in &dir.slice_plan.file_fence {
            assert!(
                !fence.starts_with(SPIKE_REFERENCE_ROOT),
                "file_fence references spike tree: {}",
                fence
            );
        }
    }

    #[test]
    fn checked_in_manifests_have_no_spike_references() {
        // No production manifest may reference a path under bench/spikes/.
        let dir = manifest_dir();
        for entry in std::fs::read_dir(&dir).expect("manifest dir exists") {
            let entry = entry.expect("read dir entry");
            let bytes = std::fs::read(entry.path()).expect("read manifest");
            let value: Value = serde_json::from_slice(&bytes).expect("manifest is JSON");
            let hits = find_spike_references(&value);
            assert!(
                hits.is_empty(),
                "{} references spike tree: {:?}",
                entry.path().display(),
                hits
            );
        }
    }

    #[test]
    fn reject_manifest_dir_outside_production_root() {
        // A spike-only manifest directory must be rejected even if its files
        // are well-formed. This path lacks the production-root segments
        // entirely.
        let spike_dir = PathBuf::from("bench/spikes/unified-rt/owned-decode-manifests");
        let result = load_manifest_dir(&spike_dir);
        assert!(matches!(
            result,
            Err(ManifestError::OutsideProductionRoot { .. })
        ));
    }

    #[test]
    fn reject_path_with_production_root_as_subsequence_not_prefix() {
        // `bench/spikes/crates/synapse-module/evil.jsonc` contains the
        // production-root segments in order but not as a leading prefix. The
        // old subsequence check accepted this; the prefix check must reject it.
        let evil = PathBuf::from("bench/spikes/crates/synapse-module/evil.jsonc");
        assert!(matches!(
            load_manifest_dir(&evil),
            Err(ManifestError::OutsideProductionRoot { .. })
        ));
    }

    #[test]
    fn reject_path_with_production_root_after_unrelated_component() {
        // `x/crates/synapse-module/y` has the root segments contiguous but
        // not leading; the root must be a prefix, not a substring.
        let path = PathBuf::from("x/crates/synapse-module/y");
        assert!(matches!(
            load_manifest_dir(&path),
            Err(ManifestError::OutsideProductionRoot { .. })
        ));
    }

    #[test]
    fn reject_absolute_path_outside_repo() {
        // An absolute path that does not pass through the production root is
        // rejected. (This path does not exist, so canonicalization would also
        // fail, but the raw check fires first with the typed error.)
        let path = PathBuf::from("/tmp/owned-decode-manifests");
        assert!(matches!(
            load_manifest_dir(&path),
            Err(ManifestError::OutsideProductionRoot { .. })
        ));
    }

    #[test]
    fn reject_parent_traversal_that_textually_contains_prefix() {
        // `../crates/synapse-module/y` textually contains the prefix, but the
        // `..` component precedes it so it is not a leading prefix of normal
        // components.
        let path = PathBuf::from("../crates/synapse-module/y");
        assert!(matches!(
            load_manifest_dir(&path),
            Err(ManifestError::OutsideProductionRoot { .. })
        ));
    }

    #[test]
    fn raw_path_check_accepts_leading_production_root() {
        // A relative path whose normal components begin with the production
        // root is accepted by the raw check (it may still fail later on I/O
        // or canonicalization, but it passes the prefix guard).
        let ok = PathBuf::from("crates/synapse-module/owned-decode-manifests");
        assert!(ensure_raw_path_inside_production_root(&ok).is_ok());
    }

    #[test]
    fn load_path_rejects_manifest_with_spike_reference() {
        // The spike-reference invariant is enforced at load time, not only in
        // a test. A manifest whose raw JSON value references `bench/spikes/`
        // in any string field fails with the existing SpikeReference error.
        let dir = std::env::temp_dir().join("synapse-module-spike-scan-test");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        // Write a slice-plan manifest whose `artifacts` array references the
        // spike tree. The typed struct would parse fine, but the raw-value
        // scan must reject it first.
        let spike_json = serde_json::json!({
            "manifest_revision": "slice-plan-v1",
            "schema_revision": CONTRACT_SCHEMA_REVISION,
            "slice_id": "s1",
            "work_item": "w",
            "file_fence": ["crates/synapse-module/"],
            "artifacts": ["bench/spikes/unified-rt/evil.jsonc"],
        });
        let path = dir.join("slice-plan-v1.json");
        std::fs::write(&path, spike_json.to_string()).expect("write manifest");
        let result =
            load_one_with_spike_scan::<SlicePlanManifest>(&dir, "slice-plan-v1.json", "slice_plan");
        assert!(
            matches!(result, Err(ManifestError::SpikeReference { .. })),
            "spike reference must be rejected at load: {result:?}"
        );
        // A clean manifest with no spike reference parses successfully.
        let clean_json = serde_json::json!({
            "manifest_revision": "slice-plan-v1",
            "schema_revision": CONTRACT_SCHEMA_REVISION,
            "slice_id": "s1",
            "work_item": "w",
            "file_fence": ["crates/synapse-module/"],
            "artifacts": ["crates/synapse-module/owned-decode-manifests/slice-plan-v1.json"],
        });
        std::fs::write(&path, clean_json.to_string()).expect("write clean manifest");
        load_one_with_spike_scan::<SlicePlanManifest>(&dir, "slice-plan-v1.json", "slice_plan")
            .expect("clean manifest parses");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn reject_unresolved_wire_literal() {
        let mut bindings = WireErrorBindingsManifest {
            manifest_revision: "owned-decode-wire-error-bindings-v1".to_string(),
            schema_revision: CONTRACT_SCHEMA_REVISION.to_string(),
            request_contract_revision: "wire-contract-v1".to_string(),
            deadline_error_id: "existing_deadline_error".to_string(),
            cancellation_error_id: "deadline_exceeded".to_string(),
            wire_changelog: Vec::new(),
        };
        let path = PathBuf::from("owned-decode-wire-error-bindings-v1.json");
        assert!(matches!(
            validate_wire_bindings(&bindings, &path),
            Err(ManifestError::UnresolvedWireLiteral { .. })
        ));
        bindings.deadline_error_id = "deadline_exceeded".to_string();
        bindings.cancellation_error_id = "existing_cancellation_error".to_string();
        assert!(matches!(
            validate_wire_bindings(&bindings, &path),
            Err(ManifestError::UnresolvedWireLiteral { .. })
        ));
        bindings.cancellation_error_id = "cancelled".to_string();
        validate_wire_bindings(&bindings, &path).expect("literal IDs are accepted");
    }

    #[test]
    fn reject_overlapping_writable_arrays() {
        let manifest = OwnershipManifest {
            manifest_revision: "decode-ownership-manifest-v1".to_string(),
            schema_revision: CONTRACT_SCHEMA_REVISION.to_string(),
            fault_sites: Vec::new(),
            concrete_writable_arrays: vec![
                ConcreteWritableArray {
                    id: "arr-a".to_string(),
                    owner: "rust".to_string(),
                    start_offset: 0,
                    end_offset: 128,
                    element_type: "f32".to_string(),
                    length: 32,
                },
                ConcreteWritableArray {
                    id: "arr-b".to_string(),
                    owner: "objc".to_string(),
                    start_offset: 64,
                    end_offset: 192,
                    element_type: "f32".to_string(),
                    length: 32,
                },
            ],
        };
        let path = PathBuf::from("decode-ownership-manifest-v1.json");
        assert!(matches!(
            validate_ownership(&manifest, &path),
            Err(ManifestError::OverlappingWritableArrays { .. })
        ));
    }

    #[test]
    fn reject_invalid_identity_fields() {
        // Two families (correct count) but one carries an unsupported family
        // name, so validation hits the InvalidIdentity branch rather than the
        // family-count check.
        let bad_family = ContextBucketsManifest {
            manifest_revision: "decode-context-buckets-v1".to_string(),
            schema_revision: CONTRACT_SCHEMA_REVISION.to_string(),
            families: vec![
                ContextBucketFamily {
                    family: "qwen3-0.6b".to_string(),
                    verified_buckets: vec![512],
                },
                ContextBucketFamily {
                    family: "qwen3-7b".to_string(),
                    verified_buckets: vec![512],
                },
            ],
            removed_buckets: Vec::new(),
        };
        let path = PathBuf::from("decode-context-buckets-v1.json");
        assert!(matches!(
            validate_context_buckets(&bad_family, &path),
            Err(ManifestError::InvalidIdentity { .. })
        ));
    }
}
