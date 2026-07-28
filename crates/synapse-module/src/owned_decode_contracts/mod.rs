//! Module-owned schemas and checked-in records for the production owned-decode
//! lane (`engine=owned-metal-decode`).
//!
//! This module is the single source of truth for the manifest shapes that gate
//! the D-009 cutover. It defines the serde schemas for every checked-in
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
        field: &'static str,
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
    // Reject paths outside the production root before canonicalizing, so a
    // non-existent spike-only directory is rejected with OutsideProductionRoot
    // rather than an I/O error.
    ensure_raw_path_inside_production_root(manifest_dir)?;
    let root = manifest_dir
        .canonicalize()
        .map_err(|source| ManifestError::Io {
            path: manifest_dir.to_path_buf(),
            source,
        })?;
    ensure_inside_production_root(&root)?;

    let context_buckets = load_one::<ContextBucketsManifest>(
        &root,
        "decode-context-buckets-v1.json",
        "context_buckets",
    )?;
    let scheduler =
        load_one::<SchedulerManifest>(&root, "decode-sched-manifest-v1.json", "scheduler")?;
    let ownership =
        load_one::<OwnershipManifest>(&root, "decode-ownership-manifest-v1.json", "ownership")?;
    let fixture_registry = load_one::<FixtureRegistryManifest>(
        &root,
        "decode-fixture-registry-v1.json",
        "fixture_registry",
    )?;
    let structural_band =
        load_one::<StructuralBandManifest>(&root, "structural-band-v1.json", "structural_band")?;
    let wire_bindings = load_one::<WireErrorBindingsManifest>(
        &root,
        "owned-decode-wire-error-bindings-v1.json",
        "wire_bindings",
    )?;
    let grammar_cost = load_one::<GrammarCostCorpusManifest>(
        &root,
        "grammar-cost-corpus-v1.json",
        "grammar_cost",
    )?;
    let ci_lane = load_one::<CiLaneManifest>(&root, "ci-lane-manifest-v1.json", "ci_lane")?;
    let slice_plan = load_one::<SlicePlanManifest>(&root, "slice-plan-v1.json", "slice_plan")?;

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

fn load_one<T: for<'de> Deserialize<'de>>(
    root: &Path,
    file: &str,
    manifest: &'static str,
) -> Result<T, ManifestError> {
    let path = root.join(file);
    let bytes = std::fs::read(&path).map_err(|source| ManifestError::Io {
        path: path.clone(),
        source,
    })?;
    serde_json::from_slice::<T>(&bytes).map_err(|source| ManifestError::Parse {
        manifest,
        path: path.clone(),
        source,
    })
}

/// Reject a raw (pre-canonicalization) manifest directory path that does not
/// reference the production root. This catches a spike-only path before the
/// canonicalization I/O error and keeps the rejection typed as
/// `OutsideProductionRoot`.
fn ensure_raw_path_inside_production_root(raw: &Path) -> Result<(), ManifestError> {
    let normal_components: Vec<String> = raw
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    let expected_segments: Vec<&str> = PRODUCTION_MANIFEST_ROOT.split('/').collect();
    let mut idx = 0usize;
    for expected in &expected_segments {
        let mut found = false;
        while idx < normal_components.len() {
            if normal_components[idx].as_str() == *expected {
                idx += 1;
                found = true;
                break;
            }
            idx += 1;
        }
        if !found {
            return Err(ManifestError::OutsideProductionRoot {
                manifest: "manifest_dir",
                path: raw.to_path_buf(),
                expected_root: PRODUCTION_MANIFEST_ROOT,
            });
        }
    }
    Ok(())
}

/// Reject a manifest directory that resolves outside the production root. This
/// prevents a spike-only artifact under `bench/spikes/` from posing as a
/// deployed record.
fn ensure_inside_production_root(resolved: &Path) -> Result<(), ManifestError> {
    // The canonicalized path is absolute (e.g. /Users/.../worktree/crates/
    // synapse-module/owned-decode-manifests). Walk its normal components and
    // require the production root segments to appear in order, so an absolute
    // path under a worktree still matches while a path under bench/spikes/
    // does not.
    let normal_components: Vec<String> = resolved
        .components()
        .filter_map(|c| match c {
            Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    let expected_segments: Vec<&str> = PRODUCTION_MANIFEST_ROOT.split('/').collect();
    let mut idx = 0usize;
    for expected in &expected_segments {
        let mut found = false;
        while idx < normal_components.len() {
            if normal_components[idx].as_str() == *expected {
                idx += 1;
                found = true;
                break;
            }
            idx += 1;
        }
        if !found {
            return Err(ManifestError::OutsideProductionRoot {
                manifest: "manifest_dir",
                path: resolved.to_path_buf(),
                expected_root: PRODUCTION_MANIFEST_ROOT,
            });
        }
    }
    Ok(())
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
                field: "file_fence",
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
        // are well-formed.
        let spike_dir = PathBuf::from("bench/spikes/unified-rt/owned-decode-manifests");
        let result = load_manifest_dir(&spike_dir);
        assert!(matches!(
            result,
            Err(ManifestError::OutsideProductionRoot { .. })
        ));
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
