//! Release verification of catalog backends against checked-in evidence.
//!
//! Every backend the compiled catalog declares must have a passing evidence
//! record from a pre-release run on reference hardware, checked in at
//! `docs/evidence/catalog-backends/<catalog_id>__<backend>.json`. A record
//! only counts when it was produced for exactly what ships: the same
//! manifest digest, self-check fixture revision and lane fingerprint as the
//! catalog, the same engine source trees as the release tag, and the
//! backend's named evidence corpus. The release workflow's
//! `verify-catalog-evidence` job runs [`verify_catalog_evidence`] before any
//! build lane, so a missing, stale or failing record stops the release.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{load_release_valid, CatalogBackend, CatalogEntry, CatalogError};

/// Where evidence records live, relative to the repository root.
pub(crate) const EVIDENCE_DIR: &str = "docs/evidence/catalog-backends";

/// The engine source directories whose git tree hashes, newline-joined in
/// this order, form a record's `engine_tree`. Tree hashes of these
/// directories do not change when a record is committed elsewhere, so
/// checking a record in does not invalidate it.
pub(crate) const ENGINE_TREE_DIRS: [&str; 3] = [
    "crates/synapse-engine-owned",
    "crates/synapse-worker-ane",
    "crates/synapse-worker-ane-direct",
];

/// The minimum cosine against the fp32 reference an embed backend must
/// reach over its whole evidence corpus.
pub(crate) const EMBED_MIN_COSINE: f64 = 0.999;

/// The evidence corpus each catalog entry's backends are measured on. The
/// corpus files live in `crates/synapse-module/src/fixtures/` under the same
/// name with a `.json` suffix.
const NAMED_CORPORA: [(&str, &str); 4] = [
    (
        "gte-modernbert-base",
        "probe_corpus_gte_modernbert_ort_fp32",
    ),
    ("qwen3-embedding-0.6b", "probe_corpus_qwen3_embedding_fp32"),
    (
        "gte-reranker-modernbert-base",
        "catalog_rerank_gte_modernbert_fp32",
    ),
    (
        "qwen3-reranker-0.6b",
        "qwen3-reranker-0.6b.ref-v1.transformers-5.16.1.seed-0",
    ),
];

/// The evidence corpus a catalog entry's backends must be measured on.
pub(crate) fn named_corpus(catalog_id: &str) -> Option<&'static str> {
    NAMED_CORPORA
        .iter()
        .find(|(id, _)| *id == catalog_id)
        .map(|(_, corpus)| *corpus)
}

/// The file name of the evidence record for one declared backend.
pub(crate) fn record_file_name(catalog_id: &str, backend: &str) -> String {
    format!("{catalog_id}__{backend}.json")
}

/// One pre-release evidence run for a `(catalog_id, backend)` pair.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvidenceRecord {
    pub catalog_id: String,
    pub backend: String,
    pub manifest_digest: String,
    pub fixture_revision: i64,
    pub fingerprint: String,
    pub engine_tree: String,
    /// Identity of the reference machine (chip, OS build, and so on).
    pub machine: BTreeMap<String, Value>,
    pub engine_build: String,
    pub dtype: String,
    pub corpus_id: String,
    pub metrics: EvidenceMetrics,
    pub thresholds: EvidenceThresholds,
    pub passed: bool,
}

/// Embed records carry `min_cosine`; rerank records carry
/// `max_abs_sigmoid_deviation` and `order_violations` (candidate pairs whose
/// reference scores differ by at least the tolerance but come out inverted).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvidenceMetrics {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_cosine: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_abs_sigmoid_deviation: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub order_violations: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvidenceThresholds {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_cosine: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank_abs_tolerance: Option<f64>,
}

/// Why the release cannot ship. Every failure names the record file or the
/// catalog violation behind it.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub(crate) enum EvidenceFailure {
    #[error("compiled catalog fails release validation: {0}")]
    Catalog(CatalogError),
    #[error("{catalog_id} backend {backend} has no evidence record {file}")]
    MissingRecord {
        catalog_id: String,
        backend: String,
        file: String,
    },
    #[error("evidence record {file} is malformed: {reason}")]
    MalformedRecord { file: String, reason: String },
    #[error("evidence record {file} has {field} {actual}, the release expects {expected}")]
    KeyMismatch {
        file: String,
        field: &'static str,
        expected: String,
        actual: String,
    },
    #[error("{catalog_id} backend {backend} has no named evidence corpus")]
    NoNamedCorpus { catalog_id: String, backend: String },
    #[error("evidence record {file} was measured on corpus {actual}, the backend's corpus is {expected}")]
    WrongCorpus {
        file: String,
        expected: String,
        actual: String,
    },
    #[error("evidence record {file} has {metric} {value}, which fails the threshold {threshold}")]
    FailingMetric {
        file: String,
        metric: &'static str,
        value: String,
        threshold: String,
    },
    #[error("evidence record {file} reports passed: false")]
    NotPassed { file: String },
}

/// Verifies that every backend the catalog declares has a matching passing
/// evidence record.
///
/// `catalog_json` is the catalog at the release tag, which must pass release
/// validation; `records` maps record file names to their contents (see
/// [`read_evidence_dir`]); `engine_tree` is [`engine_tree_at`] for the tag.
/// Returns the verified `(catalog_id, backend)` pairs, or every failure.
pub(crate) fn verify_catalog_evidence(
    catalog_json: &str,
    records: &BTreeMap<String, String>,
    engine_tree: &str,
) -> Result<Vec<(String, String)>, Vec<EvidenceFailure>> {
    let catalog =
        load_release_valid(catalog_json).map_err(|error| vec![EvidenceFailure::Catalog(error)])?;
    let mut failures = Vec::new();
    let mut verified = Vec::new();
    for entry in &catalog.models {
        for backend in &entry.backends {
            let before = failures.len();
            verify_backend(entry, backend, records, engine_tree, &mut failures);
            if failures.len() == before {
                verified.push((entry.id.clone(), backend.backend.clone()));
            }
        }
    }
    if failures.is_empty() {
        Ok(verified)
    } else {
        Err(failures)
    }
}

fn verify_backend(
    entry: &CatalogEntry,
    backend: &CatalogBackend,
    records: &BTreeMap<String, String>,
    engine_tree: &str,
    failures: &mut Vec<EvidenceFailure>,
) {
    let file = record_file_name(&entry.id, &backend.backend);
    let Some(text) = records.get(&file) else {
        failures.push(EvidenceFailure::MissingRecord {
            catalog_id: entry.id.clone(),
            backend: backend.backend.clone(),
            file,
        });
        return;
    };
    let record: EvidenceRecord = match serde_json::from_str(text) {
        Ok(record) => record,
        Err(error) => {
            failures.push(EvidenceFailure::MalformedRecord {
                file,
                reason: error.to_string(),
            });
            return;
        }
    };
    if record.machine.is_empty() || record.engine_build.trim().is_empty() {
        failures.push(EvidenceFailure::MalformedRecord {
            file: file.clone(),
            reason: "machine identity and engine_build must be non-empty".to_string(),
        });
    }

    let fixture_revision = entry
        .self_check
        .as_ref()
        .map(|check| check.fixture_revision.to_string())
        // Profile-only entries started with the first sealed parity subset.
        // Missing hardware records still fail closed before this comparison.
        .unwrap_or_else(|| "1".into());
    let keys: [(&'static str, String, String); 7] = [
        ("catalog_id", entry.id.clone(), record.catalog_id.clone()),
        ("backend", backend.backend.clone(), record.backend.clone()),
        (
            "manifest_digest",
            entry.manifest_digest(),
            record.manifest_digest.clone(),
        ),
        (
            "fixture_revision",
            fixture_revision,
            record.fixture_revision.to_string(),
        ),
        (
            "fingerprint",
            backend.fingerprint.clone(),
            record.fingerprint.clone(),
        ),
        (
            "engine_tree",
            engine_tree.to_string(),
            record.engine_tree.clone(),
        ),
        (
            "dtype",
            backend.dtype.clone().unwrap_or_default(),
            record.dtype.clone(),
        ),
    ];
    for (field, expected, actual) in keys {
        if expected != actual {
            failures.push(EvidenceFailure::KeyMismatch {
                file: file.clone(),
                field,
                expected,
                actual,
            });
        }
    }

    match named_corpus(&entry.id) {
        None => failures.push(EvidenceFailure::NoNamedCorpus {
            catalog_id: entry.id.clone(),
            backend: backend.backend.clone(),
        }),
        Some(corpus) if corpus != record.corpus_id => failures.push(EvidenceFailure::WrongCorpus {
            file: file.clone(),
            expected: corpus.to_string(),
            actual: record.corpus_id.clone(),
        }),
        Some(_) => {}
    }

    verify_metrics(entry, backend, &record, &file, failures);
    if !record.passed {
        failures.push(EvidenceFailure::NotPassed { file });
    }
}

/// Checks the record's metrics against the thresholds the catalog implies,
/// and that the record states those same thresholds. A record's own
/// `passed` flag is never trusted on its own.
fn verify_metrics(
    entry: &CatalogEntry,
    backend: &CatalogBackend,
    record: &EvidenceRecord,
    file: &str,
    failures: &mut Vec<EvidenceFailure>,
) {
    let failing = |metric: &'static str, value: Option<String>, threshold: String| {
        EvidenceFailure::FailingMetric {
            file: file.to_string(),
            metric,
            value: value.unwrap_or_else(|| "absent".to_string()),
            threshold,
        }
    };
    if entry.task == "embed" {
        if record.thresholds.min_cosine != Some(EMBED_MIN_COSINE) {
            failures.push(EvidenceFailure::KeyMismatch {
                file: file.to_string(),
                field: "thresholds.min_cosine",
                expected: EMBED_MIN_COSINE.to_string(),
                actual: format!("{:?}", record.thresholds.min_cosine),
            });
        }
        let min_cosine = record.metrics.min_cosine;
        if !min_cosine.is_some_and(|value| value.is_finite() && value >= EMBED_MIN_COSINE) {
            failures.push(failing(
                "min_cosine",
                min_cosine.map(|value| value.to_string()),
                format!(">= {EMBED_MIN_COSINE}"),
            ));
        }
    } else {
        let tolerance = backend.rerank_abs_tolerance.unwrap_or_default();
        if record.thresholds.rerank_abs_tolerance != Some(tolerance) {
            failures.push(EvidenceFailure::KeyMismatch {
                file: file.to_string(),
                field: "thresholds.rerank_abs_tolerance",
                expected: tolerance.to_string(),
                actual: format!("{:?}", record.thresholds.rerank_abs_tolerance),
            });
        }
        let deviation = record.metrics.max_abs_sigmoid_deviation;
        if !deviation.is_some_and(|value| value.is_finite() && value <= tolerance) {
            failures.push(failing(
                "max_abs_sigmoid_deviation",
                deviation.map(|value| value.to_string()),
                format!("<= {tolerance}"),
            ));
        }
        let violations = record.metrics.order_violations;
        if violations != Some(0) {
            failures.push(failing(
                "order_violations",
                violations.map(|value| value.to_string()),
                "0".to_string(),
            ));
        }
    }
}

/// Reads every `*.json` file of an evidence directory, keyed by file name.
/// A missing directory reads as no records, so every declared backend then
/// fails as missing rather than the check erroring out.
pub(crate) fn read_evidence_dir(dir: &Path) -> std::io::Result<BTreeMap<String, String>> {
    let mut records = BTreeMap::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(records),
        Err(error) => return Err(error),
    };
    for entry in entries {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if path.is_file() && name.ends_with(".json") {
            records.insert(name.to_string(), std::fs::read_to_string(&path)?);
        }
    }
    Ok(records)
}

/// The `engine_tree` key at `revision`: the newline-joined
/// `git rev-parse <revision>:<dir>` tree hashes of [`ENGINE_TREE_DIRS`].
pub(crate) fn engine_tree_at(repo_root: &Path, revision: &str) -> Result<String, String> {
    let mut trees = Vec::with_capacity(ENGINE_TREE_DIRS.len());
    for dir in ENGINE_TREE_DIRS {
        let output = synapse_core::without_launch_nonce(Command::new("git"))
            .arg("-C")
            .arg(repo_root)
            .args(["rev-parse", "--verify", &format!("{revision}:{dir}")])
            .output()
            .map_err(|error| format!("run git rev-parse for {dir}: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "git rev-parse {revision}:{dir} failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        trees.push(String::from_utf8_lossy(&output.stdout).trim().to_string());
    }
    Ok(trees.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::super::{compiled_catalog, COMPILED_CATALOG_JSON};
    use super::*;
    use serde_json::json;

    const ENGINE_TREE: &str =
        "1111111111111111111111111111111111111111\n2222222222222222222222222222222222222222";

    /// Synthetic passing records for verifier tests, not hardware evidence.
    fn passing_records() -> BTreeMap<String, Value> {
        let catalog = compiled_catalog().unwrap();
        let mut records = BTreeMap::new();
        for entry in &catalog.models {
            for backend in &entry.backends {
                let (metrics, thresholds) = if entry.task == "embed" {
                    (json!({"min_cosine": 0.9996}), json!({"min_cosine": 0.999}))
                } else {
                    (
                        json!({"max_abs_sigmoid_deviation": 0.0004, "order_violations": 0}),
                        json!({"rerank_abs_tolerance": backend.rerank_abs_tolerance}),
                    )
                };
                records.insert(
                    record_file_name(&entry.id, &backend.backend),
                    json!({
                        "catalog_id": entry.id,
                        "backend": backend.backend,
                        "manifest_digest": entry.manifest_digest(),
                        "fixture_revision": entry.self_check.as_ref().map_or(1, |check| check.fixture_revision),
                        "fingerprint": backend.fingerprint,
                        "engine_tree": ENGINE_TREE,
                        "machine": {"chip": "Apple M5", "os_build": "27A100"},
                        "engine_build": "owned-metal-v1 graph_revision=4",
                        "dtype": backend.dtype,
                        "corpus_id": named_corpus(&entry.id).unwrap(),
                        "metrics": metrics,
                        "thresholds": thresholds,
                        "passed": true
                    }),
                );
            }
        }
        records
    }

    fn texts(records: &BTreeMap<String, Value>) -> BTreeMap<String, String> {
        records
            .iter()
            .map(|(name, record)| (name.clone(), record.to_string()))
            .collect()
    }

    fn verify(
        catalog: &str,
        records: &BTreeMap<String, Value>,
    ) -> Result<Vec<(String, String)>, Vec<EvidenceFailure>> {
        verify_catalog_evidence(catalog, &texts(records), ENGINE_TREE)
    }

    /// Mutates the passing record of one backend and returns the failures.
    fn failures_after(file: &str, mutate: impl FnOnce(&mut Value)) -> Vec<EvidenceFailure> {
        let mut records = passing_records();
        mutate(records.get_mut(file).expect("record exists"));
        verify(COMPILED_CATALOG_JSON, &records).expect_err("mutated record must fail")
    }

    const GTE: &str = "gte-modernbert-base__metal.json";
    const RERANKER: &str = "gte-reranker-modernbert-base__metal.json";

    /// Release-only gate: ordinary workspace tests use synthetic fixtures.
    /// Missing hardware evidence must block a tag rather than skip this check.
    #[test]
    #[ignore = "requires checked-in full-corpus hardware evidence for the release tree"]
    fn release_catalog_evidence() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let engine_tree = engine_tree_at(&root, "HEAD").expect("release engine trees at HEAD");
        let records = read_evidence_dir(&root.join(EVIDENCE_DIR)).expect("read release evidence");
        match verify_catalog_evidence(COMPILED_CATALOG_JSON, &records, &engine_tree) {
            Ok(verified) => println!("verified {} catalog backends at HEAD", verified.len()),
            Err(failures) => {
                for failure in &failures {
                    eprintln!("{failure}");
                }
                panic!(
                    "release evidence failed: a fresh-run producer must write matching full-corpus hardware records to {EVIDENCE_DIR}"
                );
            }
        }
    }

    #[test]
    fn passing_records_verify_every_declared_backend() {
        let verified = verify(COMPILED_CATALOG_JSON, &passing_records()).expect("passing fixture");
        assert_eq!(
            verified,
            [
                ("gte-modernbert-base".to_string(), "metal".to_string()),
                ("gte-modernbert-base".to_string(), "ane".to_string()),
                (
                    "gte-reranker-modernbert-base".to_string(),
                    "metal".to_string()
                ),
                ("qwen3-embedding-0.6b".to_string(), "metal".to_string()),
                ("qwen3-embedding-0.6b".to_string(), "ane".to_string()),
                ("qwen3-reranker-0.6b".to_string(), "ane".to_string()),
            ]
        );
    }

    #[test]
    fn a_missing_record_fails() {
        let mut records = passing_records();
        records.remove(RERANKER);
        let failures = verify(COMPILED_CATALOG_JSON, &records).unwrap_err();
        assert_eq!(
            failures,
            [EvidenceFailure::MissingRecord {
                catalog_id: "gte-reranker-modernbert-base".into(),
                backend: "metal".into(),
                file: RERANKER.into(),
            }]
        );
    }

    #[test]
    fn a_failing_embed_metric_fails_even_when_the_record_claims_a_pass() {
        let failures = failures_after(GTE, |record| {
            record["metrics"]["min_cosine"] = json!(0.9985)
        });
        assert!(
            matches!(
                failures.as_slice(),
                [EvidenceFailure::FailingMetric {
                    metric: "min_cosine",
                    ..
                }]
            ),
            "{failures:?}"
        );
    }

    #[test]
    fn failing_rerank_metrics_fail() {
        let deviation = failures_after(RERANKER, |record| {
            record["metrics"]["max_abs_sigmoid_deviation"] = json!(0.0051)
        });
        assert!(
            matches!(
                deviation.as_slice(),
                [EvidenceFailure::FailingMetric {
                    metric: "max_abs_sigmoid_deviation",
                    ..
                }]
            ),
            "{deviation:?}"
        );
        let order = failures_after(RERANKER, |record| {
            record["metrics"]["order_violations"] = json!(1)
        });
        assert!(
            matches!(
                order.as_slice(),
                [EvidenceFailure::FailingMetric {
                    metric: "order_violations",
                    ..
                }]
            ),
            "{order:?}"
        );
    }

    #[test]
    fn a_record_reporting_failure_fails() {
        let failures = failures_after(GTE, |record| record["passed"] = json!(false));
        assert_eq!(failures, [EvidenceFailure::NotPassed { file: GTE.into() }]);
    }

    #[test]
    fn a_record_with_loosened_thresholds_fails() {
        let failures = failures_after(GTE, |record| {
            record["thresholds"]["min_cosine"] = json!(0.99);
            record["metrics"]["min_cosine"] = json!(0.995);
        });
        assert_eq!(failures.len(), 2, "{failures:?}");
        assert!(matches!(
            &failures[0],
            EvidenceFailure::KeyMismatch {
                field: "thresholds.min_cosine",
                ..
            }
        ));
        assert!(matches!(
            &failures[1],
            EvidenceFailure::FailingMetric {
                metric: "min_cosine",
                ..
            }
        ));
    }

    fn assert_single_key_mismatch(failures: &[EvidenceFailure], expected_field: &str) {
        assert!(
            matches!(failures, [EvidenceFailure::KeyMismatch { field, .. }] if *field == expected_field),
            "{failures:?}"
        );
    }

    #[test]
    fn a_record_for_another_manifest_fails() {
        let failures = failures_after(GTE, |record| {
            record["manifest_digest"] = json!("0".repeat(64))
        });
        assert_single_key_mismatch(&failures, "manifest_digest");
    }

    #[test]
    fn a_record_for_another_fingerprint_fails() {
        let failures = failures_after(GTE, |record| record["fingerprint"] = json!("a".repeat(64)));
        assert_single_key_mismatch(&failures, "fingerprint");
    }

    #[test]
    fn a_record_from_a_stale_engine_tree_fails() {
        let failures = failures_after(GTE, |record| {
            record["engine_tree"] =
                json!("3333333333333333333333333333333333333333\n2222222222222222222222222222222222222222")
        });
        assert_single_key_mismatch(&failures, "engine_tree");
    }

    #[test]
    fn a_record_for_another_fixture_revision_or_dtype_fails() {
        let revision = failures_after(GTE, |record| {
            record["fixture_revision"] = json!(record["fixture_revision"].as_i64().unwrap() + 1)
        });
        assert_single_key_mismatch(&revision, "fixture_revision");
        let dtype = failures_after(GTE, |record| record["dtype"] = json!("f32"));
        assert_single_key_mismatch(&dtype, "dtype");
    }

    #[test]
    fn a_record_measured_on_another_corpus_fails() {
        let failures = failures_after(RERANKER, |record| {
            record["corpus_id"] = json!("probe_rerank_gte_modernbert_v1")
        });
        assert_eq!(
            failures,
            [EvidenceFailure::WrongCorpus {
                file: RERANKER.into(),
                expected: "catalog_rerank_gte_modernbert_fp32".into(),
                actual: "probe_rerank_gte_modernbert_v1".into(),
            }]
        );
    }

    #[test]
    fn a_malformed_record_fails() {
        let failures = failures_after(GTE, |record| record["extra"] = json!(1));
        assert!(
            matches!(
                failures.as_slice(),
                [EvidenceFailure::MalformedRecord { .. }]
            ),
            "{failures:?}"
        );
        let no_machine = failures_after(GTE, |record| record["machine"] = json!({}));
        assert!(
            matches!(
                no_machine.as_slice(),
                [EvidenceFailure::MalformedRecord { .. }]
            ),
            "{no_machine:?}"
        );
    }

    /// Rewrites one backend value of `entry_id` in the compiled catalog,
    /// keeping the files consistent so only the backend value is at issue.
    fn catalog_with_backend_renamed(entry_id: &str, to: &str) -> String {
        let mut document: Value = serde_json::from_str(COMPILED_CATALOG_JSON).unwrap();
        let entry = document["models"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .find(|entry| entry["id"] == entry_id)
            .unwrap();
        let from = entry["backends"][0]["backend"].clone();
        entry["backends"][0]["backend"] = json!(to);
        for file in entry["files"].as_array_mut().unwrap() {
            for backend in file["backends"].as_array_mut().unwrap() {
                if *backend == from {
                    *backend = json!(to);
                }
            }
        }
        document.to_string()
    }

    #[test]
    fn disallowed_backend_values_fail_before_any_record_is_read() {
        for value in ["cpu", "llama"] {
            let catalog = catalog_with_backend_renamed("gte-modernbert-base", value);
            let failures = verify(&catalog, &passing_records()).unwrap_err();
            assert!(
                matches!(
                    failures.as_slice(),
                    [EvidenceFailure::Catalog(CatalogError::UnknownBackend { backend, .. })] if backend == value
                ),
                "{value}: {failures:?}"
            );
        }
    }

    #[test]
    fn a_cuda_row_on_any_entry_fails() {
        // A cuda row is schema-valid, so this is the frozen backend set
        // rejecting it, both added beside metal and replacing it.
        let mut added: Value = serde_json::from_str(COMPILED_CATALOG_JSON).unwrap();
        let entry = &mut added["models"][1];
        let mut row = entry["backends"][0].clone();
        row["backend"] = json!("cuda");
        entry["backends"].as_array_mut().unwrap().push(row);
        for file in entry["files"].as_array_mut().unwrap() {
            file["backends"] = json!(["metal", "cuda"]);
        }
        let mut records = passing_records();
        let mut cuda_record = records[RERANKER].clone();
        cuda_record["backend"] = json!("cuda");
        records.insert(
            record_file_name("gte-reranker-modernbert-base", "cuda"),
            cuda_record,
        );
        for catalog in [
            added.to_string(),
            catalog_with_backend_renamed("qwen3-embedding-0.6b", "cuda"),
        ] {
            let failures = verify(&catalog, &records).unwrap_err();
            assert!(
                matches!(
                    failures.as_slice(),
                    [EvidenceFailure::Catalog(CatalogError::FrozenMismatch { field, .. })] if field == "backends"
                ),
                "{failures:?}"
            );
        }
    }

    #[test]
    fn every_catalog_backend_names_a_checked_in_corpus() {
        let catalog = compiled_catalog().unwrap();
        for entry in catalog
            .models
            .iter()
            .filter(|entry| !entry.backends.is_empty())
        {
            if entry.self_check.is_none() {
                // Profile-only entries use the sealed parity corpus already
                // embedded by the certifier, not a second legacy corpus file.
                for backend in &entry.backends {
                    assert!(
                        synapse_certify::self_check::load(backend.profile.as_deref().unwrap())
                            .is_ok()
                    );
                }
                continue;
            }
            let corpus =
                named_corpus(&entry.id).unwrap_or_else(|| panic!("{} has no corpus", entry.id));
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("src/fixtures")
                .join(format!("{corpus}.json"));
            assert!(path.is_file(), "{} is missing", path.display());
        }
    }

    #[test]
    fn the_gte_reranker_corpus_is_a_pinned_fp32_reference_of_at_least_20_pairs() {
        let corpus: Value = serde_json::from_str(include_str!(
            "../fixtures/catalog_rerank_gte_modernbert_fp32.json"
        ))
        .unwrap();
        assert_eq!(corpus["corpus_id"], "catalog_rerank_gte_modernbert_fp32");
        let catalog = compiled_catalog().unwrap();
        let entry = catalog.entry("gte-reranker-modernbert-base").unwrap();
        // Tied to the exact upstream revision and files the catalog pins.
        assert_eq!(
            corpus["upstream"],
            serde_json::to_value(&entry.upstream).unwrap()
        );
        for (corpus_file, file) in corpus["files"].as_array().unwrap().iter().zip(&entry.files) {
            assert_eq!(corpus_file["path"], file.path);
            assert_eq!(corpus_file["sha256"], file.sha256);
            assert_eq!(corpus_file["size_bytes"], file.size_bytes);
        }
        assert!(corpus["reference_tool"]
            .as_str()
            .unwrap()
            .contains("transformers 5.16.1"));
        assert!(corpus["reference_tool"]
            .as_str()
            .unwrap()
            .contains("bench/parity/reference/generate_reference.py --catalog"));
        let mut pairs = 0;
        for item in corpus["items"].as_array().unwrap() {
            let candidates = item["candidates"].as_array().unwrap();
            let logits = item["raw_logits"].as_array().unwrap();
            let scores = item["scores"].as_array().unwrap();
            assert_eq!(candidates.len(), logits.len());
            assert_eq!(candidates.len(), scores.len());
            for (logit, score) in logits.iter().zip(scores) {
                let sigmoid = 1.0 / (1.0 + (-logit.as_f64().unwrap()).exp());
                assert!((sigmoid - score.as_f64().unwrap()).abs() < 1e-12);
            }
            pairs += candidates.len();
        }
        assert!(pairs >= 20, "{pairs} pairs");
        assert_eq!(corpus["pairs"], pairs);
    }

    #[test]
    fn engine_tree_joins_the_engine_directory_tree_hashes() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let tree = engine_tree_at(&repo_root, "HEAD").expect("git tree hashes at HEAD");
        let lines: Vec<&str> = tree.lines().collect();
        assert_eq!(lines.len(), ENGINE_TREE_DIRS.len());
        assert!(lines
            .iter()
            .all(|line| line.len() == 40 && line.bytes().all(|b| b.is_ascii_hexdigit())));
        assert!(engine_tree_at(&repo_root, "no-such-revision-anywhere").is_err());
    }

    #[test]
    fn read_evidence_dir_treats_a_missing_directory_as_no_records() {
        let missing = Path::new(env!("CARGO_MANIFEST_DIR")).join("no-such-evidence-dir");
        assert!(read_evidence_dir(&missing).unwrap().is_empty());
    }

    #[test]
    fn the_repository_evidence_dir_reads_as_record_files_only() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(EVIDENCE_DIR);
        let records = read_evidence_dir(&dir).expect("evidence dir is readable or absent");
        assert!(records.keys().all(|name| name.ends_with(".json")));
    }
}
