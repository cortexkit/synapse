//! Minimal release certification records and candidate-file validation.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

pub mod command;

pub const ROWS: [&str; 8] = [
    "vulkan-windows-amd",
    "vulkan-linux-amd",
    "vulkan-linux-nvidia",
    "vulkan-windows-nvidia",
    "cuda-linux-nvidia",
    "cuda-windows-nvidia",
    "metal-m5",
    "ane-m5",
];
pub const MODELS: [&str; 4] = [
    "gte-modernbert-base",
    "gte-reranker-modernbert-base",
    "qwen3-embedding-0.6b",
    "qwen3-reranker-0.6b",
];
pub const GATES: [&str; 10] = [
    "complete",
    "dimension",
    "finite",
    "unit_norm",
    "cosine",
    "score",
    "ranking",
    "coverage_8192",
    "coverage",
    "semantics",
];

#[derive(Debug, thiserror::Error)]
#[error("certification_refused: {0}")]
pub struct Error(pub String);
pub type Result<T> = std::result::Result<T, Error>;
fn refuse(message: impl Into<String>) -> Error {
    Error(message.into())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub role: String,
    pub file: String,
    pub sha256: String,
}

/// Retains evaluator metrics verbatim, including fields added by the evaluator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Parity {
    pub model: String,
    pub operation: String,
    pub profile_id: String,
    pub fingerprint: String,
    pub fixture_set_id: String,
    pub completed_fixtures: usize,
    pub expected_fixtures: usize,
    pub tolerance_class: String,
    pub metrics: Value,
    pub gates: BTreeMap<String, bool>,
    pub failures: Vec<String>,
}
impl Parity {
    pub fn passed(&self) -> bool {
        GATES
            .iter()
            .all(|name| self.gates.get(*name) == Some(&true))
            && self.gates.values().all(|passed| *passed)
    }
    fn failed(&self) -> bool {
        self.gates.values().any(|passed| !passed)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Outcome {
    pub outcome: String,
    pub truncated: bool,
    pub diverted: bool,
    pub worker_requests: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Admission {
    pub tokens_8192: Outcome,
    pub tokens_8193: Outcome,
}
impl Admission {
    fn passed(&self) -> bool {
        self.tokens_8192.outcome == "processed"
            && !self.tokens_8192.truncated
            && !self.tokens_8192.diverted
            && self.tokens_8193.outcome == "sequence_too_long"
            && self.tokens_8193.worker_requests == 0
            && !self.tokens_8193.truncated
            && !self.tokens_8193.diverted
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawSeries {
    pub session_id: String,
    pub ane: Vec<f64>,
    pub metal: Vec<f64>,
}
impl RawSeries {
    pub fn ratio(&self) -> Result<f64> {
        if self.session_id.is_empty() {
            return Err(refuse("missing latency session"));
        }
        Ok(median(&self.ane)? / median(&self.metal)?)
    }
}
pub fn median(series: &[f64]) -> Result<f64> {
    if series.len() != 23 || series.iter().any(|n| !n.is_finite() || *n <= 0.0) {
        return Err(refuse(
            "latency series must contain 3 warmups and 20 positive samples",
        ));
    }
    let mut measured = series[3..].to_vec();
    measured.sort_by(f64::total_cmp);
    Ok(measured[9] / 2.0 + measured[10] / 2.0)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub schema: u32,
    pub row_id: String,
    pub source_commit: String,
    pub machine: Value,
    pub executed_artifacts: Vec<Artifact>,
    pub model: String,
    pub operation: String,
    pub profile_id: String,
    pub fingerprint: String,
    pub parity: Parity,
    pub admission: Admission,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drop_cause: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_series: Option<RawSeries>,
}
impl Record {
    pub fn eligible(&self) -> bool {
        self.status == "passed"
    }
    pub fn path(&self, checkout: &Path) -> PathBuf {
        checkout
            .join("docs/evidence/certification")
            .join(&self.source_commit)
            .join(&self.row_id)
            .join(format!("{}.json", self.model))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Executable {
    pub id: String,
    pub layers: Vec<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Inventory {
    pub executables: Vec<Executable>,
    pub cpu_stages: Vec<String>,
}
impl Inventory {
    pub fn passed(&self, layers: usize, model: &str) -> bool {
        let mut counts = vec![0usize; layers];
        let mut ids = BTreeSet::new();
        for executable in &self.executables {
            if executable.id.is_empty() || !ids.insert(&executable.id) {
                return false;
            }
            for layer in &executable.layers {
                let Some(count) = counts.get_mut(*layer) else {
                    return false;
                };
                *count += 1;
            }
        }
        layers > 0
            && counts.iter().all(|count| *count == 1)
            && self.cpu_stages.iter().all(|stage| match stage.as_str() {
                "token_embedding" | "mask_position" | "final_norm" | "pooling" => true,
                "rotation_in" | "rotation_out" => model.starts_with("gte-"),
                "gte_classifier_head" => model == "gte-reranker-modernbert-base",
                "qwen_yes_no_readout" => model == "qwen3-reranker-0.6b",
                _ => false,
            })
    }
}

/// Observations collected through the candidate module. The worker's hardware
/// capability check (`--probe-floor`) is separate because Metal is in-process.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunEvidence {
    pub source_commit: String,
    pub machine: Value,
    pub artifacts: Vec<ArtifactFile>,
    pub operation: String,
    pub profile_id: String,
    pub fingerprint: String,
    pub fixture_set_id: String,
    pub expected_fixtures: usize,
    pub tolerance_class: String,
    pub parity: Option<Parity>,
    pub admission: Admission,
    pub layer_count: usize,
    pub inventories: Vec<Inventory>,
    pub raw_series: Option<RawSeries>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFile {
    pub role: String,
    pub file: String,
}

/// A live runner must execute the extracted candidate, not a locally rebuilt
/// module. Implementations return evaluator output and observed IPC admission.
pub trait Runner {
    fn probe_floor(&mut self, row: &str, model: &str) -> Result<String>;
    fn observe(&mut self, row: &str, model: &str) -> Result<RunEvidence>;
}

fn combination(row: &str, model: &str) -> Result<()> {
    if !ROWS.contains(&row) || !MODELS.contains(&model) {
        return Err(refuse("unknown row or model"));
    }
    Ok(())
}
fn droppable(row: &str, model: &str) -> bool {
    row == "ane-m5" && model.starts_with("qwen3-")
}
fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn safe_file(file: &str) -> Result<()> {
    if file.is_empty()
        || Path::new(file)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(refuse("artifact path must be relative to extracted assets"));
    }
    Ok(())
}
pub fn digest(assets: &Path, file: &str) -> Result<String> {
    safe_file(file)?;
    let base = assets.canonicalize().map_err(|e| refuse(e.to_string()))?;
    let path = base
        .join(file)
        .canonicalize()
        .map_err(|e| refuse(e.to_string()))?;
    if !path.starts_with(&base) {
        return Err(refuse("artifact escapes extracted assets"));
    }
    let bytes = std::fs::read(path).map_err(|e| refuse(e.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
fn identity(record: &Record) -> Result<()> {
    if record.schema != 1 || !hex(&record.source_commit, 40) {
        return Err(refuse("invalid schema or source commit"));
    }
    combination(&record.row_id, &record.model)?;
    let p = &record.parity;
    if p.model != record.model
        || p.operation != record.operation
        || p.profile_id != record.profile_id
        || p.fingerprint != record.fingerprint
    {
        return Err(refuse("parity identity mismatch"));
    }
    Ok(())
}

pub fn produce(runner: &mut impl Runner, assets: &Path, row: &str, model: &str) -> Result<Record> {
    combination(row, model)?;
    let floor_ok = row == "metal-m5" || runner.probe_floor(row, model)? == "ok";
    let evidence = runner.observe(row, model)?;
    let parity = evidence.parity.unwrap_or_else(|| Parity {
        model: model.into(),
        operation: evidence.operation.clone(),
        profile_id: evidence.profile_id.clone(),
        fingerprint: evidence.fingerprint.clone(),
        fixture_set_id: evidence.fixture_set_id,
        completed_fixtures: 0,
        expected_fixtures: evidence.expected_fixtures,
        tolerance_class: evidence.tolerance_class,
        metrics: serde_json::json!({}),
        gates: GATES.iter().map(|g| ((*g).into(), false)).collect(),
        failures: vec!["no parity output".into()],
    });
    let placement_ok = row != "ane-m5"
        || evidence
            .inventories
            .iter()
            .all(|i| i.passed(evidence.layer_count, model));
    if !placement_ok {
        return Err(refuse("placement inventory failed"));
    }
    let cause = if droppable(row, model) && parity.failed() {
        Some("parity")
    } else if droppable(row, model)
        && evidence
            .raw_series
            .as_ref()
            .map(RawSeries::ratio)
            .transpose()?
            .is_some_and(|ratio| ratio > 3.0)
    {
        Some("latency")
    } else {
        None
    };
    if cause.is_none()
        && (!floor_ok
            || !parity.passed()
            || !evidence.admission.passed()
            || (row == "ane-m5" && evidence.inventories.is_empty()))
    {
        return Err(refuse("floor, parity or admission failed"));
    }
    let required_worker = if row.starts_with("cuda-") {
        Some("ck-synapse-worker-cuda")
    } else if row.starts_with("vulkan-") {
        Some("ck-synapse-worker-vulkan")
    } else if row == "ane-m5" {
        Some("ck-synapse-worker-ane-direct")
    } else {
        None
    };
    for role in std::iter::once("ck-synapse").chain(required_worker) {
        if !evidence.artifacts.iter().any(|a| a.role == role) {
            return Err(refuse(format!("missing executed artifact: {role}")));
        }
    }
    let artifacts = evidence
        .artifacts
        .into_iter()
        .map(|a| {
            Ok(Artifact {
                sha256: digest(assets, &a.file)?,
                role: a.role,
                file: a.file,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let record = Record {
        schema: 1,
        row_id: row.into(),
        source_commit: evidence.source_commit,
        machine: evidence.machine,
        executed_artifacts: artifacts,
        model: model.into(),
        operation: evidence.operation,
        profile_id: evidence.profile_id,
        fingerprint: evidence.fingerprint,
        parity,
        admission: evidence.admission,
        status: if cause.is_some() { "dropped" } else { "passed" }.into(),
        drop_cause: cause.map(str::to_string),
        raw_series: if cause.is_some() {
            evidence.raw_series
        } else {
            None
        },
    };
    identity(&record)?;
    Ok(record)
}

/// Called only after all producer gates succeed; a refused run writes nothing.
pub fn write_record(record: &Record, checkout: &Path) -> Result<PathBuf> {
    identity(record)?;
    let path = record.path(checkout);
    std::fs::create_dir_all(path.parent().ok_or_else(|| refuse("record path"))?)
        .map_err(|e| refuse(e.to_string()))?;
    let bytes = serde_json::to_vec_pretty(record).map_err(|e| refuse(e.to_string()))?;
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, bytes).map_err(|e| refuse(e.to_string()))?;
    std::fs::rename(temporary, &path).map_err(|e| refuse(e.to_string()))?;
    Ok(path)
}

/// Validates status, drop evidence, extracted-file digests, parity identity and
/// matrix coverage. Hardware and admission checks belong to the producer, not
/// this offline validator, which does not execute candidate binaries.
pub fn validate(records: &[Record], assets: &Path) -> Result<Vec<(String, String)>> {
    let mut combinations = BTreeSet::new();
    let mut eligible = Vec::new();
    for record in records {
        identity(record)?;
        if !combinations.insert((record.row_id.as_str(), record.model.as_str())) {
            return Err(refuse("duplicate combination"));
        }
        match record.status.as_str() {
            "passed" => eligible.push((record.row_id.clone(), record.model.clone())),
            "dropped" if droppable(&record.row_id, &record.model) => {
                if !record.parity.failed()
                    && !record
                        .raw_series
                        .as_ref()
                        .ok_or_else(|| refuse("missing raw latency series"))?
                        .ratio()?
                        .gt(&3.0)
                {
                    return Err(refuse(
                        "drop requires failing parity or ratio strictly greater than 3",
                    ));
                }
            }
            _ => return Err(refuse("invalid certification status")),
        }
        for artifact in &record.executed_artifacts {
            if digest(assets, &artifact.file)? != artifact.sha256 {
                return Err(refuse(format!(
                    "executed artifact digest mismatch: {}",
                    artifact.file
                )));
            }
        }
    }
    if combinations.len() != ROWS.len() * MODELS.len() {
        return Err(refuse("missing certification combination"));
    }
    Ok(eligible)
}

pub fn validate_checkout(
    checkout: &Path,
    assets: &Path,
    source: &str,
) -> Result<Vec<(String, String)>> {
    if !hex(source, 40) {
        return Err(refuse("invalid source commit"));
    }
    let mut records = Vec::new();
    for row in ROWS {
        for model in MODELS {
            let path = checkout
                .join("docs/evidence/certification")
                .join(source)
                .join(row)
                .join(format!("{model}.json"));
            let bytes =
                std::fs::read(path).map_err(|e| refuse(format!("missing {row}/{model}: {e}")))?;
            let record: Record =
                serde_json::from_slice(&bytes).map_err(|e| refuse(e.to_string()))?;
            if record.row_id != row || record.model != model || record.source_commit != source {
                return Err(refuse("record does not match canonical path"));
            }
            records.push(record);
        }
    }
    validate(&records, assets)
}
