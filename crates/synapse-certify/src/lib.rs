//! Minimal release certification records and candidate-file validation.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

mod catalog;
pub mod command;
pub mod self_check;
mod validation;

pub use catalog::{derive_cells, row_backend, Catalog, CatalogLane, Cell, ROW_BACKENDS};
pub use validation::{validate, validate_checkout, validate_record};
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

/// Record schema version written by the producer and the only one the
/// validator accepts. Version 3 adds offline-verifiable ANE placement evidence;
/// hardware identities remain domain-separated SHA-256 digests.
pub const RECORD_SCHEMA: u32 = 3;

/// Domain-separation prefix hashed in front of an Apple platform UUID: the
/// bytes `synapse-certify/machine/v1` followed by one NUL byte.
pub const PLATFORM_UUID_HASH_PREFIX: &str = "synapse-certify/machine/v1\0";
/// Domain-separation prefix hashed in front of a GPU UUID: the bytes
/// `synapse-certify/gpu/v1` followed by one NUL byte.
pub const GPU_UUID_HASH_PREFIX: &str = "synapse-certify/gpu/v1\0";

fn prefixed_sha256(prefix: &str, value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix.as_bytes());
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// `machine.platform_uuid_sha256` on Apple records: lowercase hex SHA-256 of
/// `"synapse-certify/machine/v1\0"` followed by the platform UUID exactly as
/// `system_profiler SPHardwareDataType` reports it.
///
/// The raw UUID is a stable hardware identifier and never enters a record.
/// The digest still links records from one machine, because the same UUID
/// always hashes to the same value, and the versioned prefix keeps it from
/// matching a plain SHA-256 of the UUID computed anywhere else.
pub fn platform_uuid_sha256(uuid: &str) -> String {
    prefixed_sha256(PLATFORM_UUID_HASH_PREFIX, uuid)
}

/// `machine.gpu.uuid_sha256` on CUDA and Vulkan records: lowercase hex
/// SHA-256 of `"synapse-certify/gpu/v1\0"` followed by the GPU UUID exactly as
/// `nvidia-smi` or `vulkaninfo` reports it. Same reasoning as
/// [`platform_uuid_sha256`]; the separate prefix keeps a GPU digest from ever
/// equalling a platform digest.
pub fn gpu_uuid_sha256(uuid: &str) -> String {
    prefixed_sha256(GPU_UUID_HASH_PREFIX, uuid)
}

fn apple_row(row: &str) -> bool {
    matches!(row, "metal-m5" | "ane-m5")
}

/// Refuses a raw hardware UUID anywhere the collector used to put one, and
/// requires the digest that replaced it for the record's platform.
fn machine_identity(record: &Record) -> Result<()> {
    let machine = &record.machine;
    if machine.get("platform_uuid").is_some() || machine["gpu"].get("uuid").is_some() {
        return Err(refuse("machine carries a raw hardware UUID"));
    }
    let digest = if apple_row(&record.row_id) {
        &machine["platform_uuid_sha256"]
    } else {
        &machine["gpu"]["uuid_sha256"]
    };
    if !digest.as_str().is_some_and(|digest| hex(digest, 64)) {
        return Err(refuse("machine identity digest missing or malformed"));
    }
    Ok(())
}

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
    pub fn passed(&self) -> bool {
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
    pub placement: Option<Placement>,
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Placement {
    pub layer_count: usize,
    pub admitted_count: usize,
    pub inventories: Vec<Inventory>,
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
    /// Supervisor admission events counted separately from the captured inventories,
    /// so an omitted inventory cannot silently reduce both counts.
    pub admitted_count: usize,
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
    if row_backend(row).is_none() {
        return Err(refuse("unknown row or model"));
    }
    self_check::manifest()?
        .model(model)
        .map_err(|e| refuse(e.to_string()))?;
    Ok(())
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
    if record.schema != RECORD_SCHEMA {
        return Err(refuse("record_schema_mismatch"));
    }
    if !hex(&record.source_commit, 40) {
        return Err(refuse("invalid source commit"));
    }
    combination(&record.row_id, &record.model)?;
    machine_identity(record)?;
    let p = &record.parity;
    if p.model != record.model || p.fingerprint != record.fingerprint {
        return Err(refuse("parity identity mismatch"));
    }
    Ok(())
}

fn producer_identity(record: &Record) -> Result<()> {
    identity(record)?;
    if record.parity.operation != record.operation || record.parity.profile_id != record.profile_id
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
        || (evidence.admitted_count > 0
            && evidence.inventories.len() == evidence.admitted_count
            && evidence
                .inventories
                .iter()
                .all(|i| i.passed(evidence.layer_count, model)));
    if !placement_ok {
        return Err(refuse("placement inventory failed"));
    }
    if !floor_ok || !parity.passed() || !evidence.admission.passed() {
        let gate = if !floor_ok {
            "floor"
        } else if !parity.passed() {
            "parity"
        } else {
            "admission"
        };
        return Err(refuse(format!("{gate} gate failed")));
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
        schema: RECORD_SCHEMA,
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
        status: "passed".into(),
        placement: (row == "ane-m5").then_some(Placement {
            layer_count: evidence.layer_count,
            admitted_count: evidence.admitted_count,
            inventories: evidence.inventories,
        }),
    };
    producer_identity(&record)?;
    Ok(record)
}

/// Called only after all producer gates succeed; a refused run writes nothing.
pub fn write_record(record: &Record, checkout: &Path) -> Result<PathBuf> {
    producer_identity(record)?;
    let path = record.path(checkout);
    std::fs::create_dir_all(path.parent().ok_or_else(|| refuse("record path"))?)
        .map_err(|e| refuse(e.to_string()))?;
    let bytes = serde_json::to_vec_pretty(record).map_err(|e| refuse(e.to_string()))?;
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, bytes).map_err(|e| refuse(e.to_string()))?;
    std::fs::rename(temporary, &path).map_err(|e| refuse(e.to_string()))?;
    Ok(path)
}
