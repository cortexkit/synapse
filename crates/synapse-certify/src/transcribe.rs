//! Speech evidence shares the release record envelope and offline validator.
use crate::{refuse, Artifact, Catalog, Inventory, Record, Result, RECORD_SCHEMA};
use serde::{Deserialize, Deserializer, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};
use synapse_parity::{
    manifest::ToleranceClass,
    speech::{self, SpeechFixture},
};

/// Dispatch on the existing operation field without changing legacy record bytes.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum CertificationRecord {
    Legacy(Record),
    Transcribe(TranscribeRecord),
}
impl<'de> Deserialize<'de> for CertificationRecord {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        if value["operation"] == "transcribe" {
            serde_json::from_value(value)
                .map(Self::Transcribe)
                .map_err(serde::de::Error::custom)
        } else {
            serde_json::from_value(value)
                .map(Self::Legacy)
                .map_err(serde::de::Error::custom)
        }
    }
}
impl CertificationRecord {
    pub fn key(&self) -> (&str, &str, &str) {
        match self {
            Self::Legacy(r) => (&r.row_id, &r.model, &r.source_commit),
            Self::Transcribe(r) => (&r.row_id, &r.model, &r.source_commit),
        }
    }
    pub fn validate(&self, checkout: &Path, assets: &Path, catalog: &Catalog) -> Result<()> {
        match self {
            Self::Legacy(r) => crate::validate_record(r, checkout, assets, catalog),
            Self::Transcribe(r) => validate_record(r, checkout, assets, catalog),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Part {
    pub bytes: u64,
    pub sha256: String,
    pub pages: u64,
    pub reassembled_bytes: u64,
    pub reassembled_sha256: String,
}
impl Part {
    fn valid(&self) -> bool {
        self.bytes == self.reassembled_bytes
            && self.sha256 == self.reassembled_sha256
            && crate::hex(&self.sha256, 64)
            && self.pages == self.bytes.div_ceil(524288)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GradedFixture {
    pub id: String,
    pub lane_id: String,
    pub fingerprint: String,
    pub valid_encoder_frames: u64,
    pub text: String,
    pub encoder_output: Part,
    pub decode_trace: Part,
    /// Includes finite tensors, a complete trace and matching emitted
    /// (token_id, frame_index, duration) triples implied by that trace.
    pub integrity_passed: bool,
    pub trace_passed: bool,
    pub margin_exempt: bool,
    pub encoder_min_cosine: Option<f64>,
    pub encoder_error: Option<String>,
    pub trace_error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpectedRefusal {
    pub id: String,
    pub code: String,
    pub field: String,
    pub limit: String,
    pub duration_ms: u64,
    pub max_duration_ms: u64,
    pub valid_encoder_frames: u64,
    pub max_encoder_frames: u64,
    pub worker_requests: u64,
    pub shape_admissions: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shape {
    pub encoder_frames: u64,
    pub executables: Vec<crate::Executable>,
    pub cpu_stages: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurements {
    pub self_check_wall_ms: u64,
    pub longest_progress_interval_ms: u64,
    pub ane_shapes: Vec<Shape>,
    pub peak_device_memory_bytes: Option<u64>,
    pub peak_storage_buffer_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfCheck {
    pub state: String,
    pub fingerprints: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObserveRerun {
    pub fixture_ids: Vec<String>,
    pub bit_identical: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranscribeRecord {
    pub schema: u32,
    pub row_id: String,
    pub source_commit: String,
    pub machine: serde_json::Value,
    pub executed_artifacts: Vec<Artifact>,
    pub model: String,
    pub operation: String,
    pub profile_id: String,
    pub fingerprint: String,
    pub fixture_set_id: String,
    pub precision_class: ToleranceClass,
    /// The SHA-256-pinned reference ladder duration actually executed,
    /// not the profile's target.
    pub max_duration_ms: u64,
    pub max_encoder_frames: u64,
    pub graded_fixtures: Vec<GradedFixture>,
    pub expected_refusals: Vec<ExpectedRefusal>,
    pub executed_ladder: Vec<String>,
    pub split_wer: BTreeMap<String, [f64; 2]>,
    pub self_check: SelfCheck,
    pub measurements: Measurements,
    pub observe_rerun: ObserveRerun,
    pub status: String,
}
impl TranscribeRecord {
    pub fn path(&self, checkout: &Path) -> PathBuf {
        checkout
            .join("docs/evidence/certification")
            .join(&self.source_commit)
            .join(&self.row_id)
            .join(format!("{}.json", self.model))
    }
}

pub fn fixtures(checkout: &Path) -> Result<Vec<SpeechFixture>> {
    let root = checkout.join("bench/parity");
    // Bind the checkout index to the candidate's embedded index before reading it.
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("fixtures/index.json")).map_err(|e| refuse(e.to_string()))?,
    )
    .map_err(|e| refuse(e.to_string()))?;
    let id =
        synapse_parity::evaluator::fixture_set_id(&crate::self_check::manifest()?, speech::MODEL);
    if index[&id] != crate::self_check::source_index()?[&id] {
        return Err(refuse("speech fixture index differs from candidate"));
    }
    speech::load_transcribe_fixtures(&root, &crate::self_check::manifest()?)
        .map_err(|e| refuse(e.to_string()))
}

pub fn validate_record(
    record: &TranscribeRecord,
    checkout: &Path,
    assets: &Path,
    catalog: &Catalog,
) -> Result<()> {
    validate_claims(record, &fixtures(checkout)?, catalog)?;
    for artifact in &record.executed_artifacts {
        if crate::digest(assets, &artifact.file)? != artifact.sha256 {
            return Err(refuse(format!(
                "executed artifact digest mismatch: {}",
                artifact.file
            )));
        }
    }
    Ok(())
}

/// Compares a record's stored WER table with the one re-derived from its
/// graded text. The stored values have been through JSON, and serde_json's
/// default parser can land one ULP away from the digits written, so exact
/// float equality refuses honest records. A real disagreement in graded text
/// moves a WER by at least one word over the split, many orders above this
/// bound.
fn same_split_wer(
    recorded: &BTreeMap<String, [f64; 2]>,
    derived: &BTreeMap<String, [f64; 2]>,
) -> bool {
    recorded.len() == derived.len()
        && recorded.iter().zip(derived).all(|((rk, rv), (dk, dv))| {
            rk == dk && rv.iter().zip(dv).all(|(a, b)| (a - b).abs() <= 1e-12)
        })
}

pub fn validate_claims(
    r: &TranscribeRecord,
    fixtures: &[SpeechFixture],
    catalog: &Catalog,
) -> Result<()> {
    if r.schema != RECORD_SCHEMA {
        return Err(refuse("record_schema_mismatch"));
    }
    if r.operation != "transcribe"
        || r.model != speech::MODEL
        || catalog.tasks.get(&r.model).map(String::as_str) != Some("transcribe")
    {
        return Err(refuse("catalog_operation_mismatch"));
    }
    let backend =
        crate::row_backend(&r.row_id).ok_or_else(|| refuse("unknown certification row"))?;
    let cell = crate::derive_cells(&catalog.lanes)
        .into_iter()
        .find(|c| c.row_id == r.row_id && c.model == r.model)
        .ok_or_else(|| refuse("undeclared_certification_cell"))?;
    if r.fingerprint != cell.fingerprint || r.profile_id != format!("{}.{}", r.model, cell.engine) {
        return Err(refuse("catalog_pin_mismatch"));
    }
    let manifest = crate::self_check::manifest()?;
    let profile = manifest
        .profiles
        .get(&r.profile_id)
        .and_then(|p| p.transcribe.as_ref())
        .ok_or_else(|| refuse("speech profile missing"))?;
    profile.validate().map_err(|e| refuse(e.to_string()))?;
    if r.fixture_set_id != synapse_parity::evaluator::fixture_set_id(&manifest, &r.model)
        || r.precision_class != profile.precision_class
    {
        return Err(refuse("speech reference or precision mismatch"));
    }
    if !crate::hex(&r.source_commit, 40) || r.status != "passed" {
        return Err(refuse("invalid speech source or status"));
    }
    let machine_digest = if matches!(backend, "ane" | "metal") {
        &r.machine["platform_uuid_sha256"]
    } else {
        &r.machine["gpu"]["uuid_sha256"]
    };
    if r.machine.get("platform_uuid").is_some()
        || r.machine["gpu"].get("uuid").is_some()
        || !machine_digest.as_str().is_some_and(|s| crate::hex(s, 64))
    {
        return Err(refuse("invalid speech hardware identity"));
    }
    let worker = match backend {
        "ane" => Some("ck-synapse-worker-ane-direct"),
        "cuda" => Some("ck-synapse-worker-cuda"),
        "vulkan" => Some("ck-synapse-worker-vulkan"),
        _ => None,
    };
    for role in std::iter::once("ck-synapse").chain(worker) {
        if !r
            .executed_artifacts
            .iter()
            .any(|a| a.role == role && crate::hex(&a.sha256, 64))
        {
            return Err(refuse(format!("missing executed artifact: {role}")));
        }
    }
    if r.max_duration_ms == 0
        || r.max_duration_ms > profile.max_duration_ms
        || r.max_encoder_frames != speech::frame_counts(r.max_duration_ms * 16).2
        || !fixtures
            .iter()
            .any(|f| f.id.starts_with("ladder-") && f.duration_ms == r.max_duration_ms)
    {
        return Err(refuse("uncertified speech maximum"));
    }
    let lane_id = format!("{}-{backend}", r.model);
    let graded: BTreeMap<_, _> = r
        .graded_fixtures
        .iter()
        .map(|g| (g.id.as_str(), g))
        .collect();
    let refused: BTreeMap<_, _> = r
        .expected_refusals
        .iter()
        .map(|g| (g.id.as_str(), g))
        .collect();
    if graded.len() != r.graded_fixtures.len()
        || refused.len() != r.expected_refusals.len()
        || graded.len() + refused.len() != fixtures.len()
    {
        return Err(refuse("speech fixture coverage mismatch"));
    }
    let mut ladder = BTreeSet::new();
    let mut rerun = BTreeSet::new();
    let mut wer = Vec::new();
    for f in fixtures {
        if f.duration_ms > r.max_duration_ms {
            let g = refused
                .get(f.id.as_str())
                .ok_or_else(|| refuse("missing expected refusal"))?;
            if !f.id.starts_with("ladder-")
                || g.code != "input_too_large"
                || g.field != "input.audio"
                || g.limit != "max_duration_ms"
                || g.duration_ms != f.duration_ms
                || g.max_duration_ms != r.max_duration_ms
                || g.valid_encoder_frames != f.valid_encoder_frames
                || g.max_encoder_frames != r.max_encoder_frames
                || g.worker_requests != 0
                || g.shape_admissions != 0
            {
                return Err(refuse("speech duration refusal failed"));
            }
            continue;
        }
        let g = graded
            .get(f.id.as_str())
            .ok_or_else(|| refuse("missing graded fixture"))?;
        if g.lane_id != lane_id
            || g.fingerprint != r.fingerprint
            || g.valid_encoder_frames != f.valid_encoder_frames
            || !g.integrity_passed
            || !g.encoder_output.valid()
            || !g.decode_trace.valid()
            || g.encoder_output.bytes != f.valid_encoder_frames * 4096
            || g.decode_trace.bytes == 0
            || g.decode_trace.bytes % 40 != 0
            || g.decode_trace.bytes > f.valid_encoder_frames * speech::CAP as u64 * 40
        {
            return Err(refuse("speech observation integrity failed"));
        }
        if f.encoder.is_some() {
            if !g
                .encoder_min_cosine
                .is_some_and(|v| v.is_finite() && (-1.0..=1.0).contains(&v))
                && g.encoder_error.is_none()
            {
                return Err(refuse("encoder gate missing"));
            }
            if r.precision_class == ToleranceClass::Fp32
                && (!g.encoder_min_cosine.is_some_and(|v| v >= 0.999) || g.encoder_error.is_some())
            {
                return Err(refuse("encoder gate failed"));
            }
            rerun.insert(f.id.as_str());
        }
        if r.precision_class == ToleranceClass::Fp32 && (!g.trace_passed || g.trace_error.is_some())
        {
            return Err(refuse("trace gate failed"));
        }
        if !g.trace_passed && g.trace_error.is_none() {
            return Err(refuse("trace gate missing"));
        }
        if f.id.starts_with("ladder-") {
            if f.encoder.is_none() {
                return Err(refuse("cached ladder encoder reference missing"));
            }
            ladder.insert(f.id.as_str());
        }
        if let (Some(split), Some(truth)) = (&f.split, &f.ground_truth) {
            wer.push((
                split.as_str(),
                truth.as_str(),
                f.text.as_str(),
                g.text.as_str(),
            ));
        }
    }
    if r.executed_ladder
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != ladder
        || r.executed_ladder.len() != ladder.len()
    {
        return Err(refuse("executed ladder mismatch"));
    }
    let split_wer = speech::grade_wer(wer).map_err(|e| refuse(e.to_string()))?;
    if split_wer.is_empty() || !same_split_wer(&r.split_wer, &split_wer) {
        return Err(refuse("split WER mismatch"));
    }
    if !r.observe_rerun.bit_identical
        || r.observe_rerun.fixture_ids.len() != rerun.len()
        || r.observe_rerun
            .fixture_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != rerun
    {
        return Err(refuse("observe on/off rerun missing or failed"));
    }
    if r.self_check.state != "passed"
        || r.self_check.fingerprints != [r.fingerprint.clone()]
        || r.measurements.self_check_wall_ms >= 2000
        || r.measurements.longest_progress_interval_ms >= 120000
    {
        return Err(refuse("speech self-check or progress gate failed"));
    }
    if backend == "ane" {
        let expected: BTreeSet<_> = fixtures
            .iter()
            .filter(|f| f.duration_ms <= r.max_duration_ms)
            .map(|f| {
                profile
                    .encoder_rungs
                    .iter()
                    .copied()
                    .find(|v| u64::from(*v) >= f.valid_encoder_frames)
                    .map(u64::from)
            })
            .collect();
        let actual: BTreeSet<_> = r
            .measurements
            .ane_shapes
            .iter()
            .map(|s| Some(s.encoder_frames))
            .collect();
        if actual != expected
            || r.measurements.ane_shapes.len() != expected.len()
            || r.measurements.ane_shapes.iter().any(|s| {
                s.executables.len() != 24
                    || !(Inventory {
                        executables: s.executables.clone(),
                        cpu_stages: vec![],
                    })
                    .passed(24, &r.model)
            })
        {
            return Err(refuse("ANE per-shape inventory failed"));
        }
    }
    if matches!(backend, "cuda" | "vulkan")
        && !r
            .measurements
            .peak_device_memory_bytes
            .is_some_and(|v| v > 0)
    {
        return Err(refuse("peak device memory missing"));
    }
    if backend == "vulkan" {
        let p = &manifest.profiles[&r.profile_id];
        if !r
            .measurements
            .peak_device_memory_bytes
            .zip(p.vulkan_min_device_local_bytes)
            .is_some_and(|(v, floor)| v <= floor)
            || !r
                .measurements
                .peak_storage_buffer_bytes
                .zip(p.vulkan_min_storage_buffer_range)
                .is_some_and(|(v, floor)| v > 0 && v <= floor)
        {
            return Err(refuse("Vulkan memory floor exceeded"));
        }
    }
    Ok(())
}

/// Write only after the same offline checks used by the candidate have passed.
pub fn write_record(
    record: &TranscribeRecord,
    checkout: &Path,
    assets: &Path,
    catalog: &Catalog,
) -> Result<PathBuf> {
    validate_record(record, checkout, assets, catalog)?;
    let path = record.path(checkout);
    std::fs::create_dir_all(path.parent().ok_or_else(|| refuse("record path"))?)
        .map_err(|e| refuse(e.to_string()))?;
    let temp = path.with_extension("json.tmp");
    std::fs::write(
        &temp,
        serde_json::to_vec_pretty(record).map_err(|e| refuse(e.to_string()))?,
    )
    .map_err(|e| refuse(e.to_string()))?;
    std::fs::rename(temp, &path).map_err(|e| refuse(e.to_string()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }
    fn catalog() -> Catalog {
        let value: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root().join("crates/synapse-module/src/catalog/models.json")).unwrap(),
        )
        .unwrap();
        let models = value["models"].as_array().unwrap();
        Catalog {
            lanes: models
                .iter()
                .flat_map(|m| {
                    m["backends"].as_array().unwrap().iter().map(|b| {
                        (
                            m["id"].as_str().unwrap().into(),
                            b["backend"].as_str().unwrap().into(),
                            b["engine"].as_str().unwrap().into(),
                            b["fingerprint"].as_str().unwrap().into(),
                        )
                    })
                })
                .collect(),
            tasks: models
                .iter()
                .map(|m| {
                    (
                        m["id"].as_str().unwrap().into(),
                        m["task"].as_str().unwrap().into(),
                    )
                })
                .collect(),
        }
    }
    fn part(bytes: u64) -> Part {
        Part {
            bytes,
            sha256: "a".repeat(64),
            pages: bytes.div_ceil(524288),
            reassembled_bytes: bytes,
            reassembled_sha256: "a".repeat(64),
        }
    }
    fn record(row: &str, maximum: u64) -> TranscribeRecord {
        let catalog = catalog();
        let cell = crate::derive_cells(&catalog.lanes)
            .into_iter()
            .find(|c| c.row_id == row && c.model == speech::MODEL)
            .unwrap();
        let backend = crate::row_backend(row).unwrap();
        let manifest = crate::self_check::manifest().unwrap();
        let profile_id = format!("{}.{}", speech::MODEL, cell.engine);
        let profile = manifest.profiles[&profile_id].transcribe.as_ref().unwrap();
        let fixtures = fixtures(&root()).unwrap();
        let mut graded = vec![];
        let mut refused = vec![];
        let mut shapes = BTreeSet::new();
        for f in &fixtures {
            if f.duration_ms > maximum {
                refused.push(ExpectedRefusal {
                    id: f.id.clone(),
                    code: "input_too_large".into(),
                    field: "input.audio".into(),
                    limit: "max_duration_ms".into(),
                    duration_ms: f.duration_ms,
                    max_duration_ms: maximum,
                    valid_encoder_frames: f.valid_encoder_frames,
                    max_encoder_frames: speech::frame_counts(maximum * 16).2,
                    worker_requests: 0,
                    shape_admissions: 0,
                });
            } else {
                shapes.insert(
                    profile
                        .encoder_rungs
                        .iter()
                        .copied()
                        .find(|r| u64::from(*r) >= f.valid_encoder_frames)
                        .unwrap(),
                );
                graded.push(GradedFixture {
                    id: f.id.clone(),
                    lane_id: format!("{}-{backend}", speech::MODEL),
                    fingerprint: cell.fingerprint.clone(),
                    valid_encoder_frames: f.valid_encoder_frames,
                    text: f.text.clone(),
                    encoder_output: part(f.valid_encoder_frames * 4096),
                    decode_trace: part(40),
                    integrity_passed: true,
                    trace_passed: true,
                    margin_exempt: false,
                    encoder_min_cosine: f.encoder.as_ref().map(|_| 1.0),
                    encoder_error: None,
                    trace_error: None,
                });
            }
        }
        let split_wer = speech::grade_wer(fixtures.iter().filter_map(|f| {
            Some((
                f.split.as_deref()?,
                f.ground_truth.as_deref()?,
                f.text.as_str(),
                f.text.as_str(),
            ))
        }))
        .unwrap();
        let roles = match backend {
            "metal" => vec!["ck-synapse"],
            "ane" => vec!["ck-synapse", "ck-synapse-worker-ane-direct"],
            "cuda" => vec!["ck-synapse", "ck-synapse-worker-cuda"],
            _ => vec!["ck-synapse", "ck-synapse-worker-vulkan"],
        };
        TranscribeRecord {
            schema: RECORD_SCHEMA,
            row_id: row.into(),
            source_commit: "b".repeat(40),
            machine: if matches!(backend, "metal" | "ane") {
                json!({"platform_uuid_sha256":"c".repeat(64)})
            } else {
                json!({"gpu":{"uuid_sha256":"c".repeat(64)}})
            },
            executed_artifacts: roles
                .into_iter()
                .map(|r| Artifact {
                    role: r.into(),
                    file: r.into(),
                    sha256: "d".repeat(64),
                })
                .collect(),
            model: speech::MODEL.into(),
            operation: "transcribe".into(),
            profile_id,
            fingerprint: cell.fingerprint.clone(),
            fixture_set_id: synapse_parity::evaluator::fixture_set_id(&manifest, speech::MODEL),
            precision_class: profile.precision_class,
            max_duration_ms: maximum,
            max_encoder_frames: speech::frame_counts(maximum * 16).2,
            executed_ladder: graded
                .iter()
                .filter(|g| g.id.starts_with("ladder-"))
                .map(|g| g.id.clone())
                .collect(),
            observe_rerun: ObserveRerun {
                fixture_ids: fixtures
                    .iter()
                    .filter(|f| f.duration_ms <= maximum && f.encoder.is_some())
                    .map(|f| f.id.clone())
                    .collect(),
                bit_identical: true,
            },
            graded_fixtures: graded,
            expected_refusals: refused,
            split_wer,
            self_check: SelfCheck {
                state: "passed".into(),
                fingerprints: vec![cell.fingerprint],
            },
            measurements: Measurements {
                self_check_wall_ms: 1999,
                longest_progress_interval_ms: 119999,
                ane_shapes: if backend == "ane" {
                    shapes
                        .into_iter()
                        .map(|r| Shape {
                            encoder_frames: u64::from(r),
                            executables: (0..24)
                                .map(|i| crate::Executable {
                                    id: format!("{r}-{i}"),
                                    layers: vec![i],
                                })
                                .collect(),
                            cpu_stages: vec![],
                        })
                        .collect()
                } else {
                    vec![]
                },
                peak_device_memory_bytes: Some(1024),
                peak_storage_buffer_bytes: Some(512),
            },
            status: "passed".into(),
        }
    }

    #[test]
    fn transcribe_cells_follow_every_backend_and_longest_executed_ladder() {
        let catalog = catalog();
        let fixtures = fixtures(&root()).unwrap();
        let cells: Vec<_> = crate::derive_cells(&catalog.lanes)
            .into_iter()
            .filter(|c| c.model == speech::MODEL)
            .collect();
        assert_eq!(cells.len(), 8);
        for cell in cells {
            let maximum = if cell.row_id == "ane-m5" {
                81850
            } else {
                600000
            };
            let record = record(&cell.row_id, maximum);
            validate_claims(&record, &fixtures, &catalog).unwrap();
            assert!(record.graded_fixtures.iter().any(|g| fixtures
                .iter()
                .any(|f| f.id == g.id && f.duration_ms == maximum)));
        }
    }

    #[test]
    fn transcribe_validator_rederives_gates_and_refuses_missing_measurements() {
        let catalog = catalog();
        let fixtures = fixtures(&root()).unwrap();
        let good = record("cuda-linux-nvidia", 600000);
        for change in [
            (|r: &mut TranscribeRecord| r.graded_fixtures[0].lane_id = "wrong".into())
                as fn(&mut TranscribeRecord),
            |r| r.graded_fixtures[0].fingerprint = "wrong".into(),
            |r| r.graded_fixtures[0].encoder_output.reassembled_bytes += 1,
            |r| r.graded_fixtures[0].decode_trace.reassembled_sha256 = "b".repeat(64),
            |r| r.graded_fixtures[0].trace_passed = false,
            |r| r.graded_fixtures[0].integrity_passed = false,
            |r| {
                r.graded_fixtures
                    .iter_mut()
                    .find(|g| g.encoder_min_cosine.is_some())
                    .unwrap()
                    .encoder_min_cosine = Some(0.998);
            },
            |r| {
                r.graded_fixtures.pop();
            },
            |r| {
                r.executed_ladder.pop();
            },
            |r| r.expected_refusals[0].worker_requests = 1,
            |r| r.expected_refusals[0].shape_admissions = 1,
            |r| r.expected_refusals[0].code = "invalid_params".into(),
            |r| r.expected_refusals[0].max_encoder_frames += 1,
            |r| r.max_duration_ms = 655290,
            |r| r.measurements.self_check_wall_ms = 2000,
            |r| r.measurements.longest_progress_interval_ms = 120000,
            |r| r.measurements.peak_device_memory_bytes = None,
            |r| r.self_check.fingerprints.push(r.fingerprint.clone()),
            |r| r.observe_rerun.bit_identical = false,
            |r| {
                r.observe_rerun.fixture_ids.pop();
            },
            |r| {
                r.split_wer.clear();
            },
            |r| r.status = "failed".into(),
        ] {
            let mut bad = good.clone();
            change(&mut bad);
            assert!(validate_claims(&bad, &fixtures, &catalog).is_err());
        }
        let value = serde_json::to_value(&good).unwrap();
        for field in [
            "self_check_wall_ms",
            "longest_progress_interval_ms",
            "ane_shapes",
            "peak_device_memory_bytes",
        ] {
            let mut bad = value.clone();
            bad["measurements"].as_object_mut().unwrap().remove(field);
            match serde_json::from_value::<TranscribeRecord>(bad) {
                Err(_) => {}
                Ok(bad) => assert!(validate_claims(&bad, &fixtures, &catalog).is_err()),
            }
        }
    }

    #[test]
    fn split_wer_read_back_from_json_matches_the_rederived_value() {
        // serde_json's default parser reads this digit string one ULP below
        // the value the runner wrote; real speech records contain such values.
        let text = "0.037022449783379284";
        let derived: f64 = text.parse().unwrap();
        let read_back: f64 = serde_json::from_str(text).unwrap();
        let map = |v: f64| BTreeMap::from([("split".to_string(), [v, v])]);
        assert!(same_split_wer(&map(read_back), &map(derived)));
        assert!(!same_split_wer(&map(derived + 1e-9), &map(derived)));
        let mut extra = map(derived);
        extra.insert("other".into(), [0.0, 0.0]);
        assert!(!same_split_wer(&extra, &map(derived)));
    }

    #[test]
    fn transcribe_fp16_precision_and_hardware_specific_gates_are_pinned() {
        let catalog = catalog();
        let fixtures = fixtures(&root()).unwrap();
        let mut ane = record("ane-m5", 81850);
        let g = ane
            .graded_fixtures
            .iter_mut()
            .find(|g| g.encoder_min_cosine.is_some())
            .unwrap();
        g.encoder_min_cosine = None;
        g.encoder_error = Some("cosine below 0.999".into());
        g.trace_passed = false;
        g.trace_error = Some("argmax differs".into());
        validate_claims(&ane, &fixtures, &catalog).unwrap();
        ane.precision_class = ToleranceClass::Fp32;
        assert!(validate_claims(&ane, &fixtures, &catalog).is_err());
        ane.precision_class = ToleranceClass::Fp16;
        ane.measurements.ane_shapes[0].executables.pop();
        assert!(validate_claims(&ane, &fixtures, &catalog).is_err());
        let mut vulkan = record("vulkan-linux-nvidia", 600000);
        vulkan.measurements.peak_device_memory_bytes = Some(u64::MAX);
        assert!(validate_claims(&vulkan, &fixtures, &catalog).is_err());
        vulkan.measurements.peak_device_memory_bytes = Some(1024);
        vulkan.measurements.peak_storage_buffer_bytes = Some(u64::MAX);
        assert!(validate_claims(&vulkan, &fixtures, &catalog).is_err());
    }

    #[test]
    fn transcribe_uses_the_unified_offline_matrix_and_artifact_checks() {
        let mut catalog = catalog();
        catalog
            .lanes
            .retain(|lane| lane.0 == speech::MODEL && lane.1 == "metal");
        catalog.tasks.retain(|model, _| model == speech::MODEL);
        let mut record = record("metal-m5", 600000);
        let assets = root().join(format!(
            "target/scratch/transcribe-record-assets-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&assets).unwrap();
        std::fs::write(assets.join("ck-synapse"), b"synthetic candidate bytes").unwrap();
        record.executed_artifacts[0].sha256 = crate::digest(&assets, "ck-synapse").unwrap();
        let records = [CertificationRecord::Transcribe(record.clone())];
        assert_eq!(
            crate::validate_records(&records, &root(), &assets, &catalog).unwrap(),
            [("metal-m5".into(), speech::MODEL.into())]
        );
        assert!(crate::validate_records(&[], &root(), &assets, &catalog)
            .unwrap_err()
            .to_string()
            .contains("missing_certification_cell"));
        std::fs::write(assets.join("ck-synapse"), b"changed candidate bytes").unwrap();
        assert!(
            crate::validate_records(&records, &root(), &assets, &catalog)
                .unwrap_err()
                .to_string()
                .contains("digest mismatch")
        );
        std::fs::remove_dir_all(assets).unwrap();
    }

    #[test]
    fn committed_legacy_records_roundtrip_without_changing_validation_policy() {
        // This is the schema-3 serializer before speech records were introduced.
        // Keep its field order frozen. serde_json's default float parser does not
        // preserve every decimal in the original files, so compatibility means
        // matching the old serializer, not silently enabling workspace-wide
        // float parsing features that could change production fingerprints.
        #[derive(Serialize, Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RecordBeforeSpeech {
            schema: u32,
            row_id: String,
            source_commit: String,
            machine: serde_json::Value,
            executed_artifacts: Vec<Artifact>,
            model: String,
            operation: String,
            profile_id: String,
            fingerprint: String,
            parity: crate::Parity,
            admission: crate::Admission,
            status: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            placement: Option<crate::Placement>,
        }
        fn walk(path: &Path, files: &mut Vec<PathBuf>) {
            for entry in std::fs::read_dir(path).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(&path, files);
                } else if path.extension().is_some_and(|e| e == "json") {
                    files.push(path);
                }
            }
        }
        let mut files = vec![];
        walk(&root().join("docs/evidence/certification"), &mut files);
        let mut counts = [0, 0];
        let catalog = catalog();
        for path in files {
            let bytes = std::fs::read(&path).unwrap();
            let record: CertificationRecord = serde_json::from_slice(&bytes).unwrap();
            let baseline: RecordBeforeSpeech = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                serde_json::to_vec_pretty(&record).unwrap(),
                serde_json::to_vec_pretty(&baseline).unwrap(),
                "{}",
                path.display()
            );
            let CertificationRecord::Legacy(record) = record else {
                panic!("unexpected speech record");
            };
            if record.schema == 2 {
                assert_eq!(
                    crate::validation::validate_record_claims(&record, &root(), &catalog)
                        .unwrap_err()
                        .to_string(),
                    "certification_refused: record_schema_mismatch"
                );
                counts[0] += 1;
            } else {
                crate::validation::validate_record_claims(&record, &root(), &catalog).unwrap();
                counts[1] += 1;
                // Historical attested binaries are not repository fixtures. The
                // full validator must still refuse rather than skip their hashes.
                assert!(crate::validate_record(
                    &record,
                    &root(),
                    &root().join("target/missing-historical-assets"),
                    &catalog
                )
                .is_err());
            }
        }
        assert_eq!(counts, [8, 10]);
    }
}
