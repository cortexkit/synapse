//! Replay AFT's exported document chunks through Bionic or an isolated Synapse candidate.
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    future::Future,
    path::PathBuf,
    process::Command,
    sync::Arc,
    time::{Duration, Instant},
};
use synapse_parity::canonical::sha256_file;
use tokio::task::JoinSet;

#[path = "support/qwen_compare.rs"]
mod qwen_compare;
use qwen_compare::Candidate;

#[path = "support/ane_lane_profile.rs"]
mod ane_lane_profile;

const ROWS: usize = 6341;
const BATCHES: usize = 127;
const SAMPLE_COUNT: usize = 64;
/// Output width of the model under test: Qwen3-Embedding-0.6B is 1024,
/// gte-modernbert-base is 768 (see `qwen_compare::model`).
fn dimension() -> usize {
    if qwen_compare::model().starts_with("gte-") {
        768
    } else {
        1024
    }
}
const IN_FLIGHT: usize = 2;

#[derive(Deserialize)]
struct Chunk {
    seq: usize,
    text: String,
}
#[derive(Clone, Debug, Deserialize)]
struct Batch {
    seq: usize,
    first_chunk_seq: usize,
    chunk_count: usize,
}
#[derive(Deserialize)]
struct Metadata {
    batches: Vec<Batch>,
}

fn validate_plan(batches: &[Batch], rows: usize) -> Result<()> {
    ensure!(rows > 0 && !batches.is_empty(), "empty workload");
    let mut covered = vec![false; rows];
    let mut batch_ids = BTreeSet::new();
    for batch in batches {
        ensure!(batch_ids.insert(batch.seq), "duplicate batch seq");
        ensure!(
            (1..=64).contains(&batch.chunk_count),
            "batch count outside 1..=64"
        );
        let end = batch
            .first_chunk_seq
            .checked_add(batch.chunk_count)
            .context("batch range overflow")?;
        ensure!(end <= rows, "batch range out of bounds");
        for slot in &mut covered[batch.first_chunk_seq..end] {
            ensure!(!*slot, "overlapping batch ranges");
            *slot = true;
        }
    }
    ensure!(covered.iter().all(|slot| *slot), "gap in batch coverage");
    Ok(())
}

struct Workload {
    chunks: Vec<Chunk>,
    batches: Vec<Batch>,
    input_sha256: String,
    meta_sha256: String,
}
impl Workload {
    fn load(path: &std::path::Path) -> Result<Self> {
        let mut meta_path = path.as_os_str().to_os_string();
        meta_path.push(".meta.json");
        let meta_path = PathBuf::from(meta_path);
        let input =
            std::fs::read_to_string(path).context("head-to-head input missing or unreadable")?;
        let chunks = input
            .lines()
            .enumerate()
            .map(|(index, line)| {
                let chunk: Chunk = serde_json::from_str(line)
                    .with_context(|| format!("invalid input row {index}"))?;
                ensure!(
                    chunk.seq == index,
                    "input seq must be exactly 0..N-1 in file order"
                );
                Ok(chunk)
            })
            .collect::<Result<Vec<_>>>()?;
        let meta: Metadata = serde_json::from_slice(
            &std::fs::read(&meta_path).context("batch metadata missing or unreadable")?,
        )?;
        validate_plan(&meta.batches, chunks.len())?;
        ensure!(
            chunks.len() == ROWS,
            "expected {ROWS} chunks, got {}",
            chunks.len()
        );
        ensure!(meta.batches.len() == BATCHES, "expected {BATCHES} batches");
        Ok(Self {
            chunks,
            batches: meta.batches,
            input_sha256: sha256_file(path)?,
            meta_sha256: sha256_file(&meta_path)?,
        })
    }
}

struct Completion<T> {
    index: usize,
    seconds: f64,
    value: T,
}
struct Replay<T> {
    wall_seconds: f64,
    completed: Vec<Completion<T>>,
}

async fn replay<T, F, Fut>(count: usize, mut start: F) -> Result<Replay<T>>
where
    T: Send + 'static,
    F: FnMut(usize) -> Fut,
    Fut: Future<Output = T> + Send + 'static,
{
    let wall = Instant::now();
    let mut active = JoinSet::new();
    let mut next = 0;
    let mut completed = Vec::with_capacity(count);
    loop {
        while next < count && active.len() < IN_FLIGHT {
            let index = next;
            let future = start(index);
            active.spawn(async move {
                let began = Instant::now();
                let value = future.await;
                Completion {
                    index,
                    seconds: began.elapsed().as_secs_f64(),
                    value,
                }
            });
            next += 1;
        }
        match active.join_next().await {
            Some(result) => completed.push(result.context("provider task panicked")?),
            None => break,
        }
    }
    ensure!(completed.len() == count, "incomplete scheduler accounting");
    Ok(Replay {
        wall_seconds: wall.elapsed().as_secs_f64(),
        completed,
    })
}

fn percentile(samples: &[f64], fraction: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((fraction * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1)]
}
fn median(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}
#[derive(Default, Serialize, Deserialize)]
struct Timing {
    wall_seconds: f64,
    rows_per_second: f64,
    call_p50_ms: f64,
    call_p90_ms: f64,
    call_max_ms: f64,
}
fn timing(rows: usize, wall_seconds: f64, latencies: &[f64]) -> Timing {
    Timing {
        wall_seconds,
        rows_per_second: rows as f64 / wall_seconds,
        call_p50_ms: percentile(latencies, 0.5) * 1000.0,
        call_p90_ms: percentile(latencies, 0.9) * 1000.0,
        call_max_ms: latencies.iter().copied().fold(0.0, f64::max) * 1000.0,
    }
}
fn sample_seqs(rows: usize) -> Vec<usize> {
    (0..SAMPLE_COUNT)
        .map(|i| i * (rows - 1) / (SAMPLE_COUNT - 1))
        .collect()
}
fn cosine(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum();
    let norm_b: f64 = b.iter().map(|x| x * x).sum();
    let value = dot / (norm_a * norm_b).sqrt();
    value.is_finite().then_some(value.clamp(-1.0, 1.0))
}

#[derive(Debug)]
struct Failure {
    code: String,
    refusal: bool,
}
impl Failure {
    fn error(code: &str) -> Self {
        Self {
            code: code.into(),
            refusal: false,
        }
    }
    fn refusal(code: String) -> Self {
        Self {
            code,
            refusal: true,
        }
    }
}
fn wire_error(response: &Value) -> Option<Failure> {
    let error = response.get("error").filter(|e| !e.is_null()).or_else(|| {
        response
            .get("result")?
            .get("error")
            .filter(|e| !e.is_null())
    })?;
    let code = error
        .get("code")
        .map(|code| {
            code.as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| code.to_string())
        })
        .unwrap_or_else(|| "unknown_refusal".into());
    Some(Failure::refusal(code))
}
fn vector(value: &Value) -> std::result::Result<Vec<f64>, Failure> {
    value
        .as_array()
        .ok_or_else(|| Failure::error("invalid_vector"))?
        .iter()
        .map(|v| {
            v.as_f64()
                .filter(|x| x.is_finite())
                .ok_or_else(|| Failure::error("nonfinite_vector"))
        })
        .collect()
}

// Reorder by the provider's returned IDs, not by an assumed wire order.
fn decode_vectors(
    response: &Value,
    seqs: &[usize],
    bionic: bool,
) -> std::result::Result<Vec<Vec<f64>>, Failure> {
    if let Some(error) = wire_error(response) {
        return Err(error);
    }
    let entries = if bionic {
        response["data"].as_array()
    } else {
        response["result"]["vectors"].as_array()
    }
    .ok_or_else(|| Failure::error("inline_vectors_missing"))?;
    if entries.len() != seqs.len() {
        return Err(Failure::error("row_count_mismatch"));
    }
    let mut ordered = vec![None; seqs.len()];
    for entry in entries {
        let index = if bionic {
            entry["index"]
                .as_u64()
                .and_then(|i| usize::try_from(i).ok())
        } else {
            entry["id"]
                .as_str()
                .and_then(|id| seqs.iter().position(|seq| id == format!("row-{seq}")))
        }
        .filter(|index| *index < seqs.len())
        .ok_or_else(|| Failure::error("invalid_row_id"))?;
        if ordered[index].is_some() {
            return Err(Failure::error("duplicate_row_id"));
        }
        let values = vector(&entry[if bionic { "embedding" } else { "vector" }])?;
        if values.len() != dimension() {
            return Err(Failure::error("dimension_mismatch"));
        }
        if !bionic && (values.iter().map(|v| v * v).sum::<f64>().sqrt() - 1.0).abs() > 1e-3 {
            return Err(Failure::error("not_l2_normalized"));
        }
        ordered[index] = Some(values);
    }
    ordered
        .into_iter()
        .map(|v| v.ok_or_else(|| Failure::error("missing_row_id")))
        .collect()
}

enum Provider {
    Bionic {
        client: reqwest::Client,
        url: String,
    },
    Synapse {
        candidate: Arc<Candidate>,
        model: String,
    },
}
impl Provider {
    async fn embed(
        &self,
        workload: &Workload,
        index: usize,
        warmup: bool,
    ) -> std::result::Result<Vec<Vec<f64>>, Failure> {
        let batch = &workload.batches[index];
        let chunks =
            &workload.chunks[batch.first_chunk_seq..batch.first_chunk_seq + batch.chunk_count];
        let seqs = chunks.iter().map(|chunk| chunk.seq).collect::<Vec<_>>();
        let (response, bionic) = match self {
            Self::Bionic { client, url } => {
                let response = client.post(url).json(&json!({"model":"text-embedding-qwen3-embedding-0.6b", "input":chunks.iter().map(|chunk| &chunk.text).collect::<Vec<_>>() }))
                    .send().await.map_err(|_| Failure::error("http_transport"))?;
                let status = response.status();
                let body: Value = match response.json().await {
                    Ok(body) => body,
                    Err(_) if !status.is_success() => {
                        return Err(Failure {
                            code: format!("http_{}", status.as_u16()),
                            refusal: status.is_client_error(),
                        })
                    }
                    Err(_) => return Err(Failure::error("http_invalid_json")),
                };
                if let Some(error) = wire_error(&body) {
                    return Err(error);
                }
                if !status.is_success() {
                    return Err(Failure {
                        code: format!("http_{}", status.as_u16()),
                        refusal: status.is_client_error(),
                    });
                }
                (body, true)
            }
            Self::Synapse { candidate, model } => {
                let params = json!({"model":model,"input_type":"document","accept_declared":true,
                    "request_key":format!("headtohead-{}-{model}-{}-{}",std::process::id(),batch.seq,if warmup {"warmup"} else {"timed"}),
                    "items":chunks.iter().map(|chunk| json!({"id":format!("row-{}",chunk.seq),"text":chunk.text})).collect::<Vec<_>>()});
                let body = qwen_compare::raw_call(
                    &candidate.consumer,
                    &candidate.identity,
                    "embed.batch",
                    params,
                )
                .await
                .map_err(|error| {
                    match error.downcast_ref::<subc_client_rs::CallError>() {
                        Some(subc_client_rs::CallError::Module(body)) => {
                            Failure::refusal(body.code.to_string())
                        }
                        _ => Failure::error("subc_transport"),
                    }
                })?;
                (body, false)
            }
        };
        decode_vectors(&response, &seqs, bionic)
    }
}

fn load_one_minute() -> Result<f64> {
    let output = Command::new("sysctl").args(["-n", "vm.loadavg"]).output()?;
    ensure!(output.status.success(), "sysctl vm.loadavg failed");
    let load = String::from_utf8(output.stdout)?
        .split_whitespace()
        .find_map(|s| s.parse::<f64>().ok())
        .context("one-minute load missing")?;
    ensure!(load.is_finite() && load >= 0.0, "invalid one-minute load");
    // Shared-machine load is evidence, not an admission gate for this replay.
    Ok(load)
}
#[derive(Serialize, Deserialize)]
struct LoadSample {
    elapsed_seconds: f64,
    one_minute: f64,
}
#[derive(Serialize, Deserialize)]
struct CallRecord {
    batch_seq: usize,
    latency_ms: f64,
    rows_returned: usize,
    error_code: Option<String>,
}
#[derive(Serialize, Deserialize)]
struct ArmResult {
    passed: bool,
    #[serde(flatten)]
    timing: Timing,
    rows_returned: usize,
    vector_dimension: Option<usize>,
    vectors_normalized: bool,
    non_normalized_rows: usize,
    errors_by_code: BTreeMap<String, usize>,
    refusals_by_code: BTreeMap<String, usize>,
    warmup_succeeded: bool,
    load_start_1m: f64,
    load_end_1m: f64,
    load_samples: Vec<LoadSample>,
    calls: Vec<CallRecord>,
    sample_vectors: BTreeMap<usize, Vec<f64>>,
}
fn count_failure(result: &mut ArmResult, error: &Failure) {
    let counts = if error.refusal {
        &mut result.refusals_by_code
    } else {
        &mut result.errors_by_code
    };
    *counts.entry(error.code.clone()).or_default() += 1;
}
async fn run_arm(provider: Arc<Provider>, workload: Arc<Workload>) -> Result<ArmResult> {
    let load_start_1m = load_one_minute()?;
    let began = Instant::now();
    let (stop, mut stopped) = tokio::sync::oneshot::channel::<()>();
    let monitor = tokio::spawn(async move {
        let mut samples = Vec::new();
        loop {
            tokio::select! {
                _ = &mut stopped => break,
                _ = tokio::time::sleep(Duration::from_secs(30)) => samples.push(LoadSample {
                    elapsed_seconds: began.elapsed().as_secs_f64(), one_minute: load_one_minute()?,
                }),
            }
        }
        Ok::<_, anyhow::Error>(samples)
    });
    let mut result = ArmResult {
        passed: false,
        timing: Timing::default(),
        rows_returned: 0,
        vector_dimension: None,
        vectors_normalized: false,
        non_normalized_rows: 0,
        errors_by_code: BTreeMap::new(),
        refusals_by_code: BTreeMap::new(),
        warmup_succeeded: false,
        load_start_1m,
        load_end_1m: load_start_1m,
        load_samples: Vec::new(),
        calls: Vec::new(),
        sample_vectors: BTreeMap::new(),
    };
    match provider.embed(&workload, 0, true).await {
        Ok(_) => result.warmup_succeeded = true,
        Err(error) => count_failure(&mut result, &error),
    }
    let scheduled = if result.warmup_succeeded {
        Some(
            replay(workload.batches.len(), |index| {
                let provider = provider.clone();
                let workload = workload.clone();
                async move { provider.embed(&workload, index, false).await }
            })
            .await,
        )
    } else {
        None
    };
    let _ = stop.send(());
    result.load_samples = monitor.await??;
    result.load_end_1m = load_one_minute()?;
    if let Some(scheduled) = scheduled {
        let scheduled = scheduled?;
        let latencies = scheduled
            .completed
            .iter()
            .map(|call| call.seconds)
            .collect::<Vec<_>>();
        let samples = sample_seqs(workload.chunks.len())
            .into_iter()
            .collect::<BTreeSet<_>>();
        for call in scheduled.completed {
            let batch = &workload.batches[call.index];
            let mut record = CallRecord {
                batch_seq: batch.seq,
                latency_ms: call.seconds * 1000.0,
                rows_returned: 0,
                error_code: None,
            };
            match call.value {
                Ok(vectors) => {
                    record.rows_returned = vectors.len();
                    result.rows_returned += vectors.len();
                    result.vector_dimension = Some(dimension());
                    for (offset, vector) in vectors.into_iter().enumerate() {
                        let norm = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
                        if (norm - 1.0).abs() > 1e-3 {
                            result.non_normalized_rows += 1;
                        }
                        let seq = batch.first_chunk_seq + offset;
                        if samples.contains(&seq) {
                            result.sample_vectors.insert(seq, vector);
                        }
                    }
                }
                Err(error) => {
                    record.error_code = Some(error.code.clone());
                    count_failure(&mut result, &error);
                }
            }
            result.calls.push(record);
        }
        result.timing = timing(result.rows_returned, scheduled.wall_seconds, &latencies);
        if result.rows_returned != ROWS {
            count_failure(&mut result, &Failure::error("total_row_count_mismatch"));
        }
    }
    result.vectors_normalized = result.rows_returned > 0 && result.non_normalized_rows == 0;
    result.passed = result.warmup_succeeded
        && result.errors_by_code.is_empty()
        && result.refusals_by_code.is_empty()
        && result.rows_returned == ROWS;
    Ok(result)
}

#[derive(Serialize, Deserialize)]
struct CosineSummary {
    samples: usize,
    median: f64,
    min: f64,
}
#[derive(Serialize, Deserialize)]
struct Report {
    schema_version: u32,
    input_sha256: String,
    meta_sha256: String,
    rows: usize,
    batches: usize,
    calls_in_flight: usize,
    sample_seqs: Vec<usize>,
    arms: BTreeMap<String, ArmResult>,
    cosine_sanity: BTreeMap<String, CosineSummary>,
    ane_wall_over_bionic_wall: Option<f64>,
    ane_passes_2x_bionic: Option<bool>,
}
fn wall_ratio(ane: f64, bionic: f64) -> (f64, bool) {
    let ratio = ane / bionic;
    (ratio, ratio <= 2.0)
}
impl Report {
    fn load_or_new(path: &std::path::Path, workload: &Workload) -> Result<Self> {
        if path.exists() {
            let report: Self = serde_json::from_slice(&std::fs::read(path)?)?;
            ensure!(
                report.schema_version == 1
                    && report.input_sha256 == workload.input_sha256
                    && report.meta_sha256 == workload.meta_sha256,
                "output belongs to another workload or schema; use a fresh output file"
            );
            return Ok(report);
        }
        Ok(Self {
            schema_version: 1,
            input_sha256: workload.input_sha256.clone(),
            meta_sha256: workload.meta_sha256.clone(),
            rows: ROWS,
            batches: BATCHES,
            calls_in_flight: IN_FLIGHT,
            sample_seqs: sample_seqs(ROWS),
            arms: BTreeMap::new(),
            cosine_sanity: BTreeMap::new(),
            ane_wall_over_bionic_wall: None,
            ane_passes_2x_bionic: None,
        })
    }
    fn summarize(&mut self) {
        self.cosine_sanity.clear();
        self.ane_wall_over_bionic_wall = None;
        self.ane_passes_2x_bionic = None;
        if let (Some(ane), Some(bionic)) = (
            self.arms.get("ane").filter(|a| a.passed),
            self.arms.get("bionic").filter(|a| a.passed),
        ) {
            let (ratio, passes) = wall_ratio(ane.timing.wall_seconds, bionic.timing.wall_seconds);
            self.ane_wall_over_bionic_wall = Some(ratio);
            self.ane_passes_2x_bionic = Some(passes);
        }
        for other in ["metal", "bionic"] {
            if let (Some(ane), Some(reference)) = (self.arms.get("ane"), self.arms.get(other)) {
                let cosines = self
                    .sample_seqs
                    .iter()
                    .filter_map(|seq| {
                        cosine(
                            ane.sample_vectors.get(seq)?,
                            reference.sample_vectors.get(seq)?,
                        )
                    })
                    .collect::<Vec<_>>();
                if cosines.len() == SAMPLE_COUNT {
                    self.cosine_sanity.insert(
                        format!("ane_vs_{other}"),
                        CosineSummary {
                            samples: cosines.len(),
                            median: median(&cosines),
                            min: cosines.iter().copied().fold(f64::INFINITY, f64::min),
                        },
                    );
                }
            }
        }
    }
    fn save(&self, path: &std::path::Path) -> Result<()> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut temporary = path.as_os_str().to_os_string();
        temporary.push(format!(".{}.tmp", std::process::id()));
        let temporary = PathBuf::from(temporary);
        std::fs::write(&temporary, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(temporary, path)?;
        Ok(())
    }
    fn print(&self) {
        for (arm, result) in &self.arms {
            println!("{arm}: {:.2}s, {:.2} rows/s, {} rows, dim {:?}; p50/p90/max {:.1}/{:.1}/{:.1}ms; load {:.2} -> {:.2}; normalized={}; errors={:?}, refusals={:?}; passed={}",
                result.timing.wall_seconds, result.timing.rows_per_second, result.rows_returned, result.vector_dimension,
                result.timing.call_p50_ms, result.timing.call_p90_ms, result.timing.call_max_ms,
                result.load_start_1m, result.load_end_1m, result.vectors_normalized, result.errors_by_code, result.refusals_by_code, result.passed);
        }
        if let Some(ratio) = self.ane_wall_over_bionic_wall {
            println!(
                "ane_wall / bionic_wall = {ratio:.3}; ANE <= 2x Bionic: {}",
                self.ane_passes_2x_bionic.unwrap_or(false)
            );
        } else {
            println!("ane_wall / bionic_wall = unavailable (need successful ane and bionic arms)");
        }
        for (pair, summary) in &self.cosine_sanity {
            println!(
                "{pair}: {} fixed rows, cosine median {:.6}, min {:.6} (sanity only)",
                summary.samples, summary.median, summary.min
            );
        }
    }
}

fn required_path(name: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(
        env::var_os(name).with_context(|| format!("{name} is required"))?,
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    ensure!(cfg!(target_os = "macos"), "load recording requires macOS");
    let workload = Arc::new(Workload::load(&required_path("SYNAPSE_HEADTOHEAD_INPUT")?)?);
    let out = required_path("SYNAPSE_HEADTOHEAD_OUT")?;
    if env::var_os("SYNAPSE_ANE_PROFILE_CATALOG").is_some() {
        return ane_lane_profile::run(&workload, &out).await;
    }
    let arms = env::var("SYNAPSE_HEADTOHEAD_ARMS").unwrap_or_else(|_| "bionic,ane,metal".into());
    let arms = arms.split(',').map(str::trim).collect::<Vec<_>>();
    let mut seen = BTreeSet::new();
    ensure!(
        arms.iter()
            .all(|arm| ["bionic", "ane", "metal"].contains(arm) && seen.insert(*arm)),
        "select distinct bionic,ane,metal arms with SYNAPSE_HEADTOHEAD_ARMS"
    );
    let mut report = Report::load_or_new(&out, &workload)?;
    let mut candidate = None;
    let mut failed = false;
    for arm in arms {
        let provider = if arm == "bionic" {
            println!("!!! PAUSE AFT'S OWN EMBEDDING FILLS FOR THE BIONIC ARM !!! This driver does not verify that they are paused.");
            Provider::Bionic {
                client: reqwest::Client::builder()
                    .timeout(Duration::from_secs(600))
                    .build()?,
                url: env::var("SYNAPSE_HEADTOHEAD_BIONIC_URL")
                    .unwrap_or_else(|_| "http://localhost:1234/v1/embeddings".into()),
            }
        } else {
            ensure!(
                env::var_os("TMPDIR").is_none(),
                "TMPDIR must be unset for Synapse arms"
            );
            ensure!(
                env::var("DEVELOPER_DIR").as_deref()
                    == Ok("/Applications/Xcode.app/Contents/Developer"),
                "Xcode DEVELOPER_DIR required"
            );
            if candidate.is_none() {
                let assets = required_path("SYNAPSE_COMPARE_ASSETS")?.canonicalize()?;
                let weights = required_path("SYNAPSE_QWEN_WEIGHTS")?.canonicalize()?;
                // The aggregate inline budget covers 64 full-context rows; per-row model limits still apply.
                candidate = Some(Arc::new(
                    Candidate::start(
                        &env::current_dir()?,
                        &assets,
                        &weights,
                        "aft-headtohead",
                        // Preload ids must not collide with catalog lane ids,
                        // which the module reserves for models.download.
                        ["headtohead-ane", "headtohead-metal"],
                        64 * 8192,
                    )
                    .await?,
                ));
            }
            Provider::Synapse {
                candidate: candidate.as_ref().unwrap().clone(),
                model: format!("headtohead-{arm}"),
            }
        };
        let result = run_arm(Arc::new(provider), workload.clone()).await?;
        failed |= !result.passed;
        report.arms.insert(arm.into(), result);
        report.summarize();
        report.save(&out)?;
    }
    if let Some(candidate) = candidate {
        Arc::try_unwrap(candidate)
            .map_err(|_| anyhow::anyhow!("private candidate still in use"))?
            .shutdown()
            .await?;
    }
    report.print();
    ensure!(
        !failed,
        "head-to-head run failed; see errors/refusals in JSON report"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn batch(seq: usize, first_chunk_seq: usize, chunk_count: usize) -> Batch {
        Batch {
            seq,
            first_chunk_seq,
            chunk_count,
        }
    }

    #[test]
    fn batch_plan_rejects_gaps_overlaps_and_out_of_range() {
        // Meta order, rather than sorted ranges, is the replay order.
        assert!(validate_plan(&[batch(1, 2, 2), batch(0, 0, 2)], 4).is_ok());
        for (plan, rows) in [
            (vec![batch(0, 0, 1), batch(1, 2, 2)], 4),
            (vec![batch(0, 0, 3), batch(1, 2, 2)], 4),
            (vec![batch(0, 0, 5)], 4),
            (vec![batch(0, 0, 65)], 65),
            (vec![batch(0, 0, 0), batch(1, 0, 4)], 4),
            (vec![batch(0, 0, 2), batch(0, 2, 2)], 4),
            (vec![batch(0, usize::MAX, 2)], 4),
            (vec![], 4),
        ] {
            assert!(
                validate_plan(&plan, rows).is_err(),
                "accepted invalid plan: {plan:?}"
            );
        }
        assert!(validate_plan(&[batch(0, 0, 64)], 64).is_ok());
    }

    #[tokio::test]
    async fn scheduler_limits_two_and_accounts_for_every_completion() {
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(std::sync::Mutex::new(Vec::new()));
        let scheduled = replay(9, |index| {
            let active = active.clone();
            let peak = peak.clone();
            let started = started.clone();
            async move {
                started.lock().unwrap().push(index);
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(if index == 0 { 60 } else { 5 })).await;
                active.fetch_sub(1, Ordering::SeqCst);
                if index == 3 {
                    Err(index)
                } else {
                    Ok(index)
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert_eq!(active.load(Ordering::SeqCst), 0);
        assert_eq!(*started.lock().unwrap(), (0..9).collect::<Vec<_>>());
        assert_ne!(
            scheduled.completed[0].index, 0,
            "a slow first call must not block replenishment"
        );
        assert_eq!(scheduled.completed.len(), 9);
        let mut indices = scheduled
            .completed
            .iter()
            .map(|call| call.index)
            .collect::<Vec<_>>();
        indices.sort_unstable();
        assert_eq!(indices, (0..9).collect::<Vec<_>>());
        for call in scheduled.completed {
            assert_eq!(
                call.value,
                if call.index == 3 {
                    Err(3)
                } else {
                    Ok(call.index)
                }
            );
        }
        let empty = replay(0, |_| async { 0 }).await.unwrap();
        assert!(empty.completed.is_empty());
        let single = replay(1, |_| async { 42 }).await.unwrap();
        assert_eq!(single.completed[0].value, 42);
    }

    #[test]
    fn percentiles_use_nearest_rank_and_median_uses_middle_pair() {
        let samples = [10.0, 1.0, 9.0, 2.0];
        assert_eq!(percentile(&samples, 0.5), 2.0);
        assert_eq!(percentile(&samples, 0.9), 10.0);
        assert_eq!(percentile(&[7.0], 0.5), 7.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
        assert_eq!(median(&samples), 5.5);
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn summary_math_uses_wall_time_and_inclusive_two_x_bar() {
        let measured = timing(100, 4.0, &[0.010, 0.001, 0.009, 0.002]);
        assert_eq!(measured.rows_per_second, 25.0);
        assert_eq!(measured.wall_seconds, 4.0);
        assert_eq!(measured.call_max_ms, 10.0);
        assert_eq!(wall_ratio(8.0, 4.0), (2.0, true));
        assert_eq!(wall_ratio(9.0, 4.0), (2.25, false));
        assert_eq!(wall_ratio(1.0, 4.0), (0.25, true));
        assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 1.0]), Some(0.0));
        assert_eq!(cosine(&[0.0], &[1.0]), None);
        let seqs = sample_seqs(ROWS);
        assert_eq!(seqs.len(), 64);
        assert_eq!(seqs[0], 0);
        assert_eq!(seqs[63], 6340);
        assert_eq!(seqs.iter().collect::<BTreeSet<_>>().len(), 64);
    }

    fn unit_vector(scale: f64) -> Vec<f64> {
        let mut values = vec![0.0; dimension()];
        values[0] = scale;
        values
    }

    #[test]
    fn wire_validation_orders_ids_and_enforces_synapse_norm_only() {
        let response = json!({"result":{"vectors":[{"id":"row-8","vector":unit_vector(-1.0)}, {"id":"row-7","vector":unit_vector(1.0)}]}});
        let vectors = decode_vectors(&response, &[7, 8], false).unwrap();
        assert_eq!(vectors[0][0], 1.0);
        assert_eq!(vectors[1][0], -1.0);
        let wrong_norm = json!({"result":{"vectors":[{"id":"row-7","vector":unit_vector(2.0)}]}});
        assert_eq!(
            decode_vectors(&wrong_norm, &[7], false).unwrap_err().code,
            "not_l2_normalized"
        );
        let bionic = json!({"data":[{"index":0,"embedding":unit_vector(2.0)}]});
        assert_eq!(decode_vectors(&bionic, &[7], true).unwrap()[0][0], 2.0);
        let duplicates = json!({"data":[{"index":0,"embedding":unit_vector(1.0)},{"index":0,"embedding":unit_vector(1.0)}]});
        assert_eq!(
            decode_vectors(&duplicates, &[7, 8], true).unwrap_err().code,
            "duplicate_row_id"
        );
        let refused = json!({"result":{"error":{"code":"resource_busy"}}});
        let failure = decode_vectors(&refused, &[7], false).unwrap_err();
        assert_eq!(failure.code, "resource_busy");
        assert!(failure.refusal);
    }
}
