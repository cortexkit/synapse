use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use synapse_bench::parity::{load_corpus, load_reference, mean_parity, rank_overlap};
use synapse_bench::results::LaneResult;
use synapse_bench::rig_protocol::{
    read_json_frame, write_json_frame, BatchShape, CandidateMetadata, CandidateRequest,
    CandidateResponse, RerankPair, ShapePolicy, Workload, PROTOCOL_VERSION,
};
use tokenizers::{EncodeInput, Tokenizer, TruncationParams};

const BUCKET_POLICY_VERSION: u32 = 1;
const BUCKET_MAX_BATCH_ROWS: usize = 8;
const BUCKET_SEQUENCE_LADDER: &[usize] = &[64, 96, 128, 160, 192, 256, 320, 384, 448, 512];
const MAX_LARGE_CORPUS_RANK_QUERIES: usize = 100;

#[derive(Parser)]
#[command(name = "synapse-rig")]
struct Args {
    /// Candidate executable implementing the --serve-stdio contract.
    #[arg(long)]
    candidate: PathBuf,
    /// Path to a supported model snapshot or safetensors file.
    #[arg(long)]
    model: PathBuf,
    /// Path to tokenizer.json used for canonical token accounting.
    #[arg(long)]
    tokenizer: PathBuf,
    /// Corpus JSONL ({id, path, text, tokens} per line).
    #[arg(long)]
    corpus: Option<PathBuf>,
    /// Rerank request JSONL ({id, query, documents} per line).
    #[arg(long, conflicts_with = "corpus")]
    rerank_requests: Option<PathBuf>,
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long = "vectors-out", alias = "emit-vectors")]
    vectors_out: Option<PathBuf>,
    #[arg(long)]
    scores_out: Option<PathBuf>,
    #[arg(long)]
    reference: Option<PathBuf>,
    #[arg(long, default_value_t = 0.9999)]
    min_parity: f64,
    #[arg(long, default_value_t = 0.995)]
    min_rank_overlap: f64,
    #[arg(long, default_value_t = 0.999)]
    min_pearson: f64,
    #[arg(long, default_value_t = 0.98)]
    min_top1_agreement: f64,
    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    device: DeviceArg,
    #[arg(long, value_enum, default_value_t = Precision::F32)]
    dtype: Precision,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cuda_graphs: bool,
    #[arg(long, value_enum, default_value_t = Execution::Explicit)]
    execution: Execution,
    #[arg(long)]
    package_cache: Option<PathBuf>,
    #[arg(long, default_value_t = 512)]
    max_length: usize,
    #[arg(long, default_value_t = 4_000_000)]
    attention_units: usize,
    #[arg(long, value_enum, default_value_t = ShapeArg::Bucketed)]
    shapes: ShapeArg,
    #[arg(long, default_value_t = 1)]
    passes: usize,
    #[arg(long)]
    model_label: Option<String>,
}

#[derive(Clone, Copy, ValueEnum)]
enum DeviceArg {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Clone, Copy, ValueEnum)]
enum Precision {
    F32,
    F16,
}

#[derive(Clone, Copy, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum Execution {
    Explicit,
    Lazy,
}

#[derive(Clone, Copy, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum ShapeArg {
    Exact,
    Bucketed,
}

impl DeviceArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

impl Precision {
    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

impl Execution {
    fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Lazy => "lazy",
        }
    }
}

impl ShapeArg {
    fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Bucketed => "bucketed",
        }
    }

    fn protocol(self) -> ShapePolicy {
        match self {
            Self::Exact => ShapePolicy::Exact,
            Self::Bucketed => ShapePolicy::Bucketed,
        }
    }
}

#[derive(Clone)]
struct PlannedBatch {
    range: std::ops::Range<usize>,
    shape: BatchShape,
}

#[derive(Serialize)]
struct PassResult {
    pass: usize,
    label: &'static str,
    infer_wall_s: f64,
    input_tokens: u64,
    padded_tokens: u64,
    padding_waste_fraction: f64,
    tok_per_s: f64,
    items: u64,
    parity_mean_cosine: Option<f64>,
    top10_rank_overlap: Option<f64>,
}

#[derive(Default, Serialize)]
struct PackageCacheStats {
    package_count: usize,
    package_bytes: u64,
}

#[derive(Serialize)]
struct ServingResult {
    #[serde(flatten)]
    lane: LaneResult,
    shape_policy: ShapeArg,
    bucket_policy_version: Option<u32>,
    bucket_shapes: Vec<BatchShape>,
    real_tokens: u64,
    padded_tokens: u64,
    padding_waste_fraction: f64,
    package_cache: PackageCacheStats,
    passes: Vec<PassResult>,
    rig_metadata: RigMetadata,
}

#[derive(Clone, Deserialize)]
struct RerankRequest {
    id: String,
    query: String,
    documents: Vec<String>,
}

#[derive(Clone, Deserialize, Serialize)]
struct RerankScores {
    id: String,
    scores: Vec<f32>,
}

#[derive(Serialize)]
struct RerankServingResult {
    lane: String,
    workload: &'static str,
    model: String,
    provider: String,
    dtype: String,
    execution: Execution,
    shape_policy: ShapeArg,
    bucket_policy_version: Option<u32>,
    bucket_shapes: Vec<BatchShape>,
    requests: usize,
    pairs: usize,
    real_tokens: u64,
    padded_tokens: u64,
    padding_waste_fraction: f64,
    infer_wall_s: f64,
    pairs_per_s: f64,
    request_latency_p50_ms: f64,
    request_latency_p95_ms: f64,
    pearson: Option<f64>,
    tie_aware_top1_agreement: Option<f64>,
    package_cache: PackageCacheStats,
    notes: String,
    rig_metadata: RigMetadata,
}

#[derive(Serialize)]
struct RigMetadata {
    sha256: String,
    git_revision: &'static str,
    protocol_version: u32,
    candidate: CandidateMetadata,
    candidate_internal_prepare_wall_s: f64,
    candidate_internal_pass_wall_s: Vec<f64>,
    token_reconciliation: Vec<TokenReconciliation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    host_probe: Option<HostProbe>,
}

#[derive(Serialize)]
struct TokenReconciliation {
    pass: usize,
    label: &'static str,
    canonical_real_tokens: u64,
    candidate_reported_real_tokens: u64,
    divergence_fraction: f64,
}

#[derive(Serialize)]
struct HostProbe {
    tool: &'static str,
    output: String,
}

struct CandidateProcess {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl CandidateProcess {
    fn spawn(args: &Args) -> Result<(Self, CandidateMetadata, Instant)> {
        let started = Instant::now();
        let mut command = Command::new(&args.candidate);
        command
            .arg("--serve-stdio")
            .arg("--model")
            .arg(&args.model)
            .arg("--tokenizer")
            .arg(&args.tokenizer)
            .arg("--device")
            .arg(args.device.as_str())
            .arg("--dtype")
            .arg(args.dtype.as_str())
            .arg("--cuda-graphs")
            .arg(args.cuda_graphs.to_string())
            .arg("--execution")
            .arg(args.execution.as_str())
            .arg("--max-length")
            .arg(args.max_length.to_string())
            .arg("--attention-units")
            .arg(args.attention_units.to_string())
            .arg("--shapes")
            .arg(args.shapes.as_str())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        if let Some(path) = &args.package_cache {
            command.arg("--package-cache").arg(path);
        }
        if let Some(label) = &args.model_label {
            command.arg("--model-label").arg(label);
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("spawn candidate {}", args.candidate.display()))?;
        let input = BufWriter::new(child.stdin.take().context("candidate stdin unavailable")?);
        let mut output = BufReader::new(
            child
                .stdout
                .take()
                .context("candidate stdout unavailable")?,
        );
        let response: CandidateResponse = read_json_frame(&mut output)
            .context("read candidate ready frame; stdout must contain protocol frames only")?;
        let metadata = match response {
            CandidateResponse::Ready {
                protocol_version,
                metadata,
            } => {
                ensure!(
                    protocol_version == PROTOCOL_VERSION,
                    "candidate protocol version {protocol_version} does not match rig version {PROTOCOL_VERSION}"
                );
                metadata
            }
            CandidateResponse::Error { message } => bail!("candidate startup failed: {message}"),
            other => bail!("candidate sent {:?} before ready", other),
        };
        Ok((
            Self {
                child,
                input,
                output,
            },
            metadata,
            started,
        ))
    }

    fn request(&mut self, request: &CandidateRequest) -> Result<(CandidateResponse, f64)> {
        let started = Instant::now();
        write_json_frame(&mut self.input, request).context("write candidate request")?;
        let response = read_json_frame(&mut self.output).context("read candidate response")?;
        if let CandidateResponse::Error { message } = &response {
            bail!("candidate request failed: {message}");
        }
        Ok((response, started.elapsed().as_secs_f64()))
    }

    fn shutdown(mut self) -> Result<()> {
        let (response, _) = self.request(&CandidateRequest::Shutdown)?;
        ensure!(
            matches!(response, CandidateResponse::Shutdown),
            "candidate did not acknowledge shutdown"
        );
        drop(self.input);
        drop(self.output);
        let status = self.child.wait().context("wait for candidate")?;
        ensure!(status.success(), "candidate exited with {status}");
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum TokenFlavor {
    Standard { require_full_reference: bool },
    Qwen3 { eos_token_id: u32 },
}

struct CanonicalTokenizer {
    tokenizer: Tokenizer,
    flavor: TokenFlavor,
    max_length: usize,
}

impl CanonicalTokenizer {
    fn load(model: &Path, tokenizer_path: &Path, max_length: usize) -> Result<Self> {
        let root = if model.is_dir() {
            model.to_owned()
        } else {
            model
                .parent()
                .context("model path has no parent")?
                .to_owned()
        };
        let config_path = root.join("config.json");
        let config: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(&config_path)
                .with_context(|| format!("read model config {}", config_path.display()))?,
        )
        .with_context(|| format!("parse model config {}", config_path.display()))?;
        let model_type = config
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let flavor = if model_type == "qwen3" {
            let eos = config
                .get("eos_token_id")
                .and_then(serde_json::Value::as_u64)
                .context("Qwen3 config is missing eos_token_id")?;
            TokenFlavor::Qwen3 {
                eos_token_id: u32::try_from(eos).context("Qwen3 eos_token_id exceeds u32")?,
            }
        } else {
            TokenFlavor::Standard {
                require_full_reference: model_type.contains("modernbert"),
            }
        };
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| anyhow::anyhow!("tokenizer: {error}"))?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|error| anyhow::anyhow!("truncation: {error}"))?;
        Ok(Self {
            tokenizer,
            flavor,
            max_length,
        })
    }

    fn text_length(&self, text: &str) -> Result<usize> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!("canonical text tokenization: {error}"))?;
        match self.flavor {
            TokenFlavor::Standard { .. } => Ok(encoding
                .get_attention_mask()
                .iter()
                .map(|&v| usize::from(v != 0))
                .sum()),
            TokenFlavor::Qwen3 { eos_token_id } => {
                let mut ids = encoding
                    .get_ids()
                    .iter()
                    .zip(encoding.get_attention_mask())
                    .filter_map(|(&id, &mask)| (mask != 0).then_some(id))
                    .collect::<Vec<_>>();
                if ids.last() == Some(&eos_token_id) {
                    ids.pop();
                }
                ids.truncate(self.max_length.saturating_sub(1));
                ids.push(eos_token_id);
                Ok(ids.len())
            }
        }
    }

    fn pair_length(&self, query: &str, document: &str) -> Result<usize> {
        self.tokenizer
            .encode(EncodeInput::Dual(query.into(), document.into()), true)
            .map(|encoding| {
                encoding
                    .get_attention_mask()
                    .iter()
                    .map(|&v| usize::from(v != 0))
                    .sum()
            })
            .map_err(|error| anyhow::anyhow!("canonical pair tokenization: {error}"))
    }

    fn require_full_reference(&self) -> bool {
        matches!(
            self.flavor,
            TokenFlavor::Standard {
                require_full_reference: true
            }
        )
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.passes > 0, "--passes must be at least one");
    ensure!(args.max_length > 0, "--max-length must be at least one");
    ensure!(
        args.corpus.is_some() ^ args.rerank_requests.is_some(),
        "provide exactly one of --corpus or --rerank-requests"
    );
    ensure!(
        args.attention_units >= args.max_length.saturating_mul(args.max_length),
        "--attention-units must fit at least one max-length sequence"
    );
    ensure!(
        !(matches!(args.device, DeviceArg::Cpu) && matches!(args.dtype, Precision::F16)),
        "cpu + f16 is unsupported"
    );

    let canonical = CanonicalTokenizer::load(&args.model, &args.tokenizer, args.max_length)?;
    if let Some(path) = &args.rerank_requests {
        run_rerank(&args, path, &canonical)
    } else {
        run_embedding(&args, &canonical)
    }
}

fn run_embedding(args: &Args, canonical: &CanonicalTokenizer) -> Result<()> {
    let corpus_path = args.corpus.as_ref().context("embedding corpus missing")?;
    let chunks = load_corpus(corpus_path, args.limit)?;
    let lengths = chunks
        .iter()
        .map(|chunk| canonical.text_length(&chunk.text))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        lengths.iter().all(|&length| length <= args.max_length),
        "canonical tokenizer returned a sequence longer than --max-length"
    );
    let mut order = (0..chunks.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| lengths[index]);
    let bucket_shapes = bucket_shapes(args.max_length, args.attention_units);
    ensure!(
        args.shapes != ShapeArg::Bucketed || bucket_shapes.len() <= 12,
        "bucket policy produced too many shapes"
    );
    let batches = planned_batches(
        &order,
        &lengths,
        args.attention_units,
        args.shapes,
        &bucket_shapes,
    );
    let reference = args
        .reference
        .as_ref()
        .map(|path| load_reference(path))
        .transpose()?;

    let (mut candidate, candidate_metadata, candidate_started) = CandidateProcess::spawn(args)?;
    let (prepare_shapes, force_shapes) = preparation_shapes(
        args,
        &candidate_metadata,
        Workload::Embedding,
        &batches,
        &bucket_shapes,
    );
    let (prepared, _) = candidate.request(&CandidateRequest::PrepareShapes {
        workload: Workload::Embedding,
        shapes: prepare_shapes,
        max_length: args.max_length,
        force_shapes,
    })?;
    let candidate_internal_prepare_wall_s = match prepared {
        CandidateResponse::Prepared { internal_wall_s } => internal_wall_s,
        other => bail!("candidate returned {:?} for shape preparation", other),
    };
    let cold_load_s = candidate_started.elapsed().as_secs_f64();

    let mut passes = Vec::with_capacity(args.passes);
    let mut reconciliations = Vec::with_capacity(args.passes);
    let mut candidate_internal_pass_wall_s = Vec::with_capacity(args.passes);
    let mut final_vectors = Vec::new();
    for pass in 0..args.passes {
        let pass_started = Instant::now();
        let mut vectors = Vec::with_capacity(chunks.len());
        let mut candidate_tokens = 0u64;
        let mut internal_wall_s = 0.0;
        let mut padded_tokens = 0u64;
        for batch in &batches {
            let indices = &order[batch.range.clone()];
            let texts = indices
                .iter()
                .map(|&index| chunks[index].text.clone())
                .collect::<Vec<_>>();
            let (response, _) = candidate.request(&CandidateRequest::Embed {
                texts,
                max_length: args.max_length,
                shape_policy: args.shapes.protocol(),
                shape: batch.shape,
            })?;
            padded_tokens += (batch.shape.batch * batch.shape.seq) as u64;
            let (batch_vectors, reported_tokens, candidate_wall_s) = match response {
                CandidateResponse::Embedding {
                    vectors,
                    reported_real_tokens,
                    internal_infer_wall_s,
                } => (vectors, reported_real_tokens, internal_infer_wall_s),
                other => bail!("candidate returned {:?} for embedding request", other),
            };
            ensure!(
                batch_vectors.len() == indices.len(),
                "candidate returned {} vectors for {} texts",
                batch_vectors.len(),
                indices.len()
            );
            candidate_tokens += reported_tokens;
            internal_wall_s += candidate_wall_s;
            for (offset, vector) in batch_vectors.into_iter().enumerate() {
                vectors.push((chunks[indices[offset]].id.clone(), vector));
            }
        }
        let infer_wall_s = pass_started.elapsed().as_secs_f64();
        let canonical_tokens = lengths.iter().sum::<usize>() as u64;
        let reconciliation = reconcile_tokens(
            pass + 1,
            pass_label(pass, args.passes),
            canonical_tokens,
            candidate_tokens,
        )?;
        let (parity_mean_cosine, top10_rank_overlap) = embedding_gates(
            &vectors,
            reference.as_ref(),
            canonical.require_full_reference(),
            args.min_parity,
            args.min_rank_overlap,
        )?;
        let waste = padding_waste_fraction(canonical_tokens, padded_tokens);
        if args.shapes == ShapeArg::Bucketed {
            ensure!(
                waste < 0.15,
                "bucket padding waste {:.2}% exceeds the 15% serving gate",
                waste * 100.0
            );
        }
        passes.push(PassResult {
            pass: pass + 1,
            label: pass_label(pass, args.passes),
            infer_wall_s,
            input_tokens: canonical_tokens,
            padded_tokens,
            padding_waste_fraction: waste,
            tok_per_s: canonical_tokens as f64 / infer_wall_s,
            items: vectors.len() as u64,
            parity_mean_cosine,
            top10_rank_overlap,
        });
        reconciliations.push(reconciliation);
        candidate_internal_pass_wall_s.push(internal_wall_s);
        final_vectors = vectors;
    }
    candidate.shutdown()?;

    if let Some(path) = &args.vectors_out {
        write_vectors(path, &final_vectors)?;
    }
    let last = passes.last().context("at least one pass is required")?;
    let lane = LaneResult {
        lane: candidate_metadata.lane.clone(),
        workload: "embed-corpus-v1".to_owned(),
        model: candidate_metadata.model.clone(),
        cold_load_s,
        infer_wall_s: last.infer_wall_s,
        input_tokens: last.input_tokens,
        tok_per_s: last.tok_per_s,
        items: last.items,
        parity_mean_cosine: last.parity_mean_cosine,
        self_peak_rss_bytes: None,
        notes: format!(
            "{}; provider={}, dtype={}, execution={}, package_cache={}, shapes={}, policy_version={}, passes={}, length-sorted attention_units={}, max_len={}; timing and canonical token accounting owned by synapse-rig",
            candidate_metadata.notes,
            candidate_metadata.provider,
            candidate_metadata.dtype,
            candidate_metadata.execution,
            args.package_cache.as_ref().map_or_else(|| "disabled".to_owned(), |path| path.display().to_string()),
            args.shapes.as_str(),
            if args.shapes == ShapeArg::Bucketed { BUCKET_POLICY_VERSION.to_string() } else { "none".to_owned() },
            args.passes,
            args.attention_units,
            args.max_length,
        ),
    };
    let result = ServingResult {
        real_tokens: last.input_tokens,
        padded_tokens: last.padded_tokens,
        padding_waste_fraction: last.padding_waste_fraction,
        shape_policy: args.shapes,
        bucket_policy_version: (args.shapes == ShapeArg::Bucketed).then_some(BUCKET_POLICY_VERSION),
        bucket_shapes: if args.shapes == ShapeArg::Bucketed {
            bucket_shapes
        } else {
            Vec::new()
        },
        package_cache: package_cache_stats(
            candidate_metadata
                .package_cache_root
                .as_deref()
                .map(Path::new),
        )?,
        lane,
        passes,
        rig_metadata: rig_metadata(
            candidate_metadata,
            candidate_internal_prepare_wall_s,
            candidate_internal_pass_wall_s,
            reconciliations,
            args.device,
        )?,
    };
    write_result(&args.out, &result)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn run_rerank(args: &Args, path: &Path, canonical: &CanonicalTokenizer) -> Result<()> {
    ensure!(
        matches!(args.dtype, Precision::F32),
        "reranking is fp32-only"
    );
    let mut requests = load_jsonl::<RerankRequest>(path)?;
    if let Some(limit) = args.limit {
        requests.truncate(limit);
    }
    ensure!(
        !requests.is_empty(),
        "rerank request file must not be empty"
    );
    ensure!(
        requests.iter().all(|request| !request.documents.is_empty()),
        "every rerank request must contain at least one document"
    );
    let bucket_shapes = bucket_shapes(args.max_length, args.attention_units);
    let mut all_exact_batches = Vec::new();
    for request in &requests {
        let lengths = request
            .documents
            .iter()
            .map(|document| canonical.pair_length(&request.query, document))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            lengths.iter().all(|&length| length <= args.max_length),
            "canonical pair exceeds --max-length"
        );
        let mut order = (0..request.documents.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| lengths[index]);
        let batches = planned_batches(
            &order,
            &lengths,
            args.attention_units,
            args.shapes,
            &bucket_shapes,
        );
        all_exact_batches.extend(batches.iter().cloned());
    }

    let (mut candidate, candidate_metadata, candidate_started) = CandidateProcess::spawn(args)?;
    let (prepare_shapes, force_shapes) = preparation_shapes(
        args,
        &candidate_metadata,
        Workload::Rerank,
        &all_exact_batches,
        &bucket_shapes,
    );
    let (prepared, _) = candidate.request(&CandidateRequest::PrepareShapes {
        workload: Workload::Rerank,
        shapes: prepare_shapes,
        max_length: args.max_length,
        force_shapes,
    })?;
    let candidate_internal_prepare_wall_s = match prepared {
        CandidateResponse::Prepared { internal_wall_s } => internal_wall_s,
        other => bail!("candidate returned {:?} for shape preparation", other),
    };
    let cold_load_s = candidate_started.elapsed().as_secs_f64();

    let mut rows = Vec::with_capacity(requests.len());
    let mut latencies_ms = Vec::with_capacity(requests.len());
    let mut canonical_tokens = 0u64;
    let mut candidate_tokens = 0u64;
    let mut padded_tokens = 0u64;
    let mut infer_wall_s = 0.0;
    let mut internal_wall_s = 0.0;
    let mut pair_count = 0usize;
    for request in &requests {
        let request_started = Instant::now();
        let lengths = request
            .documents
            .iter()
            .map(|document| canonical.pair_length(&request.query, document))
            .collect::<Result<Vec<_>>>()?;
        let mut order = (0..request.documents.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| lengths[index]);
        let batches = planned_batches(
            &order,
            &lengths,
            args.attention_units,
            args.shapes,
            &bucket_shapes,
        );
        let mut scores = vec![0.0f32; request.documents.len()];
        for batch in &batches {
            let indices = &order[batch.range.clone()];
            let pairs = indices
                .iter()
                .map(|&index| RerankPair {
                    query: request.query.clone(),
                    document: request.documents[index].clone(),
                })
                .collect::<Vec<_>>();
            let (response, _) = candidate.request(&CandidateRequest::Rerank {
                pairs,
                max_length: args.max_length,
                shape_policy: args.shapes.protocol(),
                shape: batch.shape,
            })?;
            padded_tokens += (batch.shape.batch * batch.shape.seq) as u64;
            let (batch_scores, reported_tokens, candidate_wall_s) = match response {
                CandidateResponse::Rerank {
                    scores,
                    reported_real_tokens,
                    internal_infer_wall_s,
                } => (scores, reported_real_tokens, internal_infer_wall_s),
                other => bail!("candidate returned {:?} for rerank request", other),
            };
            ensure!(
                batch_scores.len() == indices.len(),
                "candidate rerank score count mismatch"
            );
            candidate_tokens += reported_tokens;
            internal_wall_s += candidate_wall_s;
            for (offset, score) in batch_scores.into_iter().enumerate() {
                scores[indices[offset]] = score;
            }
        }
        let request_s = request_started.elapsed().as_secs_f64();
        infer_wall_s += request_s;
        latencies_ms.push(request_s * 1_000.0);
        canonical_tokens += lengths.iter().sum::<usize>() as u64;
        pair_count += request.documents.len();
        rows.push(RerankScores {
            id: request.id.clone(),
            scores,
        });
    }
    candidate.shutdown()?;
    let reconciliation = reconcile_tokens(1, "first", canonical_tokens, candidate_tokens)?;
    let waste = padding_waste_fraction(canonical_tokens, padded_tokens);
    let (pearson, top1) = if let Some(reference) = &args.reference {
        let reference = load_jsonl::<RerankScores>(reference)?;
        let (pearson, top1) = rerank_agreement(&rows, &reference)?;
        ensure!(
            pearson >= args.min_pearson,
            "rerank Pearson {pearson:.9} below minimum {:.9}",
            args.min_pearson
        );
        ensure!(
            top1 >= args.min_top1_agreement,
            "rerank tie-aware top-1 {top1:.6} below minimum {:.6}",
            args.min_top1_agreement
        );
        (Some(pearson), Some(top1))
    } else {
        (None, None)
    };
    if let Some(path) = &args.scores_out {
        write_jsonl(path, &rows)?;
    }
    latencies_ms.sort_by(f64::total_cmp);
    let result = RerankServingResult {
        lane: candidate_metadata.lane.clone(),
        workload: "rerank-pairs-v1",
        model: candidate_metadata.model.clone(),
        provider: candidate_metadata.provider.clone(),
        dtype: candidate_metadata.dtype.clone(),
        execution: args.execution,
        shape_policy: args.shapes,
        bucket_policy_version: (args.shapes == ShapeArg::Bucketed).then_some(BUCKET_POLICY_VERSION),
        bucket_shapes: if args.shapes == ShapeArg::Bucketed { bucket_shapes } else { Vec::new() },
        requests: requests.len(),
        pairs: pair_count,
        real_tokens: canonical_tokens,
        padded_tokens,
        padding_waste_fraction: waste,
        infer_wall_s,
        pairs_per_s: pair_count as f64 / infer_wall_s,
        request_latency_p50_ms: percentile(&latencies_ms, 0.50),
        request_latency_p95_ms: percentile(&latencies_ms, 0.95),
        pearson,
        tie_aware_top1_agreement: top1,
        package_cache: package_cache_stats(
            candidate_metadata
                .package_cache_root
                .as_deref()
                .map(Path::new),
        )?,
        notes: format!(
            "{}; raw logits (no sigmoid), combined pair-length buckets, cold_load_s={cold_load_s:.6}; timing and canonical token accounting owned by synapse-rig",
            candidate_metadata.notes
        ),
        rig_metadata: rig_metadata(
            candidate_metadata,
            candidate_internal_prepare_wall_s,
            vec![internal_wall_s],
            vec![reconciliation],
            args.device,
        )?,
    };
    write_result(&args.out, &result)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn preparation_shapes(
    args: &Args,
    metadata: &CandidateMetadata,
    workload: Workload,
    actual_batches: &[PlannedBatch],
    buckets: &[BatchShape],
) -> (Vec<BatchShape>, bool) {
    if args.shapes == ShapeArg::Bucketed && !matches!(args.device, DeviceArg::Cpu) {
        return (buckets.to_vec(), true);
    }
    if args.shapes == ShapeArg::Exact
        && metadata.eager_shape_preload
        && matches!(workload, Workload::Embedding)
    {
        return (
            actual_batches.iter().map(|batch| batch.shape).collect(),
            true,
        );
    }
    (
        vec![actual_batches
            .first()
            .map_or(BatchShape { batch: 1, seq: 1 }, |batch| batch.shape)],
        false,
    )
}

fn embedding_gates(
    vectors: &[(String, Vec<f32>)],
    reference: Option<&HashMap<String, Vec<f32>>>,
    require_full_reference: bool,
    min_parity: f64,
    min_rank_overlap: f64,
) -> Result<(Option<f64>, Option<f64>)> {
    let Some(reference) = reference else {
        return Ok((None, None));
    };
    let (mean, matched) = mean_parity(vectors.iter().cloned(), reference);
    let mean = mean.context("no overlapping ids with parity reference")?;
    if require_full_reference {
        ensure!(
            matched == vectors.len(),
            "reference matched {matched} of {} produced vectors",
            vectors.len()
        );
    }
    let produced = vectors.iter().cloned().collect::<HashMap<_, _>>();
    let stride = if produced.len() > 1_000 {
        produced.len().div_ceil(MAX_LARGE_CORPUS_RANK_QUERIES)
    } else {
        1
    };
    let ranks = rank_overlap(&produced, reference, 10, stride)?;
    ensure!(
        mean >= min_parity,
        "mean parity {mean:.8} below minimum {min_parity:.8} over {matched} vectors"
    );
    ensure!(
        ranks.mean_topk_overlap >= min_rank_overlap,
        "mean top-10 rank overlap {:.6} below minimum {min_rank_overlap:.6} over {} queries",
        ranks.mean_topk_overlap,
        ranks.queries
    );
    Ok((Some(mean), Some(ranks.mean_topk_overlap)))
}

fn reconcile_tokens(
    pass: usize,
    label: &'static str,
    canonical: u64,
    candidate: u64,
) -> Result<TokenReconciliation> {
    let divergence = canonical.abs_diff(candidate) as f64 / canonical.max(1) as f64;
    ensure!(
        divergence <= 0.01,
        "candidate reported {candidate} real tokens but the rig counted {canonical}; divergence {:.2}% exceeds 1%",
        divergence * 100.0
    );
    Ok(TokenReconciliation {
        pass,
        label,
        canonical_real_tokens: canonical,
        candidate_reported_real_tokens: candidate,
        divergence_fraction: divergence,
    })
}

fn bucket_shapes(max_length: usize, attention_units: usize) -> Vec<BatchShape> {
    let mut lengths = BUCKET_SEQUENCE_LADDER
        .iter()
        .copied()
        .take_while(|&seq| seq < max_length)
        .collect::<Vec<_>>();
    lengths.push(max_length);
    lengths.sort_unstable();
    lengths.dedup();
    lengths
        .into_iter()
        .map(|seq| BatchShape {
            batch: BUCKET_MAX_BATCH_ROWS.min((attention_units / seq.saturating_mul(seq)).max(1)),
            seq,
        })
        .collect()
}

fn covering_bucket(length: usize, buckets: &[BatchShape]) -> BatchShape {
    buckets
        .iter()
        .copied()
        .find(|shape| shape.seq >= length)
        .expect("bucket policy is capped at max-length")
}

fn planned_batches(
    order: &[usize],
    lengths: &[usize],
    attention_units: usize,
    shapes: ShapeArg,
    buckets: &[BatchShape],
) -> Vec<PlannedBatch> {
    if shapes == ShapeArg::Exact {
        let mut ranges = Vec::new();
        let mut start = 0usize;
        let mut max_len = 0usize;
        for index in 0..order.len() {
            let candidate_max = max_len.max(lengths[order[index]]);
            let count = index - start;
            if count > 0 && (count + 1) * candidate_max * candidate_max > attention_units {
                ranges.push(start..index);
                start = index;
                max_len = 0;
            }
            max_len = max_len.max(lengths[order[index]]);
        }
        if start < order.len() {
            ranges.push(start..order.len());
        }
        return ranges
            .into_iter()
            .map(|range| PlannedBatch {
                shape: BatchShape {
                    batch: range.len(),
                    seq: order[range.clone()]
                        .iter()
                        .map(|&index| lengths[index])
                        .max()
                        .unwrap_or(1),
                },
                range,
            })
            .collect();
    }
    let mut batches = Vec::new();
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start;
        while end < order.len() {
            let bucket = covering_bucket(lengths[order[end]], buckets);
            if end - start + 1 > bucket.batch {
                break;
            }
            end += 1;
        }
        batches.push(PlannedBatch {
            range: start..end,
            shape: covering_bucket(lengths[order[end - 1]], buckets),
        });
        start = end;
    }
    batches
}

fn rerank_agreement(candidate: &[RerankScores], reference: &[RerankScores]) -> Result<(f64, f64)> {
    let reference = reference
        .iter()
        .map(|row| (row.id.as_str(), row.scores.as_slice()))
        .collect::<HashMap<_, _>>();
    let candidate_ids = candidate
        .iter()
        .map(|row| row.id.as_str())
        .collect::<HashSet<_>>();
    let reference_ids = reference.keys().copied().collect::<HashSet<_>>();
    ensure!(
        candidate.len() == candidate_ids.len() && candidate_ids == reference_ids,
        "rerank candidate/reference request IDs mismatch"
    );
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut top1_matches = 0usize;
    for row in candidate {
        let expected = reference
            .get(row.id.as_str())
            .with_context(|| format!("reference is missing rerank request {}", row.id))?;
        ensure!(
            row.scores.len() == expected.len(),
            "rerank pair count mismatch for {}",
            row.id
        );
        xs.extend(expected.iter().map(|&value| f64::from(value)));
        ys.extend(row.scores.iter().map(|&value| f64::from(value)));
        let expected_top = expected.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let candidate_top = row.scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if row
            .scores
            .iter()
            .zip(*expected)
            .any(|(&actual, &reference)| actual == candidate_top && reference == expected_top)
        {
            top1_matches += 1;
        }
    }
    let x_mean = xs.iter().sum::<f64>() / xs.len() as f64;
    let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let mut numerator = 0.0;
    let mut x_norm = 0.0;
    let mut y_norm = 0.0;
    for (&x, &y) in xs.iter().zip(&ys) {
        let x = x - x_mean;
        let y = y - y_mean;
        numerator += x * y;
        x_norm += x * x;
        y_norm += y * y;
    }
    ensure!(
        x_norm > 0.0 && y_norm > 0.0,
        "rerank Pearson requires non-constant scores"
    );
    Ok((
        numerator / (x_norm.sqrt() * y_norm.sqrt()),
        top1_matches as f64 / candidate.len() as f64,
    ))
}

fn pass_label(pass: usize, passes: usize) -> &'static str {
    if pass == 0 {
        "first"
    } else if pass + 1 == passes && passes > 2 {
        "steady"
    } else {
        "warm"
    }
}

fn padding_waste_fraction(real: u64, padded: u64) -> f64 {
    if padded == 0 {
        0.0
    } else {
        padded.saturating_sub(real) as f64 / padded as f64
    }
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn package_cache_stats(root: Option<&Path>) -> Result<PackageCacheStats> {
    fn directory_bytes(path: &Path) -> Result<u64> {
        let mut bytes = 0u64;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            bytes += if metadata.is_dir() {
                directory_bytes(&entry.path())?
            } else {
                metadata.len()
            };
        }
        Ok(bytes)
    }
    let Some(root) = root else {
        return Ok(PackageCacheStats::default());
    };
    if !root.exists() {
        return Ok(PackageCacheStats::default());
    }
    let mut stats = PackageCacheStats::default();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir()
            && entry.path().extension().and_then(|value| value.to_str()) == Some("mpsgraphpackage")
        {
            stats.package_count += 1;
            stats.package_bytes += directory_bytes(&entry.path())?;
        }
    }
    Ok(stats)
}

fn rig_metadata(
    candidate: CandidateMetadata,
    prepare_wall_s: f64,
    pass_wall_s: Vec<f64>,
    reconciliation: Vec<TokenReconciliation>,
    device: DeviceArg,
) -> Result<RigMetadata> {
    let exe = std::env::current_exe().context("locate synapse-rig executable")?;
    let sha256 =
        hex::encode(Sha256::digest(fs::read(&exe).with_context(|| {
            format!("read rig executable {}", exe.display())
        })?));
    Ok(RigMetadata {
        sha256,
        git_revision: env!("SYNAPSE_RIG_GIT_REV"),
        protocol_version: PROTOCOL_VERSION,
        candidate,
        candidate_internal_prepare_wall_s: prepare_wall_s,
        candidate_internal_pass_wall_s: pass_wall_s,
        token_reconciliation: reconciliation,
        host_probe: host_probe(device),
    })
}

fn host_probe(device: DeviceArg) -> Option<HostProbe> {
    let (tool, mut command) = if matches!(device, DeviceArg::Cuda) {
        let mut command = Command::new("nvidia-smi");
        command.args(["--query-gpu=name,driver_version", "--format=csv,noheader"]);
        ("nvidia-smi", command)
    } else if cfg!(target_os = "macos") {
        let mut command = Command::new("macmon");
        command.arg("--version");
        ("macmon", command)
    } else {
        return None;
    };
    command
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            let mut text = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if text.is_empty() {
                text = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            }
            text.truncate(4_096);
            HostProbe { tool, output: text }
        })
}

fn load_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    synapse_bench::parity::load_jsonl(path)
}

fn write_vectors(path: &Path, vectors: &[(String, Vec<f32>)]) -> Result<()> {
    let rows = vectors
        .iter()
        .map(|(id, vector)| serde_json::json!({ "id": id, "vec": vector }))
        .collect::<Vec<_>>();
    write_jsonl(path, &rows)
}

fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut output = String::new();
    for row in rows {
        output.push_str(&serde_json::to_string(row)?);
        output.push('\n');
    }
    fs::write(path, output).with_context(|| format!("write {}", path.display()))
}

fn write_result<T: Serialize>(path: &Path, result: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(result)?)
        .with_context(|| format!("write {}", path.display()))
}
