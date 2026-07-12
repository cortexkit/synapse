use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use safetensors::tensor::{Dtype as SafeDtype, SafeTensors};
use serde::{Deserialize, Serialize};
use synapse_bench::{
    parity::{load_corpus, load_reference, mean_parity, rank_overlap, Chunk},
    results::LaneResult,
    rig_protocol::{
        read_json_frame, write_json_frame, CandidateMetadata, CandidateRequest, CandidateResponse,
        ShapePolicy, Workload, PROTOCOL_VERSION,
    },
};
use tokenizers::{Tokenizer, TruncationParams};

mod cuda_backend;
mod modernbert;
mod qwen3;
mod vulkan_backend;

#[derive(Parser)]
#[command(name = "spike-unified-rt")]
struct Args {
    /// Path to a MiniLM or Qwen3 safetensors file or snapshot directory.
    #[arg(long)]
    model: PathBuf,
    /// Path to tokenizer.json.
    #[arg(long)]
    tokenizer: PathBuf,
    /// Corpus JSONL ({id, path, text, tokens} per line). Required for embedding mode.
    #[arg(long)]
    corpus: Option<PathBuf>,
    /// Rerank request JSONL ({id, query, documents} per line).
    #[arg(long, conflicts_with = "corpus")]
    rerank_requests: Option<PathBuf>,
    /// Optional cap for parity/throughput smoke runs.
    #[arg(long)]
    limit: Option<usize>,
    /// Output LaneResult JSON path. Not used by --serve-stdio.
    #[arg(long, required_unless_present = "serve_stdio")]
    out: Option<PathBuf>,
    /// Optional: write produced vectors (JSONL: {id, vec}). Alias kept for the spike prompt.
    #[arg(long = "vectors-out", alias = "emit-vectors")]
    vectors_out: Option<PathBuf>,
    /// Rerank score output JSONL ({id, scores} per line).
    #[arg(long)]
    scores_out: Option<PathBuf>,
    /// Optional embedding vectors or rerank scores used as the mode-specific reference.
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Minimum mean cosine when --reference is supplied.
    #[arg(long, default_value_t = 0.9999)]
    min_parity: f64,
    /// Minimum mean top-10 neighbor overlap when an embedding reference is supplied.
    #[arg(long, default_value_t = 0.995)]
    min_rank_overlap: f64,
    /// Minimum overall Pearson correlation for rerank reference scores.
    #[arg(long, default_value_t = 0.999)]
    min_pearson: f64,
    /// Minimum tie-aware top-1 agreement for rerank reference scores.
    #[arg(long, default_value_t = 0.98)]
    min_top1_agreement: f64,
    /// Kernel provider to use.
    #[arg(long, value_enum, default_value_t = DeviceArg::Cpu)]
    device: DeviceArg,
    /// Precision for the resident encoder path.
    #[arg(long, value_enum, default_value_t = Precision::F32)]
    dtype: Precision,
    /// Launch the CUDA encoder through a per-shape CUDA Graph.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    cuda_graphs: bool,
    /// GEMM implementation used by the Vulkan MiniLM block.
    #[arg(long, value_enum, default_value_t = VulkanGemm::Plain)]
    vulkan_gemm: VulkanGemm,
    /// Optional serialized VkPipelineCache file for Vulkan cold/warm measurements.
    #[arg(long)]
    vulkan_pipeline_cache: Option<PathBuf>,
    /// Metal graph execution strategy. Explicit O0 compilation is the serving default.
    #[arg(long, value_enum, default_value_t = Execution::Explicit)]
    execution: Execution,
    /// Optional directory for one compiled MPSGraph package per batch/sequence shape.
    #[arg(long)]
    package_cache: Option<PathBuf>,
    /// Tokenizer truncation max length.
    #[arg(long, default_value_t = 512)]
    max_length: usize,
    /// Greedy attention-unit budget per batch.
    #[arg(long, default_value_t = 4_000_000)]
    attention_units: usize,
    /// Shape policy used by the serving runner.
    #[arg(long, value_enum, default_value_t = Shapes::Bucketed)]
    shapes: Shapes,
    /// Bucket policy version. Version 1 remains available for A/B measurements.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=BUCKET_POLICY_VERSION as i64))]
    bucket_policy: u32,
    /// Number of in-process corpus passes.
    #[arg(long, default_value_t = 1)]
    passes: usize,
    /// Optional model label for the result.
    #[arg(long)]
    model_label: Option<String>,
    /// Load the model and serve length-prefixed JSON requests on stdin/stdout.
    #[arg(long)]
    serve_stdio: bool,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, ValueEnum)]
enum DeviceArg {
    Cpu,
    Metal,
    Cuda,
    Vulkan,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, ValueEnum)]
enum VulkanGemm {
    Plain,
    Cooperative,
}

impl VulkanGemm {
    #[cfg(all(target_os = "windows", feature = "vulkan"))]
    fn as_str(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Cooperative => "cooperative",
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, ValueEnum)]
enum Precision {
    F32,
    F16,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
enum Execution {
    Explicit,
    Lazy,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum Shapes {
    Exact,
    Bucketed,
}

const GRAPH_REVISION: u32 = 3;
const BUCKET_POLICY_VERSION: u32 = 2;
const BUCKET_V1_MAX_BATCH_ROWS: usize = 8;
const BUCKET_V2_BATCH_ROW_LADDER: &[usize] = &[16, 16, 16, 16, 16, 16, 12, 12, 8, 8];
const BUCKET_SEQUENCE_LADDER: &[usize] = &[64, 96, 128, 160, 192, 256, 320, 384, 448, 512];
const MAX_LARGE_CORPUS_RANK_QUERIES: usize = 100;

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Serialize)]
struct BatchShape {
    batch: usize,
    seq: usize,
}

#[derive(Clone, Debug)]
struct PlannedBatch {
    range: std::ops::Range<usize>,
    shape: BatchShape,
}

struct WorkloadResult {
    infer_wall_s: f64,
    input_tokens: u64,
    padded_tokens: u64,
    produced_vectors: Vec<(String, Vec<f32>)>,
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
    shape_policy: Shapes,
    bucket_policy_version: Option<u32>,
    bucket_shapes: Vec<BatchShape>,
    real_tokens: u64,
    padded_tokens: u64,
    padding_waste_fraction: f64,
    padding_waste_gate_passed: Option<bool>,
    package_cache: PackageCacheStats,
    passes: Vec<PassResult>,
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
    provider: &'static str,
    dtype: &'static str,
    execution: Execution,
    shape_policy: Shapes,
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
}

#[derive(Clone, Debug)]
struct MetalExecutionConfig {
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    execution: Execution,
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    package_root: Option<PathBuf>,
}

impl MetalExecutionConfig {
    fn from_args(args: &Args, family: &str) -> Result<Self> {
        let package_root = args.package_cache.as_ref().map(|root| {
            let model = fs::canonicalize(&args.model).unwrap_or_else(|_| args.model.clone());
            let identity = format!("{}", model.display());
            let hash = identity.bytes().fold(1469598103934665603u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
            });
            let os_build = std::process::Command::new("sw_vers")
                .arg("-buildVersion")
                .output()
                .ok()
                .filter(|output| output.status.success())
                .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
                .filter(|build| !build.is_empty())
                .unwrap_or_else(|| "unknown-os-build".to_owned());
            let shape_key = shape_cache_key(args.shapes, args.bucket_policy);
            root.join(format!(
                "{family}-graph-v{GRAPH_REVISION}-{shape_key}-{:016x}-{}-{os_build}",
                hash,
                args.dtype.as_str()
            ))
        });
        if let Some(root) = &package_root {
            fs::create_dir_all(root)
                .with_context(|| format!("create package cache {}", root.display()))?;
        }
        Ok(Self {
            execution: args.execution,
            package_root,
        })
    }

    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    fn package_path(&self, batch: usize, seq: usize) -> Option<PathBuf> {
        self.package_root
            .as_ref()
            .map(|root| root.join(format!("{batch}x{seq}.mpsgraphpackage")))
    }
}

fn shape_cache_key(shapes: Shapes, bucket_policy: u32) -> String {
    match shapes {
        Shapes::Exact => "shapes-exact".to_owned(),
        Shapes::Bucketed => format!("bucket-policy-v{bucket_policy}"),
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

fn main() -> Result<()> {
    let args = Args::parse();
    if args.serve_stdio {
        return serve_stdio(args);
    }
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
        "cpu + f16 is not supported for this spike; use --dtype f32 on cpu"
    );
    let started = Instant::now();

    let model = load_model_family(&args.model, args.dtype)?;
    let mut tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|error| anyhow::anyhow!("tokenizer: {error}"))?;
    model.configure_tokenizer(&mut tokenizer, args.max_length)?;

    let bucket_shapes = bucket_shapes(args.max_length, args.attention_units, args.bucket_policy);
    ensure!(
        args.shapes != Shapes::Bucketed || bucket_shapes.len() <= 12,
        "bucket policy produced {} shapes; serving limit is 12",
        bucket_shapes.len()
    );
    let execution = MetalExecutionConfig::from_args(&args, model.family_name())?;
    let mut provider = make_provider(
        args.device,
        args.dtype,
        execution.clone(),
        args.cuda_graphs,
        args.vulkan_gemm,
        args.vulkan_pipeline_cache.clone(),
    )?;
    let accelerator = matches!(
        args.device,
        DeviceArg::Metal | DeviceArg::Cuda | DeviceArg::Vulkan
    );
    if let Some(requests_path) = &args.rerank_requests {
        return run_rerank_cli(
            &args,
            requests_path,
            model.as_ref(),
            provider.as_mut(),
            &tokenizer,
            &bucket_shapes,
            execution,
            started,
        );
    }
    if args.shapes == Shapes::Bucketed && accelerator {
        for &shape in &bucket_shapes {
            let _ = model.embed_batch(
                provider.as_mut(),
                &tokenizer,
                &["bucket preload"],
                args.max_length,
                Some(shape),
            )?;
        }
    } else {
        let _ = model.embed_batch(
            provider.as_mut(),
            &tokenizer,
            &["warmup"],
            args.max_length,
            None,
        )?;
    }
    let initial_cold_load_s = started.elapsed().as_secs_f64();

    let corpus_path = args
        .corpus
        .as_ref()
        .context("embedding mode requires --corpus")?;
    let chunks: Vec<Chunk> = load_corpus(corpus_path, args.limit)?;
    ensure!(!chunks.is_empty(), "corpus must contain at least one row");
    let lengths = chunks
        .iter()
        .map(|chunk| model.token_length(&tokenizer, &chunk.text, args.max_length))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        lengths.iter().all(|&length| length <= args.max_length),
        "tokenizer returned a sequence longer than --max-length"
    );

    let mut order: Vec<usize> = (0..chunks.len()).collect();
    order.sort_by_key(|&index| lengths[index]);
    let batches = planned_batches(
        &order,
        &lengths,
        args.attention_units,
        args.shapes,
        &bucket_shapes,
    );
    if args.shapes == Shapes::Exact && provider.eager_shape_preload() {
        for batch in &batches {
            let texts = order[batch.range.clone()]
                .iter()
                .map(|&index| chunks[index].text.as_str())
                .collect::<Vec<_>>();
            let _ =
                model.embed_batch(provider.as_mut(), &tokenizer, &texts, args.max_length, None)?;
        }
    }
    let cold_load_s = if args.shapes == Shapes::Bucketed && accelerator
        || args.shapes == Shapes::Exact && provider.eager_shape_preload()
    {
        started.elapsed().as_secs_f64()
    } else {
        initial_cold_load_s
    };

    let reference = args
        .reference
        .as_ref()
        .map(|path| load_reference(path))
        .transpose()?;
    let mut passes = Vec::with_capacity(args.passes);
    let mut final_vectors = Vec::new();
    for pass in 0..args.passes {
        let workload = run_workload(
            model.as_ref(),
            provider.as_mut(),
            &tokenizer,
            &chunks,
            &lengths,
            &order,
            &batches,
            args.max_length,
            args.shapes,
        )?;
        let (parity_mean_cosine, top10_rank_overlap) = match &reference {
            Some(reference) => {
                let (mean, matched) = mean_parity(
                    workload
                        .produced_vectors
                        .iter()
                        .map(|(id, vector)| (id.clone(), vector.clone())),
                    reference,
                );
                let mean = mean.context("no overlapping ids with parity reference")?;
                model.validate_reference_coverage(matched, workload.produced_vectors.len())?;
                let produced: HashMap<String, Vec<f32>> =
                    workload.produced_vectors.iter().cloned().collect();
                let rank_stride = if produced.len() > 1_000 {
                    produced.len().div_ceil(MAX_LARGE_CORPUS_RANK_QUERIES)
                } else {
                    1
                };
                let ranks = rank_overlap(&produced, reference, 10, rank_stride)?;
                enforce_parity_gates(
                    mean,
                    ranks.mean_topk_overlap,
                    args.min_parity,
                    args.min_rank_overlap,
                    matched,
                    ranks.queries,
                )?;
                eprintln!(
                    "pass {} parity gate: mean cosine {mean:.8}, top-10 rank overlap {:.6}",
                    pass + 1,
                    ranks.mean_topk_overlap
                );
                (Some(mean), Some(ranks.mean_topk_overlap))
            }
            None => (None, None),
        };
        let padding_waste_fraction =
            padding_waste_fraction(workload.input_tokens, workload.padded_tokens);
        passes.push(PassResult {
            pass: pass + 1,
            label: pass_label(pass, args.passes),
            infer_wall_s: workload.infer_wall_s,
            input_tokens: workload.input_tokens,
            padded_tokens: workload.padded_tokens,
            padding_waste_fraction,
            tok_per_s: workload.input_tokens as f64 / workload.infer_wall_s,
            items: workload.produced_vectors.len() as u64,
            parity_mean_cosine,
            top10_rank_overlap,
        });
        final_vectors = workload.produced_vectors;
    }

    if let Some(path) = &args.vectors_out {
        write_vectors(path, &final_vectors)?;
    }

    let last = passes.last().context("at least one pass is required")?;
    let lane = LaneResult {
        lane: format!("owned-rt-{}", provider.name()),
        workload: "embed-corpus-v1".into(),
        model: args
            .model_label
            .unwrap_or_else(|| model.default_label(args.dtype)),
        cold_load_s,
        infer_wall_s: last.infer_wall_s,
        input_tokens: last.input_tokens,
        tok_per_s: last.tok_per_s,
        items: last.items,
        parity_mean_cosine: last.parity_mean_cosine,
        self_peak_rss_bytes: None,
        notes: format!(
            "{}, provider={}, dtype={}, execution={:?}, package_cache={}, shapes={:?}, policy_version={}, passes={}, length-sorted attention_units={}, max_len={}; {}",
            model.notes(),
            provider.name(),
            args.dtype.as_str(),
            args.execution,
            args.package_cache.as_ref().map_or("disabled".into(), |path| path.display().to_string()),
            args.shapes,
            if args.shapes == Shapes::Bucketed { args.bucket_policy.to_string() } else { "none".to_owned() },
            args.passes,
            args.attention_units,
            args.max_length,
            match (args.shapes, accelerator) {
                (Shapes::Bucketed, true) => {
                    "all bucket shapes were pre-discovered before inference"
                }
                (Shapes::Bucketed, false) => "CPU execution uses bucket padding without accelerator pre-discovery",
                (Shapes::Exact, _) => "exact shapes retain the prior A/B behavior",
            }
        ),
    };
    let result = ServingResult {
        real_tokens: last.input_tokens,
        padded_tokens: last.padded_tokens,
        padding_waste_fraction: last.padding_waste_fraction,
        padding_waste_gate_passed: (args.shapes == Shapes::Bucketed)
            .then_some(last.padding_waste_fraction < 0.15),
        shape_policy: args.shapes,
        bucket_policy_version: (args.shapes == Shapes::Bucketed).then_some(args.bucket_policy),
        bucket_shapes: if args.shapes == Shapes::Bucketed {
            bucket_shapes
        } else {
            Vec::new()
        },
        package_cache: package_cache_stats(execution.package_root.as_deref())?,
        lane,
        passes,
    };

    let out = args
        .out
        .as_ref()
        .context("standalone mode requires --out")?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&result)?)?;
    eprintln!(
        "{}: {} items, {} real / {} padded tokens, {:.1} tok/s, parity {:?}",
        result.lane.lane,
        result.lane.items,
        result.real_tokens,
        result.padded_tokens,
        result.lane.tok_per_s,
        result.lane.parity_mean_cosine
    );
    println!("{}", serde_json::to_string_pretty(&result)?);
    ensure!(
        args.shapes != Shapes::Bucketed || result.padding_waste_fraction < 0.15,
        "bucket padding waste {:.2}% exceeds the 15% serving gate",
        result.padding_waste_fraction * 100.0
    );
    Ok(())
}

fn serve_stdio(args: Args) -> Result<()> {
    ensure!(args.max_length > 0, "--max-length must be at least one");
    ensure!(
        args.attention_units >= args.max_length.saturating_mul(args.max_length),
        "--attention-units must fit at least one max-length sequence"
    );
    ensure!(
        !(matches!(args.device, DeviceArg::Cpu) && matches!(args.dtype, Precision::F16)),
        "cpu + f16 is not supported for this spike; use --dtype f32 on cpu"
    );

    let started = Instant::now();
    let model = load_model_family(&args.model, args.dtype)?;
    let mut tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|error| anyhow::anyhow!("tokenizer: {error}"))?;
    model.configure_tokenizer(&mut tokenizer, args.max_length)?;
    let execution = MetalExecutionConfig::from_args(&args, model.family_name())?;
    let package_cache_root = execution
        .package_root
        .as_ref()
        .map(|path| path.display().to_string());
    let mut provider = make_provider(
        args.device,
        args.dtype,
        execution,
        args.cuda_graphs,
        args.vulkan_gemm,
        args.vulkan_pipeline_cache.clone(),
    )?;
    let eager_shape_preload = provider.eager_shape_preload();
    let metadata = CandidateMetadata {
        lane: format!("owned-rt-{}", provider.name()),
        model: args
            .model_label
            .clone()
            .unwrap_or_else(|| model.default_label(args.dtype)),
        provider: provider.name().to_owned(),
        dtype: args.dtype.as_str().to_owned(),
        execution: match args.execution {
            Execution::Explicit => "explicit",
            Execution::Lazy => "lazy",
        }
        .to_owned(),
        notes: model.notes(),
        package_cache_root,
        internal_load_s: started.elapsed().as_secs_f64(),
        eager_shape_preload,
    };

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());
    write_json_frame(
        &mut writer,
        &CandidateResponse::Ready {
            protocol_version: PROTOCOL_VERSION,
            metadata,
        },
    )?;

    loop {
        let request: CandidateRequest = read_json_frame(&mut reader)?;
        let response = (|| -> Result<CandidateResponse> {
            match request {
                CandidateRequest::PrepareShapes {
                    workload,
                    shapes,
                    max_length,
                    force_shapes,
                } => {
                    ensure!(
                        max_length == args.max_length,
                        "rig/candidate max-length mismatch"
                    );
                    let prepare_started = Instant::now();
                    for shape in shapes {
                        ensure!(
                            shape.batch > 0 && shape.seq > 0,
                            "invalid preparation shape"
                        );
                        let shape = BatchShape {
                            batch: shape.batch,
                            seq: shape.seq,
                        };
                        match workload {
                            Workload::Embedding => {
                                let texts = vec![
                                    "shape preload";
                                    if force_shapes && args.shapes == Shapes::Exact {
                                        shape.batch
                                    } else {
                                        1
                                    }
                                ];
                                let _ = model.embed_batch(
                                    provider.as_mut(),
                                    &tokenizer,
                                    &texts,
                                    max_length,
                                    force_shapes.then_some(shape),
                                )?;
                            }
                            Workload::Rerank => {
                                let pairs = vec![
                                    ("shape preload", "shape preload document");
                                    if force_shapes && args.shapes == Shapes::Exact {
                                        shape.batch
                                    } else {
                                        1
                                    }
                                ];
                                let _ = model.rerank_batch(
                                    provider.as_mut(),
                                    &tokenizer,
                                    &pairs,
                                    max_length,
                                    force_shapes.then_some(shape),
                                )?;
                            }
                        }
                    }
                    Ok(CandidateResponse::Prepared {
                        internal_wall_s: prepare_started.elapsed().as_secs_f64(),
                    })
                }
                CandidateRequest::Embed {
                    texts,
                    max_length,
                    shape_policy,
                    shape,
                } => {
                    ensure!(
                        max_length == args.max_length,
                        "rig/candidate max-length mismatch"
                    );
                    ensure!(
                        shape_policy
                            == if args.shapes == Shapes::Bucketed {
                                ShapePolicy::Bucketed
                            } else {
                                ShapePolicy::Exact
                            },
                        "rig/candidate shape-policy mismatch"
                    );
                    ensure!(
                        texts.len() <= shape.batch,
                        "embedding batch exceeds directed shape"
                    );
                    let reported_real_tokens = texts
                        .iter()
                        .map(|text| model.token_length(&tokenizer, text, max_length))
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .sum::<usize>() as u64;
                    let refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
                    let infer_started = Instant::now();
                    let vectors = model.embed_batch(
                        provider.as_mut(),
                        &tokenizer,
                        &refs,
                        max_length,
                        (shape_policy == ShapePolicy::Bucketed).then_some(BatchShape {
                            batch: shape.batch,
                            seq: shape.seq,
                        }),
                    )?;
                    Ok(CandidateResponse::Embedding {
                        vectors,
                        reported_real_tokens,
                        internal_infer_wall_s: infer_started.elapsed().as_secs_f64(),
                    })
                }
                CandidateRequest::Rerank {
                    pairs,
                    max_length,
                    shape_policy,
                    shape,
                } => {
                    ensure!(
                        max_length == args.max_length,
                        "rig/candidate max-length mismatch"
                    );
                    ensure!(
                        shape_policy
                            == if args.shapes == Shapes::Bucketed {
                                ShapePolicy::Bucketed
                            } else {
                                ShapePolicy::Exact
                            },
                        "rig/candidate shape-policy mismatch"
                    );
                    ensure!(
                        pairs.len() <= shape.batch,
                        "rerank batch exceeds directed shape"
                    );
                    let reported_real_tokens = pairs
                        .iter()
                        .map(|pair| {
                            model.rerank_pair_length(&tokenizer, &pair.query, &pair.document)
                        })
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .sum::<usize>() as u64;
                    let refs = pairs
                        .iter()
                        .map(|pair| (pair.query.as_str(), pair.document.as_str()))
                        .collect::<Vec<_>>();
                    let infer_started = Instant::now();
                    let scores = model.rerank_batch(
                        provider.as_mut(),
                        &tokenizer,
                        &refs,
                        max_length,
                        (shape_policy == ShapePolicy::Bucketed).then_some(BatchShape {
                            batch: shape.batch,
                            seq: shape.seq,
                        }),
                    )?;
                    Ok(CandidateResponse::Rerank {
                        scores,
                        reported_real_tokens,
                        internal_infer_wall_s: infer_started.elapsed().as_secs_f64(),
                    })
                }
                CandidateRequest::Shutdown => Ok(CandidateResponse::Shutdown),
            }
        })()
        .unwrap_or_else(|error| CandidateResponse::Error {
            message: format!("{error:#}"),
        });
        let shutdown = matches!(response, CandidateResponse::Shutdown);
        write_json_frame(&mut writer, &response)?;
        if shutdown {
            return Ok(());
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_rerank_cli(
    args: &Args,
    requests_path: &Path,
    model: &dyn ModelFamily,
    provider: &mut dyn KernelProvider,
    tokenizer: &Tokenizer,
    bucket_shapes: &[BatchShape],
    execution: MetalExecutionConfig,
    started: Instant,
) -> Result<()> {
    ensure!(
        matches!(args.dtype, Precision::F32),
        "reranking is fp32-only"
    );
    let accelerator = matches!(args.device, DeviceArg::Metal | DeviceArg::Cuda);
    if args.shapes == Shapes::Bucketed && accelerator {
        for &shape in bucket_shapes {
            let _ = model.rerank_batch(
                provider,
                tokenizer,
                &[("bucket preload", "bucket preload document")],
                args.max_length,
                Some(shape),
            )?;
        }
    } else {
        let _ = model.rerank_batch(
            provider,
            tokenizer,
            &[("warmup", "warmup document")],
            args.max_length,
            None,
        )?;
    }
    let cold_load_s = started.elapsed().as_secs_f64();
    let mut requests = load_rerank_rows::<RerankRequest>(requests_path)?;
    if let Some(limit) = args.limit {
        requests.truncate(limit);
    }
    ensure!(
        !requests.is_empty(),
        "rerank request file must contain at least one row"
    );
    ensure!(
        requests.iter().all(|request| !request.documents.is_empty()),
        "every rerank request must contain at least one document"
    );

    let mut rows = Vec::with_capacity(requests.len());
    let mut latencies_ms = Vec::with_capacity(requests.len());
    let mut real_tokens = 0u64;
    let mut padded_tokens = 0u64;
    let mut infer_wall_s = 0.0;
    let mut pair_count = 0usize;
    for request in &requests {
        let request_started = Instant::now();
        let lengths = request
            .documents
            .iter()
            .map(|document| model.rerank_pair_length(tokenizer, &request.query, document))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            lengths.iter().all(|&length| length <= args.max_length),
            "pair tokenizer returned a sequence longer than --max-length"
        );
        let mut order = (0..request.documents.len()).collect::<Vec<_>>();
        order.sort_by_key(|&index| lengths[index]);
        let batches = planned_batches(
            &order,
            &lengths,
            args.attention_units,
            args.shapes,
            bucket_shapes,
        );
        let mut scores = vec![0.0f32; request.documents.len()];
        for batch in batches {
            let indices = &order[batch.range.clone()];
            let pairs = indices
                .iter()
                .map(|&index| (request.query.as_str(), request.documents[index].as_str()))
                .collect::<Vec<_>>();
            let batch_scores = model.rerank_batch(
                provider,
                tokenizer,
                &pairs,
                args.max_length,
                (args.shapes == Shapes::Bucketed).then_some(batch.shape),
            )?;
            ensure!(
                batch_scores.len() == indices.len(),
                "model returned {} scores for {} real pairs",
                batch_scores.len(),
                indices.len()
            );
            padded_tokens += (batch.shape.batch * batch.shape.seq) as u64;
            for (offset, score) in batch_scores.into_iter().enumerate() {
                scores[indices[offset]] = score;
            }
        }
        let request_s = request_started.elapsed().as_secs_f64();
        infer_wall_s += request_s;
        latencies_ms.push(request_s * 1_000.0);
        real_tokens += lengths.iter().sum::<usize>() as u64;
        pair_count += request.documents.len();
        rows.push(RerankScores {
            id: request.id.clone(),
            scores,
        });
    }
    let padding_waste_fraction = padding_waste_fraction(real_tokens, padded_tokens);

    let (pearson, tie_aware_top1_agreement) = if let Some(reference_path) = &args.reference {
        let reference_rows = load_rerank_rows::<RerankScores>(reference_path)?;
        let (pearson, top1) = rerank_agreement(&rows, &reference_rows)?;
        ensure!(
            pearson >= args.min_pearson,
            "rerank Pearson {pearson:.9} below minimum {:.9}",
            args.min_pearson
        );
        ensure!(
            top1 >= args.min_top1_agreement,
            "rerank tie-aware top-1 agreement {top1:.6} below minimum {:.6}",
            args.min_top1_agreement
        );
        eprintln!("rerank gate: Pearson {pearson:.9}, tie-aware top-1 {top1:.6}");
        (Some(pearson), Some(top1))
    } else {
        (None, None)
    };

    if let Some(path) = &args.scores_out {
        write_jsonl(path, &rows)?;
    }
    latencies_ms.sort_by(f64::total_cmp);
    let result = RerankServingResult {
        lane: format!("owned-rt-{}", provider.name()),
        workload: "rerank-pairs-v1",
        model: args
            .model_label
            .clone()
            .unwrap_or_else(|| model.default_label(args.dtype)),
        provider: provider.name(),
        dtype: args.dtype.as_str(),
        execution: args.execution,
        shape_policy: args.shapes,
        bucket_policy_version: (args.shapes == Shapes::Bucketed).then_some(args.bucket_policy),
        bucket_shapes: if args.shapes == Shapes::Bucketed {
            bucket_shapes.to_vec()
        } else {
            Vec::new()
        },
        requests: requests.len(),
        pairs: pair_count,
        real_tokens,
        padded_tokens,
        padding_waste_fraction,
        infer_wall_s,
        pairs_per_s: pair_count as f64 / infer_wall_s,
        request_latency_p50_ms: percentile(&latencies_ms, 0.50),
        request_latency_p95_ms: percentile(&latencies_ms, 0.95),
        pearson,
        tie_aware_top1_agreement,
        package_cache: package_cache_stats(execution.package_root.as_deref())?,
        notes: format!(
            "{}; raw logits (no sigmoid), combined pair-length buckets, cold_load_s={cold_load_s:.6}",
            model.notes()
        ),
    };
    let out = args
        .out
        .as_ref()
        .context("standalone mode requires --out")?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(out, serde_json::to_string_pretty(&result)?)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn load_rerank_rows<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("read rerank JSONL {}", path.display()))?;
    content
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| {
            serde_json::from_str(line)
                .with_context(|| format!("parse rerank JSONL {}:{}", path.display(), index + 1))
        })
        .collect()
}

fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = std::io::BufWriter::new(fs::File::create(path)?);
    for row in rows {
        serde_json::to_writer(&mut writer, row)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
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

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn enforce_parity_gates(
    mean_cosine: f64,
    mean_top10_overlap: f64,
    min_cosine: f64,
    min_top10_overlap: f64,
    matched: usize,
    queries: usize,
) -> Result<()> {
    ensure!(
        mean_cosine >= min_cosine,
        "mean parity {mean_cosine:.8} below minimum {min_cosine:.8} over {matched} vectors"
    );
    ensure!(
        mean_top10_overlap >= min_top10_overlap,
        "mean top-10 rank overlap {mean_top10_overlap:.6} below minimum {min_top10_overlap:.6} over {queries} queries"
    );
    Ok(())
}

/// A loaded model family owns every policy that varies between embedding and rerank graphs.
///
/// Detection remains in the registry because it runs before a model exists. Keeping
/// loading, tokenizer behavior, token accounting, output heads, labels, and provider-hook
/// installation behind this object-safe seam lets workload runners stay independent of
/// model-specific graph types.
trait ModelFamily {
    fn family_name(&self) -> &'static str;

    fn configure_tokenizer(&self, tokenizer: &mut Tokenizer, max_length: usize) -> Result<()> {
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|error| anyhow::anyhow!("truncation: {error}"))?;
        Ok(())
    }

    fn token_length(&self, tokenizer: &Tokenizer, text: &str, max_length: usize) -> Result<usize>;

    fn embed_batch(
        &self,
        provider: &mut dyn KernelProvider,
        tokenizer: &Tokenizer,
        texts: &[&str],
        max_length: usize,
        shape: Option<BatchShape>,
    ) -> Result<Vec<Vec<f32>>>;

    fn rerank_pair_length(
        &self,
        _tokenizer: &Tokenizer,
        _query: &str,
        _document: &str,
    ) -> Result<usize> {
        bail!("{} does not support reranking", self.family_name())
    }

    fn rerank_batch(
        &self,
        _provider: &mut dyn KernelProvider,
        _tokenizer: &Tokenizer,
        _pairs: &[(&str, &str)],
        _max_length: usize,
        _shape: Option<BatchShape>,
    ) -> Result<Vec<f32>> {
        bail!("{} does not support reranking", self.family_name())
    }

    fn validate_reference_coverage(&self, _matched: usize, _produced: usize) -> Result<()> {
        Ok(())
    }

    fn default_label(&self, precision: Precision) -> String;
    fn notes(&self) -> String;
}

struct FamilyRegistration {
    detect: fn(&serde_json::Value) -> bool,
    load: fn(&Path, Precision) -> Result<Box<dyn ModelFamily>>,
}

fn load_model_family(path: &Path, precision: Precision) -> Result<Box<dyn ModelFamily>> {
    let root = resolve_model_root(path)?;
    let config_path = root.join("config.json");
    let config: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(&config_path)
            .with_context(|| format!("read config {}", config_path.display()))?,
    )
    .with_context(|| format!("parse config {}", config_path.display()))?;
    let registry = [
        FamilyRegistration {
            detect: modernbert::detect_config,
            load: modernbert::load_family,
        },
        FamilyRegistration {
            detect: qwen3::detect_config,
            load: qwen3::load_family,
        },
        FamilyRegistration {
            detect: detect_minilm_config,
            load: |path, precision| Ok(Box::new(BertModel::load(path, precision)?)),
        },
    ];
    let registration = registry
        .iter()
        .find(|registration| (registration.detect)(&config))
        .context("config.json does not describe a supported embedding model family")?;
    (registration.load)(path, precision)
}

fn detect_minilm_config(config: &serde_json::Value) -> bool {
    config.get("model_type").and_then(serde_json::Value::as_str) == Some("bert")
}

fn make_provider(
    device: DeviceArg,
    dtype: Precision,
    execution: MetalExecutionConfig,
    cuda_graphs: bool,
    vulkan_gemm: VulkanGemm,
    vulkan_pipeline_cache: Option<PathBuf>,
) -> Result<Box<dyn KernelProvider>> {
    match device {
        DeviceArg::Cpu => Ok(Box::new(CpuProvider)),
        DeviceArg::Metal => Ok(Box::new(MetalProvider::new_with_config(dtype, execution)?)),
        DeviceArg::Cuda => Ok(Box::new(CudaProvider::new(dtype, execution, cuda_graphs)?)),
        DeviceArg::Vulkan => Ok(Box::new(VulkanProvider::new(
            dtype,
            execution,
            vulkan_gemm,
            vulkan_pipeline_cache,
        )?)),
    }
}

fn bucket_shapes(max_length: usize, attention_units: usize, bucket_policy: u32) -> Vec<BatchShape> {
    let mut sequence_lengths = BUCKET_SEQUENCE_LADDER
        .iter()
        .copied()
        .take_while(|&seq| seq < max_length)
        .collect::<Vec<_>>();
    sequence_lengths.push(max_length);
    sequence_lengths.sort_unstable();
    sequence_lengths.dedup();
    sequence_lengths
        .into_iter()
        .enumerate()
        .map(|(index, seq)| {
            let policy_rows = match bucket_policy {
                1 => BUCKET_V1_MAX_BATCH_ROWS,
                2 => BUCKET_V2_BATCH_ROW_LADDER
                    .get(index)
                    .copied()
                    .unwrap_or(BUCKET_V1_MAX_BATCH_ROWS),
                _ => unreachable!("bucket policy is validated by clap"),
            };
            BatchShape {
                batch: policy_rows.min((attention_units / seq.saturating_mul(seq)).max(1)),
                seq,
            }
        })
        .collect()
}

fn covering_bucket(length: usize, buckets: &[BatchShape]) -> BatchShape {
    buckets
        .iter()
        .copied()
        .find(|shape| shape.seq >= length)
        .expect("bucket policy is capped by max_length")
}

fn exact_batch_ranges(
    order: &[usize],
    lengths: &[usize],
    attention_units: usize,
) -> Vec<std::ops::Range<usize>> {
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
    ranges
}

fn planned_batches(
    order: &[usize],
    lengths: &[usize],
    attention_units: usize,
    shapes: Shapes,
    buckets: &[BatchShape],
) -> Vec<PlannedBatch> {
    if shapes == Shapes::Exact {
        let ranges = exact_batch_ranges(order, lengths, attention_units);
        return ranges
            .into_iter()
            .map(|range| {
                let shape = BatchShape {
                    batch: range.len(),
                    seq: order[range.clone()]
                        .iter()
                        .map(|&index| lengths[index])
                        .max()
                        .unwrap_or(1),
                };
                PlannedBatch { range, shape }
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
        let shape = covering_bucket(lengths[order[end - 1]], buckets);
        batches.push(PlannedBatch {
            range: start..end,
            shape,
        });
        start = end;
    }
    batches
}

#[allow(clippy::too_many_arguments)]
fn run_workload(
    model: &dyn ModelFamily,
    provider: &mut dyn KernelProvider,
    tokenizer: &Tokenizer,
    chunks: &[Chunk],
    lengths: &[usize],
    order: &[usize],
    batches: &[PlannedBatch],
    max_length: usize,
    shapes: Shapes,
) -> Result<WorkloadResult> {
    let started = Instant::now();
    let mut input_tokens = 0u64;
    let mut padded_tokens = 0u64;
    let mut produced_vectors = Vec::with_capacity(chunks.len());
    for batch in batches {
        let batch_indices = &order[batch.range.clone()];
        let batch_texts = batch_indices
            .iter()
            .map(|&index| chunks[index].text.as_str())
            .collect::<Vec<_>>();
        let vectors = model.embed_batch(
            provider,
            tokenizer,
            &batch_texts,
            max_length,
            (shapes == Shapes::Bucketed).then_some(batch.shape),
        )?;
        ensure!(
            vectors.len() == batch_indices.len(),
            "model returned {} vectors for {} real bucket rows",
            vectors.len(),
            batch_indices.len()
        );
        padded_tokens += (batch.shape.batch * batch.shape.seq) as u64;
        for (offset, vector) in vectors.into_iter().enumerate() {
            let original_index = batch_indices[offset];
            input_tokens += lengths[original_index] as u64;
            produced_vectors.push((chunks[original_index].id.clone(), vector));
        }
    }
    Ok(WorkloadResult {
        infer_wall_s: started.elapsed().as_secs_f64(),
        input_tokens,
        padded_tokens,
        produced_vectors,
    })
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

fn padding_waste_fraction(real_tokens: u64, padded_tokens: u64) -> f64 {
    if padded_tokens == 0 {
        0.0
    } else {
        padded_tokens.saturating_sub(real_tokens) as f64 / padded_tokens as f64
    }
}

fn package_cache_stats(root: Option<&Path>) -> Result<PackageCacheStats> {
    fn directory_bytes(path: &Path) -> Result<u64> {
        let mut bytes = 0u64;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                bytes += directory_bytes(&entry.path())?;
            } else {
                bytes += metadata.len();
            }
        }
        Ok(bytes)
    }

    let Some(root) = root else {
        return Ok(PackageCacheStats::default());
    };
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

fn write_vectors(path: &Path, vectors: &[(String, Vec<f32>)]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = std::io::BufWriter::new(fs::File::create(path)?);
    for (id, vector) in vectors {
        serde_json::to_writer(&mut writer, &serde_json::json!({ "id": id, "vec": vector }))?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

#[derive(Clone)]
enum BlockBackend {
    Metal,
    Cuda {
        graphs: bool,
    },
    Vulkan {
        gemm: VulkanGemm,
        pipeline_cache: Option<PathBuf>,
    },
}

type BlockContextFactory =
    fn(Precision, MetalExecutionConfig, BlockBackend) -> Result<Box<dyn Any>>;

/// A block request keeps family-specific graph inputs inside the family while the
/// provider owns context lifetime and reuse. Because embedding graphs have different
/// typed parameters, erasing those parameters into a universal tensor schema would
/// invent unsupported generality; a family key plus a typed context callback gives all
/// providers one block-level dispatch surface without central family matches.
struct BlockForwardRequest<'a> {
    family: &'static str,
    create_context: BlockContextFactory,
    run: &'a mut dyn FnMut(&mut dyn Any) -> Result<()>,
}

#[allow(clippy::too_many_arguments)]
trait KernelProvider {
    fn name(&self) -> &'static str;

    fn matmul(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
    ) -> Result<()>;

    fn matmul_static_rhs(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
    ) -> Result<()> {
        self.matmul(m, n, k, a, b, b_layout, c)
    }

    fn block_forward(&mut self, _request: BlockForwardRequest<'_>) -> Result<bool> {
        Ok(false)
    }

    fn eager_shape_preload(&self) -> bool {
        false
    }

    fn take_pooled_output(&mut self) -> Option<Vec<Vec<f32>>> {
        None
    }

    fn layer_norm(
        &mut self,
        rows: usize,
        hidden: usize,
        data: &mut [f32],
        weight: &[f32],
        bias: &[f32],
        eps: f32,
    ) -> Result<()> {
        ensure!(
            data.len() == rows * hidden,
            "layer_norm data shape mismatch"
        );
        ensure!(
            weight.len() == hidden && bias.len() == hidden,
            "layer_norm parameter shape mismatch"
        );
        for row in 0..rows {
            let start = row * hidden;
            let row_data = &mut data[start..start + hidden];
            let mean = row_data.iter().copied().sum::<f32>() / hidden as f32;
            let var = row_data
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / hidden as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for i in 0..hidden {
                row_data[i] = (row_data[i] - mean) * inv * weight[i] + bias[i];
            }
        }
        Ok(())
    }
}

#[derive(Copy, Clone)]
enum BLayout {
    RowMajorKn,
    RowMajorNkTransposed,
}

struct CpuProvider;

impl KernelProvider for CpuProvider {
    fn name(&self) -> &'static str {
        "cpu-accelerate"
    }

    fn matmul(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
    ) -> Result<()> {
        ensure!(a.len() == m * k, "matmul A shape mismatch");
        ensure!(b.len() == n * k, "matmul B shape mismatch");
        ensure!(c.len() == m * n, "matmul C shape mismatch");
        matmul_impl(m, n, k, a, b, b_layout, c);
        Ok(())
    }
}

struct MetalProvider {
    context: metal_backend::MpsGraphContext,
    block_contexts: HashMap<&'static str, Box<dyn Any>>,
    dtype: Precision,
    execution: MetalExecutionConfig,
}

impl MetalProvider {
    #[cfg(test)]
    fn new(dtype: Precision) -> Result<Self> {
        Self::new_with_config(
            dtype,
            MetalExecutionConfig {
                execution: Execution::Lazy,
                package_root: None,
            },
        )
    }

    fn new_with_config(dtype: Precision, execution: MetalExecutionConfig) -> Result<Self> {
        Ok(Self {
            context: metal_backend::MpsGraphContext::new_with_config(execution.clone())?,
            block_contexts: HashMap::new(),
            dtype,
            execution,
        })
    }
}

struct CudaProvider {
    block_contexts: HashMap<&'static str, Box<dyn Any>>,
    dtype: Precision,
    execution: MetalExecutionConfig,
    graphs: bool,
}

struct VulkanProvider {
    block_contexts: HashMap<&'static str, Box<dyn Any>>,
    dtype: Precision,
    execution: MetalExecutionConfig,
    gemm: VulkanGemm,
    pipeline_cache: Option<PathBuf>,
}

impl CudaProvider {
    fn new(dtype: Precision, execution: MetalExecutionConfig, graphs: bool) -> Result<Self> {
        cuda_backend::ensure_available()?;
        Ok(Self {
            block_contexts: HashMap::new(),
            dtype,
            execution,
            graphs,
        })
    }
}

impl VulkanProvider {
    fn new(
        dtype: Precision,
        execution: MetalExecutionConfig,
        gemm: VulkanGemm,
        pipeline_cache: Option<PathBuf>,
    ) -> Result<Self> {
        ensure!(
            matches!(dtype, Precision::F16),
            "MiniLM Vulkan requires --dtype f16"
        );
        Ok(Self {
            block_contexts: HashMap::new(),
            dtype,
            execution,
            gemm,
            pipeline_cache,
        })
    }
}

impl KernelProvider for VulkanProvider {
    fn name(&self) -> &'static str {
        match self.gemm {
            VulkanGemm::Plain => "vulkan-plain-fused-family-command-buffer",
            VulkanGemm::Cooperative => "vulkan-cooperative-matrix-family-command-buffer",
        }
    }

    fn matmul(
        &mut self,
        _m: usize,
        _n: usize,
        _k: usize,
        _a: &[f32],
        _b: &[f32],
        _b_layout: BLayout,
        _c: &mut [f32],
    ) -> Result<()> {
        bail!("Vulkan requires the MiniLM family-resident block path")
    }

    fn block_forward(&mut self, request: BlockForwardRequest<'_>) -> Result<bool> {
        ensure!(
            request.family == "minilm",
            "Vulkan day-1 supports MiniLM only"
        );
        if !self.block_contexts.contains_key(request.family) {
            let context = (request.create_context)(
                self.dtype,
                self.execution.clone(),
                BlockBackend::Vulkan {
                    gemm: self.gemm,
                    pipeline_cache: self.pipeline_cache.clone(),
                },
            )?;
            self.block_contexts.insert(request.family, context);
        }
        let context = self
            .block_contexts
            .get_mut(request.family)
            .expect("block context inserted above");
        (request.run)(context.as_mut())?;
        Ok(true)
    }

    fn eager_shape_preload(&self) -> bool {
        true
    }

    fn take_pooled_output(&mut self) -> Option<Vec<Vec<f32>>> {
        self.block_contexts
            .get_mut("minilm")
            .and_then(|context| context.downcast_mut::<MiniLmBlockContext>())
            .and_then(|context| context.last_pooled.take())
    }
}

impl KernelProvider for CudaProvider {
    fn name(&self) -> &'static str {
        if self.graphs {
            "cuda-cublaslt-family-graph"
        } else {
            "cuda-cublaslt-family-uncaptured"
        }
    }

    fn matmul(
        &mut self,
        _m: usize,
        _n: usize,
        _k: usize,
        _a: &[f32],
        _b: &[f32],
        _b_layout: BLayout,
        _c: &mut [f32],
    ) -> Result<()> {
        bail!("CUDA requires a family-resident block path")
    }

    fn block_forward(&mut self, request: BlockForwardRequest<'_>) -> Result<bool> {
        if !self.block_contexts.contains_key(request.family) {
            let context = (request.create_context)(
                self.dtype,
                self.execution.clone(),
                BlockBackend::Cuda {
                    graphs: self.graphs,
                },
            )?;
            self.block_contexts.insert(request.family, context);
        }
        let context = self
            .block_contexts
            .get_mut(request.family)
            .expect("block context inserted above");
        (request.run)(context.as_mut())?;
        Ok(true)
    }

    fn eager_shape_preload(&self) -> bool {
        true
    }

    fn take_pooled_output(&mut self) -> Option<Vec<Vec<f32>>> {
        self.block_contexts
            .get_mut("minilm")
            .and_then(|context| context.downcast_mut::<MiniLmBlockContext>())
            .and_then(|context| context.last_pooled.take())
    }
}

impl KernelProvider for MetalProvider {
    fn name(&self) -> &'static str {
        match self.dtype {
            Precision::F32 => "metal-mpsgraph-resident-encoder-fp32",
            Precision::F16 => "metal-mpsgraph-resident-encoder-f16",
        }
    }

    fn matmul(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
    ) -> Result<()> {
        ensure!(a.len() == m * k, "matmul A shape mismatch");
        ensure!(b.len() == n * k, "matmul B shape mismatch");
        ensure!(c.len() == m * n, "matmul C shape mismatch");
        self.context
            .matmul(m, n, k, a, b, b_layout, c, false, self.dtype)
    }

    fn matmul_static_rhs(
        &mut self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        b_layout: BLayout,
        c: &mut [f32],
    ) -> Result<()> {
        ensure!(a.len() == m * k, "matmul A shape mismatch");
        ensure!(b.len() == n * k, "matmul B shape mismatch");
        ensure!(c.len() == m * n, "matmul C shape mismatch");
        self.context
            .matmul(m, n, k, a, b, b_layout, c, true, self.dtype)
    }

    fn block_forward(&mut self, request: BlockForwardRequest<'_>) -> Result<bool> {
        if !self.block_contexts.contains_key(request.family) {
            let context =
                (request.create_context)(self.dtype, self.execution.clone(), BlockBackend::Metal)?;
            self.block_contexts.insert(request.family, context);
        }
        let context = self
            .block_contexts
            .get_mut(request.family)
            .expect("block context inserted above");
        (request.run)(context.as_mut())?;
        Ok(true)
    }
}

#[cfg(target_os = "macos")]
mod metal_backend {
    use std::ffi::{c_char, c_void, CStr, CString};
    use std::ptr::NonNull;

    use anyhow::{bail, ensure, Result};

    use super::{
        decode_f16_bits, encode_f16_bits, BLayout, EncoderLayer, Execution, MetalExecutionConfig,
        Precision,
    };

    #[repr(C)]
    struct SynapseMpsEncoderLayerParams {
        query_weight: *const c_void,
        query_bias: *const c_void,
        key_weight: *const c_void,
        key_bias: *const c_void,
        value_weight: *const c_void,
        value_bias: *const c_void,
        attention_output_weight: *const c_void,
        attention_output_bias: *const c_void,
        attention_ln_weight: *const c_void,
        attention_ln_bias: *const c_void,
        intermediate_weight: *const c_void,
        intermediate_bias: *const c_void,
        output_weight: *const c_void,
        output_bias: *const c_void,
        output_ln_weight: *const c_void,
        output_ln_bias: *const c_void,
    }

    #[repr(i32)]
    #[derive(Copy, Clone)]
    enum SynapseMpsDType {
        Float32 = 0,
        Float16 = 1,
    }

    impl From<Precision> for SynapseMpsDType {
        fn from(value: Precision) -> Self {
            match value {
                Precision::F32 => Self::Float32,
                Precision::F16 => Self::Float16,
            }
        }
    }

    pub struct MpsGraphContext {
        raw: NonNull<c_void>,
        execution: MetalExecutionConfig,
    }

    impl MpsGraphContext {
        pub fn new_with_config(execution: MetalExecutionConfig) -> Result<Self> {
            let raw = unsafe { synapse_mps_context_new() };
            let raw = NonNull::new(raw).ok_or_else(last_error)?;
            Ok(Self { raw, execution })
        }

        #[allow(clippy::too_many_arguments)]
        pub fn matmul(
            &mut self,
            m: usize,
            n: usize,
            k: usize,
            a: &[f32],
            b: &[f32],
            b_layout: BLayout,
            c: &mut [f32],
            cache_rhs: bool,
            dtype: Precision,
        ) -> Result<()> {
            let b_is_row_major_nk = match b_layout {
                BLayout::RowMajorKn => 0,
                BLayout::RowMajorNkTransposed => 1,
            };
            let ffi_dtype = SynapseMpsDType::from(dtype) as i32;
            let status = match dtype {
                Precision::F32 => unsafe {
                    synapse_mps_matmul(
                        self.raw.as_ptr(),
                        m as u64,
                        n as u64,
                        k as u64,
                        a.as_ptr().cast(),
                        b.as_ptr().cast(),
                        ffi_dtype,
                        b_is_row_major_nk,
                        c.as_mut_ptr().cast(),
                        i32::from(cache_rhs),
                    )
                },
                Precision::F16 => {
                    let a_half = encode_f16_bits(a);
                    let b_half = encode_f16_bits(b);
                    let mut output_half = vec![0u16; c.len()];
                    let status = unsafe {
                        synapse_mps_matmul(
                            self.raw.as_ptr(),
                            m as u64,
                            n as u64,
                            k as u64,
                            a_half.as_ptr().cast(),
                            b_half.as_ptr().cast(),
                            ffi_dtype,
                            b_is_row_major_nk,
                            output_half.as_mut_ptr().cast(),
                            i32::from(cache_rhs),
                        )
                    };
                    if status == 0 {
                        c.copy_from_slice(&decode_f16_bits(&output_half));
                    }
                    status
                }
            };
            if status != 0 {
                bail!(
                    "MPSGraph matmul failed with status {status}: {}",
                    last_error()
                );
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encoder_forward(
            &mut self,
            hidden_states: &mut [f32],
            attention_mask: &[u8],
            batch: usize,
            seq: usize,
            hidden: usize,
            heads: usize,
            intermediate: usize,
            layer_norm_eps: f32,
            layers: &[EncoderLayer],
            dtype: Precision,
        ) -> Result<()> {
            ensure!(
                batch > 0 && seq > 0 && hidden > 0 && heads > 0 && intermediate > 0,
                "encoder dimensions must be non-zero"
            );
            ensure!(hidden % heads == 0, "hidden size must divide heads");
            ensure!(
                hidden_states.len() == batch * seq * hidden,
                "encoder hidden shape mismatch"
            );
            ensure!(
                attention_mask.len() == batch * seq,
                "encoder mask shape mismatch"
            );
            ensure!(!layers.is_empty(), "encoder requires at least one layer");

            let hidden_hidden = hidden * hidden;
            let intermediate_hidden = intermediate * hidden;
            let hidden_intermediate = hidden * intermediate;
            for (index, layer) in layers.iter().enumerate() {
                ensure!(
                    layer.query.weight.data.len() == hidden_hidden
                        && layer.key.weight.data.len() == hidden_hidden
                        && layer.value.weight.data.len() == hidden_hidden
                        && layer.attention_output.weight.data.len() == hidden_hidden,
                    "encoder layer {index} attention weight shape mismatch"
                );
                ensure!(
                    layer.query.bias.len() == hidden
                        && layer.key.bias.len() == hidden
                        && layer.value.bias.len() == hidden
                        && layer.attention_output.bias.len() == hidden
                        && layer.attention_ln_weight.len() == hidden
                        && layer.attention_ln_bias.len() == hidden
                        && layer.output_ln_weight.len() == hidden
                        && layer.output_ln_bias.len() == hidden,
                    "encoder layer {index} hidden vector shape mismatch"
                );
                ensure!(
                    layer.intermediate.weight.data.len() == intermediate_hidden
                        && layer.intermediate.bias.len() == intermediate,
                    "encoder layer {index} intermediate shape mismatch"
                );
                ensure!(
                    layer.output.weight.data.len() == hidden_intermediate
                        && layer.output.bias.len() == hidden,
                    "encoder layer {index} output shape mismatch"
                );
            }

            let additive_mask: Vec<f32> = attention_mask
                .iter()
                .map(|&mask| if mask == 0 { -10_000.0 } else { 0.0 })
                .collect();
            let params: Vec<SynapseMpsEncoderLayerParams> = match dtype {
                Precision::F32 => layers
                    .iter()
                    .map(|layer| SynapseMpsEncoderLayerParams {
                        query_weight: layer.query.weight.data.as_ptr().cast(),
                        query_bias: layer.query.bias.as_slice().as_ptr().cast(),
                        key_weight: layer.key.weight.data.as_ptr().cast(),
                        key_bias: layer.key.bias.as_slice().as_ptr().cast(),
                        value_weight: layer.value.weight.data.as_ptr().cast(),
                        value_bias: layer.value.bias.as_slice().as_ptr().cast(),
                        attention_output_weight: layer.attention_output.weight.data.as_ptr().cast(),
                        attention_output_bias: layer
                            .attention_output
                            .bias
                            .as_slice()
                            .as_ptr()
                            .cast(),
                        attention_ln_weight: layer.attention_ln_weight.as_slice().as_ptr().cast(),
                        attention_ln_bias: layer.attention_ln_bias.as_slice().as_ptr().cast(),
                        intermediate_weight: layer.intermediate.weight.data.as_ptr().cast(),
                        intermediate_bias: layer.intermediate.bias.as_slice().as_ptr().cast(),
                        output_weight: layer.output.weight.data.as_ptr().cast(),
                        output_bias: layer.output.bias.as_slice().as_ptr().cast(),
                        output_ln_weight: layer.output_ln_weight.as_slice().as_ptr().cast(),
                        output_ln_bias: layer.output_ln_bias.as_slice().as_ptr().cast(),
                    })
                    .collect(),
                Precision::F16 => layers
                    .iter()
                    .map(|layer| {
                        Ok(SynapseMpsEncoderLayerParams {
                            query_weight: layer.query.weight.metal_f16_bits()?.as_ptr().cast(),
                            query_bias: layer.query.bias.metal_f16_bits()?.as_ptr().cast(),
                            key_weight: layer.key.weight.metal_f16_bits()?.as_ptr().cast(),
                            key_bias: layer.key.bias.metal_f16_bits()?.as_ptr().cast(),
                            value_weight: layer.value.weight.metal_f16_bits()?.as_ptr().cast(),
                            value_bias: layer.value.bias.metal_f16_bits()?.as_ptr().cast(),
                            attention_output_weight: layer
                                .attention_output
                                .weight
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            attention_output_bias: layer
                                .attention_output
                                .bias
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            attention_ln_weight: layer
                                .attention_ln_weight
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            attention_ln_bias: layer
                                .attention_ln_bias
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            intermediate_weight: layer
                                .intermediate
                                .weight
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            intermediate_bias: layer
                                .intermediate
                                .bias
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            output_weight: layer.output.weight.metal_f16_bits()?.as_ptr().cast(),
                            output_bias: layer.output.bias.metal_f16_bits()?.as_ptr().cast(),
                            output_ln_weight: layer
                                .output_ln_weight
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast(),
                            output_ln_bias: layer.output_ln_bias.metal_f16_bits()?.as_ptr().cast(),
                        })
                    })
                    .collect::<Result<_>>()?,
            };
            let ffi_dtype = SynapseMpsDType::from(dtype) as i32;
            if matches!(self.execution.execution, Execution::Explicit) {
                let package = self.execution.package_path(batch, seq);
                let load_package = package.as_ref().is_some_and(|path| path.exists());
                let package_c = package
                    .as_ref()
                    .map(|path| CString::new(path.to_string_lossy().as_bytes()))
                    .transpose()?;
                let mut prepare_wall_s = 0.0;
                let mut specialize_wall_s = 0.0;
                let mut serialize_wall_s = 0.0;
                let prepare_status = unsafe {
                    synapse_mps_prepare_encoder(
                        self.raw.as_ptr(),
                        batch as u64,
                        seq as u64,
                        hidden as u64,
                        heads as u64,
                        intermediate as u64,
                        layers.len() as u64,
                        layer_norm_eps,
                        ffi_dtype,
                        0,
                        package_c
                            .as_ref()
                            .map_or(std::ptr::null(), |path| path.as_ptr()),
                        i32::from(load_package),
                        0,
                        &mut prepare_wall_s,
                        &mut specialize_wall_s,
                        &mut serialize_wall_s,
                    )
                };
                if prepare_status != 0 {
                    bail!(
                        "MPSGraph encoder preparation failed with status {prepare_status}: {}",
                        last_error()
                    );
                }
                eprintln!(
                    "Metal executable {} {}x{}: prepare={prepare_wall_s:.6}s specialize={specialize_wall_s:.6}s serialize={serialize_wall_s:.6}s",
                    if load_package { "loaded" } else { "compiled" }, batch, seq
                );
            }
            let status = match dtype {
                Precision::F32 => {
                    let mut output = vec![0.0f32; hidden_states.len()];
                    let status = unsafe {
                        synapse_mps_encoder_forward(
                            self.raw.as_ptr(),
                            batch as u64,
                            seq as u64,
                            hidden as u64,
                            heads as u64,
                            intermediate as u64,
                            layers.len() as u64,
                            layer_norm_eps,
                            ffi_dtype,
                            hidden_states.as_ptr().cast(),
                            additive_mask.as_ptr(),
                            output.as_mut_ptr().cast(),
                            params.as_ptr(),
                        )
                    };
                    if status == 0 {
                        hidden_states.copy_from_slice(&output);
                    }
                    status
                }
                Precision::F16 => {
                    let input_half = encode_f16_bits(hidden_states);
                    let mut output_half = vec![0u16; hidden_states.len()];
                    let status = unsafe {
                        synapse_mps_encoder_forward(
                            self.raw.as_ptr(),
                            batch as u64,
                            seq as u64,
                            hidden as u64,
                            heads as u64,
                            intermediate as u64,
                            layers.len() as u64,
                            layer_norm_eps,
                            ffi_dtype,
                            input_half.as_ptr().cast(),
                            additive_mask.as_ptr(),
                            output_half.as_mut_ptr().cast(),
                            params.as_ptr(),
                        )
                    };
                    if status == 0 {
                        hidden_states.copy_from_slice(&decode_f16_bits(&output_half));
                    }
                    status
                }
            };
            if status != 0 {
                bail!(
                    "MPSGraph encoder forward failed with status {status}: {}",
                    last_error()
                );
            }
            Ok(())
        }
    }

    impl Drop for MpsGraphContext {
        fn drop(&mut self) {
            unsafe { synapse_mps_context_free(self.raw.as_ptr()) }
        }
    }

    fn last_error() -> anyhow::Error {
        unsafe {
            let raw = synapse_mps_last_error();
            if raw.is_null() {
                return anyhow::anyhow!("unknown MPSGraph error");
            }
            let message = CStr::from_ptr(raw).to_string_lossy();
            if message.is_empty() {
                anyhow::anyhow!("unknown MPSGraph error")
            } else {
                anyhow::anyhow!(message.into_owned())
            }
        }
    }

    unsafe extern "C" {
        fn synapse_mps_context_new() -> *mut c_void;
        fn synapse_mps_context_free(context: *mut c_void);
        fn synapse_mps_matmul(
            context: *mut c_void,
            m: u64,
            n: u64,
            k: u64,
            a: *const c_void,
            b: *const c_void,
            dtype: i32,
            b_is_row_major_nk: i32,
            c: *mut c_void,
            cache_rhs: i32,
        ) -> i32;
        fn synapse_mps_prepare_encoder(
            context: *mut c_void,
            batch: u64,
            seq: u64,
            hidden: u64,
            heads: u64,
            intermediate: u64,
            layer_count: u64,
            layer_norm_eps: f32,
            dtype: i32,
            optimization_level: i32,
            package_path: *const c_char,
            load_package: i32,
            append_package: i32,
            prepare_wall_s: *mut f64,
            specialize_wall_s: *mut f64,
            serialize_wall_s: *mut f64,
        ) -> i32;
        fn synapse_mps_encoder_forward(
            context: *mut c_void,
            batch: u64,
            seq: u64,
            hidden: u64,
            heads: u64,
            intermediate: u64,
            layer_count: u64,
            layer_norm_eps: f32,
            dtype: i32,
            input: *const c_void,
            additive_mask: *const f32,
            output: *mut c_void,
            layers: *const SynapseMpsEncoderLayerParams,
        ) -> i32;
        fn synapse_mps_last_error() -> *const c_char;
    }
}

#[cfg(not(target_os = "macos"))]
mod metal_backend {
    use anyhow::{bail, Result};

    use super::{BLayout, EncoderLayer, Precision};

    pub struct MpsGraphContext;

    impl MpsGraphContext {
        pub fn new_with_config(_execution: super::MetalExecutionConfig) -> Result<Self> {
            bail!("Metal MPSGraph provider is only available on macOS")
        }

        #[allow(clippy::too_many_arguments)]
        pub fn matmul(
            &mut self,
            _m: usize,
            _n: usize,
            _k: usize,
            _a: &[f32],
            _b: &[f32],
            _b_layout: BLayout,
            _c: &mut [f32],
            _cache_rhs: bool,
            _dtype: Precision,
        ) -> Result<()> {
            bail!("Metal MPSGraph provider is only available on macOS")
        }

        #[allow(clippy::too_many_arguments)]
        pub fn encoder_forward(
            &mut self,
            _hidden_states: &mut [f32],
            _attention_mask: &[u8],
            _batch: usize,
            _seq: usize,
            _hidden: usize,
            _heads: usize,
            _intermediate: usize,
            _layer_norm_eps: f32,
            _layers: &[EncoderLayer],
            _dtype: Precision,
        ) -> Result<()> {
            bail!("Metal MPSGraph provider is only available on macOS")
        }
    }
}

#[cfg(target_os = "macos")]
fn matmul_impl(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    b_layout: BLayout,
    c: &mut [f32],
) {
    use std::os::raw::c_int;

    const CBLAS_ROW_MAJOR: c_int = 101;
    const CBLAS_NO_TRANS: c_int = 111;
    const CBLAS_TRANS: c_int = 112;

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
    }

    let trans_b = match b_layout {
        BLayout::RowMajorKn => CBLAS_NO_TRANS,
        BLayout::RowMajorNkTransposed => CBLAS_TRANS,
    };
    let ldb = match b_layout {
        BLayout::RowMajorKn => n,
        BLayout::RowMajorNkTransposed => k,
    } as c_int;

    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            trans_b,
            m as c_int,
            n as c_int,
            k as c_int,
            1.0,
            a.as_ptr(),
            k as c_int,
            b.as_ptr(),
            ldb,
            0.0,
            c.as_mut_ptr(),
            n as c_int,
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn matmul_impl(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    b_layout: BLayout,
    c: &mut [f32],
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let b_value = match b_layout {
                    BLayout::RowMajorKn => b[p * n + j],
                    BLayout::RowMajorNkTransposed => b[j * k + p],
                };
                sum += a[i * k + p] * b_value;
            }
            c[i * n + j] = sum;
        }
    }
}

#[derive(Clone, Debug)]
struct Tensor {
    dtype: TensorDType,
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<f32>,
    metal_f16_bits: Option<Vec<u16>>,
}

#[derive(Copy, Clone, Debug)]
enum TensorDType {
    F32,
}

#[derive(Clone, Debug)]
struct ParamVector {
    values: Vec<f32>,
    metal_f16_bits: Option<Vec<u16>>,
}

impl ParamVector {
    fn new(values: Vec<f32>) -> Self {
        Self {
            values,
            metal_f16_bits: None,
        }
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn as_slice(&self) -> &[f32] {
        &self.values
    }

    fn prepare_metal_f16(&mut self) {
        if self.metal_f16_bits.is_none() {
            self.metal_f16_bits = Some(encode_f16_bits(&self.values));
        }
    }

    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    fn metal_f16_bits(&self) -> Result<&[u16]> {
        self.metal_f16_bits
            .as_deref()
            .context("f16 mirror missing for Metal parameter")
    }
}

impl Tensor {
    fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let expected = shape.iter().product::<usize>();
        ensure!(
            expected == data.len(),
            "tensor data length {} does not match shape {:?}",
            data.len(),
            shape
        );
        Ok(Self {
            dtype: TensorDType::F32,
            strides: row_major_strides(&shape),
            shape,
            data,
            metal_f16_bits: None,
        })
    }

    fn zeros(shape: Vec<usize>) -> Self {
        let len = shape.iter().product::<usize>();
        Self {
            dtype: TensorDType::F32,
            strides: row_major_strides(&shape),
            shape,
            data: vec![0.0; len],
            metal_f16_bits: None,
        }
    }

    fn dim(&self, index: usize) -> usize {
        self.shape[index]
    }

    fn prepare_metal_f16(&mut self) {
        if self.metal_f16_bits.is_none() {
            self.metal_f16_bits = Some(encode_f16_bits(&self.data));
        }
    }

    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    fn metal_f16_bits(&self) -> Result<&[u16]> {
        self.metal_f16_bits
            .as_deref()
            .context("f16 mirror missing for Metal tensor")
    }

    fn as_vector(&self) -> Result<&[f32]> {
        self.ensure_f32_contiguous()?;
        ensure!(
            self.shape.len() == 1,
            "expected vector tensor, got {:?}",
            self.shape
        );
        Ok(&self.data)
    }

    fn matrix_shape(&self) -> Result<(usize, usize)> {
        self.ensure_f32_contiguous()?;
        ensure!(
            self.shape.len() == 2,
            "expected matrix tensor, got {:?}",
            self.shape
        );
        Ok((self.shape[0], self.shape[1]))
    }

    fn ensure_f32_contiguous(&self) -> Result<()> {
        ensure!(
            matches!(self.dtype, TensorDType::F32),
            "only f32 tensors are executable"
        );
        ensure!(
            self.strides == row_major_strides(&self.shape),
            "only contiguous row-major tensors are executable"
        );
        Ok(())
    }
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

fn encode_f16_bits(values: &[f32]) -> Vec<u16> {
    values
        .iter()
        .map(|&value| half::f16::from_f32(value).to_bits())
        .collect()
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn decode_f16_bits(values: &[u16]) -> Vec<f32> {
    values
        .iter()
        .map(|&value| half::f16::from_bits(value).to_f32())
        .collect()
}

#[derive(Deserialize)]
struct BertConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    layer_norm_eps: f32,
    #[serde(default = "default_hidden_act")]
    hidden_act: String,
    #[serde(default)]
    pad_token_id: u32,
}

fn default_type_vocab_size() -> usize {
    2
}

fn default_layer_norm_eps() -> f32 {
    1e-12
}

fn default_hidden_act() -> String {
    "gelu".to_string()
}

struct BertModel {
    config: BertConfig,
    embeddings: Embeddings,
    layers: Vec<EncoderLayer>,
}

struct Embeddings {
    word: Tensor,
    position: Tensor,
    token_type: Tensor,
    layer_norm_weight: ParamVector,
    layer_norm_bias: ParamVector,
}

struct EncoderLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    attention_output: Linear,
    attention_ln_weight: ParamVector,
    attention_ln_bias: ParamVector,
    intermediate: Linear,
    output: Linear,
    output_ln_weight: ParamVector,
    output_ln_bias: ParamVector,
}

struct Linear {
    weight: Tensor,
    bias: ParamVector,
}

impl EncoderLayer {
    fn prepare_metal_f16(&mut self) {
        self.query.prepare_metal_f16();
        self.key.prepare_metal_f16();
        self.value.prepare_metal_f16();
        self.attention_output.prepare_metal_f16();
        self.attention_ln_weight.prepare_metal_f16();
        self.attention_ln_bias.prepare_metal_f16();
        self.intermediate.prepare_metal_f16();
        self.output.prepare_metal_f16();
        self.output_ln_weight.prepare_metal_f16();
        self.output_ln_bias.prepare_metal_f16();
    }
}

impl Linear {
    fn prepare_metal_f16(&mut self) {
        self.weight.prepare_metal_f16();
        self.bias.prepare_metal_f16();
    }
}

impl BertModel {
    fn load(path: &Path, precision: Precision) -> Result<Self> {
        let model_root = resolve_model_root(path)?;
        let config_path = model_root.join("config.json");
        let config: BertConfig = serde_json::from_str(
            &fs::read_to_string(&config_path)
                .with_context(|| format!("read config {}", config_path.display()))?,
        )
        .with_context(|| format!("parse config {}", config_path.display()))?;
        ensure!(
            config.hidden_act == "gelu" || config.hidden_act == "gelu_new",
            "unsupported hidden_act {}",
            config.hidden_act
        );
        ensure!(
            config.hidden_size % config.num_attention_heads == 0,
            "hidden size must divide heads"
        );

        let tensors = load_safetensor_map(&model_root, path)?;
        let embeddings = Embeddings {
            word: get_tensor(&tensors, "embeddings.word_embeddings.weight")?,
            position: get_tensor(&tensors, "embeddings.position_embeddings.weight")?,
            token_type: get_tensor(&tensors, "embeddings.token_type_embeddings.weight")?,
            layer_norm_weight: ParamVector::new(
                get_tensor(&tensors, "embeddings.LayerNorm.weight")?
                    .as_vector()?
                    .to_vec(),
            ),
            layer_norm_bias: ParamVector::new(
                get_tensor(&tensors, "embeddings.LayerNorm.bias")?
                    .as_vector()?
                    .to_vec(),
            ),
        };
        ensure!(
            embeddings.word.shape == vec![config.vocab_size, config.hidden_size],
            "word embedding shape {:?} does not match config",
            embeddings.word.shape
        );
        ensure!(
            embeddings.position.dim(0) >= config.max_position_embeddings.min(512),
            "position embedding table unexpectedly short"
        );
        ensure!(
            embeddings.token_type.dim(0) >= config.type_vocab_size.min(1),
            "token type embedding table unexpectedly short"
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{layer_index}");
            layers.push(EncoderLayer {
                query: Linear::load(&tensors, &format!("{prefix}.attention.self.query"))?,
                key: Linear::load(&tensors, &format!("{prefix}.attention.self.key"))?,
                value: Linear::load(&tensors, &format!("{prefix}.attention.self.value"))?,
                attention_output: Linear::load(
                    &tensors,
                    &format!("{prefix}.attention.output.dense"),
                )?,
                attention_ln_weight: ParamVector::new(
                    get_tensor(
                        &tensors,
                        &format!("{prefix}.attention.output.LayerNorm.weight"),
                    )?
                    .as_vector()?
                    .to_vec(),
                ),
                attention_ln_bias: ParamVector::new(
                    get_tensor(
                        &tensors,
                        &format!("{prefix}.attention.output.LayerNorm.bias"),
                    )?
                    .as_vector()?
                    .to_vec(),
                ),
                intermediate: Linear::load(&tensors, &format!("{prefix}.intermediate.dense"))?,
                output: Linear::load(&tensors, &format!("{prefix}.output.dense"))?,
                output_ln_weight: ParamVector::new(
                    get_tensor(&tensors, &format!("{prefix}.output.LayerNorm.weight"))?
                        .as_vector()?
                        .to_vec(),
                ),
                output_ln_bias: ParamVector::new(
                    get_tensor(&tensors, &format!("{prefix}.output.LayerNorm.bias"))?
                        .as_vector()?
                        .to_vec(),
                ),
            });
        }

        if matches!(precision, Precision::F16) {
            for layer in &mut layers {
                layer.prepare_metal_f16();
            }
        }

        Ok(Self {
            config,
            embeddings,
            layers,
        })
    }

    fn embed_batch(
        &self,
        provider: &mut dyn KernelProvider,
        tokenizer: &Tokenizer,
        texts: &[&str],
        shape: Option<BatchShape>,
    ) -> Result<Vec<Vec<f32>>> {
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|error| anyhow::anyhow!("encode_batch: {error}"))?;
        let real_batch = encodings.len();
        ensure!(real_batch > 0, "MiniLM batch must not be empty");
        let real_seq = encodings
            .iter()
            .map(|encoding| encoding.get_ids().len())
            .max()
            .unwrap_or(1)
            .max(1);
        let target = shape.unwrap_or(BatchShape {
            batch: real_batch,
            seq: real_seq,
        });
        ensure!(
            target.batch >= real_batch && target.seq >= real_seq,
            "MiniLM target shape {}x{} does not cover input {}x{}",
            target.batch,
            target.seq,
            real_batch,
            real_seq
        );
        let (batch, seq) = (target.batch, target.seq);
        let mut input_ids = vec![self.config.pad_token_id; batch * seq];
        let mut attention_mask = vec![0u8; batch * seq];
        for (row, encoding) in encodings.iter().enumerate() {
            for (col, (&id, &mask)) in encoding
                .get_ids()
                .iter()
                .zip(encoding.get_attention_mask())
                .enumerate()
            {
                input_ids[row * seq + col] = id;
                attention_mask[row * seq + col] = mask as u8;
            }
        }

        let hidden = self.forward(provider, &input_ids, &attention_mask, batch, seq)?;
        if let Some(mut pooled) = provider.take_pooled_output() {
            ensure!(
                pooled.len() == batch
                    && pooled
                        .iter()
                        .all(|row| row.len() == self.config.hidden_size),
                "provider returned pooled vectors with the wrong shape"
            );
            pooled.truncate(real_batch);
            return Ok(pooled);
        }
        let mut pooled = mean_pool_l2(
            &hidden,
            &attention_mask,
            batch,
            seq,
            self.config.hidden_size,
        );
        pooled.truncate(real_batch);
        Ok(pooled)
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        input_ids: &[u32],
        attention_mask: &[u8],
        batch: usize,
        seq: usize,
    ) -> Result<Tensor> {
        let hidden = self.config.hidden_size;
        let heads = self.config.num_attention_heads;
        let rows = batch * seq;
        let mut x = Tensor::zeros(vec![batch, seq, hidden]);

        for b in 0..batch {
            for s in 0..seq {
                let token_id = input_ids[b * seq + s] as usize;
                ensure!(
                    token_id < self.embeddings.word.dim(0),
                    "token id {token_id} outside vocab"
                );
                ensure!(
                    s < self.embeddings.position.dim(0),
                    "position {s} outside position embeddings"
                );
                let out = (b * seq + s) * hidden;
                for h in 0..hidden {
                    x.data[out + h] = self.embeddings.word.data[token_id * hidden + h]
                        + self.embeddings.position.data[s * hidden + h]
                        + self.embeddings.token_type.data[h];
                }
            }
        }
        provider.layer_norm(
            rows,
            hidden,
            &mut x.data,
            self.embeddings.layer_norm_weight.as_slice(),
            self.embeddings.layer_norm_bias.as_slice(),
            self.config.layer_norm_eps,
        )?;

        let mut run = |context: &mut dyn Any| {
            let context = context
                .downcast_mut::<MiniLmBlockContext>()
                .context("MiniLM provider returned the wrong block context type")?;
            context.last_pooled = context.graph.encoder_forward(
                &mut x.data,
                attention_mask,
                batch,
                seq,
                hidden,
                heads,
                self.config.intermediate_size,
                self.config.layer_norm_eps,
                &self.layers,
                context.precision,
            )?;
            Ok(())
        };
        if provider.block_forward(BlockForwardRequest {
            family: self.family_name(),
            create_context: new_minilm_block_context,
            run: &mut run,
        })? {
            return Ok(x);
        }

        encoder_layers_scalar_forward(
            provider,
            &mut x.data,
            attention_mask,
            batch,
            seq,
            hidden,
            heads,
            self.config.intermediate_size,
            self.config.layer_norm_eps,
            &self.layers,
        )?;
        Ok(x)
    }
}

impl ModelFamily for BertModel {
    fn family_name(&self) -> &'static str {
        "minilm"
    }

    fn token_length(&self, tokenizer: &Tokenizer, text: &str, _max_length: usize) -> Result<usize> {
        tokenizer
            .encode(text, true)
            .map(|encoding| encoding.get_ids().len())
            .map_err(|error| anyhow::anyhow!("encode: {error}"))
    }

    fn embed_batch(
        &self,
        provider: &mut dyn KernelProvider,
        tokenizer: &Tokenizer,
        texts: &[&str],
        _max_length: usize,
        shape: Option<BatchShape>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_batch(provider, tokenizer, texts, shape)
    }

    fn default_label(&self, precision: Precision) -> String {
        format!("all-MiniLM-L6-v2@owned-rt-{}", precision.as_str())
    }

    fn notes(&self) -> String {
        "direct BERT encoder, provider-selected mean pool+l2".to_owned()
    }
}

fn new_minilm_block_context(
    precision: Precision,
    execution: MetalExecutionConfig,
    backend: BlockBackend,
) -> Result<Box<dyn Any>> {
    let graph = match backend {
        BlockBackend::Metal => {
            MiniLmBlockGraph::Metal(metal_backend::MpsGraphContext::new_with_config(execution)?)
        }
        BlockBackend::Cuda { graphs } => {
            ensure!(
                matches!(precision, Precision::F16),
                "MiniLM CUDA requires --dtype f16"
            );
            MiniLmBlockGraph::Cuda(cuda_backend::CudaContext::new(graphs)?)
        }
        BlockBackend::Vulkan {
            gemm,
            pipeline_cache,
        } => {
            ensure!(
                matches!(precision, Precision::F16),
                "MiniLM Vulkan requires --dtype f16"
            );
            MiniLmBlockGraph::Vulkan(vulkan_backend::VulkanContext::new(gemm, pipeline_cache)?)
        }
    };
    Ok(Box::new(MiniLmBlockContext {
        graph,
        precision,
        last_pooled: None,
    }))
}

enum MiniLmBlockGraph {
    Metal(metal_backend::MpsGraphContext),
    Cuda(cuda_backend::CudaContext),
    Vulkan(vulkan_backend::VulkanContext),
}

impl MiniLmBlockGraph {
    #[allow(clippy::too_many_arguments)]
    fn encoder_forward(
        &mut self,
        hidden_states: &mut [f32],
        attention_mask: &[u8],
        batch: usize,
        seq: usize,
        hidden: usize,
        heads: usize,
        intermediate: usize,
        layer_norm_eps: f32,
        layers: &[EncoderLayer],
        precision: Precision,
    ) -> Result<Option<Vec<Vec<f32>>>> {
        match self {
            Self::Metal(graph) => graph
                .encoder_forward(
                    hidden_states,
                    attention_mask,
                    batch,
                    seq,
                    hidden,
                    heads,
                    intermediate,
                    layer_norm_eps,
                    layers,
                    precision,
                )
                .map(|()| None),
            Self::Cuda(graph) => graph
                .encoder_forward(
                    hidden_states,
                    attention_mask,
                    batch,
                    seq,
                    hidden,
                    heads,
                    intermediate,
                    layer_norm_eps,
                    layers,
                )
                .map(Some),
            Self::Vulkan(graph) => graph
                .encoder_forward(
                    hidden_states,
                    attention_mask,
                    batch,
                    seq,
                    hidden,
                    heads,
                    intermediate,
                    layer_norm_eps,
                    layers,
                )
                .map(Some),
        }
    }
}

struct MiniLmBlockContext {
    graph: MiniLmBlockGraph,
    precision: Precision,
    last_pooled: Option<Vec<Vec<f32>>>,
}

#[allow(clippy::too_many_arguments)]
fn encoder_layers_scalar_forward(
    provider: &mut dyn KernelProvider,
    hidden_states: &mut [f32],
    attention_mask: &[u8],
    batch: usize,
    seq: usize,
    hidden: usize,
    heads: usize,
    intermediate_size: usize,
    layer_norm_eps: f32,
    layers: &[EncoderLayer],
) -> Result<()> {
    let rows = batch * seq;
    let head_dim = hidden / heads;
    let mut current = hidden_states.to_vec();
    for layer in layers {
        let residual = current.clone();
        let q = layer.query.forward(provider, rows, hidden, &current)?;
        let k = layer.key.forward(provider, rows, hidden, &current)?;
        let v = layer.value.forward(provider, rows, hidden, &current)?;
        let context = self_attention(
            provider,
            &q,
            &k,
            &v,
            attention_mask,
            batch,
            seq,
            heads,
            head_dim,
        )?;
        let mut attention_out = layer
            .attention_output
            .forward(provider, rows, hidden, &context)?;
        for (value, residual_value) in attention_out.iter_mut().zip(residual) {
            *value += residual_value;
        }
        provider.layer_norm(
            rows,
            hidden,
            &mut attention_out,
            layer.attention_ln_weight.as_slice(),
            layer.attention_ln_bias.as_slice(),
            layer_norm_eps,
        )?;

        let residual = attention_out.clone();
        let mut intermediate =
            layer
                .intermediate
                .forward(provider, rows, hidden, &attention_out)?;
        for value in &mut intermediate {
            *value = gelu(*value);
        }
        let mut output = layer
            .output
            .forward(provider, rows, intermediate_size, &intermediate)?;
        for (value, residual_value) in output.iter_mut().zip(residual) {
            *value += residual_value;
        }
        provider.layer_norm(
            rows,
            hidden,
            &mut output,
            layer.output_ln_weight.as_slice(),
            layer.output_ln_bias.as_slice(),
            layer_norm_eps,
        )?;
        current = output;
    }
    hidden_states.copy_from_slice(&current);
    Ok(())
}

impl Linear {
    fn load(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        let weight = get_tensor(tensors, &format!("{prefix}.weight"))?;
        let bias = ParamVector::new(
            get_tensor(tensors, &format!("{prefix}.bias"))?
                .as_vector()?
                .to_vec(),
        );
        Ok(Self { weight, bias })
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        rows: usize,
        input: usize,
        values: &[f32],
    ) -> Result<Vec<f32>> {
        let (output, weight_input) = self.weight.matrix_shape()?;
        ensure!(
            weight_input == input,
            "linear input mismatch: weight expects {weight_input}, got {input}"
        );
        ensure!(self.bias.len() == output, "linear bias shape mismatch");
        let bias = self.bias.as_slice();
        ensure!(values.len() == rows * input, "linear values shape mismatch");
        let mut out = vec![0.0f32; rows * output];
        provider.matmul_static_rhs(
            rows,
            output,
            input,
            values,
            &self.weight.data,
            BLayout::RowMajorNkTransposed,
            &mut out,
        )?;
        for row in 0..rows {
            let start = row * output;
            for col in 0..output {
                out[start + col] += bias[col];
            }
        }
        Ok(out)
    }
}

#[allow(clippy::too_many_arguments)]
fn self_attention(
    provider: &mut dyn KernelProvider,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attention_mask: &[u8],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let hidden = heads * head_dim;
    let mut context = vec![0.0f32; batch * seq * hidden];
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut q_head = vec![0.0f32; seq * head_dim];
    let mut k_head = vec![0.0f32; seq * head_dim];
    let mut v_head = vec![0.0f32; seq * head_dim];
    let mut scores = vec![0.0f32; seq * seq];
    let mut ctx_head = vec![0.0f32; seq * head_dim];

    for b in 0..batch {
        for head in 0..heads {
            for s in 0..seq {
                let source = (b * seq + s) * hidden + head * head_dim;
                let dest = s * head_dim;
                q_head[dest..dest + head_dim].copy_from_slice(&q[source..source + head_dim]);
                k_head[dest..dest + head_dim].copy_from_slice(&k[source..source + head_dim]);
                v_head[dest..dest + head_dim].copy_from_slice(&v[source..source + head_dim]);
            }

            provider.matmul(
                seq,
                seq,
                head_dim,
                &q_head,
                &k_head,
                BLayout::RowMajorNkTransposed,
                &mut scores,
            )?;
            for query_pos in 0..seq {
                let row_start = query_pos * seq;
                let row = &mut scores[row_start..row_start + seq];
                for key_pos in 0..seq {
                    row[key_pos] *= scale;
                    if attention_mask[b * seq + key_pos] == 0 {
                        row[key_pos] = -10_000.0;
                    }
                }
                softmax(row);
            }

            provider.matmul(
                seq,
                head_dim,
                seq,
                &scores,
                &v_head,
                BLayout::RowMajorKn,
                &mut ctx_head,
            )?;
            for s in 0..seq {
                let source = s * head_dim;
                let dest = (b * seq + s) * hidden + head * head_dim;
                context[dest..dest + head_dim]
                    .copy_from_slice(&ctx_head[source..source + head_dim]);
            }
        }
    }
    Ok(context)
}

fn softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for value in row.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    let inv_sum = 1.0 / sum.max(1e-20);
    for value in row {
        *value *= inv_sum;
    }
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value * std::f32::consts::FRAC_1_SQRT_2))
}

fn mean_pool_l2(
    hidden: &Tensor,
    attention_mask: &[u8],
    batch: usize,
    seq: usize,
    hidden_size: usize,
) -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut vector = vec![0.0f32; hidden_size];
        let mut count = 0.0f32;
        for s in 0..seq {
            if attention_mask[b * seq + s] == 1 {
                count += 1.0;
                let start = (b * seq + s) * hidden_size;
                for (value, hidden_value) in vector
                    .iter_mut()
                    .zip(&hidden.data[start..start + hidden_size])
                {
                    *value += *hidden_value;
                }
            }
        }
        let denom = count.max(1.0);
        for value in &mut vector {
            *value /= denom;
        }
        normalize_l2(&mut vector);
        vectors.push(vector);
    }
    vectors
}

fn normalize_l2(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-12);
    for value in vector {
        *value /= norm;
    }
}

fn load_safetensor_map(model_root: &Path, original_path: &Path) -> Result<HashMap<String, Tensor>> {
    if original_path.is_file()
        && original_path.extension().and_then(|value| value.to_str()) == Some("safetensors")
    {
        return load_single_safetensors_file(original_path);
    }

    let single_file = model_root.join("model.safetensors");
    if single_file.is_file() {
        return load_single_safetensors_file(&single_file);
    }

    let index_file = model_root.join("model.safetensors.index.json");
    if index_file.is_file() {
        #[derive(Deserialize)]
        struct SafetensorsIndex {
            weight_map: HashMap<String, String>,
        }
        let index: SafetensorsIndex = serde_json::from_str(
            &fs::read_to_string(&index_file)
                .with_context(|| format!("read safetensors index {}", index_file.display()))?,
        )
        .with_context(|| format!("parse safetensors index {}", index_file.display()))?;
        let mut merged = HashMap::new();
        let unique_files: HashSet<_> = index.weight_map.into_values().collect();
        for shard in unique_files {
            let shard_path = model_root.join(&shard);
            merged.extend(load_single_safetensors_file(&shard_path)?);
        }
        return Ok(merged);
    }

    bail!(
        "could not find model.safetensors or model.safetensors.index.json under {}",
        model_root.display()
    )
}

fn load_single_safetensors_file(path: &Path) -> Result<HashMap<String, Tensor>> {
    let bytes = fs::read(path).with_context(|| format!("read safetensors {}", path.display()))?;
    let safetensors = SafeTensors::deserialize(&bytes)
        .map_err(|error| anyhow::anyhow!("load safetensors {}: {error}", path.display()))?;
    let mut tensors = HashMap::new();
    for name in safetensors.names() {
        let view = safetensors
            .tensor(name)
            .map_err(|error| anyhow::anyhow!("read tensor {name}: {error}"))?;
        if matches!(
            view.dtype(),
            SafeDtype::F32 | SafeDtype::F16 | SafeDtype::BF16
        ) {
            tensors.insert(
                name.to_string(),
                tensor_from_view(view.dtype(), view.shape(), view.data())?,
            );
        }
    }
    Ok(tensors)
}

fn tensor_from_view(dtype: SafeDtype, shape: &[usize], bytes: &[u8]) -> Result<Tensor> {
    let values = match dtype {
        SafeDtype::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk_exact length")))
            .collect(),
        SafeDtype::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                half::f16::from_bits(u16::from_le_bytes(
                    chunk.try_into().expect("chunk_exact length"),
                ))
                .to_f32()
            })
            .collect(),
        SafeDtype::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                half::bf16::from_bits(u16::from_le_bytes(
                    chunk.try_into().expect("chunk_exact length"),
                ))
                .to_f32()
            })
            .collect(),
        other => bail!("unsupported safetensor dtype {other:?}; expected f32/f16/bf16"),
    };
    let mut tensor = Tensor::new(shape.to_vec(), values)?;
    if matches!(dtype, SafeDtype::F16) {
        tensor.metal_f16_bits = Some(
            bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("chunk_exact length")))
                .collect(),
        );
    }
    Ok(tensor)
}

fn resolve_model_root(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
        return path
            .parent()
            .map(Path::to_path_buf)
            .context("model file has no parent directory");
    }
    bail!(
        "model path {} is neither a directory nor a safetensors file",
        path.display()
    )
}

fn get_tensor(tensors: &HashMap<String, Tensor>, base_name: &str) -> Result<Tensor> {
    let candidates = [
        base_name.to_string(),
        format!("bert.{base_name}"),
        format!("model.{base_name}"),
        format!("model.bert.{base_name}"),
    ];
    for candidate in &candidates {
        if let Some(tensor) = tensors.get(candidate) {
            return Ok(tensor.clone());
        }
    }
    bail!("missing tensor; tried {}", candidates.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_execution_defaults_to_explicit_and_lazy_is_opt_in() {
        let base = [
            "spike-unified-rt",
            "--model",
            "model",
            "--tokenizer",
            "tokenizer.json",
            "--corpus",
            "corpus.jsonl",
            "--out",
            "result.json",
        ];
        let default = Args::try_parse_from(base).expect("parse default serving arguments");
        assert_eq!(default.execution, Execution::Explicit);
        assert_eq!(default.shapes, Shapes::Bucketed);
        assert_eq!(default.bucket_policy, 1);
        assert_eq!(default.passes, 1);
        let v2 = Args::try_parse_from(base.into_iter().chain(["--bucket-policy", "2"]))
            .expect("parse policy v2 override");
        assert_eq!(v2.bucket_policy, 2);
        assert!(Args::try_parse_from(base.into_iter().chain(["--bucket-policy", "3"])).is_err());
        let lazy = Args::try_parse_from(base.into_iter().chain(["--execution", "lazy"]))
            .expect("parse lazy execution override");
        assert_eq!(lazy.execution, Execution::Lazy);
    }

    #[test]
    fn bucket_policy_is_bounded_stable_and_capped_at_max_length() {
        let v1_shapes = bucket_shapes(512, 4_000_000, 1);
        assert_eq!(v1_shapes.len(), 10);
        assert_eq!(
            v1_shapes,
            BUCKET_SEQUENCE_LADDER
                .iter()
                .map(|&seq| BatchShape { batch: 8, seq })
                .collect::<Vec<_>>()
        );
        assert_eq!(
            bucket_shapes(512, 4_000_000, 2),
            BUCKET_SEQUENCE_LADDER
                .iter()
                .zip(BUCKET_V2_BATCH_ROW_LADDER)
                .map(|(&seq, &batch)| BatchShape { batch, seq })
                .collect::<Vec<_>>()
        );
        assert_eq!(
            bucket_shapes(150, 4_000_000, 2),
            vec![
                BatchShape { batch: 16, seq: 64 },
                BatchShape { batch: 16, seq: 96 },
                BatchShape {
                    batch: 16,
                    seq: 128,
                },
                BatchShape {
                    batch: 16,
                    seq: 150,
                },
            ]
        );
    }

    #[test]
    fn bucket_batcher_maps_to_covering_shapes_and_pads_tail_rows() {
        let lengths = vec![60, 60, 60, 60, 60, 60, 60, 60, 65, 65];
        let order = (0..lengths.len()).collect::<Vec<_>>();
        let buckets = bucket_shapes(512, 4_000_000, 1);
        let batches = planned_batches(&order, &lengths, 4_000_000, Shapes::Bucketed, &buckets);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].range, 0..8);
        assert_eq!(batches[0].shape, BatchShape { batch: 8, seq: 64 });
        assert_eq!(batches[1].range, 8..10);
        assert_eq!(batches[1].shape, BatchShape { batch: 8, seq: 96 });
    }

    #[test]
    fn package_cache_identity_includes_shape_policy() {
        assert_eq!(shape_cache_key(Shapes::Exact, 1), "shapes-exact");
        assert_eq!(shape_cache_key(Shapes::Bucketed, 1), "bucket-policy-v1");
        assert_eq!(shape_cache_key(Shapes::Bucketed, 2), "bucket-policy-v2");
    }

    #[test]
    fn package_cache_uses_one_path_per_shape() {
        let config = MetalExecutionConfig {
            execution: Execution::Explicit,
            package_root: Some(PathBuf::from("cache/model-graph-v2-f16-os")),
        };
        assert_eq!(
            config.package_path(8, 128),
            Some(PathBuf::from(
                "cache/model-graph-v2-f16-os/8x128.mpsgraphpackage"
            ))
        );
        assert_ne!(config.package_path(8, 128), config.package_path(4, 256));
    }

    #[test]
    fn rerank_gate_uses_pearson_and_tie_aware_top1() {
        let reference = vec![
            RerankScores {
                id: "q1".into(),
                scores: vec![2.0, 2.0, -1.0],
            },
            RerankScores {
                id: "q2".into(),
                scores: vec![0.0, 1.0, 3.0],
            },
        ];
        let candidate = vec![
            RerankScores {
                id: "q1".into(),
                scores: vec![1.999, 2.001, -0.999],
            },
            RerankScores {
                id: "q2".into(),
                scores: vec![0.001, 1.001, 3.001],
            },
        ];
        let (pearson, agreement) = rerank_agreement(&candidate, &reference).unwrap();
        assert!(pearson >= 0.999);
        assert_eq!(agreement, 1.0);
    }

    #[test]
    fn qwen3_parity_gate_enforces_certification_thresholds() {
        enforce_parity_gates(0.9999, 0.995, 0.9999, 0.995, 400, 400)
            .expect("Qwen3 certification boundary must pass");
        assert!(enforce_parity_gates(0.99989, 1.0, 0.9999, 0.995, 400, 400).is_err());
        assert!(enforce_parity_gates(1.0, 0.9949, 0.9999, 0.995, 400, 400).is_err());
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_close_with_tolerance(actual, expected, 1e-3);
    }

    fn assert_close_with_tolerance(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
            let diff = (left - right).abs();
            assert!(
                diff <= tolerance,
                "value {index} differs: actual={left}, expected={right}, diff={diff}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_provider_matches_cpu_for_row_major_rhs() {
        let mut metal = MetalProvider::new(Precision::F32).expect("create MPSGraph provider");
        let mut cpu = CpuProvider;
        let a = vec![1.0, 2.0, 3.0, 4.0, -2.0, 0.5];
        let b = vec![
            0.5, -1.0, 2.0, 1.5, 3.0, -0.5, -2.0, 0.25, 1.25, -1.5, 0.75, 2.5,
        ];
        let mut metal_out = vec![0.0; 8];
        let mut cpu_out = vec![0.0; 8];
        metal
            .matmul(2, 4, 3, &a, &b, BLayout::RowMajorKn, &mut metal_out)
            .expect("run MPSGraph matmul");
        cpu.matmul(2, 4, 3, &a, &b, BLayout::RowMajorKn, &mut cpu_out)
            .expect("run CPU matmul");
        assert_close(&metal_out, &cpu_out);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_provider_matches_cpu_for_transposed_rhs_storage() {
        let mut metal = MetalProvider::new(Precision::F32).expect("create MPSGraph provider");
        let mut cpu = CpuProvider;
        let a = vec![1.0, 2.0, 3.0, 4.0, -2.0, 0.5];
        let b = vec![
            0.5, 3.0, 1.25, -1.0, -0.5, -1.5, 2.0, -2.0, 0.75, 1.5, 0.25, 2.5,
        ];
        let mut metal_out = vec![0.0; 8];
        let mut cpu_out = vec![0.0; 8];
        metal
            .matmul(
                2,
                4,
                3,
                &a,
                &b,
                BLayout::RowMajorNkTransposed,
                &mut metal_out,
            )
            .expect("run MPSGraph matmul");
        cpu.matmul(2, 4, 3, &a, &b, BLayout::RowMajorNkTransposed, &mut cpu_out)
            .expect("run CPU matmul");
        assert_close(&metal_out, &cpu_out);
    }

    fn patterned_values(len: usize, scale: f32, bias: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((((index * 37) % 23) as f32) - 11.0) * scale + bias)
            .collect()
    }

    fn test_linear(output: usize, input: usize, scale: f32) -> Linear {
        Linear {
            weight: Tensor::new(
                vec![output, input],
                patterned_values(output * input, scale, 0.0),
            )
            .expect("linear test weight"),
            bias: ParamVector::new(patterned_values(output, scale * 0.25, 0.0)),
        }
    }

    fn test_layer(hidden: usize, intermediate: usize) -> EncoderLayer {
        EncoderLayer {
            query: test_linear(hidden, hidden, 0.011),
            key: test_linear(hidden, hidden, -0.009),
            value: test_linear(hidden, hidden, 0.007),
            attention_output: test_linear(hidden, hidden, 0.013),
            attention_ln_weight: ParamVector::new(patterned_values(hidden, 0.01, 1.0)),
            attention_ln_bias: ParamVector::new(patterned_values(hidden, 0.003, 0.0)),
            intermediate: test_linear(intermediate, hidden, 0.008),
            output: test_linear(hidden, intermediate, -0.006),
            output_ln_weight: ParamVector::new(patterned_values(hidden, 0.012, 1.0)),
            output_ln_bias: ParamVector::new(patterned_values(hidden, -0.002, 0.0)),
        }
    }

    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn run_minilm_block(
        provider: &mut MetalProvider,
        hidden_states: &mut [f32],
        attention_mask: &[u8],
        batch: usize,
        seq: usize,
        hidden: usize,
        heads: usize,
        intermediate: usize,
        layers: &[EncoderLayer],
    ) -> Result<bool> {
        let mut run = |context: &mut dyn Any| {
            let context = context
                .downcast_mut::<MiniLmBlockContext>()
                .context("test provider returned the wrong MiniLM context")?;
            context.last_pooled = context.graph.encoder_forward(
                hidden_states,
                attention_mask,
                batch,
                seq,
                hidden,
                heads,
                intermediate,
                1e-12,
                layers,
                context.precision,
            )?;
            Ok(())
        };
        provider.block_forward(BlockForwardRequest {
            family: "minilm",
            create_context: new_minilm_block_context,
            run: &mut run,
        })
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_provider_matches_cpu_for_tiny_encoder_block() {
        let batch = 2;
        let seq = 3;
        let hidden = 4;
        let heads = 2;
        let intermediate = 8;
        let attention_mask = vec![1, 1, 0, 1, 1, 1];
        let layers = vec![test_layer(hidden, intermediate)];
        let mut expected = patterned_values(batch * seq * hidden, 0.02, 0.01);
        let mut actual = expected.clone();

        let mut cpu = CpuProvider;
        encoder_layers_scalar_forward(
            &mut cpu,
            &mut expected,
            &attention_mask,
            batch,
            seq,
            hidden,
            heads,
            intermediate,
            1e-12,
            &layers,
        )
        .expect("run CPU encoder block");

        let mut metal = MetalProvider::new(Precision::F32).expect("create MPSGraph provider");
        assert!(run_minilm_block(
            &mut metal,
            &mut actual,
            &attention_mask,
            batch,
            seq,
            hidden,
            heads,
            intermediate,
            &layers,
        )
        .expect("run resident MPSGraph encoder block"));
        assert_close_with_tolerance(&actual, &expected, 5e-3);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_provider_f16_tracks_cpu_for_tiny_encoder_block() {
        let batch = 2;
        let seq = 3;
        let hidden = 4;
        let heads = 2;
        let intermediate = 8;
        let attention_mask = vec![1, 1, 0, 1, 1, 1];
        let mut layers = vec![test_layer(hidden, intermediate)];
        for layer in &mut layers {
            layer.prepare_metal_f16();
        }
        let mut expected = patterned_values(batch * seq * hidden, 0.02, 0.01);
        let mut actual = expected.clone();

        let mut cpu = CpuProvider;
        encoder_layers_scalar_forward(
            &mut cpu,
            &mut expected,
            &attention_mask,
            batch,
            seq,
            hidden,
            heads,
            intermediate,
            1e-12,
            &layers,
        )
        .expect("run CPU encoder block");

        let mut metal = MetalProvider::new(Precision::F16).expect("create MPSGraph provider");
        assert!(run_minilm_block(
            &mut metal,
            &mut actual,
            &attention_mask,
            batch,
            seq,
            hidden,
            heads,
            intermediate,
            &layers,
        )
        .expect("run resident MPSGraph encoder block"));
        assert_close_with_tolerance(&actual, &expected, 3e-2);
    }
}
