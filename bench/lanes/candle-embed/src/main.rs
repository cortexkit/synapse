use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use clap::{Parser, ValueEnum};
use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Repo,
};
use synapse_bench::{
    parity::{load_corpus, load_reference, mean_parity, Chunk},
    results::LaneResult,
};
use tokenizers::{Tokenizer, TruncationParams};

const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const WORKLOAD: &str = "embed-corpus-v1";
const DEFAULT_MAX_LENGTH: usize = 512;
const DEFAULT_ATTENTION_UNITS: usize = 4_000_000;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum RunnerDevice {
    Cpu,
    Metal,
}

impl RunnerDevice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Precision {
    F32,
    F16,
}

impl Precision {
    fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }

    fn candle_dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "lane-candle-embed")]
struct Args {
    /// Corpus JSONL ({id, text} per line)
    #[arg(long)]
    corpus: PathBuf,
    /// Optional top-N cap for smoke or parity subsets
    #[arg(long)]
    limit: Option<usize>,
    /// Output LaneResult JSON path
    #[arg(long)]
    out: PathBuf,
    /// Optional: write produced vectors (JSONL: {id, vec})
    #[arg(long = "emit-vectors", alias = "vectors-out")]
    vectors_out: Option<PathBuf>,
    /// Optional parity reference (JSONL: {id, vec})
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Backend device to benchmark
    #[arg(long, value_enum, default_value_t = RunnerDevice::Cpu)]
    device: RunnerDevice,
    /// Weight / compute precision to request
    #[arg(long, value_enum, default_value_t = Precision::F32)]
    dtype: Precision,
    /// Tokenizer truncation max length
    #[arg(long, default_value_t = DEFAULT_MAX_LENGTH)]
    max_length: usize,
    /// Attention-unit budget per inference batch
    #[arg(long, default_value_t = DEFAULT_ATTENTION_UNITS)]
    attention_units: usize,
    /// Optional model label for the result JSON
    #[arg(long)]
    model_label: Option<String>,
}

#[derive(Debug)]
struct ModelFiles {
    root: PathBuf,
    config: PathBuf,
    tokenizer: PathBuf,
    weight_files: Vec<PathBuf>,
}

#[derive(Debug)]
struct EncodedChunk {
    original_index: usize,
    id: String,
    input_ids: Vec<u32>,
    attention_mask: Vec<u32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.max_length > 0, "max-length must be > 0");
    ensure!(args.attention_units > 0, "attention-units must be > 0");
    ensure!(
        !(matches!(args.device, RunnerDevice::Cpu) && matches!(args.dtype, Precision::F16)),
        "cpu + f16 is not supported for this spike; use --dtype f32 on cpu"
    );

    let started = Instant::now();
    let backend_label = format!("{} {}", args.device.as_str(), args.dtype.as_str());
    let device = select_device(args.device)?;
    let model_files = resolve_model_files()?;
    let tokenizer = load_tokenizer(&model_files.tokenizer, args.max_length)?;
    let (model, config) = load_model(&model_files, &device, args.dtype.candle_dtype())?;

    let warmup = EncodedChunk {
        original_index: 0,
        id: "warmup".to_string(),
        input_ids: vec![101, 4010, 102],
        attention_mask: vec![1, 1, 1],
    };
    let _ = run_batch(&model, &[warmup])
        .with_context(|| format!("warmup batch failed on {backend_label}"))?;
    let cold_load_s = started.elapsed().as_secs_f64();

    let chunks: Vec<Chunk> = load_corpus(&args.corpus, args.limit)?;
    let (mut encoded, original_ids) = encode_chunks(&tokenizer, chunks)?;
    encoded.sort_by_key(|chunk| chunk.input_ids.len());

    let infer_started = Instant::now();
    let mut batch_start = 0usize;
    let mut batch_max_len = 0usize;
    let mut index = 0usize;
    let mut input_tokens = 0u64;
    let mut restored: Vec<Option<(String, Vec<f32>)>> =
        (0..original_ids.len()).map(|_| None).collect();

    while index <= encoded.len() {
        let flush = if index == encoded.len() {
            index > batch_start
        } else {
            let candidate_max = batch_max_len.max(encoded[index].input_ids.len());
            let count = index - batch_start;
            count > 0 && (count + 1) * candidate_max * candidate_max > args.attention_units
        };

        if flush {
            let batch = &encoded[batch_start..index];
            let vectors = run_batch(&model, batch)
                .with_context(|| format!("inference batch failed on {backend_label}"))?;
            for (chunk, vector) in batch.iter().zip(vectors) {
                input_tokens += chunk.input_ids.len() as u64;
                restored[chunk.original_index] = Some((chunk.id.clone(), vector));
            }
            batch_start = index;
            batch_max_len = 0;
            if index == encoded.len() {
                break;
            }
        }

        if index < encoded.len() {
            batch_max_len = batch_max_len.max(encoded[index].input_ids.len());
        }
        index += 1;
    }
    let infer_wall_s = infer_started.elapsed().as_secs_f64();

    let produced = restored
        .into_iter()
        .enumerate()
        .map(|(index, maybe_row)| {
            maybe_row.with_context(|| format!("missing vector for original item {index}"))
        })
        .collect::<Result<Vec<_>>>()?;

    if let Some(path) = &args.vectors_out {
        write_vectors(path, &produced)?;
    }

    let parity_mean_cosine = match &args.reference {
        Some(reference_path) => Some(compare_reference(reference_path, &produced)?),
        None => None,
    };

    let result = LaneResult {
        lane: format!("candle-embed-{}-{}", args.device.as_str(), args.dtype.as_str()),
        workload: WORKLOAD.into(),
        model: args.model_label.unwrap_or_else(|| {
            format!(
                "all-MiniLM-L6-v2@candle-{}-{}",
                args.device.as_str(),
                args.dtype.as_str()
            )
        }),
        cold_load_s,
        infer_wall_s,
        input_tokens,
        tok_per_s: input_tokens as f64 / infer_wall_s.max(f64::MIN_POSITIVE),
        items: produced.len() as u64,
        parity_mean_cosine,
        self_peak_rss_bytes: None,
        notes: format!(
            "model_repo={MODEL_ID}, model_root={}, hidden_size={}, layers={}, device={}, dtype={}, attention_units={}, max_len={}, weight_files={}",
            model_files.root.display(),
            config.hidden_size,
            config.num_hidden_layers,
            args.device.as_str(),
            args.dtype.as_str(),
            args.attention_units,
            args.max_length,
            model_files.weight_files.len(),
        ),
    };

    write_lane_result(&args.out, &result)?;
    eprintln!(
        "{}: {} items, {} tokens, {:.1} tok/s, cold_load {:.2}s, infer {:.2}s{}",
        result.lane,
        result.items,
        result.input_tokens,
        result.tok_per_s,
        result.cold_load_s,
        result.infer_wall_s,
        result
            .parity_mean_cosine
            .map(|value| format!(", parity {:.6}", value))
            .unwrap_or_default(),
    );
    Ok(())
}

fn select_device(device: RunnerDevice) -> Result<Device> {
    match device {
        RunnerDevice::Cpu => Ok(Device::Cpu),
        RunnerDevice::Metal => select_metal_device(),
    }
}

#[cfg(target_os = "macos")]
fn select_metal_device() -> Result<Device> {
    Device::new_metal(0).context("create candle Metal device")
}

#[cfg(not(target_os = "macos"))]
fn select_metal_device() -> Result<Device> {
    anyhow::bail!("--device metal is only available on macOS builds")
}

fn resolve_model_files() -> Result<ModelFiles> {
    let api = ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .context("build hf-hub api")?;
    let repo = api.repo(Repo::model(MODEL_ID.to_owned()));
    let config = repo
        .get("config.json")
        .context("get config.json from hf-hub")?;
    let tokenizer = repo
        .get("tokenizer.json")
        .context("get tokenizer.json from hf-hub")?;
    let weight_files = collect_weight_files(&repo)?;
    let root = config
        .parent()
        .context("config.json path had no parent")?
        .to_path_buf();
    Ok(ModelFiles {
        root,
        config,
        tokenizer,
        weight_files,
    })
}

fn collect_weight_files(repo: &ApiRepo) -> Result<Vec<PathBuf>> {
    if let Ok(path) = repo.get("model.safetensors") {
        return Ok(vec![path]);
    }

    let mut files = Vec::new();
    for shard_index in 1..=128 {
        let mut found = None;
        for total_shards in 1..=128 {
            let candidate = format!("model-{shard_index:05}-of-{total_shards:05}.safetensors");
            if let Ok(path) = repo.get(&candidate) {
                found = Some(path);
                break;
            }
        }
        match found {
            Some(path) => files.push(path),
            None => break,
        }
    }

    ensure!(
        !files.is_empty(),
        "could not locate model.safetensors or sharded weight files for {MODEL_ID}"
    );
    Ok(files)
}

fn load_tokenizer(path: &Path, max_length: usize) -> Result<Tokenizer> {
    let mut tokenizer =
        Tokenizer::from_file(path).map_err(|err| anyhow::anyhow!("tokenizer: {err}"))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|err| anyhow::anyhow!("truncation: {err}"))?;
    Ok(tokenizer)
}

fn load_model(
    files: &ModelFiles,
    device: &Device,
    dtype: DType,
) -> Result<(BertModel, BertConfig)> {
    let config: BertConfig = serde_json::from_str(
        &std::fs::read_to_string(&files.config)
            .with_context(|| format!("read {}", files.config.display()))?,
    )
    .with_context(|| format!("parse {}", files.config.display()))?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files.weight_files, dtype, device) }
        .context("map safetensors weights")?;
    let model = BertModel::load(vb, &config).context("load bert model")?;
    Ok((model, config))
}

fn encode_chunks(
    tokenizer: &Tokenizer,
    chunks: Vec<Chunk>,
) -> Result<(Vec<EncodedChunk>, Vec<String>)> {
    let texts: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
    let encodings = tokenizer
        .encode_batch(texts, true)
        .map_err(|err| anyhow::anyhow!("encode_batch: {err}"))?;

    let original_ids = chunks.iter().map(|chunk| chunk.id.clone()).collect();
    let encoded = chunks
        .into_iter()
        .zip(encodings)
        .enumerate()
        .map(|(original_index, (chunk, encoding))| EncodedChunk {
            original_index,
            id: chunk.id,
            input_ids: encoding.get_ids().to_vec(),
            attention_mask: encoding
                .get_attention_mask()
                .iter()
                .map(|&value| value as u32)
                .collect(),
        })
        .collect();
    Ok((encoded, original_ids))
}

fn run_batch(model: &BertModel, batch: &[EncodedChunk]) -> Result<Vec<Vec<f32>>> {
    ensure!(!batch.is_empty(), "run_batch called with an empty batch");
    let max_len = batch
        .iter()
        .map(|chunk| chunk.input_ids.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let batch_size = batch.len();

    let mut input_ids = vec![0u32; batch_size * max_len];
    let mut attention_mask = vec![0u32; batch_size * max_len];
    for (row, chunk) in batch.iter().enumerate() {
        for (col, (&token_id, &attend)) in chunk
            .input_ids
            .iter()
            .zip(&chunk.attention_mask)
            .enumerate()
        {
            input_ids[row * max_len + col] = token_id;
            attention_mask[row * max_len + col] = attend;
        }
    }

    let device = &model.device;
    let input_ids = Tensor::from_vec(input_ids, (batch_size, max_len), device)
        .context("build input_ids tensor")?;
    let attention_mask = Tensor::from_vec(attention_mask, (batch_size, max_len), device)
        .context("build attention_mask tensor")?;
    let token_type_ids = Tensor::zeros((batch_size, max_len), DType::U32, device)
        .context("build token_type_ids tensor")?;

    let hidden = model
        .forward(&input_ids, &token_type_ids, Some(&attention_mask))
        .context("bert forward")?;
    let pooled = mean_pool_and_normalize(&hidden, &attention_mask)?;
    pooled
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()
        .context("extract pooled embeddings")
}

fn mean_pool_and_normalize(hidden: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let dtype = hidden.dtype();
    let float_mask = attention_mask.to_dtype(dtype)?;
    let expanded_mask = float_mask.unsqueeze(2)?;
    let summed = hidden.broadcast_mul(&expanded_mask)?.sum(1)?;
    let counts = float_mask.sum_keepdim(1)?.clamp(1e-9f32, f32::MAX)?;
    let mean = summed.broadcast_div(&counts)?;
    let eps = Tensor::try_from(1e-12f32)?
        .to_device(mean.device())?
        .to_dtype(dtype)?;
    let norms = mean.sqr()?.sum_keepdim(1)?.sqrt()?.broadcast_add(&eps)?;
    Ok(mean.broadcast_div(&norms)?)
}

fn compare_reference(reference_path: &Path, produced: &[(String, Vec<f32>)]) -> Result<f64> {
    let reference = load_reference(reference_path)?;
    let (mean, matched) = mean_parity(produced.iter().cloned(), &reference);
    ensure!(matched > 0, "no overlapping ids found in reference vectors");
    Ok(mean.expect("matched count implies a parity mean"))
}

fn write_vectors(path: &Path, produced: &[(String, Vec<f32>)]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
    for (id, vector) in produced {
        serde_json::to_writer(&mut writer, &serde_json::json!({"id": id, "vec": vector}))?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_lane_result(path: &Path, result: &LaneResult) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(result)?)?;
    Ok(())
}
