#![recursion_limit = "256"]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use burn::backend::{Metal, wgpu::WgpuDevice};
use burn::prelude::{Float, Int};
use burn::tensor::TensorData;
use clap::Parser;
use synapse_bench::{
    parity::{load_corpus, load_reference, mean_parity, Chunk},
    results::LaneResult,
};
use tokenizers::{Tokenizer, TruncationParams};

const BURN_VERSION: &str = "0.21.0";
const BURN_ONNX_VERSION: &str = "0.21.0";

type Backend = Metal<f32, i32>;
type Device = WgpuDevice;
type Tensor<const D: usize, K = Float> = burn::tensor::Tensor<Backend, D, K>;

include!(concat!(env!("OUT_DIR"), "/model_info.rs"));

type BurnModel = generated_model::Model<Backend>;

#[derive(Parser)]
struct Args {
    /// Path to model.onnx. Burn compiles this lane against a fixed ONNX snapshot,
    /// so the runtime path is validated against the compiled artifact.
    #[arg(long)]
    model: PathBuf,
    /// Path to tokenizer.json
    #[arg(long)]
    tokenizer: PathBuf,
    /// Corpus JSONL ({id, path, text, tokens} per line)
    #[arg(long)]
    corpus: PathBuf,
    /// Output LaneResult JSON path
    #[arg(long)]
    out: PathBuf,
    /// Optional: write produced vectors (JSONL: {id, vec}) for parity reference
    #[arg(long)]
    vectors_out: Option<PathBuf>,
    /// Optional parity reference (JSONL: {id, vec})
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Pooling policy over token embeddings: "mean" or "last"
    #[arg(long, default_value = "mean")]
    pooling: String,
    /// Tokenizer truncation max length
    #[arg(long, default_value_t = 512)]
    max_length: usize,
    /// Attention-unit budget per inference batch
    #[arg(long, default_value_t = 4_000_000)]
    attention_units: usize,
    /// Model label for the result
    #[arg(long)]
    model_label: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_runtime_model_path(&args.model)?;

    let started = Instant::now();
    let device = Device::default();
    let model: BurnModel = BurnModel::from_file(WEIGHTS_PATH, &device);

    let mut tokenizer =
        Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: args.max_length,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("truncation: {e}"))?;

    let _ = run_batch(&model, &device, &tokenizer, &["warmup"], &args.pooling)?;
    let cold_load_s = started.elapsed().as_secs_f64();

    let chunks: Vec<Chunk> = load_corpus(&args.corpus, None)?;

    let texts: Vec<&str> = chunks.iter().map(|chunk| chunk.text.as_str()).collect();
    let encodings = tokenizer
        .encode_batch(texts, true)
        .map_err(|e| anyhow::anyhow!("encode_batch: {e}"))?;
    // Sort by tokenized length so padded batches carry near-uniform lengths
    // (mixed-length batches pad to the batch max and waste GPU on padding).
    // Vectors are keyed by id, so output order is irrelevant.
    let mut order: Vec<usize> = (0..chunks.len()).collect();
    order.sort_by_key(|&i| encodings[i].get_ids().len());
    let chunks: Vec<Chunk> = {
        let mut src: Vec<Option<Chunk>> = chunks.into_iter().map(Some).collect();
        order.iter().map(|&i| src[i].take().expect("each index taken once")).collect()
    };
    let encodings: Vec<_> = {
        let mut src: Vec<Option<_>> = encodings.into_iter().map(Some).collect();
        order.iter().map(|&i| src[i].take().expect("each index taken once")).collect()
    };
    let lengths: Vec<usize> = encodings.iter().map(|encoding| encoding.get_ids().len()).collect();

    let infer_started = Instant::now();
    let mut input_tokens = 0u64;
    let mut items = 0u64;
    let mut produced_vectors: Vec<(String, Vec<f32>)> = Vec::with_capacity(chunks.len());

    let mut batch_start = 0usize;
    let mut batch_max_len = 0usize;
    let mut idx = 0usize;
    while idx <= chunks.len() {
        let flush = if idx == chunks.len() {
            idx > batch_start
        } else {
            let candidate_max = batch_max_len.max(lengths[idx]);
            let count = idx - batch_start;
            count > 0 && (count + 1) * candidate_max * candidate_max > args.attention_units
        };
        if flush {
            let batch: Vec<&str> = chunks[batch_start..idx]
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect();
            let embeds = run_batch(&model, &device, &tokenizer, &batch, &args.pooling)?;
            for (offset, vector) in embeds.into_iter().enumerate() {
                let chunk = &chunks[batch_start + offset];
                input_tokens += lengths[batch_start + offset] as u64;
                items += 1;
                produced_vectors.push((chunk.id.clone(), vector));
            }
            batch_start = idx;
            batch_max_len = 0;
            if idx == chunks.len() {
                break;
            }
        }
        if idx < chunks.len() {
            batch_max_len = batch_max_len.max(lengths[idx]);
        }
        idx += 1;
    }
    let infer_wall_s = infer_started.elapsed().as_secs_f64();

    if let Some(path) = &args.vectors_out {
        write_vectors(path, &produced_vectors)?;
    }

    let parity_mean_cosine = match &args.reference {
        Some(path) => Some(compare_reference(path, &produced_vectors)?),
        None => None,
    };

    let result = LaneResult {
        lane: "burn-wgpu-embed".into(),
        workload: "embed-corpus-v1".into(),
        model: args.model_label,
        cold_load_s,
        infer_wall_s,
        input_tokens,
        tok_per_s: input_tokens as f64 / infer_wall_s,
        items,
        parity_mean_cosine,
        self_peak_rss_bytes: None,
        notes: format!(
            "burn={BURN_VERSION}, burn_onnx={BURN_ONNX_VERSION}, backend=metal-f32, compiled_target={COMPILED_TARGET}, compiled_model={}, attention_units={}, pooling={}, max_len={}, build_notes={}",
            COMPILED_MODEL_PATH,
            args.attention_units,
            args.pooling,
            args.max_length,
            BUILD_NOTES,
        ),
    };

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&result)?)?;
    eprintln!(
        "burn-wgpu-embed: {} items, {} tokens, {:.1} tok/s, cold_load {:.1}s, infer {:.1}s",
        result.items, result.input_tokens, result.tok_per_s, result.cold_load_s, result.infer_wall_s
    );
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn run_batch(
    model: &BurnModel,
    device: &Device,
    tokenizer: &Tokenizer,
    texts: &[&str],
    pooling: &str,
) -> Result<Vec<Vec<f32>>> {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("encode_batch: {e}"))?;
    let batch = encodings.len();
    let max_len = encodings
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .unwrap_or(1)
        .max(1);

    let mut ids = vec![0i32; batch * max_len];
    let mut mask = vec![0i32; batch * max_len];
    let token_type_ids = vec![0i32; batch * max_len];
    for (row, encoding) in encodings.iter().enumerate() {
        for (col, (&id, &attend)) in encoding
            .get_ids()
            .iter()
            .zip(encoding.get_attention_mask())
            .enumerate()
        {
            ids[row * max_len + col] = id as i32;
            mask[row * max_len + col] = attend as i32;
        }
    }

    let input_ids = Tensor::<2, Int>::from_data(TensorData::new(ids, [batch, max_len]), device);
    let attention_mask =
        Tensor::<2, Int>::from_data(TensorData::new(mask.clone(), [batch, max_len]), device);
    let token_type_ids =
        Tensor::<2, Int>::from_data(TensorData::new(token_type_ids, [batch, max_len]), device);
    let last_hidden_state = model.forward(input_ids, attention_mask, token_type_ids);
    let shape: [usize; 3] = last_hidden_state.shape().dims();
    let [actual_batch, seq_len, hidden] = shape;
    anyhow::ensure!(actual_batch == batch, "batch mismatch: expected {batch}, got {actual_batch}");

    let tensor_data = last_hidden_state.into_data();
    let hidden_data = tensor_data
        .as_slice::<f32>()
        .context("expected f32 burn output tensor")?;

    let mut vectors = Vec::with_capacity(batch);
    for row in 0..batch {
        let mut vector = vec![0f32; hidden];
        match pooling {
            "mean" => {
                let mut count = 0f32;
                for col in 0..seq_len {
                    if mask[row * max_len + col] == 1 {
                        count += 1.0;
                        let start = (row * seq_len + col) * hidden;
                        for k in 0..hidden {
                            vector[k] += hidden_data[start + k];
                        }
                    }
                }
                let denom = count.max(1.0);
                vector.iter_mut().for_each(|value| *value /= denom);
            }
            "last" => {
                let last = (0..seq_len)
                    .rev()
                    .find(|&col| mask[row * max_len + col] == 1)
                    .unwrap_or(0);
                let start = (row * seq_len + last) * hidden;
                vector.copy_from_slice(&hidden_data[start..start + hidden]);
            }
            other => anyhow::bail!("unknown pooling: {other}"),
        }
        normalize(&mut vector);
        vectors.push(vector);
    }

    Ok(vectors)
}

fn compare_reference(reference_path: &Path, produced: &[(String, Vec<f32>)]) -> Result<f64> {
    let reference = load_reference(reference_path)?;
    let (mean, matched) = mean_parity(produced.iter().cloned(), &reference);
    anyhow::ensure!(matched > 0, "no overlapping ids found in reference vectors");
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

fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt() + 1e-12;
    vector.iter_mut().for_each(|value| *value /= norm);
}

fn validate_runtime_model_path(runtime_model: &Path) -> Result<()> {
    let runtime = runtime_model
        .canonicalize()
        .with_context(|| format!("runtime model path does not exist: {}", runtime_model.display()))?;
    let compiled = Path::new(COMPILED_MODEL_PATH)
        .canonicalize()
        .with_context(|| format!("compiled model path does not exist: {COMPILED_MODEL_PATH}"))?;
    anyhow::ensure!(
        runtime == compiled,
        "lane-burn is compiled against {} but was invoked with {}. Rebuild lane-burn for a different ONNX snapshot.",
        compiled.display(),
        runtime.display()
    );
    Ok(())
}
