//! Bench lane: raw `ort` CPU embedding, reproducing AFT's shipped production
//! policies exactly (the proven bounded-CPU floor):
//! - GraphOptimizationLevel::Level3
//! - intra-op threads = ceil(available_parallelism / 2)
//! - greedy attention-unit batching: flush when (count+1) * max_len^2 > 4M
//! - tokenizer truncation at 512 (MiniLM) or model max (Qwen3), manual zero pad
//! - mean pooling (MiniLM-class) or last-token pooling (Qwen3-Embedding),
//!   then L2 normalization
//!
//! Emits a LaneResult JSON plus optionally the raw vectors (for parity
//! reference against GPU lanes).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use synapse_bench::{
    parity::{load_corpus, Chunk},
    results::LaneResult,
};
use tokenizers::{Tokenizer, TruncationParams};

#[derive(Parser)]
struct Args {
    /// Path to model.onnx
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
    /// Pooling: "mean" (MiniLM-class) or "last" (Qwen3-Embedding-class)
    #[arg(long, default_value = "last")]
    pooling: String,
    /// Tokenizer truncation max length
    #[arg(long, default_value_t = 512)]
    max_length: usize,
    /// Attention-unit budget per inference batch (AFT production: 4M)
    #[arg(long, default_value_t = 4_000_000)]
    attention_units: usize,
    /// Model label for the result
    #[arg(long)]
    model_label: String,
    /// ORT intra-op threads. Default: ceil(cores/2), AFT's production policy
    /// (all-cores measured 1.7x slower there; sweepable here to re-verify).
    #[arg(long)]
    intra_threads: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let started = Instant::now();

    // --- Model + tokenizer load (cold-load window) ---
    let intra = args.intra_threads.unwrap_or_else(|| {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).div_ceil(2).max(1)
    });
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(intra)?
        .commit_from_file(&args.model)
        .context("ORT session")?;

    let mut tokenizer =
        Tokenizer::from_file(&args.tokenizer).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: args.max_length,
            ..Default::default()
        }))
        .map_err(|e| anyhow::anyhow!("truncation: {e}"))?;

    // Collect input names plus static dims for KV-cache inputs (the
    // onnx-community Qwen3 export carries past_key_values.* inputs; an
    // embedding pass feeds them empty with past_len=0).
    let input_names: Vec<(String, Vec<i64>)> = session
        .inputs()
        .iter()
        .map(|i| {
            let dims = match i.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => shape.to_vec(),
                _ => Vec::new(),
            };
            (i.name().to_string(), dims)
        })
        .collect();

    // Warmup: one tiny inference so cold_load includes first-run graph prep.
    run_batch(&mut session, &input_names, &tokenizer, &["warmup"], &args.pooling)?;
    let cold_load_s = started.elapsed().as_secs_f64();

    // --- Corpus ---
    let chunks: Vec<Chunk> = load_corpus(&args.corpus, None)?;

    // --- Embed with AFT's greedy attention-unit batching ---
    let mut vectors_writer = match &args.vectors_out {
        Some(p) => {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent)?;
            }
            Some(std::io::BufWriter::new(std::fs::File::create(p)?))
        }
        None => None,
    };

    let infer_started = Instant::now();
    let mut input_tokens: u64 = 0;
    let mut items: u64 = 0;

    // Pre-tokenize to know lengths (token counts contribute to input_tokens
    // as the model sees them, post-truncation).
    let encodings: Vec<usize> = {
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let encs = tokenizer
            .encode_batch(texts, true)
            .map_err(|e| anyhow::anyhow!("encode_batch: {e}"))?;
        encs.iter().map(|e| e.get_ids().len()).collect()
    };

    let mut batch_start = 0usize;
    let mut batch_max_len = 0usize;
    let mut idx = 0usize;
    while idx <= chunks.len() {
        let flush = if idx == chunks.len() {
            idx > batch_start
        } else {
            let candidate_max = batch_max_len.max(encodings[idx]);
            let count = idx - batch_start;
            count > 0 && (count + 1) * candidate_max * candidate_max > args.attention_units
        };
        if flush {
            let batch: Vec<&str> =
                chunks[batch_start..idx].iter().map(|c| c.text.as_str()).collect();
            let embeds = run_batch(&mut session, &input_names, &tokenizer, &batch, &args.pooling)?;
            for (offset, vec) in embeds.iter().enumerate() {
                let chunk = &chunks[batch_start + offset];
                input_tokens += encodings[batch_start + offset] as u64;
                items += 1;
                if let Some(w) = vectors_writer.as_mut() {
                    use std::io::Write;
                    serde_json::to_writer(&mut *w, &serde_json::json!({"id": chunk.id, "vec": vec}))?;
                    w.write_all(b"\n")?;
                }
            }
            batch_start = idx;
            batch_max_len = 0;
            if idx == chunks.len() {
                break;
            }
        }
        if idx < chunks.len() {
            batch_max_len = batch_max_len.max(encodings[idx]);
        }
        idx += 1;
    }
    let infer_wall_s = infer_started.elapsed().as_secs_f64();

    if let Some(mut w) = vectors_writer {
        use std::io::Write;
        w.flush()?;
    }

    let result = LaneResult {
        lane: "ort-cpu".into(),
        workload: "embed-corpus-v1".into(),
        model: args.model_label,
        cold_load_s,
        infer_wall_s,
        input_tokens,
        tok_per_s: input_tokens as f64 / infer_wall_s,
        items,
        parity_mean_cosine: None, // this lane IS the reference
        self_peak_rss_bytes: None,
        notes: format!(
            "Level3, intra_threads={intra}, attention_units={}, pooling={}, max_len={}",
            args.attention_units, args.pooling, args.max_length
        ),
    };
    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.out, serde_json::to_string_pretty(&result)?)?;
    eprintln!(
        "ort-cpu: {} items, {} tokens, {:.1} tok/s, cold_load {:.1}s, infer {:.1}s",
        result.items, result.input_tokens, result.tok_per_s, result.cold_load_s, result.infer_wall_s
    );
    Ok(())
}

/// Tokenize + run + pool + L2-normalize one batch. Mirrors AFT local_embed.rs.
fn run_batch(
    session: &mut Session,
    input_names: &[(String, Vec<i64>)],
    tokenizer: &Tokenizer,
    texts: &[&str],
    pooling: &str,
) -> Result<Vec<Vec<f32>>> {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow::anyhow!("encode_batch: {e}"))?;
    let batch = encodings.len();
    let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(1).max(1);

    let mut ids = vec![0i64; batch * max_len];
    let mut mask = vec![0i64; batch * max_len];
    for (row, enc) in encodings.iter().enumerate() {
        for (col, (&id, &m)) in enc.get_ids().iter().zip(enc.get_attention_mask()).enumerate() {
            ids[row * max_len + col] = id as i64;
            mask[row * max_len + col] = m as i64;
        }
    }
    let ids = Array2::<i64>::from_shape_vec((batch, max_len), ids)?;
    let mask_arr = Array2::<i64>::from_shape_vec((batch, max_len), mask.clone())?;

    let mut inputs: Vec<(&str, ort::value::DynValue)> = Vec::new();
    for (name, dims) in input_names {
        match name.as_str() {
            "input_ids" => {
                inputs.push(("input_ids", ort::value::Tensor::from_array(ids.clone())?.into_dyn()))
            }
            "attention_mask" => inputs
                .push(("attention_mask", ort::value::Tensor::from_array(mask_arr.clone())?.into_dyn())),
            "token_type_ids" => {
                let tt = Array2::<i64>::zeros((batch, max_len));
                inputs.push(("token_type_ids", ort::value::Tensor::from_array(tt)?.into_dyn()));
            }
            "position_ids" => {
                let mut pos = Array2::<i64>::zeros((batch, max_len));
                for r in 0..batch {
                    for c in 0..max_len {
                        pos[[r, c]] = c as i64;
                    }
                }
                inputs.push(("position_ids", ort::value::Tensor::from_array(pos)?.into_dyn()));
            }
            other if other.starts_with("past_key_values.") => {
                // Empty KV cache: [batch, num_kv_heads, 0, head_dim]. Static
                // dims come from the model; the two dynamic dims are batch
                // and past sequence length (0 for a fresh pass).
                anyhow::ensure!(dims.len() == 4, "unexpected past kv shape {dims:?}");
                let kv_heads = dims[1].max(1) as usize;
                let head_dim = dims[3].max(1) as usize;
                let empty = ndarray::Array4::<f32>::zeros((batch, kv_heads, 0, head_dim));
                inputs.push((other, ort::value::Tensor::from_array(empty)?.into_dyn()));
            }
            other => anyhow::bail!("unexpected model input: {other}"),
        }
    }

    let outputs = session.run(inputs)?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>().map(|(s, d)| (s.to_vec(), d.to_vec())).or_else(
        |_| -> Result<_, ort::Error> {
            let (s, d) = outputs[0].try_extract_tensor::<half::f16>()?;
            Ok((s.to_vec(), d.iter().map(|v| v.to_f32()).collect()))
        },
    )?;
    anyhow::ensure!(shape.len() == 3, "expected [batch, seq, hidden], got {shape:?}");
    let (b, s, h) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    anyhow::ensure!(b == batch, "batch mismatch");

    let mut result = Vec::with_capacity(batch);
    for row in 0..batch {
        let mut vec = vec![0f32; h];
        match pooling {
            "mean" => {
                let mut count = 0f32;
                for col in 0..s {
                    if mask[row * max_len + col] == 1 {
                        count += 1.0;
                        for k in 0..h {
                            vec[k] += data[(row * s + col) * h + k];
                        }
                    }
                }
                let denom = count.max(1.0);
                vec.iter_mut().for_each(|v| *v /= denom);
            }
            "last" => {
                // Last valid (attended) token position.
                let last = (0..s).rev().find(|&col| mask[row * max_len + col] == 1).unwrap_or(0);
                vec.copy_from_slice(&data[(row * s + last) * h..(row * s + last + 1) * h]);
            }
            other => anyhow::bail!("unknown pooling: {other}"),
        }
        let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-12;
        vec.iter_mut().for_each(|v| *v /= norm);
        result.push(vec);
    }
    Ok(result)
}
