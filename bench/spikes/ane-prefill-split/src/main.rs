use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use synapse_engine_owned::owned_decode_engine::{
    top_logits, DecodeKernel, MetalStepDecoder, MetalStepKvCache, Qwen3DecodeModel,
    WeightQuantization,
};
use synapse_engine_owned::Precision;

const LAYERS: usize = 28;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const BATCH_CHUNK: usize = 16;
const VOCAB: usize = 151_936;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    cache: PathBuf,
    #[arg(long)]
    logits: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 512)]
    cache_bucket: usize,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 20)]
    baseline_prefill_calls: usize,
    #[arg(long, default_value_t = 20)]
    upload_calls: usize,
    #[arg(long)]
    baseline_only: bool,
    #[arg(long)]
    upload_only: bool,
    #[arg(long)]
    decode_only: bool,
    #[arg(long, default_value_t = 5)]
    decode_calls: usize,
}

#[derive(Deserialize)]
struct TokenizedRow {
    id: String,
    input_ids: Vec<u32>,
    attention_mask: Vec<u8>,
}

#[derive(Serialize)]
struct ArmResult {
    prefill_or_upload_samples_ms: Vec<f64>,
    prefill_or_upload_p50_ms: f64,
    prefill_or_upload_p95_ms: f64,
    decode_wall_ms: f64,
    generated_tokens: Vec<u32>,
    top2_gaps: Vec<f32>,
}

#[derive(Serialize)]
struct Comparison {
    id: String,
    prompt_tokens: usize,
    cache_bucket: usize,
    max_new_tokens: usize,
    baseline: ArmResult,
    ane_split: ArmResult,
    token_exact: bool,
    match_depth: usize,
    divergence_depth: Option<usize>,
    baseline_gap_at_divergence: Option<f32>,
    ane_gap_at_divergence: Option<f32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        [512, 1024, 2048].contains(&args.cache_bucket),
        "cache bucket must be 512, 1024, or 2048"
    );
    ensure!(args.max_new_tokens > 0, "max-new-tokens must be positive");
    ensure!(
        args.baseline_prefill_calls > 0,
        "baseline-prefill-calls must be positive"
    );
    ensure!(args.upload_calls > 0, "upload-calls must be positive");
    ensure!(args.decode_calls > 0, "decode-calls must be positive");
    ensure!(
        usize::from(args.baseline_only)
            + usize::from(args.upload_only)
            + usize::from(args.decode_only)
            <= 1,
        "baseline-only, upload-only, and decode-only are mutually exclusive"
    );

    let row: TokenizedRow = serde_json::from_str(
        &fs::read_to_string(&args.input)
            .with_context(|| format!("read tokenized input {}", args.input.display()))?,
    )
    .context("parse tokenized input")?;
    ensure!(!row.input_ids.is_empty(), "prompt must not be empty");
    ensure!(
        row.input_ids.len() == row.attention_mask.len(),
        "input_ids and attention_mask lengths differ"
    );
    ensure!(
        row.attention_mask.iter().all(|&value| value == 1),
        "Metal comparison requires a fully occupied prompt without padding"
    );
    ensure!(
        row.input_ids.len() + args.max_new_tokens <= args.cache_bucket,
        "prompt plus generated tokens exceeds the cache bucket"
    );

    let cache_bits = read_u16(&args.cache)?;
    let expected_cache_elements = LAYERS * 2 * KV_HEADS * args.cache_bucket * HEAD_DIM;
    ensure!(
        cache_bits.len() == expected_cache_elements,
        "ANE cache has {} elements; expected {}",
        cache_bits.len(),
        expected_cache_elements
    );
    let ane_logits = read_f32(&args.logits)?;
    ensure!(
        ane_logits.len() == VOCAB,
        "ANE logits have {} values; expected {VOCAB}",
        ane_logits.len()
    );

    let model = Qwen3DecodeModel::load(&args.model, Precision::F16)
        .with_context(|| format!("load Qwen3 model {}", args.model.display()))?;
    let mut decoder = MetalStepDecoder::new(
        &model,
        Precision::F16,
        args.cache_bucket,
        WeightQuantization::None,
    )?;

    if args.upload_only {
        let mut samples = Vec::with_capacity(args.upload_calls);
        for _ in 0..args.upload_calls {
            let started = Instant::now();
            decoder.import_caches(&cache_bits)?;
            samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        return write_stage_only(&args.out, "upload", &samples);
    }
    if args.decode_only {
        let ane_top = top_logits(&ane_logits, 2);
        ensure!(ane_top.len() == 2, "ANE logits did not produce a top-2");
        let first = ane_top[0].token_id;
        let first_gap = ane_top[0].logit - ane_top[1].logit;
        let mut samples = Vec::with_capacity(args.decode_calls);
        for _ in 0..args.decode_calls {
            decoder.import_caches(&cache_bits)?;
            let mut cache = MetalStepKvCache {
                position: row.input_ids.len(),
            };
            let started = Instant::now();
            let _ = generate(
                &mut decoder,
                &mut cache,
                first,
                first_gap,
                args.max_new_tokens,
            )?;
            samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        }
        return write_stage_only(&args.out, "decode", &samples);
    }

    let mut baseline_samples = Vec::with_capacity(args.baseline_prefill_calls);
    let mut baseline_cache = MetalStepKvCache { position: 0 };
    let mut baseline_first = 0;
    for _ in 0..args.baseline_prefill_calls {
        let started = Instant::now();
        let (cache, first, _) = gpu_prefill(&mut decoder, &row.input_ids, false)?;
        baseline_samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        baseline_cache = cache;
        baseline_first = first;
    }
    if args.baseline_only {
        return write_stage_only(&args.out, "baseline_prefill", &baseline_samples);
    }
    let (_, _, baseline_first_gap) = gpu_prefill(&mut decoder, &row.input_ids, true)?;
    let baseline_decode_started = Instant::now();
    let (baseline_tokens, baseline_gaps) = generate(
        &mut decoder,
        &mut baseline_cache,
        baseline_first,
        baseline_first_gap,
        args.max_new_tokens,
    )?;
    let baseline_decode_ms = baseline_decode_started.elapsed().as_secs_f64() * 1_000.0;

    let mut upload_samples = Vec::with_capacity(args.upload_calls);
    for _ in 0..args.upload_calls {
        let started = Instant::now();
        decoder.import_caches(&cache_bits)?;
        upload_samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    let ane_top = top_logits(&ane_logits, 2);
    ensure!(ane_top.len() == 2, "ANE logits did not produce a top-2");
    let ane_first = ane_top[0].token_id;
    let ane_first_gap = ane_top[0].logit - ane_top[1].logit;
    let mut ane_cache = MetalStepKvCache {
        position: row.input_ids.len(),
    };
    let split_decode_started = Instant::now();
    let (split_tokens, split_gaps) = generate(
        &mut decoder,
        &mut ane_cache,
        ane_first,
        ane_first_gap,
        args.max_new_tokens,
    )?;
    let split_decode_ms = split_decode_started.elapsed().as_secs_f64() * 1_000.0;

    let match_depth = baseline_tokens
        .iter()
        .zip(&split_tokens)
        .take_while(|(left, right)| left == right)
        .count();
    let divergence_depth = (baseline_tokens != split_tokens).then_some(match_depth);
    let result = Comparison {
        id: row.id,
        prompt_tokens: row.input_ids.len(),
        cache_bucket: args.cache_bucket,
        max_new_tokens: args.max_new_tokens,
        baseline: ArmResult {
            prefill_or_upload_p50_ms: percentile(&baseline_samples, 0.50),
            prefill_or_upload_p95_ms: percentile(&baseline_samples, 0.95),
            prefill_or_upload_samples_ms: baseline_samples,
            decode_wall_ms: baseline_decode_ms,
            generated_tokens: baseline_tokens.clone(),
            top2_gaps: baseline_gaps.clone(),
        },
        ane_split: ArmResult {
            prefill_or_upload_p50_ms: percentile(&upload_samples, 0.50),
            prefill_or_upload_p95_ms: percentile(&upload_samples, 0.95),
            prefill_or_upload_samples_ms: upload_samples,
            decode_wall_ms: split_decode_ms,
            generated_tokens: split_tokens,
            top2_gaps: split_gaps.clone(),
        },
        token_exact: divergence_depth.is_none(),
        match_depth,
        divergence_depth,
        baseline_gap_at_divergence: divergence_depth
            .and_then(|depth| baseline_gaps.get(depth).copied()),
        ane_gap_at_divergence: divergence_depth.and_then(|depth| split_gaps.get(depth).copied()),
    };
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&result)?;
    fs::write(&args.out, format!("{json}\n"))?;
    println!("{json}");
    Ok(())
}

fn write_stage_only(path: &Path, stage: &str, samples: &[f64]) -> Result<()> {
    let value = serde_json::json!({
        "stage": stage,
        "calls": samples.len(),
        "samples_ms": samples,
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
    });
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&value)?;
    fs::write(path, format!("{json}\n"))?;
    println!("{json}");
    Ok(())
}

fn gpu_prefill(
    decoder: &mut MetalStepDecoder<'_>,
    tokens: &[u32],
    read_final_logits: bool,
) -> Result<(MetalStepKvCache, u32, f32)> {
    let mut cache = MetalStepKvCache { position: 0 };
    let mut first = 0;
    let mut first_gap = f32::NAN;
    let chunks = tokens.chunks(BATCH_CHUNK).collect::<Vec<_>>();
    for (index, chunk) in chunks.iter().enumerate() {
        let final_chunk = index + 1 == chunks.len();
        if final_chunk && read_final_logits {
            let logits = decoder.verify_tokens_batch_logits(&mut cache, chunk)?;
            let row = &logits[(chunk.len() - 1) * VOCAB..chunk.len() * VOCAB];
            let top = top_logits(row, 2);
            first = top[0].token_id;
            first_gap = top[0].logit - top[1].logit;
        } else {
            let argmaxes = decoder.verify_tokens_batch(&mut cache, chunk)?;
            if final_chunk {
                first = *argmaxes
                    .last()
                    .context("final prefill chunk has no argmax")?;
            }
        }
    }
    Ok((cache, first, first_gap))
}

fn generate(
    decoder: &mut MetalStepDecoder<'_>,
    cache: &mut MetalStepKvCache,
    first: u32,
    first_gap: f32,
    count: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    let mut tokens = Vec::with_capacity(count);
    let mut gaps = Vec::with_capacity(count);
    tokens.push(first);
    gaps.push(first_gap);
    let mut current = first;
    for _ in 1..count {
        let logits = decoder.advance(cache, current)?;
        let top = top_logits(&logits, 2);
        ensure!(top.len() == 2, "Metal decode did not produce a top-2");
        current = top[0].token_id;
        tokens.push(current);
        gaps.push(top[0].logit - top[1].logit);
    }
    Ok((tokens, gaps))
}

fn read_u16(path: &Path) -> Result<Vec<u16>> {
    let bytes = fs::read(path).with_context(|| format!("read cache {}", path.display()))?;
    ensure!(bytes.len() % 2 == 0, "cache byte length is not even");
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn read_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("read logits {}", path.display()))?;
    ensure!(
        bytes.len() % 4 == 0,
        "logits byte length is not divisible by four"
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * fraction) as usize;
    sorted[index]
}
