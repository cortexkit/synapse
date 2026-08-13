use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{ensure, Context, Result};
use clap::Parser;
use half::f16;
use serde::{Deserialize, Serialize};
use synapse_engine_owned::owned_decode_engine::{
    top_logits, DecodeKernel, MetalStepDecoder, MetalStepKvCache, Qwen3DecodeModel,
    WeightQuantization,
};
use synapse_engine_owned::Precision;

const LAYERS: usize = 28;
const KV_HEADS: usize = 8;
const HEAD_DIMENSION: usize = 128;
const PREFILL_BATCH: usize = 16;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    ane_cache: PathBuf,
    #[arg(long)]
    ane_logits: PathBuf,
    #[arg(long, requires = "cpu_logits")]
    cpu_cache: Option<PathBuf>,
    #[arg(long, requires = "cpu_cache")]
    cpu_logits: Option<PathBuf>,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 512)]
    cache_bucket: usize,
    #[arg(long, default_value_t = 64)]
    max_new_tokens: usize,
}

#[derive(Deserialize)]
struct TokenizedRow {
    id: String,
    input_ids: Vec<u32>,
    attention_mask: Vec<u8>,
}

#[derive(Serialize)]
struct Candidate {
    token_id: u32,
    logit: f32,
}

#[derive(Serialize)]
struct Divergence {
    generated_token_index: usize,
    oracle_token_id: u32,
    control_token_id: u32,
    oracle_top5: Vec<Candidate>,
    control_top5: Vec<Candidate>,
    oracle_top2_gap: f32,
    control_top2_gap: f32,
    same_top2_token_set: bool,
    max_abs_logit_difference: f32,
    mean_abs_logit_difference: f64,
}

#[derive(Serialize)]
struct DifferenceStats {
    samples: usize,
    max_abs: f32,
    mean_abs: f64,
    p50_abs: f32,
    p95_abs: f32,
    exact_bits_fraction: f64,
}

#[derive(Serialize)]
struct LayerDifference {
    layer: usize,
    stats: DifferenceStats,
}

#[derive(Serialize)]
struct KvFidelity {
    active_positions: usize,
    overall: DifferenceStats,
    per_layer: Vec<LayerDifference>,
    structural_controls: Vec<StructuralControl>,
    admission_roundtrip_bit_mismatches: usize,
}

#[derive(Serialize)]
struct StructuralControl {
    mapping: String,
    samples: usize,
    max_abs: f32,
    mean_abs: f64,
}

#[derive(Serialize)]
struct ControlResult {
    compute_units: String,
    generated_token_ids: Vec<u32>,
    match_depth: usize,
    token_exact: bool,
    divergence: Option<Divergence>,
    kv_vs_pure_gpu: KvFidelity,
}

#[derive(Serialize)]
struct Analysis {
    schema_revision: u32,
    fixture_id: String,
    prompt_tokens: usize,
    cache_bucket: usize,
    max_new_tokens: usize,
    vocabulary_size: usize,
    pure_gpu_generated_token_ids: Vec<u32>,
    controls: Vec<ControlResult>,
    ane_vs_cpu_logits: Option<DifferenceStats>,
    ane_vs_cpu_kv_active_positions: Option<DifferenceStats>,
}

struct Trace {
    tokens: Vec<u32>,
    rows: Vec<Vec<f32>>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.max_new_tokens > 0, "max-new-tokens must be positive");
    let row = read_input(&args.input)?;
    ensure!(!row.input_ids.is_empty(), "prompt must not be empty");
    ensure!(
        row.input_ids.len() == row.attention_mask.len(),
        "input_ids and attention_mask lengths differ"
    );
    ensure!(
        row.attention_mask.iter().all(|&value| value == 1),
        "diagnostic requires a width-exact prompt"
    );
    ensure!(
        row.input_ids.len() + args.max_new_tokens <= args.cache_bucket,
        "prompt plus continuation exceeds cache bucket"
    );

    let model = Qwen3DecodeModel::load(&args.model, Precision::F16)
        .with_context(|| format!("load Qwen3 model {}", args.model.display()))?;
    ensure!(
        model.config.num_key_value_heads == KV_HEADS,
        "unexpected K/V head count"
    );
    ensure!(
        model.config.head_dim == HEAD_DIMENSION,
        "unexpected head dimension"
    );
    let vocabulary_size = model.vocabulary_size();
    let mut decoder = MetalStepDecoder::new(
        &model,
        Precision::F16,
        args.cache_bucket,
        WeightQuantization::None,
    )?;

    let (pure_cache, pure_final_logits) = gpu_prefill(&mut decoder, &row.input_ids)?;
    let pure_cache_bits = collect_cache_bits(&decoder)?;
    let pure_trace = generate(
        &mut decoder,
        pure_cache,
        pure_final_logits,
        args.max_new_tokens,
    )?;

    let ane_cache = read_u16(&args.ane_cache)?;
    let ane_logits = read_f32(&args.ane_logits)?;
    let cpu_outputs = match (&args.cpu_cache, &args.cpu_logits) {
        (Some(cache_path), Some(logits_path)) => {
            Some((read_u16(cache_path)?, read_f32(logits_path)?))
        }
        (None, None) => None,
        _ => unreachable!("clap requires CPU cache and logits together"),
    };
    let expected_cache_elements = LAYERS * 2 * KV_HEADS * args.cache_bucket * HEAD_DIMENSION;
    validate_coreml_output(
        "CPU_AND_NE",
        &ane_cache,
        &ane_logits,
        expected_cache_elements,
        vocabulary_size,
    )?;
    if let Some((cpu_cache, cpu_logits)) = &cpu_outputs {
        validate_coreml_output(
            "CPU_ONLY",
            cpu_cache,
            cpu_logits,
            expected_cache_elements,
            vocabulary_size,
        )?;
    }

    let mut controls = vec![analyze_control(
        "CPU_AND_NE",
        &mut decoder,
        &pure_trace,
        &pure_cache_bits,
        &ane_cache,
        &ane_logits,
        row.input_ids.len(),
        args.cache_bucket,
        args.max_new_tokens,
    )?];
    if let Some((cpu_cache, cpu_logits)) = &cpu_outputs {
        controls.push(analyze_control(
            "CPU_ONLY",
            &mut decoder,
            &pure_trace,
            &pure_cache_bits,
            cpu_cache,
            cpu_logits,
            row.input_ids.len(),
            args.cache_bucket,
            args.max_new_tokens,
        )?);
    }
    let ane_vs_cpu_logits = cpu_outputs
        .as_ref()
        .map(|(_, cpu_logits)| difference_stats_f32(&ane_logits, cpu_logits));
    let ane_vs_cpu_kv_active_positions = cpu_outputs.as_ref().map(|(cpu_cache, _)| {
        kv_difference_stats(
            &ane_cache,
            cpu_cache,
            row.input_ids.len(),
            args.cache_bucket,
        )
        .overall
    });

    let analysis = Analysis {
        schema_revision: 1,
        fixture_id: row.id,
        prompt_tokens: row.input_ids.len(),
        cache_bucket: args.cache_bucket,
        max_new_tokens: args.max_new_tokens,
        vocabulary_size,
        pure_gpu_generated_token_ids: pure_trace.tokens,
        controls,
        ane_vs_cpu_logits,
        ane_vs_cpu_kv_active_positions,
    };
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&analysis)?;
    fs::write(&args.out, format!("{json}\n"))?;
    println!("{json}");
    Ok(())
}

fn validate_coreml_output(
    label: &str,
    cache: &[u16],
    logits: &[f32],
    expected_cache_elements: usize,
    vocabulary_size: usize,
) -> Result<()> {
    ensure!(
        cache.len() == expected_cache_elements,
        "{label} cache has {} values; expected {expected_cache_elements}",
        cache.len()
    );
    ensure!(
        logits.len() == vocabulary_size,
        "{label} logits have {} values; expected {vocabulary_size}",
        logits.len()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn analyze_control(
    compute_units: &str,
    decoder: &mut MetalStepDecoder<'_>,
    pure_trace: &Trace,
    pure_cache_bits: &[u16],
    control_cache_bits: &[u16],
    control_logits: &[f32],
    active_positions: usize,
    cache_bucket: usize,
    max_new_tokens: usize,
) -> Result<ControlResult> {
    let kv_vs_pure_gpu = kv_difference_stats(
        control_cache_bits,
        pure_cache_bits,
        active_positions,
        cache_bucket,
    );
    decoder.import_caches(control_cache_bits)?;
    let admitted = collect_cache_bits(decoder)?;
    let admission_roundtrip_bit_mismatches = admitted
        .iter()
        .zip(control_cache_bits)
        .filter(|(left, right)| left != right)
        .count();
    let trace = generate(
        decoder,
        MetalStepKvCache {
            position: active_positions,
        },
        control_logits.to_vec(),
        max_new_tokens,
    )?;
    let match_depth = pure_trace
        .tokens
        .iter()
        .zip(&trace.tokens)
        .take_while(|(left, right)| left == right)
        .count();
    let divergence = if match_depth < max_new_tokens {
        let oracle_row = &pure_trace.rows[match_depth];
        let control_row = &trace.rows[match_depth];
        let oracle_top5 = top_candidates(oracle_row, 5);
        let control_top5 = top_candidates(control_row, 5);
        let mut oracle_top2 = oracle_top5[..2]
            .iter()
            .map(|candidate| candidate.token_id)
            .collect::<Vec<_>>();
        let mut control_top2 = control_top5[..2]
            .iter()
            .map(|candidate| candidate.token_id)
            .collect::<Vec<_>>();
        oracle_top2.sort_unstable();
        control_top2.sort_unstable();
        let row_difference = difference_stats_f32(oracle_row, control_row);
        Some(Divergence {
            generated_token_index: match_depth,
            oracle_token_id: pure_trace.tokens[match_depth],
            control_token_id: trace.tokens[match_depth],
            oracle_top2_gap: oracle_top5[0].logit - oracle_top5[1].logit,
            control_top2_gap: control_top5[0].logit - control_top5[1].logit,
            same_top2_token_set: oracle_top2 == control_top2,
            oracle_top5,
            control_top5,
            max_abs_logit_difference: row_difference.max_abs,
            mean_abs_logit_difference: row_difference.mean_abs,
        })
    } else {
        None
    };
    Ok(ControlResult {
        compute_units: compute_units.to_owned(),
        generated_token_ids: trace.tokens,
        match_depth,
        token_exact: divergence.is_none(),
        divergence,
        kv_vs_pure_gpu: KvFidelity {
            admission_roundtrip_bit_mismatches,
            ..kv_vs_pure_gpu
        },
    })
}

fn read_input(path: &Path) -> Result<TokenizedRow> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read tokenized fixture {}", path.display()))?;
    let line = text.lines().next().context("tokenized fixture is empty")?;
    serde_json::from_str(line).context("parse tokenized fixture")
}

fn gpu_prefill(
    decoder: &mut MetalStepDecoder<'_>,
    prompt: &[u32],
) -> Result<(MetalStepKvCache, Vec<f32>)> {
    let mut cache = MetalStepKvCache { position: 0 };
    let mut final_logits = Vec::new();
    for chunk in prompt.chunks(PREFILL_BATCH) {
        let logits = decoder.verify_tokens_batch_logits(&mut cache, chunk)?;
        let vocabulary_size = logits.len() / chunk.len();
        final_logits = logits[(chunk.len() - 1) * vocabulary_size..].to_vec();
    }
    Ok((cache, final_logits))
}

fn collect_cache_bits(decoder: &MetalStepDecoder<'_>) -> Result<Vec<u16>> {
    let mut bits = Vec::new();
    for layer in 0..LAYERS {
        bits.extend(decoder.inspect_cache_bits(layer)?);
    }
    Ok(bits)
}

fn generate(
    decoder: &mut MetalStepDecoder<'_>,
    mut cache: MetalStepKvCache,
    mut logits: Vec<f32>,
    count: usize,
) -> Result<Trace> {
    let mut tokens = Vec::with_capacity(count);
    let mut rows = Vec::with_capacity(count);
    for index in 0..count {
        let top = top_candidates(&logits, 1);
        let token = top.first().context("logits row has no candidate")?.token_id;
        rows.push(std::mem::take(&mut logits));
        tokens.push(token);
        if index + 1 < count {
            logits = decoder.advance(&mut cache, token)?;
        }
    }
    Ok(Trace { tokens, rows })
}

fn top_candidates(values: &[f32], count: usize) -> Vec<Candidate> {
    top_logits(values, count)
        .into_iter()
        .map(|candidate| Candidate {
            token_id: candidate.token_id,
            logit: candidate.logit,
        })
        .collect()
}

fn kv_difference_stats(
    left: &[u16],
    right: &[u16],
    active_positions: usize,
    cache_bucket: usize,
) -> KvFidelity {
    let layer_elements = 2 * KV_HEADS * cache_bucket * HEAD_DIMENSION;
    let mut all_differences =
        Vec::with_capacity(LAYERS * 2 * KV_HEADS * active_positions * HEAD_DIMENSION);
    let mut all_exact = 0usize;
    let mut per_layer = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        let mut differences = Vec::with_capacity(2 * KV_HEADS * active_positions * HEAD_DIMENSION);
        let mut exact = 0usize;
        for key_or_value in 0..2 {
            for head in 0..KV_HEADS {
                for position in 0..active_positions {
                    let base = layer * layer_elements
                        + (key_or_value * KV_HEADS + head) * cache_bucket * HEAD_DIMENSION
                        + position * HEAD_DIMENSION;
                    for dimension in 0..HEAD_DIMENSION {
                        let left_bits = left[base + dimension];
                        let right_bits = right[base + dimension];
                        exact += usize::from(left_bits == right_bits);
                        differences.push(
                            (f16::from_bits(left_bits).to_f32()
                                - f16::from_bits(right_bits).to_f32())
                            .abs(),
                        );
                    }
                }
            }
        }
        all_exact += exact;
        all_differences.extend_from_slice(&differences);
        per_layer.push(LayerDifference {
            layer,
            stats: summarize_differences(differences, exact),
        });
    }
    KvFidelity {
        active_positions,
        overall: summarize_differences(all_differences, all_exact),
        per_layer,
        structural_controls: structural_controls(left, right, active_positions, cache_bucket),
        admission_roundtrip_bit_mismatches: 0,
    }
}

fn structural_controls(
    left: &[u16],
    right: &[u16],
    active_positions: usize,
    cache_bucket: usize,
) -> Vec<StructuralControl> {
    [
        "position_plus_one",
        "head_plus_one",
        "key_value_swap",
        "layer_plus_one",
    ]
    .into_iter()
    .map(|mapping| {
        let mut samples = 0usize;
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let layer_elements = 2 * KV_HEADS * cache_bucket * HEAD_DIMENSION;
        for layer in 0..LAYERS {
            for key_or_value in 0..2 {
                for head in 0..KV_HEADS {
                    for position in 0..active_positions {
                        let (mapped_layer, mapped_key_or_value, mapped_head, mapped_position) =
                            match mapping {
                                "position_plus_one" if position + 1 < active_positions => {
                                    (layer, key_or_value, head, position + 1)
                                }
                                "head_plus_one" => {
                                    (layer, key_or_value, (head + 1) % KV_HEADS, position)
                                }
                                "key_value_swap" => (layer, 1 - key_or_value, head, position),
                                "layer_plus_one" if layer + 1 < LAYERS => {
                                    (layer + 1, key_or_value, head, position)
                                }
                                _ => continue,
                            };
                        let left_base = layer * layer_elements
                            + (key_or_value * KV_HEADS + head) * cache_bucket * HEAD_DIMENSION
                            + position * HEAD_DIMENSION;
                        let right_base = mapped_layer * layer_elements
                            + (mapped_key_or_value * KV_HEADS + mapped_head)
                                * cache_bucket
                                * HEAD_DIMENSION
                            + mapped_position * HEAD_DIMENSION;
                        for dimension in 0..HEAD_DIMENSION {
                            let difference = (f16::from_bits(left[left_base + dimension]).to_f32()
                                - f16::from_bits(right[right_base + dimension]).to_f32())
                            .abs();
                            samples += 1;
                            max_abs = max_abs.max(difference);
                            sum_abs += f64::from(difference);
                        }
                    }
                }
            }
        }
        StructuralControl {
            mapping: mapping.to_owned(),
            samples,
            max_abs,
            mean_abs: sum_abs / samples as f64,
        }
    })
    .collect()
}

fn difference_stats_f32(left: &[f32], right: &[f32]) -> DifferenceStats {
    ensure_same_length(left.len(), right.len());
    let mut exact = 0usize;
    let differences = left
        .iter()
        .zip(right)
        .map(|(left, right)| {
            exact += usize::from(left.to_bits() == right.to_bits());
            (left - right).abs()
        })
        .collect();
    summarize_differences(differences, exact)
}

fn ensure_same_length(left: usize, right: usize) {
    assert_eq!(left, right, "diagnostic arrays have different lengths");
}

fn summarize_differences(mut differences: Vec<f32>, exact: usize) -> DifferenceStats {
    differences.sort_unstable_by(f32::total_cmp);
    let samples = differences.len();
    let max_abs = differences.last().copied().unwrap_or(0.0);
    let mean_abs = if samples == 0 {
        0.0
    } else {
        differences
            .iter()
            .map(|&value| f64::from(value))
            .sum::<f64>()
            / samples as f64
    };
    DifferenceStats {
        samples,
        max_abs,
        mean_abs,
        p50_abs: percentile(&differences, 0.50),
        p95_abs: percentile(&differences, 0.95),
        exact_bits_fraction: if samples == 0 {
            1.0
        } else {
            exact as f64 / samples as f64
        },
    }
}

fn percentile(sorted: &[f32], fraction: f64) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * fraction) as usize]
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
