use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::{Parser, ValueEnum};
use lane_llama_inproc::{
    default_threads, embed_sequences_with_metrics, l2_normalize, load_runtime, load_tokenizer,
    new_context, summarize_latencies_ms, tokenize_text, warmup_context, FlashAttentionSetting,
    ForwardPass, PoolingImplementation, PoolingMode, ResetPolicy, RuntimeConfig, SequenceRef,
};
use llama_cpp_2::{llama_batch::LlamaBatch, token::LlamaToken};
use serde::Serialize;

#[derive(Parser)]
#[command(name = "lane-llama-inproc-latency")]
struct Args {
    /// Path to the embedding GGUF model.
    #[arg(long)]
    model: PathBuf,
    /// Path to tokenizer.json.
    #[arg(long)]
    tokenizer: PathBuf,
    /// Text to embed repeatedly.
    #[arg(long)]
    text: String,
    /// Optional JSON output path.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Number of timed embedding calls.
    #[arg(long, default_value_t = 200)]
    iterations: usize,
    /// Pooling policy.
    #[arg(long, value_enum, default_value_t = PoolingArg::Last)]
    pooling: PoolingArg,
    /// Whether embeddings come from llama.cpp sequence pooling or local token pooling.
    #[arg(long, value_enum, default_value_t = PoolingImplementationArg::Builtin)]
    pooling_implementation: PoolingImplementationArg,
    /// Tokenizer truncation max length.
    #[arg(long, default_value_t = 512)]
    max_length: usize,
    /// Logical token budget passed to llama.cpp.
    #[arg(long, default_value_t = 4096)]
    batch_size: usize,
    /// Physical token budget passed to llama.cpp.
    #[arg(long, default_value_t = 1024)]
    ubatch_size: usize,
    /// Context window passed to llama.cpp.
    #[arg(long, default_value_t = 1024)]
    ctx_size: usize,
    /// Number of distinct sequence slots available in the context.
    #[arg(long, default_value_t = 1)]
    n_seq_max: usize,
    /// Keep the first N tokens resident and re-evaluate only the suffix on timed calls.
    #[arg(long)]
    reuse_prefix_tokens: Option<usize>,
    /// Number of layers to place on the GPU.
    #[arg(long, default_value_t = 99)]
    gpu_layers: usize,
    /// Embedding forward pass. Auto uses encode for mean/cls and decode for last.
    #[arg(long, value_enum, default_value_t = ForwardPassArg::Auto)]
    forward_pass: ForwardPassArg,
    /// Flash attention policy.
    #[arg(long, value_enum, default_value_t = FlashAttentionArg::Auto)]
    flash_attention: FlashAttentionArg,
    /// How KV state is reset before each timed call.
    #[arg(long, value_enum, default_value_t = ResetPolicyArg::Sequence)]
    reset_policy: ResetPolicyArg,
    /// How sequence ids are assigned across timed calls.
    #[arg(long, value_enum, default_value_t = SequenceStrategyArg::Fixed)]
    sequence_strategy: SequenceStrategyArg,
    /// CPU thread count used by llama.cpp.
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PoolingArg {
    Mean,
    Last,
    Cls,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ForwardPassArg {
    Auto,
    Encode,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PoolingImplementationArg {
    Builtin,
    Manual,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum FlashAttentionArg {
    Auto,
    Enabled,
    Disabled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ResetPolicyArg {
    Sequence,
    Context,
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum SequenceStrategyArg {
    Fixed,
    Rotate,
}

#[derive(Serialize)]
struct StepReport {
    p50_ms: f64,
    p95_ms: f64,
    max_ms: f64,
}

#[derive(Serialize)]
struct LatencyReport {
    cold_load_s: f64,
    iterations: usize,
    input_tokens: usize,
    p50_ms: f64,
    p95_ms: f64,
    max_ms: f64,
    batch_build_ms: StepReport,
    reset_ms: StepReport,
    infer_ms: StepReport,
    pool_ms: StepReport,
    reset_policy: &'static str,
    sequence_strategy: &'static str,
    n_seq_max: usize,
    reuse_prefix_tokens: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.iterations > 0, "iterations must be > 0");
    ensure!(args.batch_size > 0, "batch-size must be > 0");
    ensure!(args.ubatch_size > 0, "ubatch-size must be > 0");
    ensure!(args.ctx_size > 0, "ctx-size must be > 0");
    ensure!(args.n_seq_max > 0, "n-seq-max must be > 0");
    ensure!(args.max_length > 0, "max-length must be > 0");

    let tokenizer = load_tokenizer(&args.tokenizer, args.max_length)?;
    let pooling = match args.pooling {
        PoolingArg::Mean => PoolingMode::Mean,
        PoolingArg::Last => PoolingMode::Last,
        PoolingArg::Cls => PoolingMode::Cls,
    };
    let pooling_implementation = match args.pooling_implementation {
        PoolingImplementationArg::Builtin => PoolingImplementation::Builtin,
        PoolingImplementationArg::Manual => PoolingImplementation::Manual,
    };
    let flash_attention = match args.flash_attention {
        FlashAttentionArg::Auto => FlashAttentionSetting::Auto,
        FlashAttentionArg::Enabled => FlashAttentionSetting::Enabled,
        FlashAttentionArg::Disabled => FlashAttentionSetting::Disabled,
    };
    let forward_pass = resolve_forward_pass(args.forward_pass, pooling);
    let reset_policy = resolve_reset_policy(args.reset_policy);
    validate_latency_config(
        args.iterations,
        args.n_seq_max,
        reset_policy,
        args.sequence_strategy,
        args.reuse_prefix_tokens,
        pooling,
        pooling_implementation,
        forward_pass,
    )?;

    let threads = args.threads.unwrap_or_else(default_threads);
    let runtime_config = RuntimeConfig {
        ctx_size: args.ctx_size,
        batch_size: args.batch_size,
        ubatch_size: args.ubatch_size,
        n_seq_max: args.n_seq_max,
        gpu_layers: args.gpu_layers,
        threads,
        pooling,
        pooling_implementation,
        flash_attention,
        forward_pass,
    };

    let started = Instant::now();
    let runtime = load_runtime(&args.model, &runtime_config)?;
    let mut context = new_context(&runtime, &runtime_config)?;
    warmup_context(
        &tokenizer,
        &mut context,
        pooling,
        pooling_implementation,
        forward_pass,
    )?;
    let cold_load_s = started.elapsed().as_secs_f64();

    let token_ids = tokenize_text(&tokenizer, &args.text)?;
    ensure!(!token_ids.is_empty(), "text tokenized to zero tokens");

    if let Some(prefix_tokens) = args.reuse_prefix_tokens {
        ensure!(
            prefix_tokens < token_ids.len(),
            "reuse-prefix-tokens must be smaller than the token count"
        );
        let priming_sequence = [SequenceRef {
            seq_id: 0,
            token_ids: &token_ids,
        }];
        let _ = embed_sequences_with_metrics(
            &mut context,
            &priming_sequence,
            pooling,
            pooling_implementation,
            forward_pass,
            ResetPolicy::Sequence,
        )?;
    }

    context.reset_timings();

    let mut latencies_ms = Vec::with_capacity(args.iterations);
    let mut batch_build_latencies_ms = Vec::with_capacity(args.iterations);
    let mut reset_latencies_ms = Vec::with_capacity(args.iterations);
    let mut infer_latencies_ms = Vec::with_capacity(args.iterations);
    let mut pool_latencies_ms = Vec::with_capacity(args.iterations);

    for iteration in 0..args.iterations {
        let started = Instant::now();
        let metrics = if let Some(prefix_tokens) = args.reuse_prefix_tokens {
            embed_cached_suffix_with_metrics(
                &mut context,
                &token_ids,
                prefix_tokens,
                pooling,
                pooling_implementation,
                forward_pass,
            )?
        } else {
            let seq_id = resolve_sequence_id(
                iteration,
                args.n_seq_max,
                reset_policy,
                args.sequence_strategy,
            )?;
            let sequence = [SequenceRef {
                seq_id,
                token_ids: &token_ids,
            }];
            let (_, metrics) = embed_sequences_with_metrics(
                &mut context,
                &sequence,
                pooling,
                pooling_implementation,
                forward_pass,
                reset_policy,
            )?;
            metrics
        };

        latencies_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        batch_build_latencies_ms.push(metrics.batch_build_ms);
        reset_latencies_ms.push(metrics.reset_ms);
        infer_latencies_ms.push(metrics.infer_ms);
        pool_latencies_ms.push(metrics.pool_ms);
    }

    let summary = summarize_required(&latencies_ms, "latency")?;
    let report = LatencyReport {
        cold_load_s,
        iterations: args.iterations,
        input_tokens: token_ids.len(),
        p50_ms: summary.p50_ms,
        p95_ms: summary.p95_ms,
        max_ms: summary.max_ms,
        batch_build_ms: summarize_required(&batch_build_latencies_ms, "batch build")?,
        reset_ms: summarize_required(&reset_latencies_ms, "reset")?,
        infer_ms: summarize_required(&infer_latencies_ms, "infer")?,
        pool_ms: summarize_required(&pool_latencies_ms, "pool")?,
        reset_policy: reset_policy_label(args.reset_policy),
        sequence_strategy: sequence_strategy_label(args.sequence_strategy),
        n_seq_max: args.n_seq_max,
        reuse_prefix_tokens: args.reuse_prefix_tokens,
    };

    if let Some(path) = &args.out {
        write_report(path, &report)?;
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn resolve_forward_pass(forward_pass: ForwardPassArg, pooling: PoolingMode) -> ForwardPass {
    match forward_pass {
        ForwardPassArg::Auto => match pooling {
            PoolingMode::Mean | PoolingMode::Cls => ForwardPass::Encode,
            PoolingMode::Last => ForwardPass::Decode,
        },
        ForwardPassArg::Encode => ForwardPass::Encode,
        ForwardPassArg::Decode => ForwardPass::Decode,
    }
}

fn resolve_reset_policy(reset_policy: ResetPolicyArg) -> ResetPolicy {
    match reset_policy {
        ResetPolicyArg::Sequence => ResetPolicy::Sequence,
        ResetPolicyArg::Context => ResetPolicy::Context,
        ResetPolicyArg::None => ResetPolicy::None,
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_latency_config(
    iterations: usize,
    n_seq_max: usize,
    reset_policy: ResetPolicy,
    sequence_strategy: SequenceStrategyArg,
    reuse_prefix_tokens: Option<usize>,
    pooling: PoolingMode,
    pooling_implementation: PoolingImplementation,
    forward_pass: ForwardPass,
) -> Result<()> {
    if reuse_prefix_tokens.is_some() {
        ensure!(
            matches!(pooling, PoolingMode::Last),
            "reuse-prefix-tokens only supports last-token pooling"
        );
        ensure!(
            matches!(pooling_implementation, PoolingImplementation::Builtin),
            "reuse-prefix-tokens requires builtin sequence pooling"
        );
        ensure!(
            matches!(forward_pass, ForwardPass::Decode),
            "reuse-prefix-tokens requires decode mode"
        );
        ensure!(
            matches!(sequence_strategy, SequenceStrategyArg::Fixed),
            "reuse-prefix-tokens requires sequence-strategy=fixed"
        );
        ensure!(n_seq_max == 1, "reuse-prefix-tokens requires n-seq-max=1");
        ensure!(
            matches!(reset_policy, ResetPolicy::None),
            "reuse-prefix-tokens requires reset-policy=none because it manages suffix clearing internally"
        );
        return Ok(());
    }

    if matches!(sequence_strategy, SequenceStrategyArg::Rotate) {
        ensure!(
            n_seq_max > 1,
            "rotate sequence strategy requires n-seq-max > 1"
        );
    }

    if matches!(reset_policy, ResetPolicy::None) {
        ensure!(
            matches!(sequence_strategy, SequenceStrategyArg::Rotate),
            "reset-policy=none requires sequence-strategy=rotate so each timed call gets a fresh sequence slot"
        );
        ensure!(
            n_seq_max > iterations,
            "reset-policy=none needs at least iterations + 1 sequence slots because warmup occupies sequence 0"
        );
    }

    Ok(())
}

fn resolve_sequence_id(
    iteration: usize,
    n_seq_max: usize,
    reset_policy: ResetPolicy,
    sequence_strategy: SequenceStrategyArg,
) -> Result<i32> {
    let seq_id = match sequence_strategy {
        SequenceStrategyArg::Fixed => 0usize,
        SequenceStrategyArg::Rotate => {
            let first_timed_seq = if matches!(reset_policy, ResetPolicy::None) {
                1usize
            } else {
                0usize
            };
            let timed_slots = n_seq_max
                .checked_sub(first_timed_seq)
                .context("timed sequence slot calculation underflowed")?;
            ensure!(
                timed_slots > 0,
                "no sequence slots available for timed calls"
            );
            first_timed_seq + (iteration % timed_slots)
        }
    };
    i32::try_from(seq_id).context("sequence id does not fit into i32")
}

fn embed_cached_suffix_with_metrics(
    context: &mut llama_cpp_2::context::LlamaContext<'_>,
    token_ids: &[i32],
    prefix_tokens: usize,
    pooling: PoolingMode,
    pooling_implementation: PoolingImplementation,
    forward_pass: ForwardPass,
) -> Result<lane_llama_inproc::EmbedCallMetrics> {
    ensure!(
        prefix_tokens < token_ids.len(),
        "prefix must leave a non-empty suffix"
    );
    ensure!(
        matches!(pooling, PoolingMode::Last),
        "cached suffix path only supports last pooling"
    );
    ensure!(
        matches!(pooling_implementation, PoolingImplementation::Builtin),
        "cached suffix path requires builtin pooling"
    );
    ensure!(
        matches!(forward_pass, ForwardPass::Decode),
        "cached suffix path requires decode mode"
    );

    let suffix = &token_ids[prefix_tokens..];

    let batch_build_started = Instant::now();
    let mut batch = LlamaBatch::new(suffix.len(), 1);
    for (offset, token_id) in suffix.iter().copied().enumerate() {
        batch
            .add(
                LlamaToken::new(token_id),
                i32::try_from(prefix_tokens + offset)
                    .context("token position does not fit into i32")?,
                &[0],
                offset + 1 == suffix.len(),
            )
            .with_context(|| format!("add suffix token {}", prefix_tokens + offset))?;
    }
    let batch_build_ms = batch_build_started.elapsed().as_secs_f64() * 1000.0;

    let reset_started = Instant::now();
    context
        .clear_kv_cache_seq(
            Some(0),
            Some(u32::try_from(prefix_tokens).context("prefix token count does not fit into u32")?),
            None,
        )
        .map_err(|error| anyhow::anyhow!("reset cached suffix: {error}"))?;
    let reset_ms = reset_started.elapsed().as_secs_f64() * 1000.0;

    let infer_started = Instant::now();
    context.decode(&mut batch).context("llama_decode failed")?;
    let infer_ms = infer_started.elapsed().as_secs_f64() * 1000.0;

    let pool_started = Instant::now();
    let mut vector = context
        .embeddings_seq_ith(0)
        .context("read cached sequence embedding")?
        .to_vec();
    l2_normalize(&mut vector);
    let pool_ms = pool_started.elapsed().as_secs_f64() * 1000.0;

    Ok(lane_llama_inproc::EmbedCallMetrics {
        batch_build_ms,
        reset_ms,
        infer_ms,
        pool_ms,
    })
}

fn summarize_required(samples: &[f64], label: &str) -> Result<StepReport> {
    let summary = summarize_latencies_ms(samples)
        .with_context(|| format!("{label} summary requires samples"))?;
    Ok(StepReport {
        p50_ms: summary.p50_ms,
        p95_ms: summary.p95_ms,
        max_ms: summary.max_ms,
    })
}

fn reset_policy_label(reset_policy: ResetPolicyArg) -> &'static str {
    match reset_policy {
        ResetPolicyArg::Sequence => "sequence",
        ResetPolicyArg::Context => "context",
        ResetPolicyArg::None => "none",
    }
}

fn sequence_strategy_label(sequence_strategy: SequenceStrategyArg) -> &'static str {
    match sequence_strategy {
        SequenceStrategyArg::Fixed => "fixed",
        SequenceStrategyArg::Rotate => "rotate",
    }
}

fn write_report(path: &Path, report: &LatencyReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}
