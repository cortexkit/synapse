use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::{Parser, ValueEnum};
use lane_llama_inproc::{
    default_threads, embed_token_batches, load_runtime, load_tokenizer, new_context,
    summarize_latencies_ms, tokenize_text, warmup_context, FlashAttentionSetting, ForwardPass,
    PoolingImplementation, PoolingMode, RuntimeConfig,
};
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
    /// Number of layers to place on the GPU.
    #[arg(long, default_value_t = 99)]
    gpu_layers: usize,
    /// Embedding forward pass. Auto uses encode for mean/cls and decode for last.
    #[arg(long, value_enum, default_value_t = ForwardPassArg::Auto)]
    forward_pass: ForwardPassArg,
    /// Flash attention policy.
    #[arg(long, value_enum, default_value_t = FlashAttentionArg::Auto)]
    flash_attention: FlashAttentionArg,
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

#[derive(Serialize)]
struct LatencyReport {
    cold_load_s: f64,
    iterations: usize,
    input_tokens: usize,
    p50_ms: f64,
    p95_ms: f64,
    max_ms: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.iterations > 0, "iterations must be > 0");
    ensure!(args.batch_size > 0, "batch-size must be > 0");
    ensure!(args.ubatch_size > 0, "ubatch-size must be > 0");
    ensure!(args.ctx_size > 0, "ctx-size must be > 0");
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
    let threads = args.threads.unwrap_or_else(default_threads);
    let runtime_config = RuntimeConfig {
        ctx_size: args.ctx_size,
        batch_size: args.batch_size,
        ubatch_size: args.ubatch_size,
        n_seq_max: 1,
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

    let mut latencies_ms = Vec::with_capacity(args.iterations);
    for _ in 0..args.iterations {
        let started = Instant::now();
        let _ = embed_token_batches(
            &mut context,
            std::slice::from_ref(&token_ids),
            pooling,
            pooling_implementation,
            forward_pass,
        )?;
        latencies_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }

    let summary =
        summarize_latencies_ms(&latencies_ms).context("latency summary requires samples")?;
    let report = LatencyReport {
        cold_load_s,
        iterations: args.iterations,
        input_tokens: token_ids.len(),
        p50_ms: summary.p50_ms,
        p95_ms: summary.p95_ms,
        max_ms: summary.max_ms,
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

fn write_report(path: &Path, report: &LatencyReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}
