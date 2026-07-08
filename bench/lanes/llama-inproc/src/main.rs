use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use clap::{Parser, ValueEnum};
use lane_llama_inproc::{
    default_threads, embed_token_batches, format_optional_prefix, load_runtime, load_tokenizer,
    max_batch_sequences, new_context, prefixed_text, prepare_texts, rate, warmup_context,
    FlashAttentionSetting, ForwardPass, PoolingImplementation, PoolingMode, PreparedText,
    RuntimeConfig, LLAMA_CPP_2_VERSION, MAX_BATCH_SEQUENCES,
};
use synapse_bench::{
    parity::{load_corpus, load_reference, mean_parity, Chunk},
    results::LaneResult,
};

#[derive(Parser)]
#[command(name = "lane-llama-inproc")]
struct Args {
    /// Path to the embedding GGUF model.
    #[arg(long)]
    model: PathBuf,
    /// Path to tokenizer.json.
    #[arg(long)]
    tokenizer: PathBuf,
    /// Corpus JSONL ({id, text} per line).
    #[arg(long)]
    corpus: PathBuf,
    /// Output LaneResult JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Optional: write produced vectors (JSONL: {id, vec}) for parity reference.
    #[arg(long, visible_alias = "emit-vectors")]
    vectors_out: Option<PathBuf>,
    /// Optional parity reference JSONL ({id, vec}).
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Minimum allowed mean cosine when --reference is set.
    #[arg(long, default_value_t = 0.98)]
    min_parity: f64,
    /// Optional corpus limit.
    #[arg(long)]
    limit: Option<usize>,
    /// Model label for the result.
    #[arg(long)]
    model_label: String,
    /// Pooling policy.
    #[arg(long, value_enum, default_value_t = PoolingArg::Last)]
    pooling: PoolingArg,
    /// Whether embeddings come from llama.cpp sequence pooling or local token pooling.
    #[arg(long, value_enum, default_value_t = PoolingImplementationArg::Builtin)]
    pooling_implementation: PoolingImplementationArg,
    /// Tokenizer truncation max length.
    #[arg(long, default_value_t = 512)]
    max_length: usize,
    /// Optional string prepended to every corpus text before tokenization.
    #[arg(long)]
    prefix_document: Option<String>,
    /// Logical token budget per inference batch.
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

#[derive(Debug)]
struct ProducedVector {
    id: String,
    vec: Vec<f32>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.batch_size > 0, "batch-size must be > 0");
    ensure!(args.ubatch_size > 0, "ubatch-size must be > 0");
    ensure!(args.ctx_size > 0, "ctx-size must be > 0");
    ensure!(args.max_length > 0, "max-length must be > 0");

    let tokenizer = load_tokenizer(&args.tokenizer, args.max_length)?;
    let mut chunks: Vec<Chunk> = load_corpus(&args.corpus, None)?;
    if let Some(limit) = args.limit {
        chunks.truncate(limit);
    }
    ensure!(
        !chunks.is_empty(),
        "empty corpus: {}",
        args.corpus.display()
    );

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
    let forward_pass = resolve_forward_pass(args.forward_pass, pooling)?;
    let threads = args.threads.unwrap_or_else(default_threads);
    let batch_token_budget = match forward_pass {
        ForwardPass::Encode => args.batch_size.min(args.ubatch_size),
        ForwardPass::Decode => args.batch_size,
    };

    let corpus_inputs: Vec<(String, String)> = chunks
        .iter()
        .map(|chunk| {
            (
                chunk.id.clone(),
                prefixed_text(args.prefix_document.as_deref(), &chunk.text),
            )
        })
        .collect();
    let mut prepared = prepare_texts(&tokenizer, &corpus_inputs)?;
    prepared.sort_by_key(PreparedText::token_count);
    let n_seq_max = max_batch_sequences(&prepared, batch_token_budget)?;

    let runtime_config = RuntimeConfig {
        ctx_size: args.ctx_size,
        batch_size: args.batch_size,
        ubatch_size: args.ubatch_size,
        n_seq_max,
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

    let mut vectors_writer = match &args.vectors_out {
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            Some(BufWriter::new(File::create(path)?))
        }
        None => None,
    };

    let infer_started = Instant::now();
    let mut batch_start = 0usize;
    let mut batch_tokens = 0usize;
    let mut input_tokens = 0u64;
    let mut produced: Vec<Option<ProducedVector>> = std::iter::repeat_with(|| None)
        .take(prepared.len())
        .collect();

    for index in 0..=prepared.len() {
        let should_flush = if index == prepared.len() {
            index > batch_start
        } else {
            let next = prepared[index].token_count().max(1);
            let count = index - batch_start;
            count > 0
                && (batch_tokens + next > batch_token_budget || count + 1 > MAX_BATCH_SEQUENCES)
        };

        if should_flush {
            let batch = &prepared[batch_start..index];
            let batch_tokens_only: Vec<Vec<i32>> =
                batch.iter().map(|item| item.token_ids.clone()).collect();
            let embeddings = embed_token_batches(
                &mut context,
                &batch_tokens_only,
                pooling,
                pooling_implementation,
                forward_pass,
            )?;
            ensure!(
                embeddings.len() == batch.len(),
                "embedding count mismatch: got {}, expected {}",
                embeddings.len(),
                batch.len()
            );

            for (item, vector) in batch.iter().zip(embeddings) {
                input_tokens += item.token_count() as u64;
                if let Some(writer) = vectors_writer.as_mut() {
                    serde_json::to_writer(
                        &mut *writer,
                        &serde_json::json!({"id": item.id, "vec": vector}),
                    )?;
                    writer.write_all(b"\n")?;
                }
                produced[item.original_index] = Some(ProducedVector {
                    id: item.id.clone(),
                    vec: vector,
                });
            }

            batch_start = index;
            batch_tokens = 0;
            if index == prepared.len() {
                break;
            }
        }

        if index < prepared.len() {
            batch_tokens += prepared[index].token_count().max(1);
        }
    }

    if let Some(mut writer) = vectors_writer {
        writer.flush()?;
    }

    let infer_wall_s = infer_started.elapsed().as_secs_f64();
    let produced: Vec<ProducedVector> = produced
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .context("missing vectors when restoring original output order")?;

    let (parity_mean_cosine, parity_matches) = match &args.reference {
        Some(reference) => {
            let reference_vectors = load_reference(reference)?;
            let (mean_cosine, matches) = mean_parity(
                produced
                    .iter()
                    .map(|vector| (vector.id.clone(), vector.vec.clone())),
                &reference_vectors,
            );
            ensure!(matches > 0, "no overlapping ids with reference vectors");
            let mean_cosine = mean_cosine.expect("matched count implies a parity mean");
            ensure!(
                mean_cosine >= args.min_parity,
                "parity {:.6} is below {:.6}; check forward-pass and pooling",
                mean_cosine,
                args.min_parity,
            );
            (Some(mean_cosine), matches)
        }
        None => (None, 0),
    };

    let notes = format!(
        "llama-cpp-2={LLAMA_CPP_2_VERSION}, in-process=true, pooling={pooling:?}, pooling_implementation={pooling_implementation:?}, flash_attention={flash_attention:?}, forward_pass={forward_pass:?}, tokenizer=hf-tokenizers, batching=length_sorted_sum_tokens<=effective_budget&&seqs<=256, effective_batch_budget={}, ctx_size={}, batch_size={}, ubatch_size={}, ngl={}, threads={}, prefix_document={}, min_parity={}, reference_matches={}",
        batch_token_budget,
        args.ctx_size,
        args.batch_size,
        args.ubatch_size,
        args.gpu_layers,
        threads,
        format_optional_prefix(args.prefix_document.as_deref()),
        args.min_parity,
        parity_matches,
    );

    let result = LaneResult {
        lane: "llama-inproc-embed".into(),
        workload: "embed-corpus-v1".into(),
        model: args.model_label,
        cold_load_s,
        infer_wall_s,
        input_tokens,
        tok_per_s: rate(input_tokens as f64, infer_wall_s),
        items: produced.len() as u64,
        parity_mean_cosine,
        self_peak_rss_bytes: None,
        notes,
    };
    write_result(&args.out, &result)?;
    eprintln!(
        "llama-inproc-embed: {} items, {} tokens, {:.1} tok/s, cold_load {:.2}s, infer {:.2}s{}",
        result.items,
        result.input_tokens,
        result.tok_per_s,
        result.cold_load_s,
        result.infer_wall_s,
        result
            .parity_mean_cosine
            .map(|value| format!(", parity {:.6}", value))
            .unwrap_or_default()
    );
    Ok(())
}

fn resolve_forward_pass(forward_pass: ForwardPassArg, pooling: PoolingMode) -> Result<ForwardPass> {
    match forward_pass {
        ForwardPassArg::Auto => match pooling {
            PoolingMode::Mean | PoolingMode::Cls => Ok(ForwardPass::Encode),
            PoolingMode::Last => Ok(ForwardPass::Decode),
        },
        ForwardPassArg::Encode => Ok(ForwardPass::Encode),
        ForwardPassArg::Decode => Ok(ForwardPass::Decode),
    }
}

fn write_result(path: &Path, result: &LaneResult) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(result)?)?;
    Ok(())
}
