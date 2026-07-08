use std::num::NonZeroU32;
use std::path::Path;
use std::time::Instant;

use anyhow::{ensure, Context, Result};
use llama_cpp_2::{
    context::{
        kv_cache::KvCacheConversionError,
        params::{LlamaAttentionType, LlamaContextParams, LlamaPoolingType},
        LlamaContext,
    },
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, LlamaModel},
    token::LlamaToken,
};
use llama_cpp_sys_2::{
    llama_flash_attn_type, LLAMA_FLASH_ATTN_TYPE_AUTO, LLAMA_FLASH_ATTN_TYPE_DISABLED,
    LLAMA_FLASH_ATTN_TYPE_ENABLED,
};
use tokenizers::{Tokenizer, TruncationParams};

pub const WARMUP_TEXT: &str = "warmup";
pub const LLAMA_CPP_2_VERSION: &str = "0.1.151";
pub const MAX_BATCH_SEQUENCES: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoolingMode {
    Mean,
    Last,
    Cls,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PoolingImplementation {
    Builtin,
    Manual,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FlashAttentionSetting {
    Auto,
    Enabled,
    Disabled,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ForwardPass {
    Encode,
    Decode,
}

impl PoolingImplementation {
    #[must_use]
    pub fn context_pooling_type(self, pooling: PoolingMode) -> LlamaPoolingType {
        match self {
            Self::Builtin => match pooling {
                PoolingMode::Mean => LlamaPoolingType::Mean,
                PoolingMode::Last => LlamaPoolingType::Last,
                PoolingMode::Cls => LlamaPoolingType::Cls,
            },
            Self::Manual => LlamaPoolingType::None,
        }
    }
}

impl FlashAttentionSetting {
    #[must_use]
    pub fn raw_policy(self) -> llama_flash_attn_type {
        match self {
            Self::Auto => LLAMA_FLASH_ATTN_TYPE_AUTO,
            Self::Enabled => LLAMA_FLASH_ATTN_TYPE_ENABLED,
            Self::Disabled => LLAMA_FLASH_ATTN_TYPE_DISABLED,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub ctx_size: usize,
    pub batch_size: usize,
    pub ubatch_size: usize,
    pub n_seq_max: usize,
    pub gpu_layers: usize,
    pub threads: usize,
    pub pooling: PoolingMode,
    pub pooling_implementation: PoolingImplementation,
    pub flash_attention: FlashAttentionSetting,
    pub forward_pass: ForwardPass,
}

#[derive(Debug)]
pub struct LoadedRuntime {
    backend: LlamaBackend,
    model: LlamaModel,
}

#[derive(Debug, Clone)]
pub struct PreparedText {
    pub id: String,
    pub original_index: usize,
    pub token_ids: Vec<i32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResetPolicy {
    Sequence,
    Context,
    None,
}

#[derive(Clone, Copy, Debug)]
pub struct SequenceRef<'a> {
    pub seq_id: i32,
    pub token_ids: &'a [i32],
}

#[derive(Clone, Copy, Debug)]
pub struct EmbedCallMetrics {
    pub batch_build_ms: f64,
    pub reset_ms: f64,
    pub infer_ms: f64,
    pub pool_ms: f64,
}

impl PreparedText {
    #[must_use]
    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }
}

pub fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|parallelism| parallelism.get())
        .unwrap_or(1)
}

pub fn load_tokenizer(path: &Path, max_length: usize) -> Result<Tokenizer> {
    ensure!(max_length > 0, "max-length must be > 0");
    let mut tokenizer =
        Tokenizer::from_file(path).map_err(|err| anyhow::anyhow!("tokenizer: {err}"))?;
    // Some published tokenizer.json artifacts (e.g. Qdrant's MiniLM export) bake in
    // fixed-length padding. llama.cpp batches have no attention-mask concept, so pad
    // tokens would enter the forward pass as real tokens and poison pooled embeddings.
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|err| anyhow::anyhow!("truncation: {err}"))?;
    Ok(tokenizer)
}

pub fn tokenize_text(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i32>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|err| anyhow::anyhow!("encode: {err}"))?;
    encoding
        .get_ids()
        .iter()
        .copied()
        .map(|token| i32::try_from(token).context("token id does not fit into i32"))
        .collect()
}

pub fn prepare_texts(
    tokenizer: &Tokenizer,
    inputs: &[(String, String)],
) -> Result<Vec<PreparedText>> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }

    let texts: Vec<&str> = inputs.iter().map(|(_, text)| text.as_str()).collect();
    let encodings = tokenizer
        .encode_batch(texts, true)
        .map_err(|err| anyhow::anyhow!("encode_batch: {err}"))?;

    inputs
        .iter()
        .zip(encodings)
        .enumerate()
        .map(|(original_index, ((id, _), encoding))| {
            let token_ids = encoding
                .get_ids()
                .iter()
                .copied()
                .map(|token| i32::try_from(token).context("token id does not fit into i32"))
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                !token_ids.is_empty(),
                "tokenization produced no tokens for corpus item {id}"
            );
            Ok(PreparedText {
                id: id.clone(),
                original_index,
                token_ids,
            })
        })
        .collect()
}

pub fn load_runtime(model_path: &Path, config: &RuntimeConfig) -> Result<LoadedRuntime> {
    ensure!(config.ctx_size > 0, "ctx-size must be > 0");
    ensure!(config.batch_size > 0, "batch-size must be > 0");
    ensure!(config.ubatch_size > 0, "ubatch-size must be > 0");
    ensure!(config.n_seq_max > 0, "n-seq-max must be > 0");
    ensure!(config.threads > 0, "threads must be > 0");

    let backend = LlamaBackend::init().context("llama backend init")?;
    let model_params = LlamaModelParams::default().with_n_gpu_layers(
        u32::try_from(config.gpu_layers).context("gpu-layers does not fit into u32")?,
    );
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| format!("load gguf model {}", model_path.display()))?;
    Ok(LoadedRuntime { backend, model })
}

pub fn new_context<'a>(
    runtime: &'a LoadedRuntime,
    config: &RuntimeConfig,
) -> Result<LlamaContext<'a>> {
    let total_ctx_size = config
        .ctx_size
        .checked_mul(config.n_seq_max)
        .context("ctx-size * n-seq-max overflowed")?;
    let ctx_size =
        NonZeroU32::new(u32::try_from(total_ctx_size).context("ctx-size does not fit into u32")?)
            .context("ctx-size must be > 0")?;
    let batch_size =
        u32::try_from(config.batch_size).context("batch-size does not fit into u32")?;
    let ubatch_size =
        u32::try_from(config.ubatch_size).context("ubatch-size does not fit into u32")?;
    let n_seq_max = u32::try_from(config.n_seq_max.min(MAX_BATCH_SEQUENCES))
        .context("sequence cap does not fit into u32")?;
    let threads = i32::try_from(config.threads).context("threads does not fit into i32")?;
    let attention_type = match config.forward_pass {
        ForwardPass::Encode => LlamaAttentionType::NonCausal,
        ForwardPass::Decode => LlamaAttentionType::Causal,
    };

    let params = LlamaContextParams::default()
        .with_n_ctx(Some(ctx_size))
        .with_n_batch(batch_size)
        .with_n_ubatch(ubatch_size)
        .with_n_seq_max(n_seq_max)
        .with_n_threads(threads)
        .with_n_threads_batch(threads)
        .with_embeddings(true)
        .with_pooling_type(
            config
                .pooling_implementation
                .context_pooling_type(config.pooling),
        )
        .with_attention_type(attention_type)
        .with_flash_attention_policy(config.flash_attention.raw_policy());

    runtime
        .model
        .new_context(&runtime.backend, params)
        .context("create llama context")
}

pub fn warmup_context(
    tokenizer: &Tokenizer,
    context: &mut LlamaContext<'_>,
    pooling: PoolingMode,
    pooling_implementation: PoolingImplementation,
    forward_pass: ForwardPass,
) -> Result<()> {
    let warmup_ids = tokenize_text(tokenizer, WARMUP_TEXT)?;
    let _ = embed_token_batches(
        context,
        &[warmup_ids],
        pooling,
        pooling_implementation,
        forward_pass,
    )?;
    Ok(())
}

pub fn embed_token_batches(
    context: &mut LlamaContext<'_>,
    sequences: &[Vec<i32>],
    pooling: PoolingMode,
    pooling_implementation: PoolingImplementation,
    forward_pass: ForwardPass,
) -> Result<Vec<Vec<f32>>> {
    let sequences = sequences
        .iter()
        .enumerate()
        .map(|(seq_id, token_ids)| {
            Ok(SequenceRef {
                seq_id: i32::try_from(seq_id).context("sequence id does not fit into i32")?,
                token_ids,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let (embeddings, _) = embed_sequences_with_metrics(
        context,
        &sequences,
        pooling,
        pooling_implementation,
        forward_pass,
        ResetPolicy::Sequence,
    )?;
    Ok(embeddings)
}

pub fn embed_sequences_with_metrics(
    context: &mut LlamaContext<'_>,
    sequences: &[SequenceRef<'_>],
    pooling: PoolingMode,
    pooling_implementation: PoolingImplementation,
    forward_pass: ForwardPass,
    reset_policy: ResetPolicy,
) -> Result<(Vec<Vec<f32>>, EmbedCallMetrics)> {
    ensure!(!sequences.is_empty(), "cannot embed an empty batch");

    let total_tokens = sequences
        .iter()
        .map(|sequence| sequence.token_ids.len())
        .sum::<usize>();
    ensure!(total_tokens > 0, "cannot embed a batch with zero tokens");

    let batch_build_started = Instant::now();
    let seq_count =
        i32::try_from(sequences.len()).context("sequence count does not fit into i32")?;
    let mut batch = LlamaBatch::new(total_tokens, seq_count);

    for sequence in sequences {
        ensure!(sequence.seq_id >= 0, "sequence ids must be non-negative");
        ensure!(
            !sequence.token_ids.is_empty(),
            "sequence {} has zero tokens",
            sequence.seq_id
        );
        let llama_tokens: Vec<LlamaToken> = sequence
            .token_ids
            .iter()
            .copied()
            .map(LlamaToken::new)
            .collect();
        batch
            .add_sequence(&llama_tokens, sequence.seq_id, true)
            .with_context(|| format!("add sequence {} to llama batch", sequence.seq_id))?;
    }
    let batch_build_ms = batch_build_started.elapsed().as_secs_f64() * 1000.0;

    let reset_started = Instant::now();
    reset_sequence_state(context, sequences, reset_policy)?;
    let reset_ms = reset_started.elapsed().as_secs_f64() * 1000.0;

    let infer_started = Instant::now();
    match forward_pass {
        ForwardPass::Encode => context.encode(&mut batch).context("llama_encode failed")?,
        ForwardPass::Decode => context.decode(&mut batch).context("llama_decode failed")?,
    }
    let infer_ms = infer_started.elapsed().as_secs_f64() * 1000.0;

    let pool_started = Instant::now();
    let embeddings = match pooling_implementation {
        PoolingImplementation::Builtin => collect_builtin_embeddings(context, sequences)?,
        PoolingImplementation::Manual => {
            let mut pooled = Vec::with_capacity(sequences.len());
            let mut token_offset = 0i32;
            for sequence in sequences {
                pooled.push(pool_sequence(
                    context,
                    token_offset,
                    sequence.token_ids.len(),
                    pooling,
                )?);
                token_offset += i32::try_from(sequence.token_ids.len())
                    .context("token count does not fit into i32")?;
            }
            pooled
        }
    };
    let pool_ms = pool_started.elapsed().as_secs_f64() * 1000.0;

    Ok((
        embeddings,
        EmbedCallMetrics {
            batch_build_ms,
            reset_ms,
            infer_ms,
            pool_ms,
        },
    ))
}

fn collect_builtin_embeddings(
    context: &LlamaContext<'_>,
    sequences: &[SequenceRef<'_>],
) -> Result<Vec<Vec<f32>>> {
    let mut embeddings = Vec::with_capacity(sequences.len());
    for sequence in sequences {
        let mut vector = context
            .embeddings_seq_ith(sequence.seq_id)
            .with_context(|| format!("read sequence embedding {}", sequence.seq_id))?
            .to_vec();
        l2_normalize(&mut vector);
        embeddings.push(vector);
    }
    Ok(embeddings)
}

fn reset_sequence_state(
    context: &mut LlamaContext<'_>,
    sequences: &[SequenceRef<'_>],
    reset_policy: ResetPolicy,
) -> Result<()> {
    match reset_policy {
        ResetPolicy::None => Ok(()),
        ResetPolicy::Context => {
            context.clear_kv_cache();
            Ok(())
        }
        ResetPolicy::Sequence => {
            let mut seen = Vec::with_capacity(sequences.len());
            for sequence in sequences {
                if seen.contains(&sequence.seq_id) {
                    continue;
                }
                context
                    .clear_kv_cache_seq(
                        Some(
                            u32::try_from(sequence.seq_id)
                                .context("sequence id does not fit into u32")?,
                        ),
                        None,
                        None,
                    )
                    .map_err(kv_reset_error)?;
                seen.push(sequence.seq_id);
            }
            Ok(())
        }
    }
}

fn kv_reset_error(error: KvCacheConversionError) -> anyhow::Error {
    anyhow::anyhow!("reset sequence cache: {error}")
}

fn pool_sequence(
    context: &LlamaContext<'_>,
    start_token: i32,
    token_count: usize,
    pooling: PoolingMode,
) -> Result<Vec<f32>> {
    ensure!(token_count > 0, "cannot pool an empty sequence");

    let mut vector = match pooling {
        PoolingMode::Mean => {
            let first = context
                .embeddings_ith(start_token)
                .context("read first token embedding")?;
            let mut pooled = vec![0.0f32; first.len()];
            for index in 0..token_count {
                let embedding = context
                    .embeddings_ith(
                        start_token
                            + i32::try_from(index).context("token index does not fit into i32")?,
                    )
                    .with_context(|| format!("read token embedding {index}"))?;
                for (slot, value) in pooled.iter_mut().zip(embedding.iter().copied()) {
                    *slot += value;
                }
            }
            let denom = token_count as f32;
            pooled.iter_mut().for_each(|value| *value /= denom);
            pooled
        }
        PoolingMode::Last => context
            .embeddings_ith(
                start_token
                    + i32::try_from(token_count - 1)
                        .context("last token index does not fit into i32")?,
            )
            .context("read last token embedding")?
            .to_vec(),
        PoolingMode::Cls => context
            .embeddings_ith(start_token)
            .context("read cls token embedding")?
            .to_vec(),
    };

    l2_normalize(&mut vector);
    Ok(vector)
}

pub fn l2_normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt() + 1e-12;
    vector.iter_mut().for_each(|value| *value /= norm);
}

pub fn prefixed_text(prefix_document: Option<&str>, text: &str) -> String {
    match prefix_document {
        Some(prefix) => format!("{prefix}{text}"),
        None => text.to_owned(),
    }
}

pub fn format_optional_prefix(prefix_document: Option<&str>) -> String {
    prefix_document
        .map(|prefix| format!("{prefix:?}"))
        .unwrap_or_else(|| "none".to_string())
}

pub fn max_batch_sequences(prepared: &[PreparedText], batch_token_budget: usize) -> Result<usize> {
    ensure!(batch_token_budget > 0, "batch token budget must be > 0");
    let mut batch_start = 0usize;
    let mut batch_tokens = 0usize;
    let mut max_sequences = 0usize;

    for index in 0..=prepared.len() {
        let should_flush = if index == prepared.len() {
            index > batch_start
        } else {
            let next = prepared[index].token_count().max(1);
            ensure!(
                next <= batch_token_budget,
                "tokenized input length {} exceeds effective batch budget {}",
                next,
                batch_token_budget
            );
            let count = index - batch_start;
            count > 0
                && (batch_tokens + next > batch_token_budget || count + 1 > MAX_BATCH_SEQUENCES)
        };

        if should_flush {
            max_sequences = max_sequences.max(index - batch_start);
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

    Ok(max_sequences.max(1))
}

pub fn rate(tokens: f64, seconds: f64) -> f64 {
    if seconds <= f64::MIN_POSITIVE {
        0.0
    } else {
        tokens / seconds
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LatencySummary {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub max_ms: f64,
}

pub fn summarize_latencies_ms(latencies_ms: &[f64]) -> Option<LatencySummary> {
    if latencies_ms.is_empty() {
        return None;
    }
    let mut sorted = latencies_ms.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(LatencySummary {
        p50_ms: percentile(&sorted, 0.50),
        p95_ms: percentile(&sorted, 0.95),
        max_ms: *sorted.last().expect("latencies are non-empty"),
    })
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}
