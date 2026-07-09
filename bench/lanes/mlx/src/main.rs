use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use clap::{Args, Parser, Subcommand};
use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms;
use mlx_rs::{Array, Device, Dtype};
use serde::Deserialize;
use synapse_bench::{
    parity::{load_corpus, load_jsonl, load_reference, mean_parity, Chunk, Prompt},
    results::LaneResult,
};
use tokenizers::Tokenizer;

const LABELS: &[&str] = &["config", "test", "logic", "io", "types", "docs"];

#[derive(Parser)]
#[command(name = "lane-mlx")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Workload A: Qwen3 embedding on MLX/Metal.
    Embed(EmbedArgs),
    /// Workload B: Qwen3 micro-LLM one-shot classification on MLX/Metal.
    Microllm(MicrollmArgs),
}

#[derive(Args)]
struct EmbedArgs {
    /// Path to model.safetensors or its snapshot directory.
    #[arg(long)]
    model: PathBuf,
    /// Path to tokenizer.json.
    #[arg(long)]
    tokenizer: PathBuf,
    /// Corpus JSONL ({id, path, text, tokens} per line).
    #[arg(long)]
    corpus: PathBuf,
    /// Output LaneResult JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Optional: write produced vectors (JSONL: {id, vec}) for parity reference.
    #[arg(long)]
    vectors_out: Option<PathBuf>,
    /// Optional: JSONL parity reference ({id, vec}).
    #[arg(long)]
    reference: Option<PathBuf>,
    /// Model label for the result.
    #[arg(long)]
    model_label: String,
    /// Tokenizer truncation max length, including the required trailing EOS token.
    #[arg(long, default_value_t = 512)]
    max_length: usize,
    /// Attention-unit budget per inference batch.
    #[arg(long, default_value_t = 8_000_000)]
    attention_units: usize,
}

#[derive(Args)]
struct MicrollmArgs {
    /// Path to model.safetensors, a safetensors shard, or its snapshot directory.
    #[arg(long)]
    model: PathBuf,
    /// Path to tokenizer.json.
    #[arg(long)]
    tokenizer: PathBuf,
    /// Prompt JSONL ({id, prompt} per line).
    #[arg(long)]
    prompts: PathBuf,
    /// Output LaneResult JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Model label for the result.
    #[arg(long, default_value = "Qwen3-0.6B@mlx-bf16")]
    model_label: String,
    /// Maximum tokens to decode per prompt.
    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,
    /// Optional prompt cap for smoke runs.
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OneOrManyTokenIds {
    One(u32),
    Many(Vec<u32>),
}

impl OneOrManyTokenIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::One(id) => vec![id],
            Self::Many(ids) => ids,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
struct QwenGenerationConfig {
    pad_token_id: Option<u32>,
    eos_token_id: Option<OneOrManyTokenIds>,
}

#[derive(Debug, Deserialize)]
struct QwenConfig {
    hidden_size: i32,
    intermediate_size: i32,
    num_attention_heads: i32,
    num_hidden_layers: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    rms_norm_eps: f32,
    rope_theta: f32,
    vocab_size: i32,
    tie_word_embeddings: bool,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

#[derive(Clone)]
struct LinearWeight {
    weight: Array,
}

impl LinearWeight {
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(ops::matmul(x, self.weight.t())?)
    }
}

#[derive(Clone)]
struct RmsNormWeight {
    weight: Array,
    eps: f32,
}

impl RmsNormWeight {
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(fast::rms_norm(x, &self.weight, self.eps)?)
    }
}

#[derive(Clone)]
struct DecoderLayer {
    input_layernorm: RmsNormWeight,
    post_attention_layernorm: RmsNormWeight,
    q_proj: LinearWeight,
    q_norm: RmsNormWeight,
    k_proj: LinearWeight,
    k_norm: RmsNormWeight,
    v_proj: LinearWeight,
    o_proj: LinearWeight,
    gate_proj: LinearWeight,
    up_proj: LinearWeight,
    down_proj: LinearWeight,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_theta: f32,
}

#[derive(Clone, Default)]
struct LayerCache {
    keys: Option<Array>,
    values: Option<Array>,
}

struct QwenModel {
    config: QwenConfig,
    pad_token_id: u32,
    stop_token_ids: Vec<u32>,
    embed_tokens: Array,
    layers: Vec<DecoderLayer>,
    norm: RmsNormWeight,
    lm_head: Option<Array>,
}

impl QwenModel {
    fn load(model_path: &Path) -> Result<Self> {
        let model_root = resolve_model_root(model_path)?;
        let config_path = model_root.join("config.json");
        let config: QwenConfig = serde_json::from_str(
            &fs::read_to_string(&config_path)
                .with_context(|| format!("read config {}", config_path.display()))?,
        )
        .with_context(|| format!("parse config {}", config_path.display()))?;

        let generation_config = load_generation_config(&model_root)?;
        let tensors = load_safetensor_map(&model_root)?;
        ensure!(
            config.num_hidden_layers as usize > 0,
            "config reports zero hidden layers"
        );

        let embed_tokens = get_tensor(
            &tensors,
            &["embed_tokens.weight", "model.embed_tokens.weight"],
        )?;
        ensure!(
            embed_tokens.shape() == vec![config.vocab_size, config.hidden_size],
            "embed_tokens shape {:?} does not match config [{}, {}]",
            embed_tokens.shape(),
            config.vocab_size,
            config.hidden_size
        );
        let lm_head = get_optional_tensor(&tensors, &["lm_head.weight", "model.lm_head.weight"]);
        if let Some(ref lm_head) = lm_head {
            ensure!(
                lm_head.shape() == vec![config.vocab_size, config.hidden_size],
                "lm_head shape {:?} does not match config [{}, {}]",
                lm_head.shape(),
                config.vocab_size,
                config.hidden_size
            );
        } else {
            ensure!(
                config.tie_word_embeddings,
                "model is missing lm_head.weight and does not tie embeddings"
            );
        }
        let norm = RmsNormWeight {
            weight: get_tensor(&tensors, &["norm.weight", "model.norm.weight"])?,
            eps: config.rms_norm_eps,
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("layers.{layer_idx}");
            let alt_prefix = format!("model.layers.{layer_idx}");
            let q_proj = get_tensor(
                &tensors,
                &[
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    &format!("{alt_prefix}.self_attn.q_proj.weight"),
                ],
            )?;
            let k_proj = get_tensor(
                &tensors,
                &[
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    &format!("{alt_prefix}.self_attn.k_proj.weight"),
                ],
            )?;
            let v_proj = get_tensor(
                &tensors,
                &[
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    &format!("{alt_prefix}.self_attn.v_proj.weight"),
                ],
            )?;
            let o_proj = get_tensor(
                &tensors,
                &[
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    &format!("{alt_prefix}.self_attn.o_proj.weight"),
                ],
            )?;
            let gate_proj = get_tensor(
                &tensors,
                &[
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    &format!("{alt_prefix}.mlp.gate_proj.weight"),
                ],
            )?;
            let up_proj = get_tensor(
                &tensors,
                &[
                    &format!("{prefix}.mlp.up_proj.weight"),
                    &format!("{alt_prefix}.mlp.up_proj.weight"),
                ],
            )?;
            let down_proj = get_tensor(
                &tensors,
                &[
                    &format!("{prefix}.mlp.down_proj.weight"),
                    &format!("{alt_prefix}.mlp.down_proj.weight"),
                ],
            )?;
            ensure!(
                q_proj.shape()
                    == vec![
                        config.num_attention_heads * config.head_dim,
                        config.hidden_size
                    ],
                "q_proj for layer {} has shape {:?}",
                layer_idx,
                q_proj.shape()
            );
            ensure!(
                k_proj.shape()
                    == vec![
                        config.num_key_value_heads * config.head_dim,
                        config.hidden_size
                    ],
                "k_proj for layer {} has shape {:?}",
                layer_idx,
                k_proj.shape()
            );
            ensure!(
                v_proj.shape()
                    == vec![
                        config.num_key_value_heads * config.head_dim,
                        config.hidden_size
                    ],
                "v_proj for layer {} has shape {:?}",
                layer_idx,
                v_proj.shape()
            );
            ensure!(
                o_proj.shape()
                    == vec![
                        config.hidden_size,
                        config.num_attention_heads * config.head_dim
                    ],
                "o_proj for layer {} has shape {:?}",
                layer_idx,
                o_proj.shape()
            );
            ensure!(
                gate_proj.shape() == vec![config.intermediate_size, config.hidden_size],
                "gate_proj for layer {} has shape {:?}",
                layer_idx,
                gate_proj.shape()
            );
            ensure!(
                up_proj.shape() == vec![config.intermediate_size, config.hidden_size],
                "up_proj for layer {} has shape {:?}",
                layer_idx,
                up_proj.shape()
            );
            ensure!(
                down_proj.shape() == vec![config.hidden_size, config.intermediate_size],
                "down_proj for layer {} has shape {:?}",
                layer_idx,
                down_proj.shape()
            );

            layers.push(DecoderLayer {
                input_layernorm: RmsNormWeight {
                    weight: get_tensor(
                        &tensors,
                        &[
                            &format!("{prefix}.input_layernorm.weight"),
                            &format!("{alt_prefix}.input_layernorm.weight"),
                        ],
                    )?,
                    eps: config.rms_norm_eps,
                },
                post_attention_layernorm: RmsNormWeight {
                    weight: get_tensor(
                        &tensors,
                        &[
                            &format!("{prefix}.post_attention_layernorm.weight"),
                            &format!("{alt_prefix}.post_attention_layernorm.weight"),
                        ],
                    )?,
                    eps: config.rms_norm_eps,
                },
                q_proj: LinearWeight { weight: q_proj },
                q_norm: RmsNormWeight {
                    weight: get_tensor(
                        &tensors,
                        &[
                            &format!("{prefix}.self_attn.q_norm.weight"),
                            &format!("{alt_prefix}.self_attn.q_norm.weight"),
                        ],
                    )?,
                    eps: config.rms_norm_eps,
                },
                k_proj: LinearWeight { weight: k_proj },
                k_norm: RmsNormWeight {
                    weight: get_tensor(
                        &tensors,
                        &[
                            &format!("{prefix}.self_attn.k_norm.weight"),
                            &format!("{alt_prefix}.self_attn.k_norm.weight"),
                        ],
                    )?,
                    eps: config.rms_norm_eps,
                },
                v_proj: LinearWeight { weight: v_proj },
                o_proj: LinearWeight { weight: o_proj },
                gate_proj: LinearWeight { weight: gate_proj },
                up_proj: LinearWeight { weight: up_proj },
                down_proj: LinearWeight { weight: down_proj },
                num_attention_heads: config.num_attention_heads,
                num_key_value_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                scale: 1.0 / (config.head_dim as f32).sqrt(),
                rope_theta: config.rope_theta,
            });
        }

        let generation_pad_token_id = generation_config.pad_token_id;
        let mut stop_token_ids = generation_config
            .eos_token_id
            .map(OneOrManyTokenIds::into_vec)
            .unwrap_or_default();
        if let Some(eos_token_id) = config.eos_token_id {
            stop_token_ids.push(eos_token_id);
        }
        if let Some(pad_token_id) = generation_pad_token_id.or(config.pad_token_id) {
            stop_token_ids.push(pad_token_id);
        }
        stop_token_ids.sort_unstable();
        stop_token_ids.dedup();
        let pad_token_id = generation_pad_token_id
            .or(config.pad_token_id)
            .or(config.eos_token_id)
            .unwrap_or(0);

        Ok(Self {
            config,
            pad_token_id,
            stop_token_ids,
            embed_tokens,
            layers,
            norm,
            lm_head,
        })
    }

    fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    fn eos_token_id(&self) -> Option<u32> {
        self.config.eos_token_id
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    fn tied_lm_head(&self) -> &Array {
        self.lm_head.as_ref().unwrap_or(&self.embed_tokens)
    }

    fn forward_hidden(
        &self,
        input_ids: &Array,
        use_causal_mask: bool,
        mut caches: Option<&mut [LayerCache]>,
    ) -> Result<Array> {
        let mut hidden = self.embed_tokens.index(input_ids);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_cache = caches.as_deref_mut().map(|entries| &mut entries[layer_idx]);
            hidden = layer.forward(&hidden, layer_cache, use_causal_mask)?;
        }
        self.norm.forward(&hidden)
    }

    fn logits_for_last_hidden(&self, last_hidden: &Array) -> Result<Vec<f32>> {
        let logits = ops::matmul(last_hidden, self.tied_lm_head().t())?.as_dtype(Dtype::Float32)?;
        let data = logits.as_slice::<f32>();
        Ok(data.to_vec())
    }
}

impl DecoderLayer {
    fn forward(
        &self,
        x: &Array,
        cache: Option<&mut LayerCache>,
        use_causal_mask: bool,
    ) -> Result<Array> {
        let residual = x.clone();
        let hidden = self.input_layernorm.forward(x)?;

        let batch = hidden.dim(0);
        let seq_len = hidden.dim(1);

        let query_states = self
            .q_norm
            .forward(&self.q_proj.forward(&hidden)?.reshape(&[
                batch,
                seq_len,
                self.num_attention_heads,
                self.head_dim,
            ])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let key_states = self
            .k_norm
            .forward(&self.k_proj.forward(&hidden)?.reshape(&[
                batch,
                seq_len,
                self.num_key_value_heads,
                self.head_dim,
            ])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let value_states = self
            .v_proj
            .forward(&hidden)?
            .reshape(&[batch, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let past_len = cache
            .as_ref()
            .and_then(|entry| entry.keys.as_ref())
            .map(|keys| keys.dim(2))
            .unwrap_or(0);
        let query_states = fast::rope(
            &query_states,
            self.head_dim,
            false,
            self.rope_theta,
            1.0,
            past_len,
            None,
        )?;
        let key_states = fast::rope(
            &key_states,
            self.head_dim,
            false,
            self.rope_theta,
            1.0,
            past_len,
            None,
        )?;

        let (all_keys, all_values) = if let Some(cache) = cache {
            let merged_keys = if let Some(existing) = &cache.keys {
                ops::concatenate_axis(&[existing, &key_states], 2)?
            } else {
                key_states.clone()
            };
            let merged_values = if let Some(existing) = &cache.values {
                ops::concatenate_axis(&[existing, &value_states], 2)?
            } else {
                value_states.clone()
            };
            cache.keys = Some(merged_keys.clone());
            cache.values = Some(merged_values.clone());
            (merged_keys, merged_values)
        } else {
            (key_states, value_states)
        };

        let attn_output = if use_causal_mask {
            fast::scaled_dot_product_attention(
                &query_states,
                &all_keys,
                &all_values,
                self.scale,
                ScaledDotProductAttentionMask::Causal,
            )?
        } else {
            fast::scaled_dot_product_attention(
                &query_states,
                &all_keys,
                &all_values,
                self.scale,
                None,
            )?
        };
        let attn_output = attn_output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ])?;
        let attn_residual = self.o_proj.forward(&attn_output)?;
        let hidden = residual.add(&attn_residual)?;

        let residual = hidden.clone();
        let mlp_input = self.post_attention_layernorm.forward(&hidden)?;
        let gated = nn::silu(self.gate_proj.forward(&mlp_input)?)?;
        let up = self.up_proj.forward(&mlp_input)?;
        let gated_up = gated.multiply(&up)?;
        let mlp = self.down_proj.forward(&gated_up)?;
        Ok(residual.add(&mlp)?)
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    Device::set_default(&Device::gpu());
    match cli.command {
        Command::Embed(args) => run_embed(args),
        Command::Microllm(args) => run_microllm(args),
    }
}

fn run_embed(args: EmbedArgs) -> Result<()> {
    ensure!(args.max_length > 0, "max-length must be > 0");
    ensure!(args.attention_units > 0, "attention-units must be > 0");

    let started = Instant::now();
    let tokenizer = load_tokenizer(&args.tokenizer)?;
    let model = QwenModel::load(&args.model)?;
    let eos_id = model
        .eos_token_id()
        .context("embedding model config is missing eos_token_id")?;
    let pad_id = model.pad_token_id();

    let warmup = vec![encode_embedding_ids(
        &tokenizer,
        "warmup",
        args.max_length,
        eos_id,
    )?];
    let _ = embed_batch(&model, &warmup, pad_id)?;
    let cold_load_s = started.elapsed().as_secs_f64();

    let chunks: Vec<Chunk> = load_corpus(&args.corpus, None)?;

    let mut encoded = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let ids = encode_embedding_ids(&tokenizer, &chunk.text, args.max_length, eos_id)
            .with_context(|| format!("tokenize embedding chunk {}", chunk.id))?;
        encoded.push(EncodedChunk { id: chunk.id, ids });
    }

    // Sort by tokenized length so padded batches carry near-uniform lengths
    // (mixed-length batches pad to the batch max and waste GPU on padding).
    // Vectors are keyed by id, so output order is irrelevant.
    encoded.sort_by_key(|item| item.ids.len());

    let infer_started = Instant::now();
    let mut batch_start = 0usize;
    let mut batch_max_len = 0usize;
    let mut index = 0usize;
    let mut total_input_tokens = 0u64;
    let mut vectors = Vec::with_capacity(encoded.len());

    while index <= encoded.len() {
        let flush = if index == encoded.len() {
            index > batch_start
        } else {
            let candidate_max = batch_max_len.max(encoded[index].ids.len());
            let count = index - batch_start;
            count > 0 && (count + 1) * candidate_max * candidate_max > args.attention_units
        };

        if flush {
            let batch = &encoded[batch_start..index];
            let batch_ids: Vec<Vec<u32>> = batch.iter().map(|item| item.ids.clone()).collect();
            let batch_vectors = embed_batch(&model, &batch_ids, pad_id)?;
            for (item, vector) in batch.iter().zip(batch_vectors) {
                total_input_tokens += item.ids.len() as u64;
                vectors.push(ProducedVector {
                    id: item.id.clone(),
                    vec: vector,
                });
            }
            batch_start = index;
            batch_max_len = 0;
            if index == encoded.len() {
                break;
            }
        }

        if index < encoded.len() {
            batch_max_len = batch_max_len.max(encoded[index].ids.len());
        }
        index += 1;
    }

    let infer_wall_s = infer_started.elapsed().as_secs_f64();

    if let Some(vectors_out) = &args.vectors_out {
        write_vectors_jsonl(vectors_out, &vectors)?;
    }

    let parity_mean_cosine = match &args.reference {
        Some(reference_path) => {
            let reference_vectors = load_reference(reference_path)?;
            let (mean_cosine, matched) = mean_parity(
                vectors
                    .iter()
                    .map(|vector| (vector.id.clone(), vector.vec.clone())),
                &reference_vectors,
            );
            ensure!(
                matched > 0,
                "reference file had no ids in common with produced vectors"
            );
            let mean_cosine = mean_cosine.expect("matched count implies a parity mean");
            ensure!(
                mean_cosine >= 0.98,
                "parity mean cosine {:.6} is below the required debug threshold",
                mean_cosine
            );
            Some(mean_cosine)
        }
        None => None,
    };

    let result = LaneResult {
        lane: "mlx-embed".into(),
        workload: "embed-corpus-v1".into(),
        model: args.model_label,
        cold_load_s,
        infer_wall_s,
        input_tokens: total_input_tokens,
        tok_per_s: total_input_tokens as f64 / infer_wall_s.max(f64::MIN_POSITIVE),
        items: vectors.len() as u64,
        parity_mean_cosine,
        self_peak_rss_bytes: None,
        notes: format!(
            "mlx-rs=0.25.3, bf16 weights, causal attention, q_norm+k_norm, last-token pooling, manual_endoftext_eos=true, attention_units={}, max_len={}.",
            args.attention_units, args.max_length
        ),
    };
    write_lane_result(&args.out, &result)?;
    eprintln!(
        "mlx-embed: {} items, {} tokens, {:.1} tok/s, cold_load {:.2}s, infer {:.2}s{}",
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

fn run_microllm(args: MicrollmArgs) -> Result<()> {
    ensure!(args.max_new_tokens > 0, "max-new-tokens must be > 0");

    let started = Instant::now();
    let tokenizer = load_tokenizer(&args.tokenizer)?;
    let model = QwenModel::load(&args.model)?;

    let all_prompts: Vec<Prompt> = load_jsonl(&args.prompts)?;
    ensure!(!all_prompts.is_empty(), "empty prompts");
    let limit = args
        .limit
        .unwrap_or(all_prompts.len())
        .min(all_prompts.len());
    let prompts = &all_prompts[..limit];

    let infer_started = Instant::now();
    let mut cold_load_s = None;
    let mut total_input_tokens = 0u64;
    let mut total_generated_tokens = 0u64;
    let mut valid_count = 0usize;
    let mut sample_answers = Vec::new();

    for prompt in prompts {
        let prompt_ids = encode_prompt_ids(&tokenizer, &prompt.prompt)
            .with_context(|| format!("tokenize prompt {}", prompt.id))?;
        total_input_tokens += prompt_ids.len() as u64;

        let output = generate_one(
            &model,
            &tokenizer,
            &prompt_ids,
            args.max_new_tokens,
            &started,
            &mut cold_load_s,
        )
        .with_context(|| format!("generate prompt {}", prompt.id))?;
        total_generated_tokens += output.generated_ids.len() as u64;

        if is_valid_label(&output.text) {
            valid_count += 1;
        }
        if sample_answers.len() < 10 {
            sample_answers.push(format!("{}={}", prompt.id, compact_answer(&output.text)));
        }
    }

    let infer_wall_s = infer_started.elapsed().as_secs_f64();
    let decode_tok_per_s = total_generated_tokens as f64 / infer_wall_s.max(f64::MIN_POSITIVE);
    let label_validity = valid_count as f64 / prompts.len() as f64;

    let result = LaneResult {
        lane: "mlx-microllm".into(),
        workload: "microllm-oneshot-v1".into(),
        model: args.model_label,
        cold_load_s: cold_load_s.context("model never produced a first token")?,
        infer_wall_s,
        input_tokens: total_input_tokens,
        tok_per_s: total_input_tokens as f64 / infer_wall_s.max(f64::MIN_POSITIVE),
        items: prompts.len() as u64,
        parity_mean_cosine: None,
        self_peak_rss_bytes: None,
        notes: format!(
            "mlx-rs=0.25.3, bf16 weights, greedy decoding, kv_cache=true, prompt_mode=qwen_chat_template_user_turn_thinking_disabled, max_new_tokens={}, generated_tokens={}, decode_tok_per_s={:.2}, label_validity={:.3}, sample_answers=[{}].",
            args.max_new_tokens,
            total_generated_tokens,
            decode_tok_per_s,
            label_validity,
            sample_answers.join(", ")
        ),
    };
    write_lane_result(&args.out, &result)?;
    eprintln!(
        "mlx-microllm: {} prompts, {} input tokens, {} generated, decode {:.2} tok/s, validity {:.3}, cold_load {:.2}s, infer {:.2}s",
        result.items,
        result.input_tokens,
        total_generated_tokens,
        decode_tok_per_s,
        label_validity,
        result.cold_load_s,
        result.infer_wall_s,
    );
    Ok(())
}

#[derive(Clone)]
struct EncodedChunk {
    id: String,
    ids: Vec<u32>,
}

struct ProducedVector {
    id: String,
    vec: Vec<f32>,
}

struct GenerationOutput {
    generated_ids: Vec<u32>,
    text: String,
}

fn embed_batch(model: &QwenModel, batch_ids: &[Vec<u32>], pad_id: u32) -> Result<Vec<Vec<f32>>> {
    let input_ids = batch_to_array(batch_ids, pad_id)?;
    let hidden = model.forward_hidden(&input_ids, true, None)?;
    pool_last_hidden(&hidden, batch_ids)
}

fn generate_one(
    model: &QwenModel,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    started: &Instant,
    cold_load_s: &mut Option<f64>,
) -> Result<GenerationOutput> {
    let mut caches = vec![LayerCache::default(); model.layers.len()];
    let prompt_array = batch_to_array(&[prompt_ids.to_vec()], model.pad_token_id())?;
    let prompt_hidden = model.forward_hidden(&prompt_array, true, Some(&mut caches))?;
    let mut current_last_hidden = select_last_hidden(&prompt_hidden, prompt_ids.len())?;

    let mut generated_ids = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let logits = model.logits_for_last_hidden(&current_last_hidden)?;
        let next_id = argmax(&logits) as u32;
        if cold_load_s.is_none() {
            *cold_load_s = Some(started.elapsed().as_secs_f64());
        }
        generated_ids.push(next_id);
        if model.is_stop_token(next_id) {
            break;
        }

        let next_hidden = model.forward_hidden(
            &batch_to_array(&[vec![next_id]], model.pad_token_id())?,
            false,
            Some(&mut caches),
        )?;
        current_last_hidden = select_last_hidden(&next_hidden, 1)?;
    }

    let text = tokenizer
        .decode(&generated_ids, true)
        .map_err(|error| anyhow::anyhow!("decode: {error}"))?;
    Ok(GenerationOutput {
        generated_ids,
        text,
    })
}

fn batch_to_array(sequences: &[Vec<u32>], pad_id: u32) -> Result<Array> {
    ensure!(!sequences.is_empty(), "empty batch");
    let batch = sequences.len();
    let max_len = sequences
        .iter()
        .map(|ids| ids.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let mut data = vec![pad_id as i32; batch * max_len];
    for (row, ids) in sequences.iter().enumerate() {
        for (col, token_id) in ids.iter().enumerate() {
            data[row * max_len + col] = *token_id as i32;
        }
    }
    Ok(Array::from_slice(&data, &[batch as i32, max_len as i32]))
}

fn pool_last_hidden(hidden: &Array, batch_ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
    let hidden = hidden.as_dtype(Dtype::Float32)?;
    let shape = hidden.shape();
    ensure!(
        shape.len() == 3,
        "expected [batch, seq, hidden], got {shape:?}"
    );
    let batch = shape[0] as usize;
    let seq_len = shape[1] as usize;
    let hidden_size = shape[2] as usize;
    ensure!(batch == batch_ids.len(), "batch mismatch while pooling");

    let data = hidden.as_slice::<f32>();
    let mut vectors = Vec::with_capacity(batch);
    for (row, ids) in batch_ids.iter().enumerate() {
        let last = ids.len().saturating_sub(1).min(seq_len - 1);
        let start = (row * seq_len + last) * hidden_size;
        let end = start + hidden_size;
        let mut vector = data[start..end].to_vec();
        normalize_l2(&mut vector);
        vectors.push(vector);
    }
    Ok(vectors)
}

fn select_last_hidden(hidden: &Array, seq_len: usize) -> Result<Array> {
    let hidden = hidden.as_dtype(Dtype::Float32)?;
    let shape = hidden.shape();
    ensure!(
        shape.len() == 3,
        "expected [batch, seq, hidden], got {shape:?}"
    );
    ensure!(
        shape[0] == 1,
        "generation only supports batch=1, got {}",
        shape[0]
    );
    ensure!(
        seq_len > 0,
        "cannot select a hidden state from an empty sequence"
    );
    let hidden_size = shape[2] as usize;
    let last = seq_len.saturating_sub(1).min(shape[1] as usize - 1);
    let data = hidden.as_slice::<f32>();
    let start = last * hidden_size;
    Ok(Array::from_slice(
        &data[start..start + hidden_size],
        &[1, hidden_size as i32],
    ))
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|error| anyhow::anyhow!("tokenizer {}: {error}", path.display()))
}

fn encode_embedding_ids(
    tokenizer: &Tokenizer,
    text: &str,
    max_length: usize,
    eos_id: u32,
) -> Result<Vec<u32>> {
    ensure!(max_length > 0, "max_length must be > 0");
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|error| anyhow::anyhow!("encode: {error}"))?;
    let mut ids = encoding.get_ids().to_vec();
    if ids.len() + 1 > max_length {
        ids.truncate(max_length - 1);
    }
    ids.push(eos_id);
    Ok(ids)
}

fn encode_prompt_ids(tokenizer: &Tokenizer, prompt: &str) -> Result<Vec<u32>> {
    let templated_prompt = format!(
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let encoding = tokenizer
        .encode(templated_prompt, false)
        .map_err(|error| anyhow::anyhow!("encode: {error}"))?;
    Ok(encoding.get_ids().to_vec())
}

fn load_generation_config(model_root: &Path) -> Result<QwenGenerationConfig> {
    let generation_config_path = model_root.join("generation_config.json");
    if !generation_config_path.is_file() {
        return Ok(QwenGenerationConfig::default());
    }
    serde_json::from_str(
        &fs::read_to_string(&generation_config_path).with_context(|| {
            format!(
                "read generation config {}",
                generation_config_path.display()
            )
        })?,
    )
    .with_context(|| {
        format!(
            "parse generation config {}",
            generation_config_path.display()
        )
    })
}

fn load_safetensor_map(model_root: &Path) -> Result<HashMap<String, Array>> {
    let single_file = model_root.join("model.safetensors");
    if single_file.is_file() {
        return load_single_safetensors_file(&single_file);
    }

    let index_file = model_root.join("model.safetensors.index.json");
    if index_file.is_file() {
        let index: SafetensorsIndex = serde_json::from_str(
            &fs::read_to_string(&index_file)
                .with_context(|| format!("read safetensors index {}", index_file.display()))?,
        )
        .with_context(|| format!("parse safetensors index {}", index_file.display()))?;
        let mut merged = HashMap::new();
        let unique_files: HashSet<_> = index.weight_map.into_values().collect();
        for shard in unique_files {
            let shard_path = model_root.join(&shard);
            let shard_tensors = load_single_safetensors_file(&shard_path)?;
            merged.extend(shard_tensors);
        }
        return Ok(merged);
    }

    bail!(
        "could not find model.safetensors or model.safetensors.index.json under {}",
        model_root.display()
    )
}

fn load_single_safetensors_file(path: &Path) -> Result<HashMap<String, Array>> {
    let tensors = Array::load_safetensors(path)
        .with_context(|| format!("load safetensors {}", path.display()))?
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect::<HashMap<_, _>>();
    transforms::eval(tensors.values())?;
    Ok(tensors)
}

fn resolve_model_root(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    if file_name.ends_with(".safetensors") {
        return path
            .parent()
            .map(Path::to_path_buf)
            .context("model path has no parent directory");
    }
    bail!(
        "model path {} is neither a directory nor a safetensors file",
        path.display()
    )
}

fn get_tensor(
    tensors: &HashMap<String, Array>,
    candidate_names: &[impl AsRef<str>],
) -> Result<Array> {
    for name in candidate_names {
        if let Some(value) = tensors.get(name.as_ref()) {
            return Ok(value.clone());
        }
    }
    bail!(
        "missing tensor; tried [{}]",
        candidate_names
            .iter()
            .map(|name| name.as_ref().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn get_optional_tensor(
    tensors: &HashMap<String, Array>,
    candidate_names: &[impl AsRef<str>],
) -> Option<Array> {
    candidate_names
        .iter()
        .find_map(|name| tensors.get(name.as_ref()).cloned())
}

fn normalize_l2(vector: &mut [f32]) {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    let denom = norm.max(1e-12);
    for value in vector {
        *value /= denom;
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    best_index
}

fn is_valid_label(text: &str) -> bool {
    normalized_label(text).is_some()
}

fn normalized_label(text: &str) -> Option<&'static str> {
    let first_line = text.trim().lines().next()?.trim();
    let cleaned = first_line.trim_matches(|ch: char| !ch.is_ascii_alphabetic());
    if cleaned.is_empty() || !cleaned.chars().all(|ch| ch.is_ascii_alphabetic()) {
        return None;
    }
    let lowered = cleaned.to_ascii_lowercase();
    LABELS.iter().copied().find(|label| *label == lowered)
}

fn compact_answer(text: &str) -> String {
    let first_line = text.trim().lines().next().unwrap_or("").trim();
    if first_line.is_empty() {
        return "<empty>".into();
    }
    let compact = first_line.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.len() > 48 {
        format!("{}…", &compact[..48])
    } else {
        compact
    }
}

fn write_vectors_jsonl(path: &Path, vectors: &[ProducedVector]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = std::io::BufWriter::new(fs::File::create(path)?);
    for vector in vectors {
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({"id": vector.id, "vec": vector.vec}),
        )?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn write_lane_result(path: &Path, result: &LaneResult) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(result)?)?;
    Ok(())
}
