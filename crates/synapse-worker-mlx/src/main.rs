#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::net::Shutdown;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, ensure, Context, Result};
use clap::Parser;
use mlx_rs::fast::{self, ScaledDotProductAttentionMask};
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms;
use mlx_rs::{Array, Dtype};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use synapse_core::{
    decode_i32_frame, encode_f32_frame, EngineIdentity, WorkerHello, WorkerHelloAck, WorkerPooling,
    WorkerRequest, WorkerResponse, WorkerTokenItem, DEFAULT_MAX_FRAME_BYTES,
    WORKER_PROTOCOL_VERSION,
};

const ENGINE_VERSION: &str = "mlx-rs-0.25.3";
const DEFAULT_MAX_BATCH_SEQUENCES: usize = 256;

#[derive(Parser)]
#[command(name = "synapse-worker-mlx")]
struct Args {
    #[arg(long)]
    socket: PathBuf,
    #[arg(long)]
    nonce: String,
    #[arg(long = "test-abort", hide = true)]
    test_abort: bool,
    #[arg(long, hide = true)]
    test_abort_on_request: bool,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Clone)]
struct BertConfig {
    vocab_size: i32,
    hidden_size: i32,
    num_attention_heads: i32,
    num_hidden_layers: i32,
    max_position_embeddings: i32,
    #[serde(default = "default_type_vocab_size")]
    type_vocab_size: i32,
    #[serde(default = "default_layer_norm_eps")]
    layer_norm_eps: f32,
    #[serde(default = "default_hidden_act")]
    hidden_act: String,
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
struct DenseWeight {
    weight: Array,
    bias: Option<Array>,
}

impl DenseWeight {
    fn forward(&self, x: &Array) -> Result<Array> {
        let projected = ops::matmul(x, self.weight.t())?;
        if let Some(bias) = &self.bias {
            Ok(projected.add(bias)?)
        } else {
            Ok(projected)
        }
    }
}

#[derive(Clone)]
struct LayerNormWeight {
    weight: Array,
    bias: Array,
    eps: f32,
}

impl LayerNormWeight {
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(fast::layer_norm(
            x,
            Some(&self.weight),
            Some(&self.bias),
            self.eps,
        )?)
    }
}

#[derive(Clone)]
struct BertLayer {
    query: DenseWeight,
    key: DenseWeight,
    value: DenseWeight,
    attention_output: DenseWeight,
    attention_layer_norm: LayerNormWeight,
    intermediate: DenseWeight,
    output: DenseWeight,
    output_layer_norm: LayerNormWeight,
    num_attention_heads: i32,
    head_dim: i32,
    hidden_act: String,
}

struct MiniLmModel {
    config: BertConfig,
    word_embeddings: Array,
    position_embeddings: Array,
    token_type_embeddings: Array,
    embeddings_layer_norm: LayerNormWeight,
    layers: Vec<BertLayer>,
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

struct QwenModel {
    config: QwenConfig,
    pad_token_id: u32,
    embed_tokens: Array,
    layers: Vec<DecoderLayer>,
    norm: RmsNormWeight,
}

struct LoadedRuntime {
    model: MlxModel,
    dims: usize,
    max_batch_sequences: usize,
}

enum MlxModel {
    MiniLm(MiniLmModel),
    Qwen(QwenModel),
}

struct WorkerState {
    models: HashMap<String, LoadedRuntime>,
    next_model: u64,
}

impl WorkerState {
    fn new() -> Self {
        Self {
            models: HashMap::new(),
            next_model: 0,
        }
    }
}

fn default_type_vocab_size() -> i32 {
    2
}

fn default_layer_norm_eps() -> f32 {
    1e-12
}

fn default_hidden_act() -> String {
    "gelu".to_string()
}

impl MiniLmModel {
    fn load(
        model_root: &Path,
        config: BertConfig,
        tensors: &HashMap<String, Array>,
    ) -> Result<Self> {
        ensure!(
            config.num_hidden_layers > 0,
            "config reports zero hidden layers"
        );
        ensure!(
            config.hidden_size % config.num_attention_heads == 0,
            "hidden_size must divide num_attention_heads"
        );
        let word_embeddings = get_bert_tensor(tensors, "embeddings.word_embeddings.weight")?;
        ensure!(
            word_embeddings.shape() == vec![config.vocab_size, config.hidden_size],
            "word_embeddings shape {:?} does not match [{}, {}] in {}",
            word_embeddings.shape(),
            config.vocab_size,
            config.hidden_size,
            model_root.display()
        );
        let position_embeddings =
            get_bert_tensor(tensors, "embeddings.position_embeddings.weight")?;
        ensure!(
            position_embeddings.shape()[1] == config.hidden_size,
            "position_embeddings hidden size mismatch"
        );
        let token_type_embeddings =
            get_bert_tensor(tensors, "embeddings.token_type_embeddings.weight")?;
        ensure!(
            token_type_embeddings.shape() == vec![config.type_vocab_size, config.hidden_size],
            "token_type_embeddings shape {:?} does not match [{}, {}]",
            token_type_embeddings.shape(),
            config.type_vocab_size,
            config.hidden_size
        );
        let embeddings_layer_norm = LayerNormWeight {
            weight: get_bert_tensor(tensors, "embeddings.LayerNorm.weight")?,
            bias: get_bert_tensor(tensors, "embeddings.LayerNorm.bias")?,
            eps: config.layer_norm_eps,
        };

        let head_dim = config.hidden_size / config.num_attention_heads;
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{layer_idx}");
            layers.push(BertLayer {
                query: dense_bert(tensors, &format!("{prefix}.attention.self.query"), true)?,
                key: dense_bert(tensors, &format!("{prefix}.attention.self.key"), true)?,
                value: dense_bert(tensors, &format!("{prefix}.attention.self.value"), true)?,
                attention_output: dense_bert(
                    tensors,
                    &format!("{prefix}.attention.output.dense"),
                    true,
                )?,
                attention_layer_norm: LayerNormWeight {
                    weight: get_bert_tensor(
                        tensors,
                        &format!("{prefix}.attention.output.LayerNorm.weight"),
                    )?,
                    bias: get_bert_tensor(
                        tensors,
                        &format!("{prefix}.attention.output.LayerNorm.bias"),
                    )?,
                    eps: config.layer_norm_eps,
                },
                intermediate: dense_bert(tensors, &format!("{prefix}.intermediate.dense"), true)?,
                output: dense_bert(tensors, &format!("{prefix}.output.dense"), true)?,
                output_layer_norm: LayerNormWeight {
                    weight: get_bert_tensor(tensors, &format!("{prefix}.output.LayerNorm.weight"))?,
                    bias: get_bert_tensor(tensors, &format!("{prefix}.output.LayerNorm.bias"))?,
                    eps: config.layer_norm_eps,
                },
                num_attention_heads: config.num_attention_heads,
                head_dim,
                hidden_act: config.hidden_act.clone(),
            });
        }

        Ok(Self {
            config,
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embeddings_layer_norm,
            layers,
        })
    }

    fn embed(
        &self,
        batch_ids: &[Vec<u32>],
        pooling: WorkerPooling,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let (input_ids, mask_values, additive_mask) =
            bert_inputs(batch_ids, self.config.max_position_embeddings)?;
        let hidden = self.forward_hidden(&input_ids, &additive_mask)?;
        pool_hidden(&hidden, batch_ids, &mask_values, pooling, normalize)
    }

    fn forward_hidden(&self, input_ids: &Array, additive_mask: &Array) -> Result<Array> {
        let batch = input_ids.dim(0);
        let seq_len = input_ids.dim(1);
        let mut hidden = self.word_embeddings.index(input_ids);
        let position_ids = Array::from_iter(0..seq_len, &[seq_len]);
        let position_embeddings = self.position_embeddings.index(&position_ids);
        let token_type_ids = vec![0_i32; (batch * seq_len) as usize];
        let token_type_ids = Array::from_slice(&token_type_ids, &[batch, seq_len]);
        let token_type_embeddings = self.token_type_embeddings.index(&token_type_ids);
        hidden = hidden
            .add(&position_embeddings)?
            .add(&token_type_embeddings)?;
        hidden = self.embeddings_layer_norm.forward(&hidden)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, additive_mask)?;
        }
        Ok(hidden)
    }
}

impl BertLayer {
    fn forward(&self, x: &Array, additive_mask: &Array) -> Result<Array> {
        let batch = x.dim(0);
        let seq_len = x.dim(1);
        let query = self
            .query
            .forward(x)?
            .reshape(&[batch, seq_len, self.num_attention_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let key = self
            .key
            .forward(x)?
            .reshape(&[batch, seq_len, self.num_attention_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let value = self
            .value
            .forward(x)?
            .reshape(&[batch, seq_len, self.num_attention_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        // MLX fused attention requires the mask dtype to promote to the output
        // dtype; an f32 mask against bf16 Q/K/V promotes to f32 and is rejected,
        // so the additive mask is cast to the computation dtype here.
        let additive_mask = additive_mask.as_dtype(query.dtype())?;
        let attention = fast::scaled_dot_product_attention(
            &query,
            &key,
            &value,
            1.0 / (self.head_dim as f32).sqrt(),
            &additive_mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, seq_len, self.num_attention_heads * self.head_dim])?;
        let attention_output = self.attention_output.forward(&attention)?;
        let hidden = self
            .attention_layer_norm
            .forward(&x.add(&attention_output)?)?;
        let intermediate = match self.hidden_act.as_str() {
            "gelu" => nn::gelu(self.intermediate.forward(&hidden)?)?,
            "gelu_new" | "gelu_fast" | "gelu_pytorch_tanh" => {
                nn::gelu_approximate(self.intermediate.forward(&hidden)?)?
            }
            other => bail!("unsupported BERT hidden_act '{other}'"),
        };
        let output = self.output.forward(&intermediate)?;
        self.output_layer_norm.forward(&hidden.add(&output)?)
    }
}

impl QwenModel {
    fn load(
        model_root: &Path,
        config: QwenConfig,
        tensors: &HashMap<String, Array>,
    ) -> Result<Self> {
        ensure!(
            config.num_hidden_layers as usize > 0,
            "config reports zero hidden layers"
        );
        let embed_tokens = get_tensor(
            tensors,
            &["embed_tokens.weight", "model.embed_tokens.weight"],
        )?;
        ensure!(
            embed_tokens.shape() == vec![config.vocab_size, config.hidden_size],
            "embed_tokens shape {:?} does not match config [{}, {}]",
            embed_tokens.shape(),
            config.vocab_size,
            config.hidden_size
        );
        let lm_head = get_optional_tensor(tensors, &["lm_head.weight", "model.lm_head.weight"]);
        if lm_head.is_none() {
            ensure!(
                config.tie_word_embeddings,
                "model is missing lm_head.weight and does not tie embeddings"
            );
        }
        let norm = RmsNormWeight {
            weight: get_tensor(tensors, &["norm.weight", "model.norm.weight"])?,
            eps: config.rms_norm_eps,
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for layer_idx in 0..config.num_hidden_layers {
            let prefix = format!("layers.{layer_idx}");
            let alt_prefix = format!("model.layers.{layer_idx}");
            let q_proj = get_tensor(
                tensors,
                &[
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    &format!("{alt_prefix}.self_attn.q_proj.weight"),
                ],
            )?;
            let k_proj = get_tensor(
                tensors,
                &[
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    &format!("{alt_prefix}.self_attn.k_proj.weight"),
                ],
            )?;
            let v_proj = get_tensor(
                tensors,
                &[
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    &format!("{alt_prefix}.self_attn.v_proj.weight"),
                ],
            )?;
            let o_proj = get_tensor(
                tensors,
                &[
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    &format!("{alt_prefix}.self_attn.o_proj.weight"),
                ],
            )?;
            let gate_proj = get_tensor(
                tensors,
                &[
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    &format!("{alt_prefix}.mlp.gate_proj.weight"),
                ],
            )?;
            let up_proj = get_tensor(
                tensors,
                &[
                    &format!("{prefix}.mlp.up_proj.weight"),
                    &format!("{alt_prefix}.mlp.up_proj.weight"),
                ],
            )?;
            let down_proj = get_tensor(
                tensors,
                &[
                    &format!("{prefix}.mlp.down_proj.weight"),
                    &format!("{alt_prefix}.mlp.down_proj.weight"),
                ],
            )?;
            layers.push(DecoderLayer {
                input_layernorm: RmsNormWeight {
                    weight: get_tensor(
                        tensors,
                        &[
                            &format!("{prefix}.input_layernorm.weight"),
                            &format!("{alt_prefix}.input_layernorm.weight"),
                        ],
                    )?,
                    eps: config.rms_norm_eps,
                },
                post_attention_layernorm: RmsNormWeight {
                    weight: get_tensor(
                        tensors,
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
                        tensors,
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
                        tensors,
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

        let generation_config = load_generation_config(model_root)?;
        let pad_token_id = generation_config
            .pad_token_id
            .or(config.pad_token_id)
            .or(config.eos_token_id)
            .unwrap_or(0);
        let _stop_token_ids = generation_config
            .eos_token_id
            .map(OneOrManyTokenIds::into_vec)
            .unwrap_or_default();

        Ok(Self {
            config,
            pad_token_id,
            embed_tokens,
            layers,
            norm,
        })
    }

    fn embed(
        &self,
        batch_ids: &[Vec<u32>],
        pooling: WorkerPooling,
        normalize: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let input_ids = batch_to_array(batch_ids, self.pad_token_id)?;
        let hidden = self.forward_hidden(&input_ids, true)?;
        pool_hidden_without_mask(&hidden, batch_ids, pooling, normalize)
    }

    fn forward_hidden(&self, input_ids: &Array, use_causal_mask: bool) -> Result<Array> {
        let mut hidden = self.embed_tokens.index(input_ids);
        for layer in &self.layers {
            hidden = layer.forward(&hidden, use_causal_mask)?;
        }
        self.norm.forward(&hidden)
    }
}

impl DecoderLayer {
    fn forward(&self, x: &Array, use_causal_mask: bool) -> Result<Array> {
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

        let query_states = fast::rope(
            &query_states,
            self.head_dim,
            false,
            self.rope_theta,
            1.0,
            0,
            None,
        )?;
        let key_states = fast::rope(
            &key_states,
            self.head_dim,
            false,
            self.rope_theta,
            1.0,
            0,
            None,
        )?;

        let attn_output = if use_causal_mask {
            fast::scaled_dot_product_attention(
                &query_states,
                &key_states,
                &value_states,
                self.scale,
                ScaledDotProductAttentionMask::Causal,
            )?
        } else {
            fast::scaled_dot_product_attention(
                &query_states,
                &key_states,
                &value_states,
                self.scale,
                None,
            )?
        };
        let attn_output = attn_output.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            seq_len,
            self.num_attention_heads * self.head_dim,
        ])?;
        let hidden = residual.add(&self.o_proj.forward(&attn_output)?)?;
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
    let args = Args::parse();
    let mut stream = UnixStream::connect(&args.socket)
        .with_context(|| format!("connect worker socket {}", args.socket.display()))?;
    let hello = WorkerHello {
        v: WORKER_PROTOCOL_VERSION,
        nonce: args.nonce,
        engine: engine_identity(),
        pid: std::process::id(),
        max_frame: DEFAULT_MAX_FRAME_BYTES,
    };
    write_json_frame(&mut stream, &hello, DEFAULT_MAX_FRAME_BYTES)?;
    let ack: WorkerHelloAck = read_json_frame(&mut stream, DEFAULT_MAX_FRAME_BYTES)?;
    ensure!(
        ack.v == WORKER_PROTOCOL_VERSION,
        "module replied with protocol v{}",
        ack.v
    );
    ensure!(ack.accept, "module rejected worker handshake");
    let max_frame = ack.max_frame.min(DEFAULT_MAX_FRAME_BYTES);

    let mut state = WorkerState::new();
    loop {
        let frame = match read_frame(&mut stream, max_frame) {
            Ok(frame) => frame,
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error).context("read request frame"),
        };
        let request: WorkerRequest =
            serde_json::from_slice(&frame).context("decode request JSON")?;
        let should_abort = args.test_abort_on_request
            || (args.test_abort && !matches!(request, WorkerRequest::Load { .. }));
        if should_abort {
            std::process::abort();
        }
        match request {
            WorkerRequest::Load {
                req_id,
                artifact_path,
                artifact_digest,
                format,
                runtime_config,
            } => {
                let response = match handle_load(
                    &mut state,
                    req_id.clone(),
                    &artifact_path,
                    &artifact_digest,
                    &format,
                    &runtime_config,
                ) {
                    Ok(response) => response,
                    Err(error) => WorkerResponse::Err {
                        req_id: Some(req_id),
                        code: classify_load_error(&error).to_string(),
                        msg: error.to_string(),
                    },
                };
                write_json_frame(&mut stream, &response, max_frame)?;
            }
            WorkerRequest::EmbedBatch {
                req_id,
                model_ref,
                pooling,
                normalize,
                items,
            } => {
                let raw = read_frame(&mut stream, max_frame).context("read EMBED_BATCH ids")?;
                let response = match handle_embed_batch(
                    &state, &req_id, &model_ref, pooling, normalize, &items, &raw,
                ) {
                    Ok((response, vectors)) => {
                        write_json_frame(&mut stream, &response, max_frame)?;
                        write_frame(&mut stream, &vectors, max_frame)?;
                        continue;
                    }
                    Err(error) => WorkerResponse::Err {
                        req_id: Some(req_id),
                        code: "inference_failed".to_string(),
                        msg: error.to_string(),
                    },
                };
                write_json_frame(&mut stream, &response, max_frame)?;
            }
            WorkerRequest::Rerank { req_id, .. } => {
                let _ = read_frame(&mut stream, max_frame).context("read RERANK ids")?;
                write_json_frame(
                    &mut stream,
                    &WorkerResponse::Err {
                        req_id: Some(req_id),
                        code: "unknown_type".to_string(),
                        msg: "synapse-worker-mlx v1 supports LOAD, EMBED_BATCH, PING, UNLOAD, and SHUTDOWN only".to_string(),
                    },
                    max_frame,
                )?;
            }
            WorkerRequest::Generate { req_id, .. } => {
                let _ = read_frame(&mut stream, max_frame).context("read GENERATE ids")?;
                write_json_frame(
                    &mut stream,
                    &WorkerResponse::Err {
                        req_id: Some(req_id),
                        code: "unknown_type".to_string(),
                        msg: "synapse-worker-mlx v1 supports LOAD, EMBED_BATCH, PING, UNLOAD, and SHUTDOWN only".to_string(),
                    },
                    max_frame,
                )?;
            }
            WorkerRequest::Unload { req_id, model_ref } => {
                state.models.remove(&model_ref);
                write_json_frame(&mut stream, &WorkerResponse::Unloaded { req_id }, max_frame)?;
            }
            WorkerRequest::Ping { req_id } => {
                write_json_frame(
                    &mut stream,
                    &WorkerResponse::Pong {
                        req_id,
                        rss_mb: 0,
                        models_loaded: state.models.len(),
                        placement_share: None,
                    },
                    max_frame,
                )?;
            }
            WorkerRequest::Shutdown {} => {
                let _ = stream.shutdown(Shutdown::Both);
                break;
            }
        }
    }
    Ok(())
}

fn engine_identity() -> EngineIdentity {
    let mut build_flags = BTreeMap::new();
    build_flags.insert("risk_class".to_string(), "abort_capable".to_string());
    build_flags.insert("backend".to_string(), "metal".to_string());
    build_flags.insert("numeric_profile".to_string(), "bf16-distinct".to_string());
    EngineIdentity {
        engine: "mlx".to_string(),
        version: ENGINE_VERSION.to_string(),
        build_flags,
    }
}

fn handle_load(
    state: &mut WorkerState,
    req_id: String,
    artifact_path: &str,
    artifact_digest: &str,
    format: &str,
    runtime_config: &BTreeMap<String, String>,
) -> Result<WorkerResponse> {
    ensure!(
        matches!(format, "safetensors" | "mlx-safetensors"),
        "MLX worker only loads safetensors artifacts, got {format}"
    );
    let started = Instant::now();
    let path = Path::new(artifact_path);
    verify_digest(path, artifact_digest)?;
    let model_root = resolve_model_root(path)?;
    let config_path = model_root.join("config.json");
    let config_text = fs::read_to_string(&config_path)
        .with_context(|| format!("read config {}", config_path.display()))?;
    let config_json: serde_json::Value = serde_json::from_str(&config_text)
        .with_context(|| format!("parse config {}", config_path.display()))?;
    let tensors = load_safetensor_map(&model_root)?;
    let architecture = runtime_config
        .get("architecture")
        .or_else(|| runtime_config.get("model_family"))
        .map(String::as_str)
        .unwrap_or("auto");
    let model = match resolve_architecture(architecture, &config_json)? {
        ModelArchitecture::MiniLm => {
            let config: BertConfig = serde_json::from_value(config_json)
                .with_context(|| format!("parse BERT config {}", config_path.display()))?;
            MlxModel::MiniLm(MiniLmModel::load(&model_root, config, &tensors)?)
        }
        ModelArchitecture::Qwen => {
            let config: QwenConfig = serde_json::from_value(config_json)
                .with_context(|| format!("parse Qwen config {}", config_path.display()))?;
            MlxModel::Qwen(QwenModel::load(&model_root, config, &tensors)?)
        }
    };
    let dims = match &model {
        MlxModel::MiniLm(model) => model.config.hidden_size as usize,
        MlxModel::Qwen(model) => model.config.hidden_size as usize,
    };
    let max_batch_sequences = runtime_config
        .get("max_batch_sequences")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_BATCH_SEQUENCES);
    ensure!(max_batch_sequences > 0, "max_batch_sequences must be > 0");
    let model_ref = format!("mlx:{}", state.next_model);
    state.next_model += 1;
    state.models.insert(
        model_ref.clone(),
        LoadedRuntime {
            model,
            dims,
            max_batch_sequences,
        },
    );
    Ok(WorkerResponse::Loaded {
        req_id,
        model_ref,
        dims,
        cold_load_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
    })
}

fn handle_embed_batch(
    state: &WorkerState,
    req_id: &str,
    model_ref: &str,
    pooling: WorkerPooling,
    normalize: bool,
    items: &[WorkerTokenItem],
    raw: &[u8],
) -> Result<(WorkerResponse, Vec<u8>)> {
    ensure!(!items.is_empty(), "EMBED_BATCH requires at least one item");
    let runtime = state
        .models
        .get(model_ref)
        .ok_or_else(|| anyhow!("unknown model_ref '{model_ref}'"))?;
    ensure!(
        items.len() <= runtime.max_batch_sequences,
        "too many sequences in one worker request"
    );
    let ids = decode_i32_frame(raw).map_err(|error| anyhow!(error.to_string()))?;
    let expected_tokens = items.iter().map(|item| item.n_tokens).sum::<usize>();
    ensure!(
        ids.len() == expected_tokens,
        "raw id frame has {} tokens, expected {expected_tokens}",
        ids.len()
    );
    let mut sequences = Vec::with_capacity(items.len());
    let mut offset = 0_usize;
    for item in items {
        ensure!(item.n_tokens > 0, "item '{}' has zero tokens", item.id);
        let end = offset + item.n_tokens;
        let sequence = ids[offset..end]
            .iter()
            .copied()
            .map(|value| u32::try_from(value).context("token id must be non-negative"))
            .collect::<Result<Vec<_>>>()?;
        sequences.push(sequence);
        offset = end;
    }

    let mut order = (0..sequences.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| sequences[index].len());
    let sorted = order
        .iter()
        .map(|&index| sequences[index].clone())
        .collect::<Vec<_>>();
    let sorted_vectors = match &runtime.model {
        MlxModel::MiniLm(model) => model.embed(&sorted, pooling, normalize)?,
        MlxModel::Qwen(model) => model.embed(&sorted, pooling, normalize)?,
    };
    ensure!(
        sorted_vectors.len() == sequences.len(),
        "embedding count mismatch"
    );
    let dims = sorted_vectors.first().map(Vec::len).unwrap_or(runtime.dims);
    ensure!(
        dims == runtime.dims,
        "embedding dims mismatch: got {dims}, expected {}",
        runtime.dims
    );
    let mut restored = vec![Vec::<f32>::new(); sequences.len()];
    for (sorted_index, original_index) in order.into_iter().enumerate() {
        restored[original_index] = sorted_vectors[sorted_index].clone();
    }
    let flat = restored.into_iter().flatten().collect::<Vec<_>>();
    Ok((
        WorkerResponse::Vectors {
            req_id: req_id.to_string(),
            dims,
            n: items.len(),
        },
        encode_f32_frame(&flat),
    ))
}

#[derive(Clone, Copy)]
enum ModelArchitecture {
    MiniLm,
    Qwen,
}

fn resolve_architecture(value: &str, config: &serde_json::Value) -> Result<ModelArchitecture> {
    match value.trim().to_ascii_lowercase().as_str() {
        "bert" | "minilm" | "mini-lm" => return Ok(ModelArchitecture::MiniLm),
        "qwen" | "qwen3" => return Ok(ModelArchitecture::Qwen),
        "auto" | "" => {}
        other => bail!("unsupported MLX architecture '{other}'"),
    }
    if config.get("head_dim").is_some()
        || config
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|model_type| model_type.contains("qwen"))
    {
        Ok(ModelArchitecture::Qwen)
    } else {
        Ok(ModelArchitecture::MiniLm)
    }
}

fn bert_inputs(
    sequences: &[Vec<u32>],
    max_position_embeddings: i32,
) -> Result<(Array, Vec<i64>, Array)> {
    ensure!(!sequences.is_empty(), "empty batch");
    let batch = sequences.len();
    let max_len = sequences.iter().map(Vec::len).max().unwrap_or(1).max(1);
    ensure!(
        max_len <= max_position_embeddings as usize,
        "batch sequence length {max_len} exceeds model max_position_embeddings {max_position_embeddings}"
    );
    let mut ids = vec![0_i32; batch * max_len];
    let mut mask = vec![0_i64; batch * max_len];
    let mut additive = vec![-10_000.0_f32; batch * max_len];
    for (row, token_ids) in sequences.iter().enumerate() {
        for (col, token_id) in token_ids.iter().copied().enumerate() {
            ids[row * max_len + col] = i32::try_from(token_id).context("token id exceeds i32")?;
            mask[row * max_len + col] = 1;
            additive[row * max_len + col] = 0.0;
        }
    }
    let input_ids = Array::from_slice(&ids, &[batch as i32, max_len as i32]);
    let additive_mask = Array::from_slice(&additive, &[batch as i32, 1, 1, max_len as i32]);
    Ok((input_ids, mask, additive_mask))
}

fn batch_to_array(sequences: &[Vec<u32>], pad_id: u32) -> Result<Array> {
    ensure!(!sequences.is_empty(), "empty batch");
    let batch = sequences.len();
    let max_len = sequences.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut data = vec![pad_id as i32; batch * max_len];
    for (row, ids) in sequences.iter().enumerate() {
        for (col, token_id) in ids.iter().enumerate() {
            data[row * max_len + col] = i32::try_from(*token_id).context("token id exceeds i32")?;
        }
    }
    Ok(Array::from_slice(&data, &[batch as i32, max_len as i32]))
}

fn pool_hidden(
    hidden: &Array,
    batch_ids: &[Vec<u32>],
    mask: &[i64],
    pooling: WorkerPooling,
    normalize: bool,
) -> Result<Vec<Vec<f32>>> {
    let hidden = hidden.as_dtype(Dtype::Float32)?;
    transforms::eval([&hidden])?;
    let shape = hidden.shape();
    ensure!(
        shape.len() == 3,
        "expected [batch, seq, hidden], got {shape:?}"
    );
    let batch = shape[0] as usize;
    let seq_len = shape[1] as usize;
    let hidden_size = shape[2] as usize;
    ensure!(batch == batch_ids.len(), "batch mismatch while pooling");
    ensure!(
        mask.len() == batch * seq_len,
        "mask shape mismatch while pooling"
    );
    let data = hidden.as_slice::<f32>();
    let mut vectors = Vec::with_capacity(batch);
    for row in 0..batch {
        let mut vector = vec![0.0_f32; hidden_size];
        match pooling {
            WorkerPooling::Mean => {
                let mut count = 0.0_f32;
                for col in 0..seq_len {
                    if mask[row * seq_len + col] == 1 {
                        count += 1.0;
                        for dim in 0..hidden_size {
                            vector[dim] += data[(row * seq_len + col) * hidden_size + dim];
                        }
                    }
                }
                ensure!(count > 0.0, "attention mask is all zeros for row {row}");
                for value in &mut vector {
                    *value /= count;
                }
            }
            WorkerPooling::Cls => vector.copy_from_slice(
                &data[row * seq_len * hidden_size..row * seq_len * hidden_size + hidden_size],
            ),
            WorkerPooling::Last => {
                let last = (0..seq_len)
                    .rev()
                    .find(|&col| mask[row * seq_len + col] == 1)
                    .unwrap_or(0);
                vector.copy_from_slice(
                    &data[(row * seq_len + last) * hidden_size
                        ..(row * seq_len + last + 1) * hidden_size],
                );
            }
        }
        if normalize {
            normalize_l2(&mut vector);
        }
        vectors.push(vector);
    }
    Ok(vectors)
}

fn pool_hidden_without_mask(
    hidden: &Array,
    batch_ids: &[Vec<u32>],
    pooling: WorkerPooling,
    normalize: bool,
) -> Result<Vec<Vec<f32>>> {
    let hidden = hidden.as_dtype(Dtype::Float32)?;
    transforms::eval([&hidden])?;
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
        let mut vector = vec![0.0_f32; hidden_size];
        match pooling {
            WorkerPooling::Mean => {
                for col in 0..ids.len().min(seq_len) {
                    for dim in 0..hidden_size {
                        vector[dim] += data[(row * seq_len + col) * hidden_size + dim];
                    }
                }
                let denom = ids.len().max(1) as f32;
                for value in &mut vector {
                    *value /= denom;
                }
            }
            WorkerPooling::Cls => vector.copy_from_slice(
                &data[row * seq_len * hidden_size..row * seq_len * hidden_size + hidden_size],
            ),
            WorkerPooling::Last => {
                let last = ids.len().saturating_sub(1).min(seq_len - 1);
                vector.copy_from_slice(
                    &data[(row * seq_len + last) * hidden_size
                        ..(row * seq_len + last + 1) * hidden_size],
                );
            }
        }
        if normalize {
            normalize_l2(&mut vector);
        }
        vectors.push(vector);
    }
    Ok(vectors)
}

fn dense_bert(tensors: &HashMap<String, Array>, prefix: &str, bias: bool) -> Result<DenseWeight> {
    Ok(DenseWeight {
        weight: get_bert_tensor(tensors, &format!("{prefix}.weight"))?,
        bias: bias
            .then(|| get_bert_tensor(tensors, &format!("{prefix}.bias")))
            .transpose()?,
    })
}

fn get_bert_tensor(tensors: &HashMap<String, Array>, name: &str) -> Result<Array> {
    let prefixes = ["", "bert.", "model.", "0.auto_model."];
    for prefix in prefixes {
        let candidate = format!("{prefix}{name}");
        if let Some(value) = tensors.get(&candidate) {
            return Ok(value.clone());
        }
    }
    bail!(
        "missing tensor; tried [{}]",
        prefixes
            .iter()
            .map(|prefix| format!("{prefix}{name}"))
            .collect::<Vec<_>>()
            .join(", ")
    )
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

fn classify_load_error(error: &anyhow::Error) -> &'static str {
    let message = error.to_string();
    if message.contains("digest mismatch") || message.contains("only loads") {
        "artifact_invalid"
    } else if message.contains("unsupported") || message.contains("invalid") {
        "config_invalid"
    } else {
        "artifact_invalid"
    }
}

fn verify_digest(path: &Path, digest: &str) -> Result<()> {
    if digest.trim().is_empty() {
        return Ok(());
    }
    let expected = digest.strip_prefix("sha256:").unwrap_or(digest);
    let actual = sha256_path(path).with_context(|| format!("hash {}", path.display()))?;
    ensure!(
        actual == expected,
        "artifact digest mismatch for {}: expected {expected}, got {actual}",
        path.display()
    );
    Ok(())
}

fn sha256_path(path: &Path) -> io::Result<String> {
    if path.is_dir() {
        let mut files = Vec::new();
        collect_files(path, &mut files)?;
        files.sort();
        let mut hasher = Sha256::new();
        for file in files {
            let relative = file
                .strip_prefix(path)
                .unwrap_or(&file)
                .to_string_lossy()
                .replace('\\', "/");
            hasher.update(relative.as_bytes());
            hasher.update([0]);
            hash_file_into(&file, &mut hasher)?;
        }
        Ok(hex::encode(hasher.finalize()))
    } else {
        let mut hasher = Sha256::new();
        hash_file_into(path, &mut hasher)?;
        Ok(hex::encode(hasher.finalize()))
    }
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn hash_file_into(path: &Path, hasher: &mut Sha256) -> io::Result<()> {
    let mut file = File::open(path)?;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(())
}

fn read_json_frame<T: serde::de::DeserializeOwned>(
    stream: &mut UnixStream,
    max_frame: u32,
) -> Result<T> {
    let frame = read_frame(stream, max_frame)?;
    Ok(serde_json::from_slice(&frame)?)
}

fn write_json_frame<T: serde::Serialize>(
    stream: &mut UnixStream,
    value: &T,
    max_frame: u32,
) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    write_frame(stream, &bytes, max_frame)
}

fn read_frame(stream: &mut UnixStream, max_frame: u32) -> io::Result<Vec<u8>> {
    let mut len_bytes = [0_u8; 4];
    stream.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes);
    if len > max_frame {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {len} exceeds max {max_frame}"),
        ));
    }
    let mut frame = vec![0_u8; len as usize];
    stream.read_exact(&mut frame)?;
    Ok(frame)
}

fn write_frame(stream: &mut UnixStream, bytes: &[u8], max_frame: u32) -> Result<()> {
    let len = u32::try_from(bytes.len()).context("frame too large for u32 length")?;
    ensure!(
        len <= max_frame,
        "frame length {len} exceeds max {max_frame}"
    );
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(bytes)?;
    stream.flush()?;
    Ok(())
}
