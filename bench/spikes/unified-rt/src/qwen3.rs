//! Qwen3-Embedding-0.6B model graph for the owned runtime.
//!
//! The CPU path intentionally keeps pointwise operations in Rust and delegates
//! every dense/attention product to `KernelProvider`. Metal providers may take
//! the whole decoder stack through the Qwen3-specific block hook.

use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::{
    get_tensor, load_safetensor_map, normalize_l2, resolve_model_root, BLayout, KernelProvider,
    Tensor,
};

#[derive(Debug, Deserialize)]
struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    vocab_size: usize,
    eos_token_id: Option<u32>,
}

pub(crate) struct Model {
    config: Config,
    eos_token_id: u32,
    embeddings: Tensor,
    pub(crate) layers: Vec<Layer>,
    final_norm: RmsNorm,
}

pub(crate) struct Layer {
    pub(crate) input_norm: RmsNorm,
    pub(crate) post_attention_norm: RmsNorm,
    pub(crate) q_proj: Weight,
    pub(crate) q_norm: RmsNorm,
    pub(crate) k_proj: Weight,
    pub(crate) k_norm: RmsNorm,
    pub(crate) v_proj: Weight,
    pub(crate) o_proj: Weight,
    pub(crate) gate_proj: Weight,
    pub(crate) up_proj: Weight,
    pub(crate) down_proj: Weight,
}

pub(crate) struct Weight {
    pub(crate) tensor: Tensor,
}

pub(crate) struct RmsNorm {
    pub(crate) weight: Tensor,
    pub(crate) eps: f32,
}

impl Model {
    pub(crate) fn load(path: &Path, precision: super::Precision) -> Result<Self> {
        let root = resolve_model_root(path)?;
        let config_path = root.join("config.json");
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("read config {}", config_path.display()))?,
        )
        .with_context(|| format!("parse config {}", config_path.display()))?;
        ensure!(config.num_hidden_layers > 0, "Qwen3 config has no layers");
        ensure!(
            config.num_attention_heads > 0,
            "Qwen3 config has no query heads"
        );
        ensure!(
            config.num_key_value_heads > 0,
            "Qwen3 config has no KV heads"
        );
        ensure!(
            config.num_attention_heads % config.num_key_value_heads == 0,
            "Qwen3 query heads must divide evenly across KV heads"
        );
        let tensors = load_safetensor_map(&root, path)?;
        let embeddings = get_qwen_tensor(&tensors, "embed_tokens.weight")?;
        ensure!(
            embeddings.shape == vec![config.vocab_size, config.hidden_size],
            "Qwen3 embedding shape {:?} does not match config",
            embeddings.shape
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("layers.{index}");
            layers.push(Layer {
                input_norm: load_norm(
                    &tensors,
                    &format!("{prefix}.input_layernorm"),
                    config.rms_norm_eps,
                )?,
                post_attention_norm: load_norm(
                    &tensors,
                    &format!("{prefix}.post_attention_layernorm"),
                    config.rms_norm_eps,
                )?,
                q_proj: load_weight(&tensors, &format!("{prefix}.self_attn.q_proj"))?,
                q_norm: load_norm(
                    &tensors,
                    &format!("{prefix}.self_attn.q_norm"),
                    config.rms_norm_eps,
                )?,
                k_proj: load_weight(&tensors, &format!("{prefix}.self_attn.k_proj"))?,
                k_norm: load_norm(
                    &tensors,
                    &format!("{prefix}.self_attn.k_norm"),
                    config.rms_norm_eps,
                )?,
                v_proj: load_weight(&tensors, &format!("{prefix}.self_attn.v_proj"))?,
                o_proj: load_weight(&tensors, &format!("{prefix}.self_attn.o_proj"))?,
                gate_proj: load_weight(&tensors, &format!("{prefix}.mlp.gate_proj"))?,
                up_proj: load_weight(&tensors, &format!("{prefix}.mlp.up_proj"))?,
                down_proj: load_weight(&tensors, &format!("{prefix}.mlp.down_proj"))?,
            });
        }
        validate_layers(&config, &layers)?;
        if matches!(precision, super::Precision::F16) {
            for layer in &mut layers {
                layer.input_norm.weight.prepare_metal_f16();
                layer.post_attention_norm.weight.prepare_metal_f16();
                layer.q_proj.tensor.prepare_metal_f16();
                layer.q_norm.weight.prepare_metal_f16();
                layer.k_proj.tensor.prepare_metal_f16();
                layer.k_norm.weight.prepare_metal_f16();
                layer.v_proj.tensor.prepare_metal_f16();
                layer.o_proj.tensor.prepare_metal_f16();
                layer.gate_proj.tensor.prepare_metal_f16();
                layer.up_proj.tensor.prepare_metal_f16();
                layer.down_proj.tensor.prepare_metal_f16();
            }
        }

        let eos_token_id = config
            .eos_token_id
            .context("Qwen3 embedding config is missing eos_token_id")?;
        let mut final_norm = load_norm(&tensors, "norm", config.rms_norm_eps)?;
        if matches!(precision, super::Precision::F16) {
            final_norm.weight.prepare_metal_f16();
        }
        Ok(Self {
            config,
            eos_token_id,
            embeddings,
            layers,
            final_norm,
        })
    }

    pub(crate) fn encode(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        max_length: usize,
    ) -> Result<Vec<u32>> {
        ensure!(max_length > 0, "max_length must be positive");
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!("encode Qwen3 input: {error}"))?;
        // Some tokenizer.json files carry a padding policy. Remove those baked
        // pads before adding Qwen3's required terminal embedding token.
        let mut ids: Vec<u32> = encoding
            .get_ids()
            .iter()
            .zip(encoding.get_attention_mask())
            .filter_map(|(&id, &mask)| (mask != 0).then_some(id))
            .collect();
        if ids.last() == Some(&self.eos_token_id) {
            ids.pop();
        }
        ids.truncate(max_length.saturating_sub(1));
        ids.push(self.eos_token_id);
        Ok(ids)
    }

    pub(crate) fn embed_batch(
        &self,
        provider: &mut dyn KernelProvider,
        tokenizer: &Tokenizer,
        texts: &[&str],
        max_length: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let ids = texts
            .iter()
            .map(|text| self.encode(tokenizer, text, max_length))
            .collect::<Result<Vec<_>>>()?;
        self.embed_ids(provider, &ids)
    }

    fn embed_ids(
        &self,
        provider: &mut dyn KernelProvider,
        sequences: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>> {
        let batch = sequences.len();
        ensure!(batch > 0, "Qwen3 batch must not be empty");
        let seq = sequences.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let hidden = self.config.hidden_size;
        let mut hidden_states = vec![0.0f32; batch * seq * hidden];
        let mut attention_mask = vec![0u8; batch * seq];
        for (row, ids) in sequences.iter().enumerate() {
            for (col, &id) in ids.iter().enumerate() {
                let id = id as usize;
                ensure!(
                    id < self.config.vocab_size,
                    "token id {id} outside Qwen3 vocab"
                );
                attention_mask[row * seq + col] = 1;
                let source = id * hidden;
                let target = (row * seq + col) * hidden;
                hidden_states[target..target + hidden]
                    .copy_from_slice(&self.embeddings.data[source..source + hidden]);
            }
        }

        if !provider.qwen3_forward(
            &mut hidden_states,
            &attention_mask,
            batch,
            seq,
            hidden,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            self.config.intermediate_size,
            self.config.rms_norm_eps,
            self.config.rope_theta,
            &self.layers,
            &self.final_norm,
        )? {
            scalar_forward(
                provider,
                &mut hidden_states,
                &attention_mask,
                batch,
                seq,
                &self.config,
                &self.layers,
                &self.final_norm,
            )?;
        }
        Ok(last_token_pool_l2(
            &hidden_states,
            &attention_mask,
            batch,
            seq,
            hidden,
        ))
    }

    pub(crate) fn default_label(&self, precision: super::Precision) -> String {
        format!("Qwen3-Embedding-0.6B@owned-rt-{}", precision.as_str())
    }

    pub(crate) fn notes(&self) -> String {
        format!(
            "direct Qwen3 decoder encoder, {} layers, causal GQA {}/{}, q_norm+k_norm, RoPE theta={}, pre-RMSNorm, SwiGLU, manual EOS, last-token pool+l2",
            self.config.num_hidden_layers,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.rope_theta
        )
    }
}

fn get_qwen_tensor(
    tensors: &std::collections::HashMap<String, Tensor>,
    name: &str,
) -> Result<Tensor> {
    get_tensor(tensors, name)
}

fn load_weight(
    tensors: &std::collections::HashMap<String, Tensor>,
    prefix: &str,
) -> Result<Weight> {
    Ok(Weight {
        tensor: get_qwen_tensor(tensors, &format!("{prefix}.weight"))?,
    })
}

fn load_norm(
    tensors: &std::collections::HashMap<String, Tensor>,
    prefix: &str,
    eps: f32,
) -> Result<RmsNorm> {
    Ok(RmsNorm {
        weight: get_qwen_tensor(tensors, &format!("{prefix}.weight"))?,
        eps,
    })
}

fn validate_layers(config: &Config, layers: &[Layer]) -> Result<()> {
    let q = config.num_attention_heads * config.head_dim;
    let kv = config.num_key_value_heads * config.head_dim;
    for (index, layer) in layers.iter().enumerate() {
        let expected = [
            (&layer.q_proj, vec![q, config.hidden_size], "q_proj"),
            (&layer.k_proj, vec![kv, config.hidden_size], "k_proj"),
            (&layer.v_proj, vec![kv, config.hidden_size], "v_proj"),
            (&layer.o_proj, vec![config.hidden_size, q], "o_proj"),
            (
                &layer.gate_proj,
                vec![config.intermediate_size, config.hidden_size],
                "gate_proj",
            ),
            (
                &layer.up_proj,
                vec![config.intermediate_size, config.hidden_size],
                "up_proj",
            ),
            (
                &layer.down_proj,
                vec![config.hidden_size, config.intermediate_size],
                "down_proj",
            ),
        ];
        for (weight, shape, name) in expected {
            ensure!(
                weight.tensor.shape == shape,
                "Qwen3 layer {index} {name} shape {:?}, expected {shape:?}",
                weight.tensor.shape
            );
        }
        for (norm, width, name) in [
            (&layer.input_norm, config.hidden_size, "input norm"),
            (
                &layer.post_attention_norm,
                config.hidden_size,
                "post-attention norm",
            ),
            (&layer.q_norm, config.head_dim, "query norm"),
            (&layer.k_norm, config.head_dim, "key norm"),
        ] {
            ensure!(
                norm.weight.shape == vec![width],
                "Qwen3 layer {index} {name} shape mismatch"
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scalar_forward(
    provider: &mut dyn KernelProvider,
    hidden_states: &mut [f32],
    attention_mask: &[u8],
    batch: usize,
    seq: usize,
    config: &Config,
    layers: &[Layer],
    final_norm: &RmsNorm,
) -> Result<()> {
    let rows = batch * seq;
    let hidden = config.hidden_size;
    let mut current = hidden_states.to_vec();
    for layer in layers {
        let residual = current.clone();
        rms_norm_rows(&mut current, rows, hidden, &layer.input_norm)?;
        let mut q = linear(provider, &current, rows, hidden, &layer.q_proj)?;
        let mut k = linear(provider, &current, rows, hidden, &layer.k_proj)?;
        let v = linear(provider, &current, rows, hidden, &layer.v_proj)?;
        rms_norm_heads(
            &mut q,
            rows,
            config.num_attention_heads,
            config.head_dim,
            &layer.q_norm,
        )?;
        rms_norm_heads(
            &mut k,
            rows,
            config.num_key_value_heads,
            config.head_dim,
            &layer.k_norm,
        )?;
        apply_rope(
            &mut q,
            batch,
            seq,
            config.num_attention_heads,
            config.head_dim,
            config.rope_theta,
        );
        apply_rope(
            &mut k,
            batch,
            seq,
            config.num_key_value_heads,
            config.head_dim,
            config.rope_theta,
        );
        let context = causal_gqa(
            provider,
            &q,
            &k,
            &v,
            attention_mask,
            batch,
            seq,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )?;
        current = linear(
            provider,
            &context,
            rows,
            config.num_attention_heads * config.head_dim,
            &layer.o_proj,
        )?;
        for (value, residual) in current.iter_mut().zip(residual) {
            *value += residual;
        }

        let residual = current.clone();
        rms_norm_rows(&mut current, rows, hidden, &layer.post_attention_norm)?;
        let mut gate = linear(provider, &current, rows, hidden, &layer.gate_proj)?;
        let up = linear(provider, &current, rows, hidden, &layer.up_proj)?;
        for (gate, up) in gate.iter_mut().zip(up) {
            *gate = silu(*gate) * up;
        }
        current = linear(
            provider,
            &gate,
            rows,
            config.intermediate_size,
            &layer.down_proj,
        )?;
        for (value, residual) in current.iter_mut().zip(residual) {
            *value += residual;
        }
    }
    rms_norm_rows(&mut current, rows, hidden, final_norm)?;
    hidden_states.copy_from_slice(&current);
    Ok(())
}

fn linear(
    provider: &mut dyn KernelProvider,
    values: &[f32],
    rows: usize,
    input: usize,
    weight: &Weight,
) -> Result<Vec<f32>> {
    let (output, weight_input) = weight.tensor.matrix_shape()?;
    ensure!(weight_input == input, "Qwen3 linear input shape mismatch");
    ensure!(
        values.len() == rows * input,
        "Qwen3 linear value shape mismatch"
    );
    let mut output_values = vec![0.0; rows * output];
    provider.matmul_static_rhs(
        rows,
        output,
        input,
        values,
        &weight.tensor.data,
        BLayout::RowMajorNkTransposed,
        &mut output_values,
    )?;
    Ok(output_values)
}

fn rms_norm_rows(data: &mut [f32], rows: usize, width: usize, norm: &RmsNorm) -> Result<()> {
    ensure!(
        norm.weight.shape == vec![width],
        "RMSNorm weight shape mismatch"
    );
    ensure!(data.len() == rows * width, "RMSNorm data shape mismatch");
    for row in data.chunks_exact_mut(width) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / width as f32;
        let inv = 1.0 / (mean_square + norm.eps).sqrt();
        for (value, weight) in row.iter_mut().zip(&norm.weight.data) {
            *value *= inv * weight;
        }
    }
    Ok(())
}

fn rms_norm_heads(
    data: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    norm: &RmsNorm,
) -> Result<()> {
    rms_norm_rows(data, rows * heads, head_dim, norm)
}

fn apply_rope(
    data: &mut [f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    theta: f32,
) {
    let half = head_dim / 2;
    debug_assert_eq!(head_dim % 2, 0);
    for b in 0..batch {
        for position in 0..seq {
            for head in 0..heads {
                let start = ((b * seq + position) * heads + head) * head_dim;
                for index in 0..half {
                    let frequency = 1.0 / theta.powf((2 * index) as f32 / head_dim as f32);
                    let angle = position as f32 * frequency;
                    let (sin, cos) = angle.sin_cos();
                    let first = data[start + index];
                    let second = data[start + half + index];
                    data[start + index] = first * cos - second * sin;
                    data[start + half + index] = second * cos + first * sin;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn causal_gqa(
    provider: &mut dyn KernelProvider,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    attention_mask: &[u8],
    batch: usize,
    seq: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let query_width = query_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let groups = query_heads / kv_heads;
    let mut output = vec![0.0; batch * seq * query_width];
    let mut q_head = vec![0.0; seq * head_dim];
    let mut k_head = vec![0.0; seq * head_dim];
    let mut v_head = vec![0.0; seq * head_dim];
    let mut scores = vec![0.0; seq * seq];
    let mut context = vec![0.0; seq * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for b in 0..batch {
        for query_head in 0..query_heads {
            let kv_head = query_head / groups;
            for position in 0..seq {
                let q_source = (b * seq + position) * query_width + query_head * head_dim;
                let kv_source = (b * seq + position) * kv_width + kv_head * head_dim;
                let target = position * head_dim;
                q_head[target..target + head_dim]
                    .copy_from_slice(&q[q_source..q_source + head_dim]);
                k_head[target..target + head_dim]
                    .copy_from_slice(&k[kv_source..kv_source + head_dim]);
                v_head[target..target + head_dim]
                    .copy_from_slice(&v[kv_source..kv_source + head_dim]);
            }
            provider.matmul(
                seq,
                seq,
                head_dim,
                &q_head,
                &k_head,
                BLayout::RowMajorNkTransposed,
                &mut scores,
            )?;
            for query_position in 0..seq {
                let row = &mut scores[query_position * seq..(query_position + 1) * seq];
                for (key_position, score) in row.iter_mut().enumerate() {
                    *score = if key_position > query_position
                        || attention_mask[b * seq + key_position] == 0
                    {
                        -10_000.0
                    } else {
                        *score * scale
                    };
                }
                super::softmax(row);
            }
            provider.matmul(
                seq,
                head_dim,
                seq,
                &scores,
                &v_head,
                BLayout::RowMajorKn,
                &mut context,
            )?;
            for position in 0..seq {
                let source = position * head_dim;
                let target = (b * seq + position) * query_width + query_head * head_dim;
                output[target..target + head_dim]
                    .copy_from_slice(&context[source..source + head_dim]);
            }
        }
    }
    Ok(output)
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn last_token_pool_l2(
    hidden: &[f32],
    attention_mask: &[u8],
    batch: usize,
    seq: usize,
    hidden_size: usize,
) -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(batch);
    for row in 0..batch {
        let last = (0..seq)
            .rev()
            .find(|&position| attention_mask[row * seq + position] != 0)
            .unwrap_or(0);
        let start = (row * seq + last) * hidden_size;
        let mut vector = hidden[start..start + hidden_size].to_vec();
        normalize_l2(&mut vector);
        vectors.push(vector);
    }
    vectors
}

#[cfg(target_os = "macos")]
mod metal {
    use std::ffi::{c_char, c_void, CStr};
    use std::ptr::NonNull;

    use anyhow::{bail, ensure, Result};

    use super::{Layer, RmsNorm};
    use crate::{decode_f16_bits, encode_f16_bits, Execution, MetalExecutionConfig, Precision};

    #[repr(C)]
    struct LayerParams {
        input_norm: *const c_void,
        post_attention_norm: *const c_void,
        q_weight: *const c_void,
        q_norm: *const c_void,
        k_weight: *const c_void,
        k_norm: *const c_void,
        v_weight: *const c_void,
        o_weight: *const c_void,
        gate_weight: *const c_void,
        up_weight: *const c_void,
        down_weight: *const c_void,
    }

    pub(crate) struct Context {
        raw: NonNull<c_void>,
        precision: Precision,
        execution: MetalExecutionConfig,
    }

    impl Context {
        pub(crate) fn new(precision: Precision, execution: MetalExecutionConfig) -> Result<Self> {
            let raw = unsafe { synapse_qwen3_context_new() };
            Ok(Self {
                raw: NonNull::new(raw).ok_or_else(last_error)?,
                precision,
                execution,
            })
        }

        #[allow(clippy::too_many_arguments)]
        pub(crate) fn forward(
            &mut self,
            hidden_states: &mut [f32],
            attention_mask: &[u8],
            batch: usize,
            seq: usize,
            hidden: usize,
            query_heads: usize,
            kv_heads: usize,
            head_dim: usize,
            intermediate: usize,
            epsilon: f32,
            rope_theta: f32,
            layers: &[Layer],
            final_norm: &RmsNorm,
        ) -> Result<()> {
            ensure!(
                hidden_states.len() == batch * seq * hidden,
                "Qwen3 Metal hidden shape mismatch"
            );
            ensure!(
                attention_mask.len() == batch * seq,
                "Qwen3 Metal mask shape mismatch"
            );
            ensure!(
                final_norm.weight.data.len() == hidden,
                "Qwen3 final norm shape mismatch"
            );
            let mut additive_mask = vec![0.0f32; batch * seq * seq];
            for b in 0..batch {
                for query in 0..seq {
                    for key in 0..seq {
                        if key > query || attention_mask[b * seq + key] == 0 {
                            additive_mask[(b * seq + query) * seq + key] = -10_000.0;
                        }
                    }
                }
            }
            let mut rope_cos = vec![0.0f32; seq * head_dim];
            let mut rope_sin = vec![0.0f32; seq * head_dim];
            for position in 0..seq {
                for index in 0..head_dim / 2 {
                    let frequency = 1.0 / rope_theta.powf((2 * index) as f32 / head_dim as f32);
                    let (sin, cos) = (position as f32 * frequency).sin_cos();
                    for offset in [index, index + head_dim / 2] {
                        rope_cos[position * head_dim + offset] = cos;
                        rope_sin[position * head_dim + offset] = sin;
                    }
                }
            }
            let f16 = matches!(self.precision, Precision::F16);
            let params: Vec<LayerParams> = layers
                .iter()
                .map(|layer| -> Result<_> {
                    Ok(LayerParams {
                        input_norm: if f16 {
                            layer.input_norm.weight.metal_f16_bits()?.as_ptr().cast()
                        } else {
                            layer.input_norm.weight.data.as_ptr().cast()
                        },
                        post_attention_norm: if f16 {
                            layer
                                .post_attention_norm
                                .weight
                                .metal_f16_bits()?
                                .as_ptr()
                                .cast()
                        } else {
                            layer.post_attention_norm.weight.data.as_ptr().cast()
                        },
                        q_weight: if f16 {
                            layer.q_proj.tensor.metal_f16_bits()?.as_ptr().cast()
                        } else {
                            layer.q_proj.tensor.data.as_ptr().cast()
                        },
                        q_norm: if f16 {
                            layer.q_norm.weight.metal_f16_bits()?.as_ptr().cast()
                        } else {
                            layer.q_norm.weight.data.as_ptr().cast()
                        },
                        k_weight: if f16 {
                            layer.k_proj.tensor.metal_f16_bits()?.as_ptr().cast()
                        } else {
                            layer.k_proj.tensor.data.as_ptr().cast()
                        },
                        k_norm: if f16 {
                            layer.k_norm.weight.metal_f16_bits()?.as_ptr().cast()
                        } else {
                            layer.k_norm.weight.data.as_ptr().cast()
                        },
                        v_weight: if f16 {
                            layer.v_proj.tensor.metal_f16_bits()?.as_ptr().cast()
                        } else {
                            layer.v_proj.tensor.data.as_ptr().cast()
                        },
                        o_weight: if f16 {
                            layer.o_proj.tensor.metal_f16_bits()?.as_ptr().cast()
                        } else {
                            layer.o_proj.tensor.data.as_ptr().cast()
                        },
                        gate_weight: if f16 {
                            layer.gate_proj.tensor.metal_f16_bits()?.as_ptr().cast()
                        } else {
                            layer.gate_proj.tensor.data.as_ptr().cast()
                        },
                        up_weight: if f16 {
                            layer.up_proj.tensor.metal_f16_bits()?.as_ptr().cast()
                        } else {
                            layer.up_proj.tensor.data.as_ptr().cast()
                        },
                        down_weight: if f16 {
                            layer.down_proj.tensor.metal_f16_bits()?.as_ptr().cast()
                        } else {
                            layer.down_proj.tensor.data.as_ptr().cast()
                        },
                    })
                })
                .collect::<Result<_>>()?;
            let input_f16 = f16.then(|| encode_f16_bits(hidden_states));
            let cos_f16 = f16.then(|| encode_f16_bits(&rope_cos));
            let sin_f16 = f16.then(|| encode_f16_bits(&rope_sin));
            let package = self.execution.package_path(batch, seq);
            let package_c = package
                .as_ref()
                .map(|path| std::ffi::CString::new(path.to_string_lossy().as_bytes()))
                .transpose()?;
            let mut output_f32 = vec![0.0f32; hidden_states.len()];
            let mut output_f16 = vec![0u16; hidden_states.len()];
            let status = unsafe {
                synapse_qwen3_forward(
                    self.raw.as_ptr(),
                    batch as u64,
                    seq as u64,
                    hidden as u64,
                    query_heads as u64,
                    kv_heads as u64,
                    head_dim as u64,
                    intermediate as u64,
                    layers.len() as u64,
                    epsilon,
                    i32::from(f16),
                    i32::from(matches!(self.execution.execution, Execution::Explicit)),
                    package_c
                        .as_ref()
                        .map_or(std::ptr::null(), |path| path.as_ptr()),
                    input_f16
                        .as_ref()
                        .map_or(hidden_states.as_ptr().cast(), |values| {
                            values.as_ptr().cast()
                        }),
                    additive_mask.as_ptr(),
                    cos_f16
                        .as_ref()
                        .map_or(rope_cos.as_ptr().cast(), |values| values.as_ptr().cast()),
                    sin_f16
                        .as_ref()
                        .map_or(rope_sin.as_ptr().cast(), |values| values.as_ptr().cast()),
                    params.as_ptr(),
                    if f16 {
                        final_norm.weight.metal_f16_bits()?.as_ptr().cast()
                    } else {
                        final_norm.weight.data.as_ptr().cast()
                    },
                    if f16 {
                        output_f16.as_mut_ptr().cast()
                    } else {
                        output_f32.as_mut_ptr().cast()
                    },
                )
            };
            if status != 0 {
                bail!(
                    "Qwen3 MPSGraph forward failed with status {status}: {}",
                    last_error()
                );
            }
            if f16 {
                hidden_states.copy_from_slice(&decode_f16_bits(&output_f16));
            } else {
                hidden_states.copy_from_slice(&output_f32);
            }
            Ok(())
        }
    }

    impl Drop for Context {
        fn drop(&mut self) {
            unsafe { synapse_qwen3_context_free(self.raw.as_ptr()) }
        }
    }

    fn last_error() -> anyhow::Error {
        unsafe {
            let raw = synapse_qwen3_last_error();
            if raw.is_null() {
                anyhow::anyhow!("unknown Qwen3 MPSGraph error")
            } else {
                anyhow::anyhow!(CStr::from_ptr(raw).to_string_lossy().into_owned())
            }
        }
    }

    unsafe extern "C" {
        fn synapse_qwen3_context_new() -> *mut c_void;
        fn synapse_qwen3_context_free(context: *mut c_void);
        fn synapse_qwen3_forward(
            context: *mut c_void,
            batch: u64,
            seq: u64,
            hidden: u64,
            query_heads: u64,
            kv_heads: u64,
            head_dim: u64,
            intermediate: u64,
            layer_count: u64,
            epsilon: f32,
            dtype: i32,
            explicit_execution: i32,
            package_path: *const c_char,
            input: *const c_void,
            mask: *const f32,
            rope_cos: *const c_void,
            rope_sin: *const c_void,
            layers: *const LayerParams,
            final_norm: *const c_void,
            output: *mut c_void,
        ) -> i32;
        fn synapse_qwen3_last_error() -> *const c_char;
    }
}

#[cfg(target_os = "macos")]
pub(crate) use metal::Context as MetalContext;

#[cfg(not(target_os = "macos"))]
pub(crate) struct MetalContext;

#[cfg(not(target_os = "macos"))]
impl MetalContext {
    pub(crate) fn new(
        _precision: super::Precision,
        _execution: super::MetalExecutionConfig,
    ) -> Result<Self> {
        anyhow::bail!("Qwen3 Metal MPSGraph is only available on macOS")
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &mut self,
        _hidden_states: &mut [f32],
        _attention_mask: &[u8],
        _batch: usize,
        _seq: usize,
        _hidden: usize,
        _query_heads: usize,
        _kv_heads: usize,
        _head_dim: usize,
        _intermediate: usize,
        _epsilon: f32,
        _rope_theta: f32,
        _layers: &[Layer],
        _final_norm: &RmsNorm,
    ) -> Result<()> {
        anyhow::bail!("Qwen3 Metal MPSGraph is only available on macOS")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: Vec<usize>, values: Vec<f32>) -> Tensor {
        Tensor::new(shape, values).expect("test tensor")
    }

    #[test]
    fn rope_rotates_both_halves_with_qwen_layout() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut values, 1, 2, 1, 4, 10_000.0);
        assert_eq!(&values[..4], &[1.0, 2.0, 3.0, 4.0]);
        let (sin, cos) = 1.0f32.sin_cos();
        assert!((values[4] - (1.0 * cos - 3.0 * sin)).abs() < 1e-6);
        assert!((values[6] - (3.0 * cos + 1.0 * sin)).abs() < 1e-6);
    }

    #[test]
    fn per_head_rms_norm_uses_head_dim_weights() {
        let norm = RmsNorm {
            weight: tensor(vec![2], vec![1.0, 2.0]),
            eps: 0.0,
        };
        let mut values = vec![3.0, 4.0, 0.0, 5.0];
        rms_norm_heads(&mut values, 1, 2, 2, &norm).expect("normalize heads");
        let first_inv = 1.0 / 12.5f32.sqrt();
        let second_inv = 1.0 / 12.5f32.sqrt();
        assert!((values[0] - 3.0 * first_inv).abs() < 1e-6);
        assert!((values[1] - 8.0 * first_inv).abs() < 1e-6);
        assert!((values[3] - 10.0 * second_inv).abs() < 1e-6);
    }

    #[test]
    fn last_pool_ignores_padding_and_normalizes() {
        let hidden = vec![1.0, 0.0, 3.0, 4.0, 9.0, 9.0];
        let pooled = last_token_pool_l2(&hidden, &[1, 1, 0], 1, 3, 2);
        assert!((pooled[0][0] - 0.6).abs() < 1e-6);
        assert!((pooled[0][1] - 0.8).abs() < 1e-6);
    }
}
