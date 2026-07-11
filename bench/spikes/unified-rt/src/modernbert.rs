use std::any::Any;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationParams};

use super::{
    decode_f16_bits, encode_f16_bits, get_tensor, load_safetensor_map, normalize_l2,
    resolve_model_root, BLayout, BlockForwardRequest, Execution, KernelProvider,
    MetalExecutionConfig, ModelFamily, Precision, Tensor,
};

#[derive(Clone, Deserialize)]
struct ModernBertConfig {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    pad_token_id: u32,
    #[serde(default = "default_norm_eps")]
    norm_eps: f32,
    #[serde(default = "default_local_attention")]
    local_attention: usize,
    #[serde(default = "default_global_interval")]
    global_attn_every_n_layers: usize,
    #[serde(default = "default_global_rope_theta")]
    global_rope_theta: f32,
    #[serde(default = "default_local_rope_theta")]
    local_rope_theta: f32,
    #[serde(default = "default_hidden_activation")]
    hidden_activation: String,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    layer_types: Option<Vec<String>>,
}

fn default_norm_eps() -> f32 {
    1e-5
}

fn default_local_attention() -> usize {
    128
}

fn default_global_interval() -> usize {
    3
}

fn default_global_rope_theta() -> f32 {
    160_000.0
}

fn default_local_rope_theta() -> f32 {
    10_000.0
}

fn default_hidden_activation() -> String {
    "gelu".to_owned()
}

#[derive(Clone)]
struct Linear {
    weight: Tensor,
}

impl Linear {
    fn load(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Self> {
        Ok(Self {
            weight: get_tensor(tensors, &format!("{name}.weight"))?,
        })
    }

    fn forward(&self, rows: usize, input_size: usize, input: &[f32]) -> Result<Vec<f32>> {
        let (output_size, weight_input) = self.weight.matrix_shape()?;
        ensure!(
            weight_input == input_size,
            "ModernBERT linear {output_size}x{weight_input} received input width {input_size}"
        );
        ensure!(
            input.len() == rows * input_size,
            "ModernBERT linear input shape mismatch"
        );
        let mut output = vec![0.0; rows * output_size];
        super::matmul_impl(
            rows,
            output_size,
            input_size,
            input,
            &self.weight.data,
            BLayout::RowMajorNkTransposed,
            &mut output,
        );
        Ok(output)
    }
}

#[derive(Clone)]
struct Layer {
    qkv: Linear,
    attention_output: Linear,
    attention_norm: Option<Vec<f32>>,
    mlp_input: Linear,
    mlp_output: Linear,
    mlp_norm: Vec<f32>,
    attention_type: AttentionType,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum AttentionType {
    Full,
    Sliding,
}

impl AttentionType {
    fn from_name(name: &str) -> Result<Self> {
        match name {
            "full_attention" => Ok(Self::Full),
            "sliding_attention" => Ok(Self::Sliding),
            other => bail!("unsupported ModernBERT attention type {other}"),
        }
    }
}

struct ModernBertModel {
    config: ModernBertConfig,
    embeddings: Tensor,
    embedding_norm: Vec<f32>,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
}

impl ModernBertModel {
    fn load(path: &Path, precision: Precision) -> Result<Self> {
        let model_root = resolve_model_root(path)?;
        let config = load_config(&model_root)?;
        ensure!(config.model_type == "modernbert", "model is not ModernBERT");
        ensure!(
            config.hidden_size % config.num_attention_heads == 0,
            "ModernBERT hidden size must divide attention heads"
        );
        ensure!(
            config.hidden_activation == "gelu",
            "unsupported ModernBERT hidden activation {}",
            config.hidden_activation
        );
        ensure!(
            !config.attention_bias && !config.mlp_bias,
            "ModernBERT linear biases are not supported by this graph"
        );
        ensure!(
            config.global_attn_every_n_layers > 0,
            "ModernBERT global attention interval must be positive"
        );

        let tensors = load_safetensor_map(&model_root, path)?;
        let embeddings = get_tensor(&tensors, "embeddings.tok_embeddings.weight")?;
        ensure!(
            embeddings.shape == vec![config.vocab_size, config.hidden_size],
            "ModernBERT token embedding shape does not match config"
        );
        let embedding_norm = vector(&tensors, "embeddings.norm.weight", config.hidden_size)?;
        let final_norm = vector(&tensors, "final_norm.weight", config.hidden_size)?;
        let layer_types = resolved_layer_types(&config)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (index, &attention_type) in layer_types.iter().enumerate() {
            let prefix = format!("layers.{index}");
            layers.push(Layer {
                qkv: Linear::load(&tensors, &format!("{prefix}.attn.Wqkv"))?,
                attention_output: Linear::load(&tensors, &format!("{prefix}.attn.Wo"))?,
                attention_norm: (index > 0)
                    .then(|| {
                        vector(
                            &tensors,
                            &format!("{prefix}.attn_norm.weight"),
                            config.hidden_size,
                        )
                    })
                    .transpose()?,
                mlp_input: Linear::load(&tensors, &format!("{prefix}.mlp.Wi"))?,
                mlp_output: Linear::load(&tensors, &format!("{prefix}.mlp.Wo"))?,
                mlp_norm: vector(
                    &tensors,
                    &format!("{prefix}.mlp_norm.weight"),
                    config.hidden_size,
                )?,
                attention_type,
            });
        }

        if matches!(precision, Precision::F16) {
            for layer in &mut layers {
                layer.qkv.weight.prepare_metal_f16();
                layer.attention_output.weight.prepare_metal_f16();
                layer.mlp_input.weight.prepare_metal_f16();
                layer.mlp_output.weight.prepare_metal_f16();
            }
        }

        Ok(Self {
            config,
            embeddings,
            embedding_norm,
            layers,
            final_norm,
        })
    }

    fn embed_batch(
        &self,
        provider: &mut dyn KernelProvider,
        tokenizer: &Tokenizer,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>> {
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|error| anyhow::anyhow!("encode_batch: {error}"))?;
        let batch = encodings.len();
        let seq = encodings
            .iter()
            .map(|encoding| encoding.get_ids().len())
            .max()
            .unwrap_or(1)
            .max(1);
        ensure!(
            seq <= self.config.max_position_embeddings,
            "sequence length {seq} exceeds ModernBERT maximum {}",
            self.config.max_position_embeddings
        );

        let mut input_ids = vec![self.config.pad_token_id; batch * seq];
        let mut attention_mask = vec![0u8; batch * seq];
        for (row, encoding) in encodings.iter().enumerate() {
            for (col, &id) in encoding.get_ids().iter().enumerate() {
                input_ids[row * seq + col] = id;
                attention_mask[row * seq + col] = u8::from(
                    id != self.config.pad_token_id && encoding.get_attention_mask()[col] != 0,
                );
            }
        }

        let hidden = self.forward(provider, &input_ids, &attention_mask, batch, seq)?;
        let mut vectors = Vec::with_capacity(batch);
        for row in 0..batch {
            let start = row * seq * self.config.hidden_size;
            let mut vector = hidden[start..start + self.config.hidden_size].to_vec();
            normalize_l2(&mut vector);
            vectors.push(vector);
        }
        Ok(vectors)
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        input_ids: &[u32],
        attention_mask: &[u8],
        batch: usize,
        seq: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let rows = batch * seq;
        let mut current = vec![0.0; rows * hidden];
        for (row, &token_id) in input_ids.iter().enumerate() {
            let token = token_id as usize;
            ensure!(
                token < self.embeddings.dim(0),
                "token id {token} outside vocabulary"
            );
            current[row * hidden..(row + 1) * hidden]
                .copy_from_slice(&self.embeddings.data[token * hidden..(token + 1) * hidden]);
        }
        layer_norm(
            &mut current,
            rows,
            hidden,
            &self.embedding_norm,
            self.config.norm_eps,
        );

        let mut run = |context: &mut dyn Any| {
            let context = context
                .downcast_mut::<MetalContext>()
                .context("ModernBERT provider returned the wrong block context type")?;
            context.forward(self, &mut current, attention_mask, batch, seq)
        };
        if !provider.block_forward(BlockForwardRequest {
            family: self.family_name(),
            create_context: new_block_context,
            run: &mut run,
        })? {
            self.forward_cpu(&mut current, attention_mask, batch, seq)?;
        }
        Ok(current)
    }

    fn forward_cpu(
        &self,
        current: &mut [f32],
        attention_mask: &[u8],
        batch: usize,
        seq: usize,
    ) -> Result<()> {
        let hidden = self.config.hidden_size;
        let rows = batch * seq;
        for layer in &self.layers {
            let mut attention_input = current.to_vec();
            if let Some(weight) = &layer.attention_norm {
                layer_norm(
                    &mut attention_input,
                    rows,
                    hidden,
                    weight,
                    self.config.norm_eps,
                );
            }
            let qkv = layer.qkv.forward(rows, hidden, &attention_input)?;
            let context = attention(
                &qkv,
                attention_mask,
                batch,
                seq,
                self.config.num_attention_heads,
                layer.attention_type,
                self.config.local_attention / 2,
                rope_theta(&self.config, layer.attention_type),
            )?;
            let attention_output = layer.attention_output.forward(rows, hidden, &context)?;
            add_in_place(current, &attention_output);

            let mut mlp_input = current.to_vec();
            layer_norm(
                &mut mlp_input,
                rows,
                hidden,
                &layer.mlp_norm,
                self.config.norm_eps,
            );
            let projected = layer.mlp_input.forward(rows, hidden, &mlp_input)?;
            ensure!(
                projected.len() == rows * self.config.intermediate_size * 2,
                "ModernBERT GeGLU projection shape mismatch"
            );
            let mut activated = vec![0.0; rows * self.config.intermediate_size];
            for row in 0..rows {
                let source = row * self.config.intermediate_size * 2;
                let destination = row * self.config.intermediate_size;
                for column in 0..self.config.intermediate_size {
                    activated[destination + column] = gelu(projected[source + column])
                        * projected[source + self.config.intermediate_size + column];
                }
            }
            let mlp_output =
                layer
                    .mlp_output
                    .forward(rows, self.config.intermediate_size, &activated)?;
            add_in_place(current, &mlp_output);
        }
        layer_norm(
            current,
            rows,
            hidden,
            &self.final_norm,
            self.config.norm_eps,
        );
        Ok(())
    }
}

fn resolved_layer_types(config: &ModernBertConfig) -> Result<Vec<AttentionType>> {
    let names = config.layer_types.clone().unwrap_or_else(|| {
        (0..config.num_hidden_layers)
            .map(|index| {
                if index % config.global_attn_every_n_layers == 0 {
                    "full_attention".to_owned()
                } else {
                    "sliding_attention".to_owned()
                }
            })
            .collect()
    });
    ensure!(
        names.len() == config.num_hidden_layers,
        "ModernBERT layer_types length does not match num_hidden_layers"
    );
    names
        .iter()
        .map(|name| AttentionType::from_name(name))
        .collect()
}

fn rope_theta(config: &ModernBertConfig, attention_type: AttentionType) -> f32 {
    match attention_type {
        AttentionType::Full => config.global_rope_theta,
        AttentionType::Sliding => config.local_rope_theta,
    }
}

#[allow(clippy::too_many_arguments)]
fn attention(
    qkv: &[f32],
    attention_mask: &[u8],
    batch: usize,
    seq: usize,
    heads: usize,
    attention_type: AttentionType,
    half_window: usize,
    theta: f32,
) -> Result<Vec<f32>> {
    let rows = batch * seq;
    ensure!(qkv.len() % (rows * 3) == 0, "ModernBERT QKV shape mismatch");
    let hidden = qkv.len() / (rows * 3);
    ensure!(hidden % heads == 0, "ModernBERT head shape mismatch");
    let head_dim = hidden / heads;
    ensure!(
        head_dim % 2 == 0,
        "ModernBERT RoPE head dimension must be even"
    );

    let mut output = vec![0.0; rows * hidden];
    let mut q = vec![0.0; seq * head_dim];
    let mut k = vec![0.0; seq * head_dim];
    let mut v = vec![0.0; seq * head_dim];
    let mut scores = vec![0.0; seq * seq];
    let mut head_output = vec![0.0; seq * head_dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for batch_index in 0..batch {
        for head in 0..heads {
            for position in 0..seq {
                let qkv_row = (batch_index * seq + position) * 3 * hidden;
                let head_start = head * head_dim;
                for dimension in 0..head_dim {
                    q[position * head_dim + dimension] = qkv[qkv_row + head_start + dimension];
                    k[position * head_dim + dimension] =
                        qkv[qkv_row + hidden + head_start + dimension];
                    v[position * head_dim + dimension] =
                        qkv[qkv_row + 2 * hidden + head_start + dimension];
                }
                apply_rope(
                    &mut q[position * head_dim..(position + 1) * head_dim],
                    position,
                    theta,
                );
                apply_rope(
                    &mut k[position * head_dim..(position + 1) * head_dim],
                    position,
                    theta,
                );
            }
            super::matmul_impl(
                seq,
                seq,
                head_dim,
                &q,
                &k,
                BLayout::RowMajorNkTransposed,
                &mut scores,
            );
            for query in 0..seq {
                let row = &mut scores[query * seq..(query + 1) * seq];
                for key in 0..seq {
                    let outside_window = attention_type == AttentionType::Sliding
                        && query.abs_diff(key) > half_window;
                    row[key] = if attention_mask[batch_index * seq + key] == 0 || outside_window {
                        -10_000.0
                    } else {
                        row[key] * scale
                    };
                }
                softmax(row);
            }
            super::matmul_impl(
                seq,
                head_dim,
                seq,
                &scores,
                &v,
                BLayout::RowMajorKn,
                &mut head_output,
            );
            for position in 0..seq {
                let destination = (batch_index * seq + position) * hidden + head * head_dim;
                output[destination..destination + head_dim]
                    .copy_from_slice(&head_output[position * head_dim..(position + 1) * head_dim]);
            }
        }
    }
    Ok(output)
}

fn apply_rope(values: &mut [f32], position: usize, theta: f32) {
    let half = values.len() / 2;
    let original = values.to_vec();
    for index in 0..half {
        let frequency = theta.powf(-((2 * index) as f32) / values.len() as f32);
        let angle = position as f32 * frequency;
        let (sin, cos) = angle.sin_cos();
        values[index] = original[index] * cos - original[index + half] * sin;
        values[index + half] = original[index + half] * cos + original[index] * sin;
    }
}

fn layer_norm(data: &mut [f32], rows: usize, hidden: usize, weight: &[f32], eps: f32) {
    debug_assert_eq!(data.len(), rows * hidden);
    debug_assert_eq!(weight.len(), hidden);
    for row in data.chunks_exact_mut(hidden) {
        let mean = row.iter().sum::<f32>() / hidden as f32;
        let variance = row
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / hidden as f32;
        let inverse_stddev = (variance + eps).sqrt().recip();
        for (value, scale) in row.iter_mut().zip(weight) {
            *value = (*value - mean) * inverse_stddev * scale;
        }
    }
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value * std::f32::consts::FRAC_1_SQRT_2))
}

fn softmax(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn add_in_place(destination: &mut [f32], source: &[f32]) {
    debug_assert_eq!(destination.len(), source.len());
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += *source;
    }
}

fn vector(tensors: &HashMap<String, Tensor>, name: &str, expected: usize) -> Result<Vec<f32>> {
    let tensor = get_tensor(tensors, name)?;
    let values = tensor.as_vector()?.to_vec();
    ensure!(
        values.len() == expected,
        "ModernBERT vector {name} shape mismatch"
    );
    Ok(values)
}

fn load_config(model_root: &Path) -> Result<ModernBertConfig> {
    let path = model_root.join("config.json");
    serde_json::from_str(
        &fs::read_to_string(&path).with_context(|| format!("read config {}", path.display()))?,
    )
    .with_context(|| format!("parse ModernBERT config {}", path.display()))
}

pub(super) fn detect_config(config: &serde_json::Value) -> bool {
    config.get("model_type").and_then(serde_json::Value::as_str) == Some("modernbert")
}

pub(super) fn load_family(path: &Path, precision: Precision) -> Result<Box<dyn ModelFamily>> {
    Ok(Box::new(ModernBertModel::load(path, precision)?))
}

impl ModelFamily for ModernBertModel {
    fn family_name(&self) -> &'static str {
        "gte-modernbert"
    }

    fn configure_tokenizer(&self, tokenizer: &mut Tokenizer, max_length: usize) -> Result<()> {
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|error| anyhow::anyhow!("truncation: {error}"))?;
        Ok(())
    }

    fn token_length(&self, tokenizer: &Tokenizer, text: &str, _max_length: usize) -> Result<usize> {
        tokenizer
            .encode(text, true)
            .map(|encoding| encoding.get_ids().len())
            .map_err(|error| anyhow::anyhow!("encode: {error}"))
    }

    fn embed_batch(
        &self,
        provider: &mut dyn KernelProvider,
        tokenizer: &Tokenizer,
        texts: &[&str],
        _max_length: usize,
    ) -> Result<Vec<Vec<f32>>> {
        ModernBertModel::embed_batch(self, provider, tokenizer, texts)
    }

    fn validate_reference_coverage(&self, matched: usize, produced: usize) -> Result<()> {
        ensure!(
            matched == produced,
            "ModernBERT reference matched {matched} of {produced} produced vectors"
        );
        Ok(())
    }

    fn default_label(&self, precision: Precision) -> String {
        format!(
            "Alibaba-NLP/gte-modernbert-base@owned-rt-{}",
            precision.as_str()
        )
    }

    fn notes(&self) -> String {
        format!(
            "ModernBERT CLS+l2, RoPE, alternating full/local-{} attention, GeGLU, pre-norm",
            self.config.local_attention
        )
    }
}

fn new_block_context(
    precision: Precision,
    execution: MetalExecutionConfig,
) -> Result<Box<dyn Any>> {
    Ok(Box::new(MetalContext::new(precision, execution)?))
}

#[cfg(test)]
fn validate_parity(cosine: f64, rank: f64, minimum_cosine: f64, minimum_rank: f64) -> Result<()> {
    ensure!(
        cosine >= minimum_cosine,
        "ModernBERT mean cosine {cosine:.8} below minimum {minimum_cosine:.8}"
    );
    ensure!(
        rank >= minimum_rank,
        "ModernBERT top-10 rank overlap {rank:.8} below minimum {minimum_rank:.8}"
    );
    Ok(())
}

#[cfg(target_os = "macos")]
struct MetalContext {
    raw: std::ptr::NonNull<std::ffi::c_void>,
    buckets: HashMap<usize, MetalBucket>,
    precision: Precision,
    execution: super::MetalExecutionConfig,
}

#[cfg(target_os = "macos")]
struct MetalBucket {
    band_mask: Vec<f32>,
    global_cos: Vec<f32>,
    global_sin: Vec<f32>,
    local_cos: Vec<f32>,
    local_sin: Vec<f32>,
}

#[cfg(target_os = "macos")]
impl MetalContext {
    fn new(precision: Precision, execution: super::MetalExecutionConfig) -> Result<Self> {
        let raw = unsafe { synapse_modernbert_mps_context_new() };
        let raw = std::ptr::NonNull::new(raw).ok_or_else(metal_error)?;
        Ok(Self {
            raw,
            buckets: HashMap::new(),
            precision,
            execution,
        })
    }

    fn forward(
        &mut self,
        model: &ModernBertModel,
        hidden_states: &mut [f32],
        attention_mask: &[u8],
        batch: usize,
        seq: usize,
    ) -> Result<()> {
        let bucket = self.buckets.entry(seq).or_insert_with(|| {
            let head_dim = model.config.hidden_size / model.config.num_attention_heads;
            let (global_cos, global_sin) =
                rope_tables(seq, head_dim, model.config.global_rope_theta);
            let (local_cos, local_sin) = rope_tables(seq, head_dim, model.config.local_rope_theta);
            MetalBucket {
                band_mask: band_mask(seq, model.config.local_attention / 2),
                global_cos,
                global_sin,
                local_cos,
                local_sin,
            }
        });
        let masks = additive_masks_with_band(attention_mask, batch, seq, &bucket.band_mask);
        let input_f16 =
            matches!(self.precision, Precision::F16).then(|| encode_f16_bits(hidden_states));
        let params: Vec<ModernBertLayerParams> = model
            .layers
            .iter()
            .map(|layer| -> Result<_> {
                let f16 = matches!(self.precision, Precision::F16);
                Ok(ModernBertLayerParams {
                    qkv_weight: if f16 {
                        layer.qkv.weight.metal_f16_bits()?.as_ptr().cast()
                    } else {
                        layer.qkv.weight.data.as_ptr().cast()
                    },
                    attention_output_weight: if f16 {
                        layer
                            .attention_output
                            .weight
                            .metal_f16_bits()?
                            .as_ptr()
                            .cast()
                    } else {
                        layer.attention_output.weight.data.as_ptr().cast()
                    },
                    attention_norm_weight: layer
                        .attention_norm
                        .as_ref()
                        .map_or(std::ptr::null(), |weight| weight.as_ptr().cast()),
                    mlp_input_weight: if f16 {
                        layer.mlp_input.weight.metal_f16_bits()?.as_ptr().cast()
                    } else {
                        layer.mlp_input.weight.data.as_ptr().cast()
                    },
                    mlp_output_weight: if f16 {
                        layer.mlp_output.weight.metal_f16_bits()?.as_ptr().cast()
                    } else {
                        layer.mlp_output.weight.data.as_ptr().cast()
                    },
                    mlp_norm_weight: layer.mlp_norm.as_ptr().cast(),
                    attention_type: match layer.attention_type {
                        AttentionType::Full => 0,
                        AttentionType::Sliding => 1,
                    },
                })
            })
            .collect::<Result<_>>()?;
        let package = self.execution.package_path(batch, seq);
        let package_c = package
            .as_ref()
            .map(|path| std::ffi::CString::new(path.to_string_lossy().as_bytes()))
            .transpose()?;
        let mut output_f32 = vec![0.0; hidden_states.len()];
        let mut output_f16 = vec![0u16; hidden_states.len()];
        let f16 = matches!(self.precision, Precision::F16);
        let status = unsafe {
            synapse_modernbert_mps_forward(
                self.raw.as_ptr(),
                batch as u64,
                seq as u64,
                model.config.hidden_size as u64,
                model.config.num_attention_heads as u64,
                model.config.intermediate_size as u64,
                model.layers.len() as u64,
                model.config.norm_eps,
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
                masks.full.as_ptr(),
                masks.local.as_ptr(),
                bucket.global_cos.as_ptr(),
                bucket.global_sin.as_ptr(),
                bucket.local_cos.as_ptr(),
                bucket.local_sin.as_ptr(),
                model.final_norm.as_ptr(),
                params.as_ptr(),
                if f16 {
                    output_f16.as_mut_ptr().cast()
                } else {
                    output_f32.as_mut_ptr().cast()
                },
            )
        };
        ensure!(
            status == 0,
            "ModernBERT MPSGraph forward failed: {}",
            metal_error()
        );
        if f16 {
            hidden_states.copy_from_slice(&decode_f16_bits(&output_f16));
        } else {
            hidden_states.copy_from_slice(&output_f32);
        }
        Ok(())
    }
}

#[cfg(target_os = "macos")]
impl Drop for MetalContext {
    fn drop(&mut self) {
        unsafe { synapse_modernbert_mps_context_free(self.raw.as_ptr()) };
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
struct ModernBertLayerParams {
    qkv_weight: *const std::ffi::c_void,
    attention_output_weight: *const std::ffi::c_void,
    attention_norm_weight: *const std::ffi::c_void,
    mlp_input_weight: *const std::ffi::c_void,
    mlp_output_weight: *const std::ffi::c_void,
    mlp_norm_weight: *const std::ffi::c_void,
    attention_type: i32,
}

#[cfg(target_os = "macos")]
fn metal_error() -> anyhow::Error {
    unsafe {
        let pointer = synapse_modernbert_mps_last_error();
        if pointer.is_null() {
            anyhow::anyhow!("unknown MPSGraph error")
        } else {
            anyhow::anyhow!(std::ffi::CStr::from_ptr(pointer)
                .to_string_lossy()
                .into_owned())
        }
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn synapse_modernbert_mps_context_new() -> *mut std::ffi::c_void;
    fn synapse_modernbert_mps_context_free(context: *mut std::ffi::c_void);
    fn synapse_modernbert_mps_last_error() -> *const std::ffi::c_char;
    fn synapse_modernbert_mps_forward(
        context: *mut std::ffi::c_void,
        batch: u64,
        seq: u64,
        hidden: u64,
        heads: u64,
        intermediate: u64,
        layers: u64,
        epsilon: f32,
        dtype: i32,
        explicit_execution: i32,
        package_path: *const std::ffi::c_char,
        input: *const std::ffi::c_void,
        full_mask: *const f32,
        local_mask: *const f32,
        global_cos: *const f32,
        global_sin: *const f32,
        local_cos: *const f32,
        local_sin: *const f32,
        final_norm: *const f32,
        layer_params: *const ModernBertLayerParams,
        output: *mut std::ffi::c_void,
    ) -> i32;
}

#[cfg(not(target_os = "macos"))]
struct MetalContext;

#[cfg(not(target_os = "macos"))]
impl MetalContext {
    fn new(_precision: Precision, _execution: super::MetalExecutionConfig) -> Result<Self> {
        bail!("ModernBERT Metal MPSGraph is only available on macOS")
    }

    fn forward(
        &mut self,
        _model: &ModernBertModel,
        _hidden_states: &mut [f32],
        _attention_mask: &[u8],
        _batch: usize,
        _seq: usize,
    ) -> Result<()> {
        bail!("ModernBERT Metal MPSGraph is only available on macOS")
    }
}

struct AdditiveMasks {
    full: Vec<f32>,
    local: Vec<f32>,
}

fn band_mask(seq: usize, half_window: usize) -> Vec<f32> {
    let mut band = vec![0.0; seq * seq];
    for query in 0..seq {
        for key in 0..seq {
            if query.abs_diff(key) > half_window {
                band[query * seq + key] = -10_000.0;
            }
        }
    }
    band
}

fn additive_masks_with_band(mask: &[u8], batch: usize, seq: usize, band: &[f32]) -> AdditiveMasks {
    debug_assert_eq!(band.len(), seq * seq);
    let mut full = vec![0.0; batch * seq * seq];
    let mut local = vec![0.0; batch * seq * seq];
    for batch_index in 0..batch {
        for query in 0..seq {
            for key in 0..seq {
                let index = (batch_index * seq + query) * seq + key;
                let padding: f32 = if mask[batch_index * seq + key] == 0 {
                    -10_000.0
                } else {
                    0.0
                };
                full[index] = padding;
                local[index] = padding.min(band[query * seq + key]);
            }
        }
    }
    AdditiveMasks { full, local }
}

#[cfg(test)]
fn additive_masks(mask: &[u8], batch: usize, seq: usize, half_window: usize) -> AdditiveMasks {
    additive_masks_with_band(mask, batch, seq, &band_mask(seq, half_window))
}

fn rope_tables(seq: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0; seq * head_dim];
    let mut sin = vec![0.0; seq * head_dim];
    for position in 0..seq {
        for index in 0..half {
            let frequency = theta.powf(-((2 * index) as f32) / head_dim as f32);
            let (sine, cosine) = (position as f32 * frequency).sin_cos();
            cos[position * head_dim + index] = cosine;
            cos[position * head_dim + half + index] = cosine;
            sin[position * head_dim + index] = sine;
            sin[position * head_dim + half + index] = sine;
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use synapse_bench::parity::{load_reference, mean_parity, rank_overlap};

    const PARITY_K: usize = 10;
    const PARITY_STRIDE: usize = 1;

    #[test]
    fn modernbert_layer_pattern_starts_with_global_attention() {
        let config = ModernBertConfig {
            model_type: "modernbert".into(),
            hidden_size: 8,
            intermediate_size: 12,
            num_hidden_layers: 7,
            num_attention_heads: 2,
            vocab_size: 32,
            max_position_embeddings: 64,
            pad_token_id: 0,
            norm_eps: 1e-5,
            local_attention: 4,
            global_attn_every_n_layers: 3,
            global_rope_theta: 160_000.0,
            local_rope_theta: 10_000.0,
            hidden_activation: "gelu".into(),
            attention_bias: false,
            mlp_bias: false,
            layer_types: None,
        };
        assert_eq!(
            resolved_layer_types(&config).unwrap(),
            vec![
                AttentionType::Full,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Full,
                AttentionType::Sliding,
                AttentionType::Sliding,
                AttentionType::Full,
            ]
        );
    }

    #[test]
    fn local_mask_uses_inclusive_half_window_and_padding() {
        let masks = additive_masks(&[1, 1, 1, 0], 1, 4, 1);
        assert_eq!(&masks.local[0..4], &[0.0, 0.0, -10_000.0, -10_000.0]);
        assert_eq!(&masks.full[0..4], &[0.0, 0.0, 0.0, -10_000.0]);
    }

    #[test]
    fn parity_gate_asserts_both_certification_thresholds() {
        validate_parity(0.9999, 0.995, 0.9999, 0.995).unwrap();
        assert!(validate_parity(0.99989, 1.0, 0.9999, 0.995).is_err());
        assert!(validate_parity(1.0, 0.9949, 0.9999, 0.995).is_err());
    }

    #[test]
    fn rope_preserves_values_at_position_zero() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut values, 0, 10_000.0);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires MODERNBERT_F16_MODEL and a Metal device"]
    fn modernbert_f16_static_feeds_survive_multiple_calls() {
        const CHILD_OUTPUT: &str = "MODERNBERT_F16_MULTICALL_CHILD_OUTPUT";
        const TARGET_TEXTS: &[&str] = &[
            "A persistent cache must preserve every learned normalization scale.",
            "The third invocation uses a different batch shape and different content.",
            "Fresh-process output is the reference for detecting cross-call pointer reuse.",
        ];

        let model_path = std::path::PathBuf::from(
            std::env::var("MODERNBERT_F16_MODEL").expect("set MODERNBERT_F16_MODEL"),
        );
        let tokenizer_path = std::env::var("MODERNBERT_F16_TOKENIZER")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| model_path.join("tokenizer.json"));
        let model = ModernBertModel::load(&model_path, Precision::F16).unwrap();
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();
        tokenizer.with_padding(None);
        let execution = super::super::MetalExecutionConfig {
            execution: Execution::Lazy,
            package_root: None,
        };
        let mut backend = crate::MetalProvider::new_with_config(Precision::F16, execution).unwrap();

        if let Ok(output_path) = std::env::var(CHILD_OUTPUT) {
            let baseline = model
                .embed_batch(&mut backend, &tokenizer, TARGET_TEXTS)
                .unwrap();
            fs::write(output_path, serde_json::to_vec(&baseline).unwrap()).unwrap();
            return;
        }

        model
            .embed_batch(&mut backend, &tokenizer, &["first call"])
            .unwrap();
        model
            .embed_batch(
                &mut backend,
                &tokenizer,
                &[
                    "second call has two rows",
                    "and enough distinct content to use another shape",
                ],
            )
            .unwrap();
        let third_call = model
            .embed_batch(&mut backend, &tokenizer, TARGET_TEXTS)
            .unwrap();

        let baseline_path = std::env::temp_dir().join(format!(
            "modernbert-f16-multicall-baseline-{}.json",
            std::process::id()
        ));
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "modernbert::tests::modernbert_f16_static_feeds_survive_multiple_calls",
                "--ignored",
            ])
            .env(CHILD_OUTPUT, &baseline_path)
            .output()
            .unwrap();
        assert!(
            child.status.success(),
            "fresh-process baseline failed: stdout={} stderr={}",
            String::from_utf8_lossy(&child.stdout),
            String::from_utf8_lossy(&child.stderr)
        );
        let baseline: Vec<Vec<f32>> =
            serde_json::from_slice(&fs::read(&baseline_path).unwrap()).unwrap();
        fs::remove_file(&baseline_path).unwrap();

        assert_eq!(third_call.len(), baseline.len());
        for (index, (actual, expected)) in third_call.iter().zip(&baseline).enumerate() {
            let similarity = synapse_bench::parity::cosine(actual, expected);
            assert!(
                similarity >= 0.999999,
                "row {index} changed after repeated calls: cosine={similarity:.9}"
            );
        }
    }

    #[test]
    #[ignore = "requires MODERNBERT_PARITY_VECTORS and MODERNBERT_REFERENCE_VECTORS"]
    fn modernbert_400_chunk_parity_gate() {
        let produced_path = std::env::var("MODERNBERT_PARITY_VECTORS").unwrap();
        let reference_path = std::env::var("MODERNBERT_REFERENCE_VECTORS").unwrap();
        let produced = load_reference(Path::new(&produced_path)).unwrap();
        let reference = load_reference(Path::new(&reference_path)).unwrap();
        assert_eq!(produced.len(), 400);
        assert_eq!(reference.len(), 400);
        let (cosine, matched) = mean_parity(produced.clone(), &reference);
        assert_eq!(matched, 400);
        let rank = rank_overlap(&produced, &reference, PARITY_K, PARITY_STRIDE).unwrap();
        validate_parity(cosine.unwrap(), rank.mean_topk_overlap, 0.9999, 0.995).unwrap();
    }
}
