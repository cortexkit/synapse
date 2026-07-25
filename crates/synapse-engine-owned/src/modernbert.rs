#![allow(dead_code)]

use std::any::Any;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

#[cfg(target_os = "macos")]
use super::{decode_f16_bits, encode_f16_bits, Execution};
use super::{
    get_tensor, load_safetensor_map, normalize_l2, resolve_model_root, BLayout, BatchShape,
    BlockBackend, BlockForwardRequest, KernelProvider, MetalExecutionConfig, ModelFamily,
    Precision, Tensor,
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
    #[serde(default)]
    classifier_pooling: Option<String>,
    #[serde(default = "default_classifier_activation")]
    classifier_activation: String,
    #[serde(default)]
    classifier_bias: bool,
    #[serde(default)]
    norm_bias: bool,
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

fn default_classifier_activation() -> String {
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

#[derive(Clone)]
struct ClassificationHead {
    dense: Linear,
    norm: Vec<f32>,
    classifier_weight: Vec<f32>,
    classifier_bias: f32,
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
    classification_head: Option<ClassificationHead>,
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

        let classification_head = if has_classification_head_tensors(&tensors) {
            let pooling = config
                .classifier_pooling
                .as_deref()
                .context("ModernBERT classifier tensors require classifier_pooling")?;
            ensure!(
                pooling == "mean",
                "unsupported ModernBERT classifier pooling {pooling}; rerank reference requires mean"
            );
            ensure!(
                config.classifier_activation == "gelu",
                "unsupported ModernBERT classifier activation {}",
                config.classifier_activation
            );
            ensure!(
                !config.classifier_bias && !config.norm_bias,
                "ModernBERT classifier dense/norm biases are not supported"
            );
            let classifier = get_tensor(&tensors, "classifier.weight")?;
            ensure!(
                classifier.shape == vec![1, config.hidden_size],
                "ModernBERT classifier weight must have shape [1, hidden_size]"
            );
            Some(ClassificationHead {
                dense: Linear::load(&tensors, "head.dense")?,
                norm: vector(&tensors, "head.norm.weight", config.hidden_size)?,
                classifier_weight: classifier.data,
                classifier_bias: vector(&tensors, "classifier.bias", 1)?[0],
            })
        } else {
            None
        };

        if matches!(precision, Precision::F16) {
            ensure!(
                classification_head.is_none(),
                "ModernBERT reranking is fp32-only; f16 is a later serving experiment"
            );
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
            classification_head,
        })
    }

    fn embed_ids(
        &self,
        provider: &mut dyn KernelProvider,
        sequences: &[Vec<u32>],
        shape: Option<BatchShape>,
    ) -> Result<Vec<Vec<f32>>> {
        let profile = crate::embed_profile_enabled();
        let started = Instant::now();
        let real_batch = sequences.len();
        ensure!(real_batch > 0, "ModernBERT batch must not be empty");
        ensure!(
            sequences.iter().all(|ids| !ids.is_empty()),
            "ModernBERT token sequences must not be empty"
        );
        let real_seq = sequences.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let target = shape.unwrap_or(BatchShape {
            batch: real_batch,
            seq: real_seq,
        });
        ensure!(
            target.batch >= real_batch && target.seq >= real_seq,
            "ModernBERT target shape {}x{} does not cover input {}x{}",
            target.batch,
            target.seq,
            real_batch,
            real_seq
        );
        let (batch, seq) = (target.batch, target.seq);
        ensure!(
            seq <= self.config.max_position_embeddings,
            "sequence length {seq} exceeds ModernBERT maximum {}",
            self.config.max_position_embeddings
        );

        let mut input_ids = vec![self.config.pad_token_id; batch * seq];
        let mut attention_mask = vec![0u8; batch * seq];
        for (row, ids) in sequences.iter().enumerate() {
            for (col, &id) in ids.iter().enumerate() {
                input_ids[row * seq + col] = id;
                attention_mask[row * seq + col] = u8::from(id != self.config.pad_token_id);
            }
        }

        let forward_started = Instant::now();
        let hidden = self.forward(provider, &input_ids, &attention_mask, batch, seq)?;
        if profile {
            eprintln!(
                "[synapse-embed-profile] modernbert_forward batch={} seq={} forward_ms={:.3}",
                batch,
                seq,
                forward_started.elapsed().as_secs_f64() * 1_000.0
            );
        }
        let mut vectors = Vec::with_capacity(real_batch);
        for row in 0..real_batch {
            let start = row * seq * self.config.hidden_size;
            let mut vector = hidden[start..start + self.config.hidden_size].to_vec();
            normalize_l2(&mut vector);
            vectors.push(vector);
        }
        if profile {
            eprintln!(
                "[synapse-embed-profile] modernbert_embed_ids items={} shape={}x{} total_ms={:.3}",
                real_batch,
                batch,
                seq,
                started.elapsed().as_secs_f64() * 1_000.0
            );
        }
        Ok(vectors)
    }

    fn initial_hidden(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let rows = input_ids.len();
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
        Ok(current)
    }

    fn forward_rerank(
        &self,
        provider: &mut dyn KernelProvider,
        input_ids: &[u32],
        attention_mask: &[u8],
        batch: usize,
        seq: usize,
    ) -> Result<Vec<f32>> {
        let head = self
            .classification_head
            .as_ref()
            .context("this ModernBERT checkpoint has no sequence-classification head")?;
        let mut current = self.initial_hidden(input_ids)?;
        let mut metal_scores = None;
        let mut run = |context: &mut dyn Any| {
            let context = context
                .downcast_mut::<MetalContext>()
                .context("ModernBERT provider returned the wrong block context type")?;
            metal_scores = Some(context.forward_rerank(
                self,
                head,
                &mut current,
                attention_mask,
                batch,
                seq,
            )?);
            Ok(())
        };
        if provider.block_forward(BlockForwardRequest {
            family: self.family_name(),
            create_context: new_block_context,
            run: &mut run,
        })? {
            return metal_scores.context("ModernBERT Metal rerank did not return scores");
        }

        self.forward_cpu(&mut current, attention_mask, batch, seq)?;
        self.classify_cpu(head, &current, attention_mask, batch, seq)
    }

    fn classify_cpu(
        &self,
        head: &ClassificationHead,
        hidden_states: &[f32],
        attention_mask: &[u8],
        batch: usize,
        seq: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let mut pooled = vec![0.0; batch * hidden];
        for row in 0..batch {
            let token_count = attention_mask[row * seq..(row + 1) * seq]
                .iter()
                .map(|&value| usize::from(value))
                .sum::<usize>();
            if token_count == 0 {
                continue;
            }
            for col in 0..seq {
                if attention_mask[row * seq + col] == 0 {
                    continue;
                }
                for feature in 0..hidden {
                    pooled[row * hidden + feature] +=
                        hidden_states[(row * seq + col) * hidden + feature];
                }
            }
            for feature in 0..hidden {
                pooled[row * hidden + feature] /= token_count as f32;
            }
        }
        let mut activated = head.dense.forward(batch, hidden, &pooled)?;
        for value in &mut activated {
            *value = gelu(*value);
        }
        layer_norm(
            &mut activated,
            batch,
            hidden,
            &head.norm,
            self.config.norm_eps,
        );
        Ok(activated
            .chunks_exact(hidden)
            .map(|row| {
                row.iter()
                    .zip(&head.classifier_weight)
                    .map(|(&value, &weight)| value * weight)
                    .sum::<f32>()
                    + head.classifier_bias
            })
            .collect())
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        input_ids: &[u32],
        attention_mask: &[u8],
        batch: usize,
        seq: usize,
    ) -> Result<Vec<f32>> {
        let mut current = self.initial_hidden(input_ids)?;

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

fn has_classification_head_tensors(tensors: &HashMap<String, Tensor>) -> bool {
    [
        "head.dense.weight",
        "head.norm.weight",
        "classifier.weight",
        "classifier.bias",
    ]
    .iter()
    .any(|name| has_tensor(tensors, name))
}

fn has_tensor(tensors: &HashMap<String, Tensor>, base_name: &str) -> bool {
    [
        base_name.to_string(),
        format!("bert.{base_name}"),
        format!("model.{base_name}"),
        format!("model.bert.{base_name}"),
    ]
    .iter()
    .any(|candidate| tensors.contains_key(candidate))
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

    fn tokenizer_policy(&self) -> super::FamilyTokenizerPolicy {
        super::FamilyTokenizerPolicy {
            pad_token_id: self.config.pad_token_id,
            terminal_token_id: None,
        }
    }

    fn embed_batch(
        &self,
        provider: &mut dyn KernelProvider,
        sequences: &[Vec<u32>],
        shape: Option<BatchShape>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_ids(provider, sequences, shape)
    }
}

fn new_block_context(
    precision: Precision,
    execution: MetalExecutionConfig,
    _backend: BlockBackend,
) -> Result<Box<dyn Any + Send>> {
    Ok(Box::new(MetalContext::new(precision, execution)?))
}

#[cfg(target_os = "macos")]
struct MetalContext {
    raw: std::ptr::NonNull<std::ffi::c_void>,
    buckets: HashMap<usize, MetalBucket>,
    precision: Precision,
    execution: super::MetalExecutionConfig,
}

// The engine serializes all access to this context with its model mutex.
#[cfg(target_os = "macos")]
unsafe impl Send for MetalContext {}

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
                self.execution.optimization_level(),
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
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0.0,
                0,
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

    fn forward_rerank(
        &mut self,
        model: &ModernBertModel,
        head: &ClassificationHead,
        hidden_states: &mut [f32],
        attention_mask: &[u8],
        batch: usize,
        seq: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            matches!(self.precision, Precision::F32),
            "ModernBERT reranking is fp32-only"
        );
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
        let pooling_mask = attention_mask
            .iter()
            .map(|&value| f32::from(value))
            .collect::<Vec<_>>();
        let params: Vec<ModernBertLayerParams> = model
            .layers
            .iter()
            .map(|layer| ModernBertLayerParams {
                qkv_weight: layer.qkv.weight.data.as_ptr().cast(),
                attention_output_weight: layer.attention_output.weight.data.as_ptr().cast(),
                attention_norm_weight: layer
                    .attention_norm
                    .as_ref()
                    .map_or(std::ptr::null(), |weight| weight.as_ptr().cast()),
                mlp_input_weight: layer.mlp_input.weight.data.as_ptr().cast(),
                mlp_output_weight: layer.mlp_output.weight.data.as_ptr().cast(),
                mlp_norm_weight: layer.mlp_norm.as_ptr().cast(),
                attention_type: match layer.attention_type {
                    AttentionType::Full => 0,
                    AttentionType::Sliding => 1,
                },
            })
            .collect();
        let package = self
            .execution
            .package_path(batch, seq)
            .map(|path| path.with_file_name(format!("{batch}x{seq}-rerank.mpsgraphpackage")));
        let package_c = package
            .as_ref()
            .map(|path| std::ffi::CString::new(path.to_string_lossy().as_bytes()))
            .transpose()?;
        let mut scores = vec![0.0f32; batch];
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
                0,
                i32::from(matches!(self.execution.execution, Execution::Explicit)),
                self.execution.optimization_level(),
                package_c
                    .as_ref()
                    .map_or(std::ptr::null(), |path| path.as_ptr()),
                hidden_states.as_ptr().cast(),
                masks.full.as_ptr(),
                masks.local.as_ptr(),
                bucket.global_cos.as_ptr(),
                bucket.global_sin.as_ptr(),
                bucket.local_cos.as_ptr(),
                bucket.local_sin.as_ptr(),
                model.final_norm.as_ptr(),
                pooling_mask.as_ptr(),
                head.dense.weight.data.as_ptr(),
                head.norm.as_ptr(),
                head.classifier_weight.as_ptr(),
                head.classifier_bias,
                1,
                params.as_ptr(),
                scores.as_mut_ptr().cast(),
            )
        };
        ensure!(
            status == 0,
            "ModernBERT MPSGraph rerank failed: {}",
            metal_error()
        );
        Ok(scores)
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
        optimization_level: i32,
        package_path: *const std::ffi::c_char,
        input: *const std::ffi::c_void,
        full_mask: *const f32,
        local_mask: *const f32,
        global_cos: *const f32,
        global_sin: *const f32,
        local_cos: *const f32,
        local_sin: *const f32,
        final_norm: *const f32,
        pooling_mask: *const f32,
        head_dense: *const f32,
        head_norm: *const f32,
        classifier_weight: *const f32,
        classifier_bias: f32,
        rerank: i32,
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

    fn forward_rerank(
        &mut self,
        _model: &ModernBertModel,
        _head: &ClassificationHead,
        _hidden_states: &mut [f32],
        _attention_mask: &[u8],
        _batch: usize,
        _seq: usize,
    ) -> Result<Vec<f32>> {
        bail!("ModernBERT Metal MPSGraph is only available on macOS")
    }
}

#[cfg_attr(not(any(test, target_os = "macos")), allow(dead_code))]
struct AdditiveMasks {
    full: Vec<f32>,
    local: Vec<f32>,
}

#[cfg_attr(not(any(test, target_os = "macos")), allow(dead_code))]
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

#[cfg_attr(not(any(test, target_os = "macos")), allow(dead_code))]
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

#[cfg_attr(not(any(test, target_os = "macos")), allow(dead_code))]
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
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
    use serde_json::json;

    use super::*;
    use crate::runtime::CpuProvider;

    static NEXT_FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    #[derive(Clone, Copy)]
    enum HeadFixture {
        None,
        Complete,
        MissingClassifierWeight,
    }

    struct ModelFixture {
        path: PathBuf,
    }

    impl ModelFixture {
        fn new(head: HeadFixture, classifier_activation: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "synapse-modernbert-loader-{}-{}",
                std::process::id(),
                NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).expect("create ModernBERT fixture directory");
            fs::write(
                path.join("config.json"),
                serde_json::to_vec(&json!({
                    "model_type": "modernbert",
                    "hidden_size": 2,
                    "intermediate_size": 2,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "vocab_size": 3,
                    "max_position_embeddings": 4,
                    "pad_token_id": 0,
                    "classifier_pooling": "mean",
                    "classifier_activation": classifier_activation
                }))
                .expect("serialize ModernBERT fixture config"),
            )
            .expect("write ModernBERT fixture config");

            let mut tensors = vec![
                tensor(
                    "embeddings.tok_embeddings.weight",
                    &[3, 2],
                    &[0.0, 0.0, 1.0, -1.0, 0.5, 1.0],
                ),
                tensor("embeddings.norm.weight", &[2], &[1.0, 1.0]),
                tensor("final_norm.weight", &[2], &[1.0, 1.0]),
                tensor("layers.0.attn.Wqkv.weight", &[6, 2], &[0.0; 12]),
                tensor("layers.0.attn.Wo.weight", &[2, 2], &[0.0; 4]),
                tensor("layers.0.mlp.Wi.weight", &[4, 2], &[0.0; 8]),
                tensor("layers.0.mlp.Wo.weight", &[2, 2], &[0.0; 4]),
                tensor("layers.0.mlp_norm.weight", &[2], &[1.0, 1.0]),
            ];
            if !matches!(head, HeadFixture::None) {
                tensors.extend([
                    tensor("head.dense.weight", &[2, 2], &[1.0, 0.0, 0.0, 1.0]),
                    tensor("head.norm.weight", &[2], &[1.0, 1.0]),
                    tensor("classifier.bias", &[1], &[0.0]),
                ]);
            }
            if matches!(head, HeadFixture::Complete) {
                tensors.push(tensor("classifier.weight", &[1, 2], &[1.0, -1.0]));
            }
            let views = tensors
                .iter()
                .map(|(name, shape, data)| {
                    (
                        name.as_str(),
                        TensorView::new(Dtype::F32, shape.clone(), data)
                            .expect("create ModernBERT fixture tensor"),
                    )
                })
                .collect::<Vec<_>>();
            serialize_to_file(views, None, &path.join("model.safetensors"))
                .expect("write ModernBERT fixture tensors");
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for ModelFixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn tensor(name: &str, shape: &[usize], values: &[f32]) -> (String, Vec<usize>, Vec<u8>) {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        (
            name.to_string(),
            shape.to_vec(),
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        )
    }

    #[test]
    fn classifier_pooling_without_head_tensors_loads_embed_only() {
        let fixture = ModelFixture::new(HeadFixture::None, "gelu");
        let model = ModernBertModel::load(fixture.path(), Precision::F32)
            .expect("embed checkpoint should load without classifier tensors");
        assert!(model.classification_head.is_none());

        let mut provider = CpuProvider;
        let vectors = model
            .embed_ids(&mut provider, &[vec![1]], None)
            .expect("embed-only checkpoint should embed");
        assert_eq!(vectors.len(), 1);
        assert!((vectors[0].iter().map(|value| value * value).sum::<f32>() - 1.0).abs() < 1e-5);

        let error = model
            .forward_rerank(&mut provider, &[1], &[1], 1, 1)
            .expect_err("embed-only checkpoint must refuse reranking");
        assert!(error
            .to_string()
            .contains("no sequence-classification head"));
    }

    #[test]
    fn classifier_tensors_load_head_and_enforce_config() {
        let fixture = ModelFixture::new(HeadFixture::Complete, "gelu");
        let model = ModernBertModel::load(fixture.path(), Precision::F32)
            .expect("reranker checkpoint should load its classifier head");
        assert!(model.classification_head.is_some());

        let invalid = ModelFixture::new(HeadFixture::Complete, "relu");
        let error = ModernBertModel::load(invalid.path(), Precision::F32)
            .err()
            .expect("classifier config mismatch must fail loading");
        assert!(error.to_string().contains("classifier activation relu"));
    }

    #[test]
    fn partial_classifier_tensors_fail_loading() {
        let fixture = ModelFixture::new(HeadFixture::MissingClassifierWeight, "gelu");
        let error = ModernBertModel::load(fixture.path(), Precision::F32)
            .err()
            .expect("partial reranker head must fail loading");
        assert!(error
            .to_string()
            .contains("missing tensor; tried classifier.weight"));
    }
}
