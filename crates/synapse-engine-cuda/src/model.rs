#![allow(dead_code, private_interfaces)]

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};
use safetensors::tensor::{Dtype as SafeDtype, SafeTensors};
use serde::Deserialize;

use crate::cuda::{MiniLmContext, ModernBertContext, Qwen3Context};
use crate::{encode_f16_bits, Precision};

#[derive(Clone, Debug)]
struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    fn vector(self, name: &str) -> Result<Vec<f32>> {
        ensure!(
            self.shape.len() == 1,
            "{name} is not a vector: {:?}",
            self.shape
        );
        Ok(self.data)
    }

    fn matrix(self, name: &str) -> Result<Vec<f32>> {
        ensure!(
            self.shape.len() == 2,
            "{name} is not a matrix: {:?}",
            self.shape
        );
        Ok(self.data)
    }
}

#[derive(Clone)]
pub(crate) struct Linear {
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct MiniLmLayer {
    pub(crate) query: Linear,
    pub(crate) key: Linear,
    pub(crate) value: Linear,
    pub(crate) attention_output: Linear,
    pub(crate) attention_norm: Linear,
    pub(crate) intermediate: Linear,
    pub(crate) output: Linear,
    pub(crate) output_norm: Linear,
}

pub(crate) struct MiniLmModel {
    pub(crate) hidden: usize,
    heads: usize,
    intermediate: usize,
    epsilon: f32,
    pad_token_id: u32,
    word_embeddings: Tensor,
    position_embeddings: Tensor,
    token_type_embeddings: Tensor,
    embedding_norm: Linear,
    layers: Vec<MiniLmLayer>,
}

#[derive(Clone)]
pub(crate) struct ModernBertLayer {
    pub(crate) qkv_weight: Vec<f32>,
    pub(crate) attention_output_weight: Vec<f32>,
    pub(crate) attention_norm_weight: Option<Vec<f32>>,
    pub(crate) mlp_input_weight: Vec<f32>,
    pub(crate) mlp_output_weight: Vec<f32>,
    pub(crate) mlp_norm_weight: Vec<f32>,
    pub(crate) sliding_attention: bool,
}

pub(crate) struct ModernBertModel {
    pub(crate) hidden: usize,
    pub(crate) heads: usize,
    pub(crate) intermediate: usize,
    pub(crate) epsilon: f32,
    pub(crate) global_rope_theta: f32,
    pub(crate) local_rope_theta: f32,
    pub(crate) local_attention: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) pad_token_id: u32,
    pub(crate) embeddings: Tensor,
    pub(crate) embedding_norm: Vec<f32>,
    pub(crate) layers: Vec<ModernBertLayer>,
    pub(crate) final_norm: Vec<f32>,
}

#[derive(Clone)]
pub(crate) struct Qwen3Layer {
    pub(crate) input_norm: Vec<f32>,
    pub(crate) post_attention_norm: Vec<f32>,
    pub(crate) q_weight: Vec<f32>,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_weight: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) v_weight: Vec<f32>,
    pub(crate) o_weight: Vec<f32>,
    pub(crate) gate_weight: Vec<f32>,
    pub(crate) up_weight: Vec<f32>,
    pub(crate) down_weight: Vec<f32>,
}

pub(crate) struct Qwen3Model {
    pub(crate) hidden: usize,
    pub(crate) query_heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) intermediate: usize,
    pub(crate) epsilon: f32,
    pub(crate) rope_theta: f32,
    pub(crate) vocab_size: usize,
    pub(crate) eos_token_id: u32,
    pub(crate) embeddings: Tensor,
    pub(crate) layers: Vec<Qwen3Layer>,
    pub(crate) final_norm: Vec<f32>,
}

pub(crate) fn resolve_model_root(path: &Path) -> Result<PathBuf> {
    if path.is_dir() {
        return Ok(path.to_path_buf());
    }
    if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
        return path
            .parent()
            .map(Path::to_path_buf)
            .context("model file has no parent directory");
    }
    bail!(
        "model path {} is neither a directory nor a safetensors file",
        path.display()
    )
}

fn load_safetensor_map(root: &Path, original: &Path) -> Result<HashMap<String, Tensor>> {
    if original.is_file() && original.extension().and_then(|v| v.to_str()) == Some("safetensors") {
        return load_safetensors_file(original);
    }
    let single = root.join("model.safetensors");
    if single.is_file() {
        return load_safetensors_file(&single);
    }
    let index_file = root.join("model.safetensors.index.json");
    if !index_file.is_file() {
        bail!(
            "could not find model.safetensors or model.safetensors.index.json under {}",
            root.display()
        );
    }
    #[derive(Deserialize)]
    struct Index {
        weight_map: HashMap<String, String>,
    }
    let index: Index = serde_json::from_str(
        &fs::read_to_string(&index_file)
            .with_context(|| format!("read safetensors index {}", index_file.display()))?,
    )
    .with_context(|| format!("parse safetensors index {}", index_file.display()))?;
    let mut merged = HashMap::new();
    let unique: HashSet<String> = index.weight_map.into_values().collect();
    for name in unique {
        merged.extend(load_safetensors_file(&root.join(name))?);
    }
    Ok(merged)
}

fn load_safetensors_file(path: &Path) -> Result<HashMap<String, Tensor>> {
    let bytes = fs::read(path).with_context(|| format!("read safetensors {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|error| anyhow::anyhow!("load safetensors {}: {error}", path.display()))?;
    let mut result = HashMap::new();
    for name in tensors.names() {
        let view = tensors
            .tensor(name)
            .map_err(|error| anyhow::anyhow!("read tensor {name}: {error}"))?;
        if matches!(
            view.dtype(),
            SafeDtype::F32 | SafeDtype::F16 | SafeDtype::BF16
        ) {
            result.insert(
                name.to_string(),
                tensor_from_bytes(view.dtype(), view.shape(), view.data())?,
            );
        }
    }
    Ok(result)
}

fn tensor_from_bytes(dtype: SafeDtype, shape: &[usize], bytes: &[u8]) -> Result<Tensor> {
    let data: Vec<f32> = match dtype {
        SafeDtype::F32 => bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
            .collect(),
        SafeDtype::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                half::f16::from_bits(u16::from_le_bytes(chunk.try_into().expect("f16 chunk")))
                    .to_f32()
            })
            .collect(),
        SafeDtype::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| {
                half::bf16::from_bits(u16::from_le_bytes(chunk.try_into().expect("bf16 chunk")))
                    .to_f32()
            })
            .collect(),
        other => bail!("unsupported safetensor dtype {other:?}; expected f32/f16/bf16"),
    };
    ensure!(
        bytes.len()
            == data.len()
                * match dtype {
                    SafeDtype::F32 => 4,
                    _ => 2,
                }
    );
    ensure!(
        shape.iter().product::<usize>() == data.len(),
        "tensor shape/data mismatch for {shape:?}"
    );
    Ok(Tensor {
        shape: shape.to_vec(),
        data,
    })
}

fn tensor_candidates(name: &str) -> [String; 4] {
    [
        name.to_owned(),
        format!("bert.{name}"),
        format!("model.{name}"),
        format!("model.bert.{name}"),
    ]
}

fn get_tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    let candidates = tensor_candidates(name);
    candidates
        .iter()
        .find_map(|candidate| tensors.get(candidate))
        .ok_or_else(|| anyhow::anyhow!("missing tensor; tried {}", candidates.join(", ")))
}

fn take_matrix(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Vec<f32>> {
    get_tensor(tensors, name)?.clone().matrix(name)
}

fn take_vector(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Vec<f32>> {
    get_tensor(tensors, name)?.clone().vector(name)
}

const MODERNBERT_EMBEDDINGS: &str = "embeddings.tok_embeddings.weight";
const MODERNBERT_EMBEDDING_NORM: &str = "embeddings.norm.weight";
const MODERNBERT_FINAL_NORM: &str = "final_norm.weight";
const QWEN3_EMBEDDINGS: &str = "embed_tokens.weight";
const QWEN3_FINAL_NORM: &str = "norm.weight";

fn family_layer_tensor_name(index: usize, name: &str) -> String {
    format!("layers.{index}.{name}.weight")
}

#[derive(Deserialize)]
struct BertConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    layer_norm_eps: f32,
    #[serde(default = "default_hidden_act")]
    hidden_act: String,
    #[serde(default)]
    pad_token_id: u32,
}

fn default_layer_norm_eps() -> f32 {
    1e-12
}
fn default_hidden_act() -> String {
    "gelu".to_owned()
}

impl MiniLmModel {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let root = resolve_model_root(path)?;
        let config: BertConfig =
            serde_json::from_str(&fs::read_to_string(root.join("config.json"))?)?;
        ensure!(config.hidden_act == "gelu" || config.hidden_act == "gelu_new");
        ensure!(config.hidden_size % config.num_attention_heads == 0);
        let tensors = load_safetensor_map(&root, path)?;
        let word_embeddings = get_tensor(&tensors, "embeddings.word_embeddings.weight")?.clone();
        let position_embeddings =
            get_tensor(&tensors, "embeddings.position_embeddings.weight")?.clone();
        let token_type_embeddings =
            get_tensor(&tensors, "embeddings.token_type_embeddings.weight")?.clone();
        let embedding_norm = Linear {
            weight: take_vector(&tensors, "embeddings.LayerNorm.weight")?,
            bias: take_vector(&tensors, "embeddings.LayerNorm.bias")?,
        };
        ensure!(word_embeddings.shape == vec![config.vocab_size, config.hidden_size]);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layer.{index}");
            let linear = |name: &str| -> Result<Linear> {
                Ok(Linear {
                    weight: take_matrix(&tensors, &format!("{prefix}.{name}.weight"))?,
                    bias: take_vector(&tensors, &format!("{prefix}.{name}.bias"))?,
                })
            };
            layers.push(MiniLmLayer {
                query: linear("attention.self.query")?,
                key: linear("attention.self.key")?,
                value: linear("attention.self.value")?,
                attention_output: linear("attention.output.dense")?,
                attention_norm: Linear {
                    weight: take_vector(
                        &tensors,
                        &format!("{prefix}.attention.output.LayerNorm.weight"),
                    )?,
                    bias: take_vector(
                        &tensors,
                        &format!("{prefix}.attention.output.LayerNorm.bias"),
                    )?,
                },
                intermediate: linear("intermediate.dense")?,
                output: linear("output.dense")?,
                output_norm: Linear {
                    weight: take_vector(&tensors, &format!("{prefix}.output.LayerNorm.weight"))?,
                    bias: take_vector(&tensors, &format!("{prefix}.output.LayerNorm.bias"))?,
                },
            });
        }
        Ok(Self {
            hidden: config.hidden_size,
            heads: config.num_attention_heads,
            intermediate: config.intermediate_size,
            epsilon: config.layer_norm_eps,
            pad_token_id: config.pad_token_id,
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embedding_norm,
            layers,
        })
    }

    pub(crate) fn embed(
        &self,
        context: &mut MiniLmContext,
        sequences: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>> {
        let real_batch = sequences.len();
        ensure!(real_batch > 0 && sequences.iter().all(|ids| !ids.is_empty()));
        let seq = sequences.iter().map(Vec::len).max().unwrap_or(1);
        ensure!(seq <= self.position_embeddings.shape[0]);
        let mut hidden = vec![0.0; real_batch * seq * self.hidden];
        let mut mask = vec![0u8; real_batch * seq];
        for (row, ids) in sequences.iter().enumerate() {
            for (position, &token) in ids.iter().enumerate() {
                let token = token as usize;
                ensure!(token < self.word_embeddings.shape[0]);
                let destination = (row * seq + position) * self.hidden;
                for feature in 0..self.hidden {
                    hidden[destination + feature] = self.word_embeddings.data
                        [token * self.hidden + feature]
                        + self.position_embeddings.data[position * self.hidden + feature]
                        + self.token_type_embeddings.data[feature];
                }
                mask[row * seq + position] = 1;
            }
        }
        layer_norm(
            &mut hidden,
            real_batch * seq,
            self.hidden,
            &self.embedding_norm.weight,
            &self.embedding_norm.bias,
            self.epsilon,
        );
        let output = context.forward(
            &encode_f16_bits(&hidden),
            &mask,
            real_batch,
            seq,
            self.hidden,
            self.heads,
            self.intermediate,
            self.epsilon,
            &self.layers,
        )?;
        Ok(output)
    }

    pub(crate) fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }
}

#[derive(Deserialize)]
struct ModernConfig {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    max_position_embeddings: usize,
    pad_token_id: u32,
    #[serde(default = "default_modern_eps")]
    norm_eps: f32,
    #[serde(default = "default_local_attention")]
    local_attention: usize,
    #[serde(default = "default_global_interval")]
    global_attn_every_n_layers: usize,
    #[serde(default = "default_global_theta")]
    global_rope_theta: f32,
    #[serde(default = "default_local_theta")]
    local_rope_theta: f32,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
}
fn default_modern_eps() -> f32 {
    1e-5
}
fn default_local_attention() -> usize {
    128
}
fn default_global_interval() -> usize {
    3
}
fn default_global_theta() -> f32 {
    160_000.0
}
fn default_local_theta() -> f32 {
    10_000.0
}

impl ModernBertModel {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let root = resolve_model_root(path)?;
        let config: ModernConfig =
            serde_json::from_str(&fs::read_to_string(root.join("config.json"))?)?;
        ensure!(config.model_type == "modernbert");
        ensure!(config.hidden_size % config.num_attention_heads == 0);
        let tensors = load_safetensor_map(&root, path)?;
        let embeddings = get_tensor(&tensors, MODERNBERT_EMBEDDINGS)?.clone();
        let embedding_norm = take_vector(&tensors, MODERNBERT_EMBEDDING_NORM)?;
        let final_norm = take_vector(&tensors, MODERNBERT_FINAL_NORM)?;
        let layer_types = config.layer_types.clone().unwrap_or_else(|| {
            (0..config.num_hidden_layers)
                .map(|index| {
                    if index % config.global_attn_every_n_layers == 0 {
                        "full_attention"
                    } else {
                        "sliding_attention"
                    }
                    .to_owned()
                })
                .collect()
        });
        ensure!(layer_types.len() == config.num_hidden_layers);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (index, layer_type) in layer_types.iter().enumerate() {
            let attention = match layer_type.as_str() {
                "full_attention" => false,
                "sliding_attention" => true,
                other => bail!("unsupported ModernBERT attention type {other}"),
            };
            layers.push(ModernBertLayer {
                qkv_weight: take_matrix(&tensors, &family_layer_tensor_name(index, "attn.Wqkv"))?,
                attention_output_weight: take_matrix(
                    &tensors,
                    &family_layer_tensor_name(index, "attn.Wo"),
                )?,
                attention_norm_weight: (index > 0)
                    .then(|| take_vector(&tensors, &family_layer_tensor_name(index, "attn_norm")))
                    .transpose()?,
                mlp_input_weight: take_matrix(
                    &tensors,
                    &family_layer_tensor_name(index, "mlp.Wi"),
                )?,
                mlp_output_weight: take_matrix(
                    &tensors,
                    &family_layer_tensor_name(index, "mlp.Wo"),
                )?,
                mlp_norm_weight: take_vector(
                    &tensors,
                    &family_layer_tensor_name(index, "mlp_norm"),
                )?,
                sliding_attention: attention,
            });
        }
        Ok(Self {
            hidden: config.hidden_size,
            heads: config.num_attention_heads,
            intermediate: config.intermediate_size,
            epsilon: config.norm_eps,
            global_rope_theta: config.global_rope_theta,
            local_rope_theta: config.local_rope_theta,
            local_attention: config.local_attention,
            max_position_embeddings: config.max_position_embeddings,
            pad_token_id: config.pad_token_id,
            embeddings,
            embedding_norm,
            layers,
            final_norm,
        })
    }

    pub(crate) fn embed(
        &self,
        context: &mut ModernBertContext,
        sequences: &[Vec<u32>],
        precision: Precision,
    ) -> Result<Vec<Vec<f32>>> {
        let real_batch = sequences.len();
        ensure!(real_batch > 0 && sequences.iter().all(|ids| !ids.is_empty()));
        let seq = sequences.iter().map(Vec::len).max().unwrap_or(1);
        ensure!(seq <= self.max_position_embeddings);
        let mut hidden = vec![0.0; real_batch * seq * self.hidden];
        let mut mask = vec![0u8; real_batch * seq];
        for (row, ids) in sequences.iter().enumerate() {
            for (position, &token) in ids.iter().enumerate() {
                let token = token as usize;
                ensure!(token < self.embeddings.shape[0]);
                let destination = (row * seq + position) * self.hidden;
                hidden[destination..destination + self.hidden].copy_from_slice(
                    &self.embeddings.data[token * self.hidden..(token + 1) * self.hidden],
                );
                mask[row * seq + position] = u8::from(token as u32 != self.pad_token_id);
            }
        }
        layer_norm(
            &mut hidden,
            real_batch * seq,
            self.hidden,
            &self.embedding_norm,
            &vec![0.0; self.hidden],
            self.epsilon,
        );
        context.forward(
            &mut hidden,
            &mask,
            real_batch,
            seq,
            self.hidden,
            self.heads,
            self.intermediate,
            self.epsilon,
            self.global_rope_theta,
            self.local_rope_theta,
            self.local_attention / 2,
            &self.layers,
            &self.final_norm,
        )?;
        let mut vectors = Vec::with_capacity(real_batch);
        for row in 0..real_batch {
            let offset = row * seq * self.hidden;
            let mut vector = hidden[offset..offset + self.hidden].to_vec();
            normalize_l2(&mut vector);
            vectors.push(vector);
        }
        let _ = precision;
        Ok(vectors)
    }
}

#[derive(Deserialize)]
struct QwenConfig {
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

impl Qwen3Model {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let root = resolve_model_root(path)?;
        let config: QwenConfig =
            serde_json::from_str(&fs::read_to_string(root.join("config.json"))?)?;
        ensure!(config.num_hidden_layers > 0);
        ensure!(config.num_attention_heads % config.num_key_value_heads == 0);
        let tensors = load_safetensor_map(&root, path)?;
        let embeddings = get_tensor(&tensors, QWEN3_EMBEDDINGS)?.clone();
        ensure!(embeddings.shape == vec![config.vocab_size, config.hidden_size]);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let norm = |name: &str| take_vector(&tensors, &family_layer_tensor_name(index, name));
            let weight = |name: &str| take_matrix(&tensors, &family_layer_tensor_name(index, name));
            layers.push(Qwen3Layer {
                input_norm: norm("input_layernorm")?,
                post_attention_norm: norm("post_attention_layernorm")?,
                q_weight: weight("self_attn.q_proj")?,
                q_norm: norm("self_attn.q_norm")?,
                k_weight: weight("self_attn.k_proj")?,
                k_norm: norm("self_attn.k_norm")?,
                v_weight: weight("self_attn.v_proj")?,
                o_weight: weight("self_attn.o_proj")?,
                gate_weight: weight("mlp.gate_proj")?,
                up_weight: weight("mlp.up_proj")?,
                down_weight: weight("mlp.down_proj")?,
            });
        }
        let final_norm = take_vector(&tensors, QWEN3_FINAL_NORM)?;
        Ok(Self {
            hidden: config.hidden_size,
            query_heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            intermediate: config.intermediate_size,
            epsilon: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            vocab_size: config.vocab_size,
            eos_token_id: config
                .eos_token_id
                .context("Qwen3 config is missing eos_token_id")?,
            embeddings,
            layers,
            final_norm,
        })
    }

    pub(crate) fn embed(
        &self,
        context: &mut Qwen3Context,
        sequences: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>> {
        let real_batch = sequences.len();
        ensure!(real_batch > 0 && sequences.iter().all(|ids| !ids.is_empty()));
        let seq = sequences.iter().map(Vec::len).max().unwrap_or(1);
        let mut hidden = vec![0.0; real_batch * seq * self.hidden];
        let mut mask = vec![0u8; real_batch * seq];
        for (row, ids) in sequences.iter().enumerate() {
            for (position, &token) in ids.iter().enumerate() {
                let token = token as usize;
                ensure!(token < self.vocab_size);
                let destination = (row * seq + position) * self.hidden;
                hidden[destination..destination + self.hidden].copy_from_slice(
                    &self.embeddings.data[token * self.hidden..(token + 1) * self.hidden],
                );
                mask[row * seq + position] = 1;
            }
        }
        context.forward(
            &mut hidden,
            &mask,
            real_batch,
            seq,
            self.hidden,
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.intermediate,
            self.epsilon,
            self.rope_theta,
            &self.layers,
            &self.final_norm,
        )?;
        let mut vectors = Vec::with_capacity(real_batch);
        for row in 0..real_batch {
            let last = (0..seq)
                .rev()
                .find(|&position| mask[row * seq + position] != 0)
                .unwrap_or(0);
            let offset = (row * seq + last) * self.hidden;
            let mut vector = hidden[offset..offset + self.hidden].to_vec();
            normalize_l2(&mut vector);
            vectors.push(vector);
        }
        Ok(vectors)
    }

    pub(crate) fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }
}

pub(crate) fn normalize_l2(vector: &mut [f32]) {
    let norm = vector
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt()
        .max(1e-12) as f32;
    for value in vector {
        *value /= norm;
    }
}

fn layer_norm(
    data: &mut [f32],
    rows: usize,
    hidden: usize,
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) {
    for row in 0..rows {
        let values = &mut data[row * hidden..(row + 1) * hidden];
        let mean = values.iter().sum::<f32>() / hidden as f32;
        let variance = values
            .iter()
            .map(|value| (*value - mean).powi(2))
            .sum::<f32>()
            / hidden as f32;
        let inverse = 1.0 / (variance + epsilon).sqrt();
        for index in 0..hidden {
            values[index] = (values[index] - mean) * inverse * weight[index] + bias[index];
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use serde_json::Value;
    use sha2::{Digest, Sha256};

    use super::{
        family_layer_tensor_name, tensor_candidates, MODERNBERT_EMBEDDINGS,
        MODERNBERT_EMBEDDING_NORM, MODERNBERT_FINAL_NORM, QWEN3_EMBEDDINGS, QWEN3_FINAL_NORM,
    };

    const GTE_HEADER: &[u8] =
        include_bytes!("../tests/fixtures/gte-modernbert-e7f32e3c.safetensors.header");
    const QWEN3_HEADER: &[u8] =
        include_bytes!("../tests/fixtures/qwen3-embedding-97b0c614.safetensors.header");

    fn header_names(bytes: &[u8]) -> HashSet<String> {
        let length = u64::from_le_bytes(bytes[..8].try_into().expect("safetensors length"));
        let end = 8 + usize::try_from(length).expect("header length fits usize");
        let header: Value =
            serde_json::from_slice(&bytes[8..end]).expect("safetensors header JSON");
        header
            .as_object()
            .expect("safetensors header object")
            .keys()
            .filter(|name| name.as_str() != "__metadata__")
            .cloned()
            .collect()
    }

    fn assert_resolves(names: &HashSet<String>, base: &str) {
        let candidates = tensor_candidates(base);
        assert!(
            candidates.iter().any(|name| names.contains(name)),
            "real checkpoint header has none of: {}",
            candidates.join(", ")
        );
    }

    #[test]
    fn loader_ignores_non_weight_integer_tensors() {
        let mut header = br#"{"float":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"position_ids":{"dtype":"I64","shape":[1],"data_offsets":[4,12]}}"#.to_vec();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(1.5_f32.to_le_bytes());
        bytes.extend(0_i64.to_le_bytes());
        let path = std::env::temp_dir().join(format!(
            "synapse-owned-cuda-mixed-dtype-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        let tensors = super::load_safetensors_file(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors["float"].data, vec![1.5]);
    }

    #[test]
    fn pinned_gte_header_resolves_every_production_tensor_name() {
        assert_eq!(
            format!("{:x}", Sha256::digest(GTE_HEADER)),
            "6cb5eb4efd5757d8f4ef3ef789e3d75168568388a5ce7c816e335c2342cbc8fc"
        );
        let names = header_names(GTE_HEADER);
        assert_resolves(&names, MODERNBERT_EMBEDDINGS);
        assert_resolves(&names, MODERNBERT_EMBEDDING_NORM);
        assert_resolves(&names, MODERNBERT_FINAL_NORM);
        for index in 0..22 {
            for name in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo", "mlp_norm"] {
                assert_resolves(&names, &family_layer_tensor_name(index, name));
            }
            if index > 0 {
                assert_resolves(&names, &family_layer_tensor_name(index, "attn_norm"));
            }
        }
    }

    #[test]
    fn pinned_qwen3_header_resolves_every_production_tensor_name() {
        assert_eq!(
            format!("{:x}", Sha256::digest(QWEN3_HEADER)),
            "1ff013283b0190994f5557f04301841482bcba6c006dd557fcb15563d4f1768b"
        );
        let names = header_names(QWEN3_HEADER);
        assert_resolves(&names, QWEN3_EMBEDDINGS);
        assert_resolves(&names, QWEN3_FINAL_NORM);
        for index in 0..28 {
            for name in [
                "input_layernorm",
                "post_attention_layernorm",
                "self_attn.q_proj",
                "self_attn.q_norm",
                "self_attn.k_proj",
                "self_attn.k_norm",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                assert_resolves(&names, &family_layer_tensor_name(index, name));
            }
        }
    }
}
