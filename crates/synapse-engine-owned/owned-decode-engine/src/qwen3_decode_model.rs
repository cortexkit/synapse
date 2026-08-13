//! Qwen3-0.6B causal-LM decode model for the owned Metal step engine.
//!
//! Ported from the proven `bench/spikes/unified-rt/src/qwen3.rs` spike model.
//! The embedding-only model in `crate::runtime::qwen3` does not carry an LM
//! head or Q8 quantization; this decode model adds both so the Metal step
//! engine can drive causal prefill and greedy token stepping in f16 and Q8_0.
//!
//! The weight layout, config parsing, and Q8 quantization are byte-identical
//! to the spike so a production-loaded model reproduces the spike engine's
//! pinned fixtures exactly. The spike tree is read-only reference material;
//! this is the production-owned copy.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use serde::Deserialize;

use crate::runtime::{
    encode_f16_bits, get_tensor, load_safetensor_map, resolve_model_root, Precision, Tensor,
};

use super::quant::{Q8_0Tensor, WeightQuantization};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OneOrManyTokenIds {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Debug, Deserialize, Default)]
struct GenerationConfig {
    eos_token_id: Option<OneOrManyTokenIds>,
}

#[derive(Debug, Deserialize)]
pub struct Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) vocab_size: usize,
    #[serde(default)]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) eos_token_id: Option<u32>,
}

pub struct Model {
    pub config: Config,
    pub(crate) eos_token_id: u32,
    pub(crate) generation_stop_ids: Vec<u32>,
    pub(crate) embeddings: Tensor,
    pub layers: Vec<Layer>,
    pub(crate) final_norm: RmsNorm,
    pub(crate) lm_head: Option<Weight>,
    pub(crate) tied_lm_head_q8_0: Option<Q8_0Tensor>,
    pub(crate) weight_quantization: WeightQuantization,
}

pub struct Layer {
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
    pub(crate) q8_0: Option<Q8_0Tensor>,
}

pub(crate) struct RmsNorm {
    pub(crate) weight: Tensor,
    pub(crate) eps: f32,
}

fn layer_weights(layer: &Layer) -> [&Weight; 7] {
    [
        &layer.q_proj,
        &layer.k_proj,
        &layer.v_proj,
        &layer.o_proj,
        &layer.gate_proj,
        &layer.up_proj,
        &layer.down_proj,
    ]
}

impl Model {
    pub fn load(path: &Path, precision: Precision) -> Result<Self> {
        Self::load_with_quant(path, precision, WeightQuantization::None)
    }

    pub fn load_with_quant(
        path: &Path,
        precision: Precision,
        weight_quantization: WeightQuantization,
    ) -> Result<Self> {
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
        let mut embeddings = get_qwen_tensor(&tensors, "embed_tokens.weight")?;
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
                q_proj: load_weight(
                    &tensors,
                    &format!("{prefix}.self_attn.q_proj"),
                    weight_quantization,
                )?,
                q_norm: load_norm(
                    &tensors,
                    &format!("{prefix}.self_attn.q_norm"),
                    config.rms_norm_eps,
                )?,
                k_proj: load_weight(
                    &tensors,
                    &format!("{prefix}.self_attn.k_proj"),
                    weight_quantization,
                )?,
                k_norm: load_norm(
                    &tensors,
                    &format!("{prefix}.self_attn.k_norm"),
                    config.rms_norm_eps,
                )?,
                v_proj: load_weight(
                    &tensors,
                    &format!("{prefix}.self_attn.v_proj"),
                    weight_quantization,
                )?,
                o_proj: load_weight(
                    &tensors,
                    &format!("{prefix}.self_attn.o_proj"),
                    weight_quantization,
                )?,
                gate_proj: load_weight(
                    &tensors,
                    &format!("{prefix}.mlp.gate_proj"),
                    weight_quantization,
                )?,
                up_proj: load_weight(
                    &tensors,
                    &format!("{prefix}.mlp.up_proj"),
                    weight_quantization,
                )?,
                down_proj: load_weight(
                    &tensors,
                    &format!("{prefix}.mlp.down_proj"),
                    weight_quantization,
                )?,
            });
        }
        validate_layers(&config, &layers)?;
        // Embedding-only Qwen3 snapshots legitimately omit lm_head. Causal-LM
        // decode validates an untied head when the decode context is created.
        let mut lm_head = if config.tie_word_embeddings {
            None
        } else {
            load_weight(&tensors, "lm_head", weight_quantization).ok()
        };
        if let Some(weight) = &lm_head {
            ensure!(
                weight.tensor.shape == vec![config.vocab_size, config.hidden_size],
                "Qwen3 LM head shape {:?} does not match config",
                weight.tensor.shape
            );
        }
        if matches!(precision, Precision::F16) {
            embeddings.prepare_metal_f16();
            if let Some(weight) = &mut lm_head {
                weight.tensor.prepare_metal_f16();
            }
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
        let generation_config_path = root.join("generation_config.json");
        let generation_config: GenerationConfig = if generation_config_path.exists() {
            serde_json::from_str(
                &std::fs::read_to_string(&generation_config_path).with_context(|| {
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
            })?
        } else {
            GenerationConfig::default()
        };
        let mut generation_stop_ids = match generation_config.eos_token_id {
            Some(OneOrManyTokenIds::One(token)) => vec![token],
            Some(OneOrManyTokenIds::Many(tokens)) => tokens,
            None => vec![eos_token_id],
        };
        if !generation_stop_ids.contains(&eos_token_id) {
            generation_stop_ids.push(eos_token_id);
        }
        generation_stop_ids.sort_unstable();
        generation_stop_ids.dedup();
        let mut final_norm = load_norm(&tensors, "norm", config.rms_norm_eps)?;
        if matches!(precision, Precision::F16) {
            final_norm.weight.prepare_metal_f16();
        }
        let tied_lm_head_q8_0 = if config.tie_word_embeddings
            && matches!(weight_quantization, WeightQuantization::Q8_0)
        {
            Some(Q8_0Tensor::quantize(&embeddings.data, config.hidden_size)?)
        } else {
            None
        };
        Ok(Self {
            config,
            eos_token_id,
            generation_stop_ids,
            embeddings,
            layers,
            final_norm,
            lm_head,
            tied_lm_head_q8_0,
            weight_quantization,
        })
    }

    pub fn generation_stop_ids(&self) -> &[u32] {
        &self.generation_stop_ids
    }

    pub fn vocabulary_size(&self) -> usize {
        self.config.vocab_size
    }

    pub(crate) fn lm_head(&self) -> Result<&Tensor> {
        if self.config.tie_word_embeddings {
            Ok(&self.embeddings)
        } else {
            self.lm_head
                .as_ref()
                .map(|weight| &weight.tensor)
                .context("untied Qwen3 causal LM is missing lm_head.weight")
        }
    }

    pub(crate) fn lm_head_q8_0(&self) -> Option<&Q8_0Tensor> {
        if self.config.tie_word_embeddings {
            self.tied_lm_head_q8_0.as_ref()
        } else {
            self.lm_head
                .as_ref()
                .and_then(|weight| weight.q8_0.as_ref())
        }
    }

    pub(crate) fn quantized_weight_sha256(&self) -> Option<String> {
        self.weight_quantization.is_quantized().then(|| {
            super::quant::quantized_sha256(
                self.layers
                    .iter()
                    .flat_map(layer_weights)
                    .filter_map(|weight| weight.q8_0.as_ref())
                    .chain(self.lm_head_q8_0()),
            )
        })
    }
}

fn get_qwen_tensor(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    get_tensor(tensors, name)
}

fn load_weight(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    weight_quantization: WeightQuantization,
) -> Result<Weight> {
    let tensor = get_qwen_tensor(tensors, &format!("{prefix}.weight"))?;
    let q8_0 = if matches!(weight_quantization, WeightQuantization::Q8_0) {
        let (_, row_width) = tensor.matrix_shape()?;
        Some(Q8_0Tensor::quantize(&tensor.data, row_width)?)
    } else {
        None
    };
    Ok(Weight { tensor, q8_0 })
}

fn load_norm(tensors: &HashMap<String, Tensor>, prefix: &str, eps: f32) -> Result<RmsNorm> {
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
        for (weight, want, name) in expected {
            ensure!(
                weight.tensor.shape == want,
                "Qwen3 layer {index} {name} shape {:?}, expected {want:?}",
                weight.tensor.shape
            );
        }
        for (norm, name) in [
            (&layer.input_norm, "input_layernorm"),
            (&layer.post_attention_norm, "post_attention_layernorm"),
            (&layer.q_norm, "q_norm"),
            (&layer.k_norm, "k_norm"),
        ] {
            ensure!(
                norm.weight.shape == vec![config.hidden_size]
                    || norm.weight.shape == vec![config.head_dim],
                "Qwen3 layer {index} {name} shape {:?} does not match hidden or head dim",
                norm.weight.shape
            );
        }
    }
    Ok(())
}

// Keep encode_f16_bits reachable so the import is not flagged as unused when
// only the non-macOS path is compiled.
const _: fn(&[f32]) -> Vec<u16> = encode_f16_bits;
