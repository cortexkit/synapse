//! LFM2-1.2B hybrid causal-LM decode model for the owned Metal step engine.
//!
//! Ported from the proven `bench/spikes/unified-rt/src/lfm2.rs` spike model.
//! The model carries the hybrid backbone (short-convolution layers +
//! group-query attention layers), Q8 quantization, and tied/united LM head
//! support that the Metal hybrid step engine drives. Weight layout, config
//! parsing, and Q8 quantization are byte-identical to the spike so the pinned
//! fixture batteries reproduce exactly.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;

use crate::runtime::{get_tensor, load_safetensor_map, resolve_model_root, Precision, Tensor};

use super::quant::{Q8_0Tensor, WeightQuantization};

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum OneOrManyTokenIds {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Debug, Default, Deserialize)]
struct GenerationConfig {
    eos_token_id: Option<OneOrManyTokenIds>,
}

/// Nested `rope_parameters` object carried by newer LFM2 checkpoints.
#[derive(Debug, Deserialize)]
struct RopeParameters {
    #[serde(default)]
    rope_theta: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    norm_eps: Option<f32>,
    #[serde(default)]
    block_norm_eps: Option<f32>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    block_ff_dim: Option<usize>,
    #[serde(default)]
    block_auto_adjust_ff_dim: bool,
    #[serde(default = "one_f32")]
    block_ffn_dim_multiplier: f32,
    #[serde(default = "default_multiple_of")]
    block_multiple_of: usize,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default)]
    full_attn_idxs: Vec<usize>,
    #[serde(rename = "conv_L_cache", default)]
    conv_l_cache: Option<usize>,
    #[serde(default)]
    conv_bias: bool,
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
    #[serde(default)]
    tie_embedding: Option<bool>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

fn one_f32() -> f32 {
    1.0
}

fn default_multiple_of() -> usize {
    256
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LayerType {
    Conv,
    FullAttention,
}

#[derive(Debug)]
pub(crate) struct Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) serialized_intermediate_size: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) vocab_size: usize,
    pub(crate) layer_types: Vec<LayerType>,
    pub(crate) conv_kernel_size: usize,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) bos_token_id: Option<u32>,
    pub(crate) eos_token_id: u32,
    pub(crate) pad_token_id: Option<u32>,
}

impl RawConfig {
    fn into_config(self) -> Result<Config> {
        let head_dim = self
            .head_dim
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads.max(1));
        ensure!(
            head_dim > 0,
            "LFM2 head_dim is zero (hidden {} / heads {})",
            self.hidden_size,
            self.num_attention_heads
        );
        let serialized_intermediate_size = self
            .intermediate_size
            .or(self.block_ff_dim)
            .context("LFM2 config is missing block_ff_dim/intermediate_size")?;
        let expected_intermediate_size = if self.block_auto_adjust_ff_dim {
            let adjusted = (2 * serialized_intermediate_size) / 3;
            let scaled = (adjusted as f32 * self.block_ffn_dim_multiplier) as usize;
            let multiple = self.block_multiple_of.max(1);
            scaled.div_ceil(multiple) * multiple
        } else {
            serialized_intermediate_size
        };
        ensure!(
            expected_intermediate_size > 0,
            "LFM2 adjusted feed-forward width is zero"
        );

        let layer_types = if let Some(layer_types) = self.layer_types {
            ensure!(
                layer_types.len() == self.num_hidden_layers,
                "LFM2 layer_types has {} entries for {} layers",
                layer_types.len(),
                self.num_hidden_layers
            );
            layer_types
                .iter()
                .map(|name| parse_layer_type(name))
                .collect::<Result<Vec<_>>>()?
        } else {
            let attention_indices = self.full_attn_idxs.into_iter().collect::<HashSet<_>>();
            ensure!(
                attention_indices
                    .iter()
                    .all(|&index| index < self.num_hidden_layers),
                "LFM2 full_attn_idxs contains an out-of-range layer"
            );
            (0..self.num_hidden_layers)
                .map(|index| {
                    if attention_indices.contains(&index) {
                        LayerType::FullAttention
                    } else {
                        LayerType::Conv
                    }
                })
                .collect()
        };
        ensure!(
            layer_types.contains(&LayerType::Conv)
                && layer_types.contains(&LayerType::FullAttention),
            "LFM2 hybrid config must contain convolution and attention layers"
        );
        ensure!(!self.conv_bias, "biasful LFM2 convolution is not supported");
        let conv_kernel_size = self.conv_l_cache.unwrap_or(3);
        ensure!(
            conv_kernel_size > 0,
            "LFM2 convolution cache length must be non-zero"
        );
        let tie_word_embeddings = self
            .tie_word_embeddings
            .or(self.tie_embedding)
            .unwrap_or(true);
        let rms_norm_eps = self
            .norm_eps
            .or(self.block_norm_eps)
            .context("LFM2 config is missing norm_eps")?;
        let eos_token_id = self
            .eos_token_id
            .context("LFM2 config is missing eos_token_id")?;

        Ok(Config {
            hidden_size: self.hidden_size,
            intermediate_size: expected_intermediate_size,
            serialized_intermediate_size,
            num_attention_heads: self.num_attention_heads,
            num_hidden_layers: self.num_hidden_layers,
            num_key_value_heads: self.num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta: self
                .rope_theta
                .or_else(|| {
                    self.rope_parameters
                        .as_ref()
                        .and_then(|params| params.rope_theta)
                })
                .unwrap_or(1_000_000.0),
            vocab_size: self.vocab_size,
            layer_types,
            conv_kernel_size,
            tie_word_embeddings,
            bos_token_id: self.bos_token_id,
            eos_token_id,
            pad_token_id: self.pad_token_id,
        })
    }
}

fn parse_layer_type(name: &str) -> Result<LayerType> {
    match name {
        "conv" | "short_conv" => Ok(LayerType::Conv),
        "full_attention" | "attention" => Ok(LayerType::FullAttention),
        other => bail!("unsupported LFM2 layer type {other:?}"),
    }
}

pub struct Model {
    pub(crate) config: Config,
    pub(crate) embeddings: Tensor,
    pub(crate) layers: Vec<Layer>,
    pub(crate) final_norm: RmsNorm,
    pub(crate) lm_head: Option<Weight>,
    pub(crate) tied_lm_head_q8_0: Option<Q8_0Tensor>,
    pub(crate) weight_quantization: WeightQuantization,
    pub(crate) generation_stop_ids: Vec<u32>,
}

pub(crate) struct Layer {
    pub(crate) operator_norm: RmsNorm,
    pub(crate) ffn_norm: RmsNorm,
    pub(crate) mixer: Mixer,
    pub(crate) w1: Weight,
    pub(crate) w2: Weight,
    pub(crate) w3: Weight,
}

pub(crate) enum Mixer {
    Conv(Box<ConvMixer>),
    Attention(Box<AttentionMixer>),
}

pub(crate) struct ConvMixer {
    pub(crate) in_proj: Weight,
    pub(crate) conv_weight: Tensor,
    pub(crate) out_proj: Weight,
    pub(crate) kernel_size: usize,
}

pub(crate) struct AttentionMixer {
    pub(crate) q_proj: Weight,
    pub(crate) q_norm: RmsNorm,
    pub(crate) k_proj: Weight,
    pub(crate) k_norm: RmsNorm,
    pub(crate) v_proj: Weight,
    pub(crate) out_proj: Weight,
}

pub(crate) struct Weight {
    pub(crate) tensor: Tensor,
    pub(crate) q8_0: Option<Q8_0Tensor>,
}

pub(crate) struct RmsNorm {
    pub(crate) weight: Tensor,
    pub(crate) eps: f32,
}

impl Model {
    pub(crate) fn load(path: &Path, precision: Precision) -> Result<Self> {
        Self::load_with_quant(path, precision, WeightQuantization::None)
    }

    pub(crate) fn load_with_quant(
        path: &Path,
        _precision: Precision,
        weight_quantization: WeightQuantization,
    ) -> Result<Self> {
        let root = resolve_model_root(path)?;
        let config_path = root.join("config.json");
        let config_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("read config {}", config_path.display()))?,
        )
        .with_context(|| format!("parse config {}", config_path.display()))?;
        let raw_config: RawConfig =
            serde_json::from_value(config_json.get("lfm").cloned().unwrap_or(config_json))
                .context("parse LFM2 backbone config")?;
        let mut config = raw_config.into_config()?;
        let tensors = load_safetensor_map(&root, path)?;
        let embeddings = get_lfm2_tensor(&tensors, "embed_tokens.weight")?;
        ensure!(
            embeddings.shape == vec![config.vocab_size, config.hidden_size],
            "LFM2 embedding shape {:?} does not match config",
            embeddings.shape
        );

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let mut actual_intermediate_size = None;
        for (index, layer_type) in config.layer_types.iter().copied().enumerate() {
            let prefix = format!("layers.{index}");
            let w1 = load_weight(
                &tensors,
                &format!("{prefix}.feed_forward.w1"),
                weight_quantization,
            )?;
            let w2 = load_weight(
                &tensors,
                &format!("{prefix}.feed_forward.w2"),
                weight_quantization,
            )?;
            let w3 = load_weight(
                &tensors,
                &format!("{prefix}.feed_forward.w3"),
                weight_quantization,
            )?;
            let layer_intermediate = w1
                .tensor
                .shape
                .first()
                .copied()
                .context("LFM2 w1 is missing its output dimension")?;
            match actual_intermediate_size {
                Some(width) => ensure!(
                    width == layer_intermediate,
                    "LFM2 layer {index} feed-forward width {layer_intermediate} differs from {width}"
                ),
                None => actual_intermediate_size = Some(layer_intermediate),
            }
            let mixer = match layer_type {
                LayerType::Conv => Mixer::Conv(Box::new(load_conv_mixer(
                    &tensors,
                    &prefix,
                    config.hidden_size,
                    config.conv_kernel_size,
                    weight_quantization,
                )?)),
                LayerType::FullAttention => Mixer::Attention(Box::new(load_attention_mixer(
                    &tensors,
                    &prefix,
                    &config,
                    weight_quantization,
                )?)),
            };
            layers.push(Layer {
                operator_norm: load_norm(
                    &tensors,
                    &format!("{prefix}.operator_norm"),
                    config.rms_norm_eps,
                )?,
                ffn_norm: load_norm(&tensors, &format!("{prefix}.ffn_norm"), config.rms_norm_eps)?,
                mixer,
                w1,
                w2,
                w3,
            });
        }
        config.intermediate_size = actual_intermediate_size.context("LFM2 has no MLP weights")?;
        validate_layers(&config, &layers)?;

        let final_norm = load_norm(&tensors, "embedding_norm", config.rms_norm_eps)?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(load_weight(&tensors, "lm_head", weight_quantization)?)
        };
        if let Some(head) = &lm_head {
            ensure!(
                head.tensor.shape == vec![config.vocab_size, config.hidden_size],
                "LFM2 LM head shape {:?} does not match config",
                head.tensor.shape
            );
        } else {
            ensure!(
                !tensors.contains_key("lm_head.weight")
                    && !tensors.contains_key("model.lm_head.weight"),
                "LFM2 config declares tied embeddings but checkpoint contains a separate LM head"
            );
        }

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
            None => vec![config.eos_token_id],
        };
        if !generation_stop_ids.contains(&config.eos_token_id) {
            generation_stop_ids.push(config.eos_token_id);
        }
        generation_stop_ids.sort_unstable();
        generation_stop_ids.dedup();

        let tied_lm_head_q8_0 = if config.tie_word_embeddings
            && matches!(weight_quantization, WeightQuantization::Q8_0)
        {
            Some(Q8_0Tensor::quantize(&embeddings.data, config.hidden_size)?)
        } else {
            None
        };
        Ok(Self {
            config,
            embeddings,
            layers,
            final_norm,
            lm_head,
            tied_lm_head_q8_0,
            weight_quantization,
            generation_stop_ids,
        })
    }

    pub(crate) fn generation_stop_ids(&self) -> &[u32] {
        &self.generation_stop_ids
    }

    pub(crate) fn lm_head(&self) -> Result<&Tensor> {
        if self.config.tie_word_embeddings {
            Ok(&self.embeddings)
        } else {
            self.lm_head
                .as_ref()
                .map(|head| &head.tensor)
                .context("untied LFM2 causal LM is missing lm_head.weight")
        }
    }

    pub(crate) fn lm_head_q8_0(&self) -> Option<&Q8_0Tensor> {
        if self.config.tie_word_embeddings {
            self.tied_lm_head_q8_0.as_ref()
        } else {
            self.lm_head.as_ref().and_then(|head| head.q8_0.as_ref())
        }
    }
}

fn get_lfm2_tensor(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    get_tensor(tensors, name).or_else(|_| get_tensor(tensors, &format!("lfm.{name}")))
}

fn load_weight(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    weight_quantization: WeightQuantization,
) -> Result<Weight> {
    let tensor = get_lfm2_tensor(tensors, &format!("{prefix}.weight"))?;
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
        weight: get_lfm2_tensor(tensors, &format!("{prefix}.weight"))?,
        eps,
    })
}

fn load_conv_mixer(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    hidden: usize,
    configured_kernel: usize,
    weight_quantization: WeightQuantization,
) -> Result<ConvMixer> {
    for name in ["in_proj.bias", "conv.bias", "out_proj.bias"] {
        let full_name = format!("model.{prefix}.conv.{name}");
        ensure!(
            !tensors.contains_key(&full_name),
            "biasful LFM2 convolution tensor {full_name} is not supported"
        );
    }
    let conv_weight = get_lfm2_tensor(tensors, &format!("{prefix}.conv.conv.weight"))?;
    ensure!(
        conv_weight.shape.len() == 3 && conv_weight.shape[0] == hidden && conv_weight.shape[1] == 1,
        "LFM2 {prefix} convolution shape {:?} is not depthwise",
        conv_weight.shape
    );
    let kernel_size = conv_weight.shape[2];
    ensure!(
        kernel_size == configured_kernel,
        "LFM2 {prefix} convolution kernel {kernel_size} differs from conv_L_cache {configured_kernel}"
    );
    Ok(ConvMixer {
        in_proj: load_weight(
            tensors,
            &format!("{prefix}.conv.in_proj"),
            weight_quantization,
        )?,
        conv_weight,
        out_proj: load_weight(
            tensors,
            &format!("{prefix}.conv.out_proj"),
            weight_quantization,
        )?,
        kernel_size,
    })
}

fn load_attention_mixer(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    config: &Config,
    weight_quantization: WeightQuantization,
) -> Result<AttentionMixer> {
    Ok(AttentionMixer {
        q_proj: load_weight(
            tensors,
            &format!("{prefix}.self_attn.q_proj"),
            weight_quantization,
        )?,
        q_norm: load_norm(
            tensors,
            &format!("{prefix}.self_attn.q_layernorm"),
            config.rms_norm_eps,
        )?,
        k_proj: load_weight(
            tensors,
            &format!("{prefix}.self_attn.k_proj"),
            weight_quantization,
        )?,
        k_norm: load_norm(
            tensors,
            &format!("{prefix}.self_attn.k_layernorm"),
            config.rms_norm_eps,
        )?,
        v_proj: load_weight(
            tensors,
            &format!("{prefix}.self_attn.v_proj"),
            weight_quantization,
        )?,
        out_proj: load_weight(
            tensors,
            &format!("{prefix}.self_attn.out_proj"),
            weight_quantization,
        )?,
    })
}

fn validate_layers(config: &Config, layers: &[Layer]) -> Result<()> {
    ensure!(
        config.intermediate_size > 0,
        "LFM2 checkpoint has zero feed-forward width"
    );
    let q_width = config.num_attention_heads * config.head_dim;
    let kv_width = config.num_key_value_heads * config.head_dim;
    for (index, layer) in layers.iter().enumerate() {
        for (norm, name) in [
            (&layer.operator_norm, "operator_norm"),
            (&layer.ffn_norm, "ffn_norm"),
        ] {
            ensure!(
                norm.weight.shape == vec![config.hidden_size],
                "LFM2 layer {index} {name} shape mismatch"
            );
        }
        for (weight, expected, name) in [
            (
                &layer.w1,
                vec![config.intermediate_size, config.hidden_size],
                "w1",
            ),
            (
                &layer.w3,
                vec![config.intermediate_size, config.hidden_size],
                "w3",
            ),
            (
                &layer.w2,
                vec![config.hidden_size, config.intermediate_size],
                "w2",
            ),
        ] {
            ensure!(
                weight.tensor.shape == expected,
                "LFM2 layer {index} {name} shape {:?}, expected {expected:?}",
                weight.tensor.shape
            );
        }
        match &layer.mixer {
            Mixer::Conv(mixer) => {
                for (weight, expected, name) in [
                    (
                        &mixer.in_proj,
                        vec![3 * config.hidden_size, config.hidden_size],
                        "conv.in_proj",
                    ),
                    (
                        &mixer.out_proj,
                        vec![config.hidden_size, config.hidden_size],
                        "conv.out_proj",
                    ),
                ] {
                    ensure!(
                        weight.tensor.shape == expected,
                        "LFM2 layer {index} {name} shape {:?}, expected {expected:?}",
                        weight.tensor.shape
                    );
                }
            }
            Mixer::Attention(mixer) => {
                for (weight, expected, name) in [
                    (&mixer.q_proj, vec![q_width, config.hidden_size], "q_proj"),
                    (&mixer.k_proj, vec![kv_width, config.hidden_size], "k_proj"),
                    (&mixer.v_proj, vec![kv_width, config.hidden_size], "v_proj"),
                    (
                        &mixer.out_proj,
                        vec![config.hidden_size, q_width],
                        "out_proj",
                    ),
                ] {
                    ensure!(
                        weight.tensor.shape == expected,
                        "LFM2 layer {index} {name} shape {:?}, expected {expected:?}",
                        weight.tensor.shape
                    );
                }
                ensure!(
                    mixer.q_norm.weight.shape == vec![config.head_dim]
                        && mixer.k_norm.weight.shape == vec![config.head_dim],
                    "LFM2 layer {index} per-head Q/K norm shape mismatch"
                );
            }
        }
    }
    Ok(())
}
