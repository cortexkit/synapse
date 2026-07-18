//! LFM2 hybrid causal-language-model graph for the owned runtime.
//!
//! Dense products use the selected `KernelProvider`. The short convolution uses
//! the provider's depthwise primitive, while norms, RoPE, residuals, and gating
//! remain explicit so the CPU and Metal paths share one architecture definition.

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::{
    get_tensor, load_safetensor_map, resolve_model_root, BLayout, BatchShape, BlockBackend,
    BlockForwardRequest, KernelProvider, MetalExecutionConfig, ModelFamily, Precision, Tensor,
};

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

#[derive(Debug, Deserialize)]
struct RawConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    vocab_size: usize,
    rope_theta: f32,
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

pub(crate) struct Model {
    pub(crate) config: Config,
    pub(crate) embeddings: Tensor,
    pub(crate) layers: Vec<Layer>,
    pub(crate) final_norm: RmsNorm,
    pub(crate) lm_head: Option<Weight>,
    generation_stop_ids: Vec<u32>,
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
}

pub(crate) struct RmsNorm {
    pub(crate) weight: Tensor,
    pub(crate) eps: f32,
}

pub(crate) struct DecodeCache {
    pub(crate) position: usize,
    pub(crate) capacity: usize,
    pub(crate) layers: Vec<LayerCache>,
}

pub(crate) enum LayerCache {
    Conv { state: Vec<f32> },
    Attention { keys: Vec<f32>, values: Vec<f32> },
}

fn new_cuda_block_context(
    precision: Precision,
    _execution: MetalExecutionConfig,
    backend: BlockBackend,
) -> Result<Box<dyn Any>> {
    let BlockBackend::Cuda { graphs } = backend else {
        bail!("LFM2 resident block context is only implemented for CUDA")
    };
    Ok(Box::new(CudaBlockContext {
        backend: super::cuda_backend::Lfm2Context::new(graphs, precision)?,
    }))
}

struct CudaBlockContext {
    backend: super::cuda_backend::Lfm2Context,
}

impl RawConfig {
    fn into_config(self) -> Result<Config> {
        ensure!(self.hidden_size > 0, "LFM2 config has zero hidden size");
        ensure!(
            self.num_hidden_layers > 0,
            "LFM2 config has no decoder layers"
        );
        ensure!(
            self.num_attention_heads > 0 && self.num_key_value_heads > 0,
            "LFM2 attention head counts must be non-zero"
        );
        ensure!(
            self.num_attention_heads % self.num_key_value_heads == 0,
            "LFM2 query heads must divide evenly across KV heads"
        );
        let head_dim = self
            .head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads);
        ensure!(
            head_dim * self.num_attention_heads == self.hidden_size,
            "LFM2 query heads do not cover hidden size"
        );
        ensure!(head_dim % 2 == 0, "LFM2 RoPE head dimension must be even");

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
            rope_theta: self.rope_theta,
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

impl Model {
    pub(crate) fn load(path: &Path, _precision: Precision) -> Result<Self> {
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
            let w1 = load_weight(&tensors, &format!("{prefix}.feed_forward.w1"))?;
            let w2 = load_weight(&tensors, &format!("{prefix}.feed_forward.w2"))?;
            let w3 = load_weight(&tensors, &format!("{prefix}.feed_forward.w3"))?;
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
                )?)),
                LayerType::FullAttention => {
                    Mixer::Attention(Box::new(load_attention_mixer(&tensors, &prefix, &config)?))
                }
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
            Some(load_weight(&tensors, "lm_head")?)
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

        Ok(Self {
            config,
            embeddings,
            layers,
            final_norm,
            lm_head,
            generation_stop_ids,
        })
    }

    pub(crate) fn encode_generation(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        capacity: usize,
    ) -> Result<Vec<u32>> {
        ensure!(capacity > 0, "decode cache capacity must be positive");
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|error| anyhow::anyhow!("encode LFM2 generation prompt: {error}"))?;
        let ids = encoding.get_ids();
        ensure!(!ids.is_empty(), "LFM2 generation prompt produced no tokens");
        if let Some(bos_token_id) = self.config.bos_token_id {
            ensure!(
                ids.first() == Some(&bos_token_id),
                "LFM2 tokenizer did not prepend configured BOS token {bos_token_id}"
            );
        }
        ensure!(
            ids.len() <= capacity,
            "LFM2 generation prompt has {} tokens, exceeding cache capacity {capacity}",
            ids.len()
        );
        Ok(ids.to_vec())
    }

    pub(crate) fn generation_stop_ids(&self) -> &[u32] {
        &self.generation_stop_ids
    }

    fn lm_head(&self) -> Result<&Tensor> {
        if self.config.tie_word_embeddings {
            Ok(&self.embeddings)
        } else {
            self.lm_head
                .as_ref()
                .map(|head| &head.tensor)
                .context("untied LFM2 causal LM is missing lm_head.weight")
        }
    }

    fn cuda_layer_params(&self) -> Vec<super::cuda_backend::Lfm2LayerParams> {
        self.layers
            .iter()
            .map(|layer| {
                let mut params = super::cuda_backend::Lfm2LayerParams {
                    mixer_type: 0,
                    operator_norm: layer.operator_norm.weight.data.as_ptr(),
                    ffn_norm: layer.ffn_norm.weight.data.as_ptr(),
                    conv_in_weight: std::ptr::null(),
                    conv_weight: std::ptr::null(),
                    conv_out_weight: std::ptr::null(),
                    q_weight: std::ptr::null(),
                    q_norm: std::ptr::null(),
                    k_weight: std::ptr::null(),
                    k_norm: std::ptr::null(),
                    v_weight: std::ptr::null(),
                    attention_out_weight: std::ptr::null(),
                    w1_weight: layer.w1.tensor.data.as_ptr(),
                    w2_weight: layer.w2.tensor.data.as_ptr(),
                    w3_weight: layer.w3.tensor.data.as_ptr(),
                };
                match &layer.mixer {
                    Mixer::Conv(mixer) => {
                        params.conv_in_weight = mixer.in_proj.tensor.data.as_ptr();
                        params.conv_weight = mixer.conv_weight.data.as_ptr();
                        params.conv_out_weight = mixer.out_proj.tensor.data.as_ptr();
                    }
                    Mixer::Attention(mixer) => {
                        params.mixer_type = 1;
                        params.q_weight = mixer.q_proj.tensor.data.as_ptr();
                        params.q_norm = mixer.q_norm.weight.data.as_ptr();
                        params.k_weight = mixer.k_proj.tensor.data.as_ptr();
                        params.k_norm = mixer.k_norm.weight.data.as_ptr();
                        params.v_weight = mixer.v_proj.tensor.data.as_ptr();
                        params.attention_out_weight = mixer.out_proj.tensor.data.as_ptr();
                    }
                }
                params
            })
            .collect()
    }

    fn cuda_full_forward(
        &self,
        provider: &mut dyn KernelProvider,
        hidden_states: &mut [f32],
        attention_mask: &[u8],
        seq: usize,
    ) -> Result<bool> {
        let params = self.cuda_layer_params();
        let lm_head = self.lm_head()?;
        let config = &self.config;
        let mut run = |context: &mut dyn Any| {
            let context = context
                .downcast_mut::<CudaBlockContext>()
                .context("LFM2 CUDA block context type mismatch")?;
            context.backend.full_forward(
                hidden_states,
                attention_mask,
                seq,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.intermediate_size,
                config.conv_kernel_size,
                config.vocab_size,
                config.rms_norm_eps,
                config.rope_theta,
                &params,
                &self.final_norm.weight.data,
                &lm_head.data,
            )
        };
        provider.lfm2_forward(BlockForwardRequest {
            family: "lfm2-causal",
            create_context: new_cuda_block_context,
            run: &mut run,
        })
    }

    pub(crate) fn prefill_embeddings(
        &self,
        provider: &mut dyn KernelProvider,
        embeddings: &[Vec<f32>],
        capacity: usize,
    ) -> Result<Option<(DecodeCache, Vec<f32>)>> {
        let hidden = self.config.hidden_size;
        ensure!(
            embeddings.iter().all(|embedding| embedding.len() == hidden),
            "LFM2 prefill embedding width mismatch"
        );
        let flattened = embeddings.concat();
        let params = self.cuda_layer_params();
        let lm_head = self.lm_head()?;
        let config = &self.config;
        let mut logits = None;
        let mut run = |context: &mut dyn Any| {
            let context = context
                .downcast_mut::<CudaBlockContext>()
                .context("LFM2 CUDA block context type mismatch")?;
            logits = Some(context.backend.prefill(
                &flattened,
                embeddings.len(),
                capacity,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.intermediate_size,
                config.conv_kernel_size,
                config.vocab_size,
                config.rms_norm_eps,
                config.rope_theta,
                &params,
                &self.final_norm.weight.data,
                &lm_head.data,
            )?);
            Ok(())
        };
        if !provider.lfm2_forward(BlockForwardRequest {
            family: "lfm2-causal",
            create_context: new_cuda_block_context,
            run: &mut run,
        })? {
            return Ok(None);
        }
        let mut cache = self.empty_decode_cache(capacity);
        cache.position = embeddings.len();
        Ok(Some((
            cache,
            logits.context("LFM2 CUDA prefill did not return logits")?,
        )))
    }

    fn cuda_decode_embedding(
        &self,
        provider: &mut dyn KernelProvider,
        cache: &DecodeCache,
        embedding: &[f32],
    ) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        let params = self.cuda_layer_params();
        let lm_head = self.lm_head()?;
        let config = &self.config;
        let mut output = None;
        let mut run = |context: &mut dyn Any| {
            let context = context
                .downcast_mut::<CudaBlockContext>()
                .context("LFM2 CUDA block context type mismatch")?;
            output = Some(context.backend.decode(
                embedding,
                cache.position,
                cache.capacity,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.intermediate_size,
                config.conv_kernel_size,
                config.vocab_size,
                config.rms_norm_eps,
                config.rope_theta,
                &params,
                &self.final_norm.weight.data,
                &lm_head.data,
            )?);
            Ok(())
        };
        if !provider.lfm2_forward(BlockForwardRequest {
            family: "lfm2-causal",
            create_context: new_cuda_block_context,
            run: &mut run,
        })? {
            return Ok(None);
        }
        output
            .map(Some)
            .context("LFM2 CUDA decode did not return an output")
    }

    pub(crate) fn empty_decode_cache(&self, capacity: usize) -> DecodeCache {
        let hidden = self.config.hidden_size;
        let kv_width = self.config.num_key_value_heads * self.config.head_dim;
        let layers = self
            .layers
            .iter()
            .map(|layer| match layer.mixer {
                Mixer::Conv(_) => LayerCache::Conv {
                    state: vec![0.0; self.config.conv_kernel_size * hidden],
                },
                Mixer::Attention(_) => LayerCache::Attention {
                    keys: Vec::with_capacity(capacity * kv_width),
                    values: Vec::with_capacity(capacity * kv_width),
                },
            })
            .collect();
        DecodeCache {
            position: 0,
            capacity,
            layers,
        }
    }

    pub(crate) fn token_embedding(&self, token: u32) -> Result<&[f32]> {
        let hidden = self.config.hidden_size;
        let token = token as usize;
        ensure!(
            token < self.config.vocab_size,
            "token id {token} outside LFM2 vocab"
        );
        Ok(&self.embeddings.data[token * hidden..(token + 1) * hidden])
    }

    pub(crate) fn decode_token(
        &self,
        provider: &mut dyn KernelProvider,
        cache: &mut DecodeCache,
        token: u32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.decode_embedding(provider, cache, self.token_embedding(token)?)
    }

    pub(crate) fn decode_embedding(
        &self,
        provider: &mut dyn KernelProvider,
        cache: &mut DecodeCache,
        embedding: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(cache.position < cache.capacity, "LFM2 decode cache is full");
        ensure!(
            cache.layers.len() == self.layers.len(),
            "LFM2 decode cache layer count mismatch"
        );
        let hidden = self.config.hidden_size;
        ensure!(
            embedding.len() == hidden,
            "LFM2 input embedding width {} does not match hidden size {hidden}",
            embedding.len()
        );
        if let Some(result) = self.cuda_decode_embedding(provider, cache, embedding)? {
            cache.position += 1;
            return Ok(result);
        }
        let mut current = embedding.to_vec();

        for (layer, layer_cache) in self.layers.iter().zip(&mut cache.layers) {
            let residual = current.clone();
            rms_norm_rows(&mut current, 1, hidden, &layer.operator_norm)?;
            current = match (&layer.mixer, layer_cache) {
                (Mixer::Conv(mixer), LayerCache::Conv { state }) => {
                    decode_conv(provider, &current, state, mixer, hidden)?
                }
                (Mixer::Attention(mixer), LayerCache::Attention { keys, values }) => {
                    decode_attention(
                        provider,
                        &current,
                        keys,
                        values,
                        mixer,
                        &self.config,
                        cache.position,
                    )?
                }
                _ => bail!("LFM2 decode cache type does not match layer layout"),
            };
            add_residual(&mut current, &residual);

            let residual = current.clone();
            rms_norm_rows(&mut current, 1, hidden, &layer.ffn_norm)?;
            current = feed_forward(provider, &current, 1, hidden, layer)?;
            add_residual(&mut current, &residual);
        }
        rms_norm_rows(&mut current, 1, hidden, &self.final_norm)?;
        let final_hidden = current.clone();
        let logits = linear_tensor(provider, &current, 1, hidden, self.lm_head()?, "LM head")?;
        cache.position += 1;
        Ok((final_hidden, logits))
    }

    pub(crate) fn forward_hidden(
        &self,
        provider: &mut dyn KernelProvider,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(!tokens.is_empty(), "LFM2 input must not be empty");
        let hidden = self.config.hidden_size;
        let mut hidden_states = vec![0.0; tokens.len() * hidden];
        for (position, &token) in tokens.iter().enumerate() {
            let token = token as usize;
            ensure!(
                token < self.config.vocab_size,
                "token id {token} outside LFM2 vocab"
            );
            hidden_states[position * hidden..(position + 1) * hidden]
                .copy_from_slice(&self.embeddings.data[token * hidden..(token + 1) * hidden]);
        }
        let attention_mask = vec![1u8; tokens.len()];
        if !self.cuda_full_forward(provider, &mut hidden_states, &attention_mask, tokens.len())? {
            scalar_forward(
                provider,
                &mut hidden_states,
                &attention_mask,
                1,
                tokens.len(),
                &self.config,
                &self.layers,
                &self.final_norm,
            )?;
        }
        Ok(hidden_states
            .chunks_exact(hidden)
            .map(<[f32]>::to_vec)
            .collect())
    }

    pub(crate) fn forward_logits(
        &self,
        provider: &mut dyn KernelProvider,
        tokens: &[u32],
    ) -> Result<Vec<f32>> {
        let hidden = self.forward_hidden(provider, tokens)?;
        let last = hidden.last().context("LFM2 input must not be empty")?;
        linear_tensor(
            provider,
            last,
            1,
            self.config.hidden_size,
            self.lm_head()?,
            "LM head",
        )
    }

    pub(crate) fn weight_count(&self) -> usize {
        let globals = 2 + usize::from(self.lm_head.is_some());
        globals
            + self
                .layers
                .iter()
                .map(|layer| {
                    5 + match layer.mixer {
                        Mixer::Conv(_) => 3,
                        Mixer::Attention(_) => 6,
                    }
                })
                .sum::<usize>()
    }

    fn embed_ids(
        &self,
        provider: &mut dyn KernelProvider,
        sequences: &[Vec<u32>],
        shape: Option<BatchShape>,
    ) -> Result<Vec<Vec<f32>>> {
        let real_batch = sequences.len();
        ensure!(real_batch > 0, "LFM2 batch must not be empty");
        let real_seq = sequences.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let target = shape.unwrap_or(BatchShape {
            batch: real_batch,
            seq: real_seq,
        });
        ensure!(
            target.batch >= real_batch && target.seq >= real_seq,
            "LFM2 target shape {}x{} does not cover input {}x{}",
            target.batch,
            target.seq,
            real_batch,
            real_seq
        );
        let hidden = self.config.hidden_size;
        let mut hidden_states = vec![0.0; target.batch * target.seq * hidden];
        let mut attention_mask = vec![0u8; target.batch * target.seq];
        for (batch, ids) in sequences.iter().enumerate() {
            for (position, &token) in ids.iter().enumerate() {
                let token = token as usize;
                ensure!(
                    token < self.config.vocab_size,
                    "token id {token} outside LFM2 vocab"
                );
                attention_mask[batch * target.seq + position] = 1;
                hidden_states[(batch * target.seq + position) * hidden
                    ..(batch * target.seq + position + 1) * hidden]
                    .copy_from_slice(&self.embeddings.data[token * hidden..(token + 1) * hidden]);
            }
        }
        scalar_forward(
            provider,
            &mut hidden_states,
            &attention_mask,
            target.batch,
            target.seq,
            &self.config,
            &self.layers,
            &self.final_norm,
        )?;
        let mut result = Vec::with_capacity(real_batch);
        for (batch, ids) in sequences.iter().enumerate() {
            let last = ids.len().saturating_sub(1);
            let start = (batch * target.seq + last) * hidden;
            result.push(hidden_states[start..start + hidden].to_vec());
        }
        Ok(result)
    }

    pub(crate) fn notes(&self) -> String {
        let conv_layers = self
            .config
            .layer_types
            .iter()
            .filter(|kind| matches!(kind, LayerType::Conv))
            .count();
        format!(
            "direct LFM2 hybrid decoder, {} conv + {} causal GQA layers, short-conv cache {}, GQA {}/{}, RoPE theta={}, SwiGLU actual width {} (serialized {}), BOS {:?}, EOS {}, PAD {:?}",
            conv_layers,
            self.layers.len() - conv_layers,
            self.config.conv_kernel_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.rope_theta,
            self.config.intermediate_size,
            self.config.serialized_intermediate_size,
            self.config.bos_token_id,
            self.config.eos_token_id,
            self.config.pad_token_id,
        )
    }
}

impl ModelFamily for Model {
    fn family_name(&self) -> &'static str {
        "lfm2"
    }

    fn token_length(&self, tokenizer: &Tokenizer, text: &str, max_length: usize) -> Result<usize> {
        Ok(self.encode_generation(tokenizer, text, max_length)?.len())
    }

    fn embed_batch(
        &self,
        provider: &mut dyn KernelProvider,
        tokenizer: &Tokenizer,
        texts: &[&str],
        max_length: usize,
        shape: Option<BatchShape>,
    ) -> Result<Vec<Vec<f32>>> {
        let ids = texts
            .iter()
            .map(|text| self.encode_generation(tokenizer, text, max_length))
            .collect::<Result<Vec<_>>>()?;
        self.embed_ids(provider, &ids, shape)
    }

    fn default_label(&self, precision: Precision) -> String {
        format!("LFM2@owned-rt-{}", precision.as_str())
    }

    fn notes(&self) -> String {
        Model::notes(self)
    }
}

pub(super) fn detect_config(config: &serde_json::Value) -> bool {
    if config.get("lfm").is_some() {
        return false;
    }
    config.get("model_type").and_then(serde_json::Value::as_str) == Some("lfm2")
        || config
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|architectures| {
                architectures.iter().any(|name| {
                    name.as_str()
                        .is_some_and(|name| name.to_ascii_lowercase().contains("lfm2"))
                })
            })
}

pub(super) fn load_family(path: &Path, precision: Precision) -> Result<Box<dyn ModelFamily>> {
    Ok(Box::new(Model::load(path, precision)?))
}

fn get_lfm2_tensor(tensors: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    get_tensor(tensors, name).or_else(|_| get_tensor(tensors, &format!("lfm.{name}")))
}

fn load_weight(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Weight> {
    Ok(Weight {
        tensor: get_lfm2_tensor(tensors, &format!("{prefix}.weight"))?,
    })
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
        in_proj: load_weight(tensors, &format!("{prefix}.conv.in_proj"))?,
        conv_weight,
        out_proj: load_weight(tensors, &format!("{prefix}.conv.out_proj"))?,
        kernel_size,
    })
}

fn load_attention_mixer(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    config: &Config,
) -> Result<AttentionMixer> {
    Ok(AttentionMixer {
        q_proj: load_weight(tensors, &format!("{prefix}.self_attn.q_proj"))?,
        q_norm: load_norm(
            tensors,
            &format!("{prefix}.self_attn.q_layernorm"),
            config.rms_norm_eps,
        )?,
        k_proj: load_weight(tensors, &format!("{prefix}.self_attn.k_proj"))?,
        k_norm: load_norm(
            tensors,
            &format!("{prefix}.self_attn.k_layernorm"),
            config.rms_norm_eps,
        )?,
        v_proj: load_weight(tensors, &format!("{prefix}.self_attn.v_proj"))?,
        out_proj: load_weight(tensors, &format!("{prefix}.self_attn.out_proj"))?,
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
        rms_norm_rows(&mut current, rows, hidden, &layer.operator_norm)?;
        current = match &layer.mixer {
            Mixer::Conv(mixer) => full_conv(provider, &current, batch, seq, hidden, mixer)?,
            Mixer::Attention(mixer) => full_attention(
                provider,
                &current,
                attention_mask,
                batch,
                seq,
                config,
                mixer,
            )?,
        };
        add_residual(&mut current, &residual);

        let residual = current.clone();
        rms_norm_rows(&mut current, rows, hidden, &layer.ffn_norm)?;
        current = feed_forward(provider, &current, rows, hidden, layer)?;
        add_residual(&mut current, &residual);
    }
    rms_norm_rows(&mut current, rows, hidden, final_norm)?;
    hidden_states.copy_from_slice(&current);
    Ok(())
}

fn full_conv(
    provider: &mut dyn KernelProvider,
    current: &[f32],
    batch: usize,
    seq: usize,
    hidden: usize,
    mixer: &ConvMixer,
) -> Result<Vec<f32>> {
    let rows = batch * seq;
    let projected = linear(provider, current, rows, hidden, &mixer.in_proj)?;
    let mut c_gate = vec![0.0; rows * hidden];
    let mut product = vec![0.0; rows * hidden];
    for row in 0..rows {
        for channel in 0..hidden {
            let base = row * 3 * hidden;
            let target = row * hidden + channel;
            c_gate[target] = projected[base + hidden + channel];
            product[target] = projected[base + channel] * projected[base + 2 * hidden + channel];
        }
    }
    let mut convolved = vec![0.0; product.len()];
    provider.depthwise_causal_conv1d(
        &product,
        batch,
        seq,
        hidden,
        &mixer.conv_weight.data,
        mixer.kernel_size,
        &mut convolved,
    )?;
    for (value, gate) in convolved.iter_mut().zip(c_gate) {
        *value *= gate;
    }
    linear(provider, &convolved, rows, hidden, &mixer.out_proj)
}

fn decode_conv(
    provider: &mut dyn KernelProvider,
    current: &[f32],
    state: &mut [f32],
    mixer: &ConvMixer,
    hidden: usize,
) -> Result<Vec<f32>> {
    ensure!(
        state.len() == mixer.kernel_size * hidden,
        "LFM2 convolution cache shape mismatch"
    );
    let projected = linear(provider, current, 1, hidden, &mixer.in_proj)?;
    let mut product = vec![0.0; hidden];
    let mut gate = vec![0.0; hidden];
    for channel in 0..hidden {
        product[channel] = projected[channel] * projected[2 * hidden + channel];
        gate[channel] = projected[hidden + channel];
    }
    state.copy_within(hidden.., 0);
    state[(mixer.kernel_size - 1) * hidden..].copy_from_slice(&product);
    let mut convolved_state = vec![0.0; state.len()];
    provider.depthwise_causal_conv1d(
        state,
        1,
        mixer.kernel_size,
        hidden,
        &mixer.conv_weight.data,
        mixer.kernel_size,
        &mut convolved_state,
    )?;
    let last = (mixer.kernel_size - 1) * hidden;
    for channel in 0..hidden {
        gate[channel] *= convolved_state[last + channel];
    }
    linear(provider, &gate, 1, hidden, &mixer.out_proj)
}

fn full_attention(
    provider: &mut dyn KernelProvider,
    current: &[f32],
    attention_mask: &[u8],
    batch: usize,
    seq: usize,
    config: &Config,
    mixer: &AttentionMixer,
) -> Result<Vec<f32>> {
    let rows = batch * seq;
    let hidden = config.hidden_size;
    let mut q = linear(provider, current, rows, hidden, &mixer.q_proj)?;
    let mut k = linear(provider, current, rows, hidden, &mixer.k_proj)?;
    let v = linear(provider, current, rows, hidden, &mixer.v_proj)?;
    rms_norm_heads(
        &mut q,
        rows,
        config.num_attention_heads,
        config.head_dim,
        &mixer.q_norm,
    )?;
    rms_norm_heads(
        &mut k,
        rows,
        config.num_key_value_heads,
        config.head_dim,
        &mixer.k_norm,
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
    linear(
        provider,
        &context,
        rows,
        config.num_attention_heads * config.head_dim,
        &mixer.out_proj,
    )
}

fn decode_attention(
    provider: &mut dyn KernelProvider,
    current: &[f32],
    keys: &mut Vec<f32>,
    values: &mut Vec<f32>,
    mixer: &AttentionMixer,
    config: &Config,
    position: usize,
) -> Result<Vec<f32>> {
    let hidden = config.hidden_size;
    let mut q = linear(provider, current, 1, hidden, &mixer.q_proj)?;
    let mut k = linear(provider, current, 1, hidden, &mixer.k_proj)?;
    let v = linear(provider, current, 1, hidden, &mixer.v_proj)?;
    rms_norm_heads(
        &mut q,
        1,
        config.num_attention_heads,
        config.head_dim,
        &mixer.q_norm,
    )?;
    rms_norm_heads(
        &mut k,
        1,
        config.num_key_value_heads,
        config.head_dim,
        &mixer.k_norm,
    )?;
    apply_rope_position(
        &mut q,
        config.num_attention_heads,
        config.head_dim,
        config.rope_theta,
        position,
    );
    apply_rope_position(
        &mut k,
        config.num_key_value_heads,
        config.head_dim,
        config.rope_theta,
        position,
    );
    keys.extend_from_slice(&k);
    values.extend_from_slice(&v);
    let context = causal_gqa_decode(
        provider,
        &q,
        keys,
        values,
        position + 1,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )?;
    linear(
        provider,
        &context,
        1,
        config.num_attention_heads * config.head_dim,
        &mixer.out_proj,
    )
}

fn feed_forward(
    provider: &mut dyn KernelProvider,
    current: &[f32],
    rows: usize,
    hidden: usize,
    layer: &Layer,
) -> Result<Vec<f32>> {
    let mut gate = linear(provider, current, rows, hidden, &layer.w1)?;
    let up = linear(provider, current, rows, hidden, &layer.w3)?;
    for (gate, up) in gate.iter_mut().zip(up) {
        *gate = silu(*gate) * up;
    }
    linear(provider, &gate, rows, layer.w1.tensor.shape[0], &layer.w2)
}

fn linear(
    provider: &mut dyn KernelProvider,
    values: &[f32],
    rows: usize,
    input: usize,
    weight: &Weight,
) -> Result<Vec<f32>> {
    linear_tensor(provider, values, rows, input, &weight.tensor, "linear")
}

fn linear_tensor(
    provider: &mut dyn KernelProvider,
    values: &[f32],
    rows: usize,
    input: usize,
    weight: &Tensor,
    label: &str,
) -> Result<Vec<f32>> {
    let (output, weight_input) = weight.matrix_shape()?;
    ensure!(weight_input == input, "LFM2 {label} input shape mismatch");
    ensure!(
        values.len() == rows * input,
        "LFM2 {label} value shape mismatch"
    );
    let mut output_values = vec![0.0; rows * output];
    provider.matmul_static_rhs(
        rows,
        output,
        input,
        values,
        &weight.data,
        BLayout::RowMajorNkTransposed,
        &mut output_values,
    )?;
    Ok(output_values)
}

fn rms_norm_rows(data: &mut [f32], rows: usize, width: usize, norm: &RmsNorm) -> Result<()> {
    ensure!(
        norm.weight.shape == vec![width],
        "LFM2 RMSNorm weight shape mismatch"
    );
    ensure!(
        data.len() == rows * width,
        "LFM2 RMSNorm data shape mismatch"
    );
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
    for batch_index in 0..batch {
        for position in 0..seq {
            let start = (batch_index * seq + position) * heads * head_dim;
            apply_rope_position(
                &mut data[start..start + heads * head_dim],
                heads,
                head_dim,
                theta,
                position,
            );
        }
    }
}

fn apply_rope_position(
    data: &mut [f32],
    heads: usize,
    head_dim: usize,
    theta: f32,
    position: usize,
) {
    let half = head_dim / 2;
    for head in 0..heads {
        let start = head * head_dim;
        for index in 0..half {
            let frequency = 1.0 / theta.powf((2 * index) as f32 / head_dim as f32);
            let (sin, cos) = (position as f32 * frequency).sin_cos();
            let first = data[start + index];
            let second = data[start + half + index];
            data[start + index] = first * cos - second * sin;
            data[start + half + index] = second * cos + first * sin;
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

    for batch_index in 0..batch {
        for query_head in 0..query_heads {
            let kv_head = query_head / groups;
            for position in 0..seq {
                let q_source = (batch_index * seq + position) * query_width + query_head * head_dim;
                let kv_source = (batch_index * seq + position) * kv_width + kv_head * head_dim;
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
                        || attention_mask[batch_index * seq + key_position] == 0
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
                let target = (batch_index * seq + position) * query_width + query_head * head_dim;
                output[target..target + head_dim]
                    .copy_from_slice(&context[source..source + head_dim]);
            }
        }
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn causal_gqa_decode(
    provider: &mut dyn KernelProvider,
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    seq: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let query_width = query_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    ensure!(q.len() == query_width, "LFM2 decode query shape mismatch");
    ensure!(
        keys.len() == seq * kv_width && values.len() == seq * kv_width,
        "LFM2 decode KV cache shape mismatch"
    );
    let groups = query_heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0; query_width];
    let mut k_head = vec![0.0; seq * head_dim];
    let mut v_head = vec![0.0; seq * head_dim];
    let mut scores = vec![0.0; seq];
    let mut context = vec![0.0; head_dim];
    for query_head in 0..query_heads {
        let kv_head = query_head / groups;
        for position in 0..seq {
            let source = position * kv_width + kv_head * head_dim;
            let target = position * head_dim;
            k_head[target..target + head_dim].copy_from_slice(&keys[source..source + head_dim]);
            v_head[target..target + head_dim].copy_from_slice(&values[source..source + head_dim]);
        }
        provider.matmul(
            1,
            seq,
            head_dim,
            &q[query_head * head_dim..(query_head + 1) * head_dim],
            &k_head,
            BLayout::RowMajorNkTransposed,
            &mut scores,
        )?;
        for score in &mut scores {
            *score *= scale;
        }
        super::softmax(&mut scores);
        provider.matmul(
            1,
            head_dim,
            seq,
            &scores,
            &v_head,
            BLayout::RowMajorKn,
            &mut context,
        )?;
        output[query_head * head_dim..(query_head + 1) * head_dim].copy_from_slice(&context);
    }
    Ok(output)
}

fn add_residual(values: &mut [f32], residual: &[f32]) {
    for (value, residual) in values.iter_mut().zip(residual) {
        *value += *residual;
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[cfg(test)]
pub(crate) fn tiny_test_model() -> Model {
    fn tensor(shape: Vec<usize>, seed: usize) -> Tensor {
        let elements = shape.iter().product();
        let values = (0..elements)
            .map(|index| (((index + seed) % 17) as f32 - 8.0) / 40.0)
            .collect();
        Tensor::new(shape, values).unwrap()
    }

    fn weight(output: usize, input: usize, seed: usize) -> Weight {
        Weight {
            tensor: tensor(vec![output, input], seed),
        }
    }

    fn norm(width: usize) -> RmsNorm {
        RmsNorm {
            weight: Tensor::new(vec![width], vec![1.0; width]).unwrap(),
            eps: 1e-5,
        }
    }

    let config = Config {
        hidden_size: 4,
        intermediate_size: 6,
        serialized_intermediate_size: 6,
        num_attention_heads: 2,
        num_hidden_layers: 2,
        num_key_value_heads: 1,
        head_dim: 2,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        vocab_size: 8,
        layer_types: vec![LayerType::Conv, LayerType::FullAttention],
        conv_kernel_size: 3,
        tie_word_embeddings: true,
        bos_token_id: Some(1),
        eos_token_id: 7,
        pad_token_id: Some(0),
    };
    let common = |seed| Layer {
        operator_norm: norm(4),
        ffn_norm: norm(4),
        mixer: Mixer::Conv(Box::new(ConvMixer {
            in_proj: weight(12, 4, seed),
            conv_weight: Tensor::new(
                vec![4, 1, 3],
                vec![
                    0.2, -0.1, 0.4, -0.3, 0.2, 0.1, 0.1, 0.3, -0.2, 0.4, 0.1, 0.2,
                ],
            )
            .unwrap(),
            out_proj: weight(4, 4, seed + 1),
            kernel_size: 3,
        })),
        w1: weight(6, 4, seed + 2),
        w2: weight(4, 6, seed + 3),
        w3: weight(6, 4, seed + 4),
    };
    let conv = common(1);
    let attention = Layer {
        operator_norm: norm(4),
        ffn_norm: norm(4),
        mixer: Mixer::Attention(Box::new(AttentionMixer {
            q_proj: weight(4, 4, 9),
            q_norm: norm(2),
            k_proj: weight(2, 4, 10),
            k_norm: norm(2),
            v_proj: weight(2, 4, 11),
            out_proj: weight(4, 4, 12),
        })),
        w1: weight(6, 4, 13),
        w2: weight(4, 6, 14),
        w3: weight(6, 4, 15),
    };
    Model {
        config,
        embeddings: tensor(vec![8, 4], 21),
        layers: vec![conv, attention],
        final_norm: norm(4),
        lm_head: None,
        generation_stop_ids: vec![7],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_config_synthesizes_the_checkpoint_layer_layout_and_ff_width() {
        let raw: RawConfig = serde_json::from_value(serde_json::json!({
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_hidden_layers": 16,
            "num_key_value_heads": 8,
            "vocab_size": 65536,
            "rope_theta": 1000000.0,
            "norm_eps": 1e-5,
            "block_ff_dim": 12288,
            "block_auto_adjust_ff_dim": true,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 256,
            "full_attn_idxs": [2, 5, 8, 10, 12, 14],
            "conv_L_cache": 3,
            "conv_bias": false,
            "bos_token_id": 1,
            "eos_token_id": 7,
            "pad_token_id": 0
        }))
        .unwrap();
        let config = raw.into_config().unwrap();
        assert_eq!(config.intermediate_size, 8192);
        assert_eq!(
            config.layer_types,
            [
                LayerType::Conv,
                LayerType::Conv,
                LayerType::FullAttention,
                LayerType::Conv,
                LayerType::Conv,
                LayerType::FullAttention,
                LayerType::Conv,
                LayerType::Conv,
                LayerType::FullAttention,
                LayerType::Conv,
                LayerType::FullAttention,
                LayerType::Conv,
                LayerType::FullAttention,
                LayerType::Conv,
                LayerType::FullAttention,
                LayerType::Conv,
            ]
        );
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn causal_conv_uses_unflipped_cross_correlation_taps() {
        struct Provider;
        impl KernelProvider for Provider {
            fn name(&self) -> &'static str {
                "test"
            }

            fn matmul(
                &mut self,
                _m: usize,
                _n: usize,
                _k: usize,
                _a: &[f32],
                _b: &[f32],
                _b_layout: BLayout,
                _c: &mut [f32],
            ) -> Result<()> {
                unreachable!()
            }
        }
        let mut provider = Provider;
        let mut output = vec![0.0; 4];
        provider
            .depthwise_causal_conv1d(
                &[1.0, 2.0, 3.0, 4.0],
                1,
                4,
                1,
                &[10.0, 20.0, 30.0],
                3,
                &mut output,
            )
            .unwrap();
        assert_eq!(output, [30.0, 80.0, 140.0, 200.0]);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_f16_static_rhs_cache_distinguishes_weights() {
        let mut provider = super::super::MetalProvider::new(Precision::F16).unwrap();
        let input = [2.0, 3.0];
        let first_weight = [4.0, 5.0];
        let second_weight = [6.0, 7.0];
        let mut first = [0.0];
        let mut second = [0.0];
        provider
            .matmul_static_rhs(
                1,
                1,
                2,
                &input,
                &first_weight,
                BLayout::RowMajorNkTransposed,
                &mut first,
            )
            .unwrap();
        provider
            .matmul_static_rhs(
                1,
                1,
                2,
                &input,
                &second_weight,
                BLayout::RowMajorNkTransposed,
                &mut second,
            )
            .unwrap();
        assert_eq!(first[0], 23.0);
        assert_eq!(second[0], 33.0);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn metal_depthwise_primitive_matches_cpu() {
        let values = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let weights = vec![0.5, -0.25, 0.75, -0.5, 0.125, 0.25];
        let mut expected = vec![0.0; values.len()];
        let mut actual = vec![0.0; values.len()];
        let mut cpu = super::super::CpuProvider::platform_for_test();
        cpu.depthwise_causal_conv1d(&values, 1, 6, 2, &weights, 3, &mut expected)
            .unwrap();
        let mut metal = super::super::MetalProvider::new(Precision::F32).unwrap();
        metal
            .depthwise_causal_conv1d(&values, 1, 6, 2, &weights, 3, &mut actual)
            .unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_uses_half_split_pairs() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope_position(&mut values, 1, 4, 10_000.0, 1);
        let (sin, cos) = 1.0f32.sin_cos();
        assert!((values[0] - (1.0 * cos - 3.0 * sin)).abs() < 1e-6);
        assert!((values[2] - (3.0 * cos + 1.0 * sin)).abs() < 1e-6);
    }
}
