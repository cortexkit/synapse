//! LFM2-Audio speech frontend and noncausal FastConformer encoder.
//!
//! The learned tower is kept separate from the text tokenizer path. Audio is
//! converted to projected 2048-wide vectors and then spliced into the LFM2
//! backbone at the placeholder selected by the modality flag.

use std::collections::HashMap;
use std::f32::consts::PI;
use std::path::Path;
use std::sync::Arc;

use anyhow::{ensure, Context, Result};
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use serde::Deserialize;
use tokenizers::Tokenizer;

use super::{
    get_tensor, load_safetensor_map, resolve_model_root, BLayout, BatchShape, KernelProvider,
    ModelFamily, Precision, Tensor,
};
use crate::lfm2;

pub(crate) const SAMPLE_RATE: u32 = 16_000;
pub(crate) const TEXT_END_TOKEN: u32 = 130;

#[derive(Debug, Deserialize)]
struct RawAudioConfig {
    preprocessor: PreprocessorConfig,
    encoder: EncoderConfig,
}

#[derive(Clone, Debug, Deserialize)]
struct PreprocessorConfig {
    sample_rate: usize,
    normalize: String,
    window_size: f32,
    window_stride: f32,
    window: String,
    features: usize,
    n_fft: usize,
    log: bool,
    frame_splicing: usize,
    dither: f32,
    pad_to: usize,
    pad_value: f32,
}

#[derive(Clone, Debug, Deserialize)]
struct EncoderConfig {
    feat_in: usize,
    n_layers: usize,
    d_model: usize,
    subsampling: String,
    subsampling_factor: usize,
    subsampling_conv_channels: usize,
    causal_downsampling: bool,
    ff_expansion_factor: usize,
    self_attention_model: String,
    n_heads: usize,
    att_context_size: Vec<isize>,
    xscaling: bool,
    untie_biases: bool,
    pos_emb_max_len: usize,
    conv_kernel_size: usize,
    conv_norm_type: String,
}

#[derive(Clone)]
struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
    label: String,
}

#[derive(Clone)]
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

struct Conv2dPreEncoder {
    conv0_weight: Tensor,
    conv0_bias: Tensor,
    conv2_weight: Tensor,
    conv2_bias: Tensor,
    conv3: Linear,
    conv5_weight: Tensor,
    conv5_bias: Tensor,
    conv6: Linear,
    out: Linear,
    channels: usize,
}

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

struct RelativeAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
}

struct ConvModule {
    pointwise1: Linear,
    depthwise_weight: Tensor,
    depthwise_bias: Tensor,
    batch_norm_weight: Tensor,
    batch_norm_bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    pointwise2: Linear,
}

struct ConformerLayer {
    norm_feed_forward1: LayerNorm,
    feed_forward1: FeedForward,
    norm_self_att: LayerNorm,
    self_attn: RelativeAttention,
    norm_conv: LayerNorm,
    conv: ConvModule,
    norm_feed_forward2: LayerNorm,
    feed_forward2: FeedForward,
    norm_out: LayerNorm,
}

struct FastConformer {
    config: EncoderConfig,
    pre_encode: Conv2dPreEncoder,
    layers: Vec<ConformerLayer>,
}

struct AudioAdapter {
    norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

pub(crate) struct MelOutput {
    /// Frame-major normalized log-mel values, `[frames, 128]`.
    pub(crate) values: Vec<f32>,
    pub(crate) frames: usize,
    pub(crate) features: usize,
}

pub(crate) struct AudioProjection {
    pub(crate) samples: usize,
    pub(crate) mel: MelOutput,
    /// Frame-major projected encoder values, `[encoded_frames, 2048]`.
    pub(crate) embeddings: Vec<Vec<f32>>,
}

pub(crate) struct AudioModel {
    pub(crate) backbone: lfm2::Model,
    frontend: MelFrontend,
    conformer: FastConformer,
    adapter: AudioAdapter,
}

struct AudioFamily {
    backbone: lfm2::Model,
}

struct MelFrontend {
    config: PreprocessorConfig,
    hop_length: usize,
    fft: Arc<dyn Fft<f32>>,
    fft_window: Vec<f32>,
    mel_filter: Vec<f32>,
}

impl AudioModel {
    pub(crate) fn load(path: &Path, precision: Precision) -> Result<Self> {
        let root = resolve_model_root(path)?;
        let config_path = root.join("config.json");
        let raw: RawAudioConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("read audio config {}", config_path.display()))?,
        )
        .with_context(|| format!("parse audio config {}", config_path.display()))?;
        validate_audio_config(&raw)?;

        let backbone = lfm2::Model::load(path, precision)?;
        ensure!(
            backbone.config.hidden_size == 2048,
            "LFM2-Audio adapter requires a 2048-wide backbone"
        );
        let tensors = load_safetensor_map(&root, path)?;
        let frontend = MelFrontend::new(raw.preprocessor)?;
        let conformer = FastConformer::load(raw.encoder, &tensors)?;
        let adapter = AudioAdapter::load(&tensors)?;
        Ok(Self {
            backbone,
            frontend,
            conformer,
            adapter,
        })
    }

    pub(crate) fn project_wav(
        &self,
        provider: &mut dyn KernelProvider,
        path: &Path,
    ) -> Result<AudioProjection> {
        let samples = read_mono_wav(path)?;
        self.project_samples(provider, &samples)
            .with_context(|| format!("encode audio {}", path.display()))
    }

    pub(crate) fn project_samples(
        &self,
        provider: &mut dyn KernelProvider,
        samples: &[f32],
    ) -> Result<AudioProjection> {
        let mel = self.frontend.extract(samples)?;
        let encoded = self
            .conformer
            .forward(provider, &mel.values, mel.frames, mel.features)?;
        let embeddings = self
            .adapter
            .forward(provider, &encoded)?
            .chunks_exact(2048)
            .map(<[f32]>::to_vec)
            .collect();
        Ok(AudioProjection {
            samples: samples.len(),
            mel,
            embeddings,
        })
    }

    /// Scatters the shorter text stream and projected audio stream according to
    /// the modality sequence produced by `ChatState`.
    pub(crate) fn splice_prefill(
        &self,
        token_ids: &[u32],
        modality_flag: &[bool],
        audio_embeddings: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        ensure!(
            !audio_embeddings.is_empty(),
            "audio encoder returned no frames"
        );
        let text_slots = modality_flag.iter().filter(|&&is_audio| !is_audio).count();
        let audio_slots = modality_flag.len() - text_slots;
        ensure!(
            text_slots == token_ids.len(),
            "modality sequence has {text_slots} text slots for {} text tokens",
            token_ids.len()
        );
        ensure!(
            audio_slots == audio_embeddings.len(),
            "modality sequence has {audio_slots} audio slots for {} projected frames",
            audio_embeddings.len()
        );
        let mut tokens = token_ids.iter();
        let mut audio = audio_embeddings.iter();
        let mut result = Vec::with_capacity(modality_flag.len());
        for &is_audio in modality_flag {
            if is_audio {
                let embedding = audio.next().context("missing projected audio frame")?;
                ensure!(
                    embedding.len() == self.backbone.config.hidden_size,
                    "projected audio embedding width mismatch"
                );
                result.push(embedding.clone());
            } else {
                let token = *tokens.next().context("missing text token")?;
                result.push(self.backbone.token_embedding(token)?.to_vec());
            }
        }
        Ok(result)
    }

    /// Builds the exact text stream and longer modality stream used by the
    /// reference `ChatState` ASR example. Audio is inserted out-of-band between
    /// the user prefix and its end-of-turn marker.
    pub(crate) fn asr_prompt(
        &self,
        tokenizer: &Tokenizer,
        audio_frames: usize,
    ) -> Result<(Vec<u32>, Vec<bool>)> {
        let mut tokens = vec![1];
        let mut modality = vec![false];
        append_text(
            tokenizer,
            "<|im_start|>system\n",
            &mut tokens,
            &mut modality,
        )?;
        append_text(tokenizer, "Perform ASR.", &mut tokens, &mut modality)?;
        append_text(tokenizer, "<|im_end|>\n", &mut tokens, &mut modality)?;
        append_text(tokenizer, "<|im_start|>user\n", &mut tokens, &mut modality)?;
        modality.extend(std::iter::repeat_n(true, audio_frames));
        append_text(tokenizer, "<|im_end|>\n", &mut tokens, &mut modality)?;
        append_text(
            tokenizer,
            "<|im_start|>assistant\n",
            &mut tokens,
            &mut modality,
        )?;
        Ok((tokens, modality))
    }
}

impl MelFrontend {
    fn new(config: PreprocessorConfig) -> Result<Self> {
        let window_length = (config.sample_rate as f32 * config.window_size).round() as usize;
        let hop_length = (config.sample_rate as f32 * config.window_stride).round() as usize;
        ensure!(
            window_length == 400 && hop_length == 160,
            "checkpoint requires a 400-sample window and 160-sample hop"
        );
        ensure!(
            config.n_fft >= window_length,
            "FFT must cover the analysis window"
        );
        let mut fft_window = vec![0.0; config.n_fft];
        let window_offset = (config.n_fft - window_length) / 2;
        for index in 0..window_length {
            // The reference explicitly requests a symmetric (non-periodic) Hann window.
            fft_window[window_offset + index] =
                0.5 - 0.5 * (2.0 * PI * index as f32 / (window_length - 1) as f32).cos();
        }
        let mel_filter = slaney_mel_filter(
            config.sample_rate,
            config.n_fft,
            config.features,
            0.0,
            config.sample_rate as f32 / 2.0,
        );
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(config.n_fft);
        Ok(Self {
            config,
            hop_length,
            fft,
            fft_window,
            mel_filter,
        })
    }

    fn frame_count(&self, samples: usize) -> usize {
        samples / self.hop_length + 1
    }

    fn extract(&self, samples: &[f32]) -> Result<MelOutput> {
        ensure!(!samples.is_empty(), "audio waveform is empty");
        let mut emphasized = Vec::with_capacity(samples.len());
        emphasized.push(samples[0]);
        emphasized.extend(
            samples[1..]
                .iter()
                .zip(samples)
                .map(|(&current, &previous)| current - 0.97 * previous),
        );

        let pad = self.config.n_fft / 2;
        let mut padded = vec![0.0; emphasized.len() + 2 * pad];
        padded[pad..pad + emphasized.len()].copy_from_slice(&emphasized);

        let frames = self.frame_count(samples.len());
        let bins = self.config.n_fft / 2 + 1;
        let mut values = vec![0.0; frames * self.config.features];
        let mut buffer = vec![Complex32::new(0.0, 0.0); self.config.n_fft];
        for frame in 0..frames {
            let start = frame * self.hop_length;
            for index in 0..self.config.n_fft {
                buffer[index] = Complex32::new(padded[start + index] * self.fft_window[index], 0.0);
            }
            self.fft.process(&mut buffer);
            for mel in 0..self.config.features {
                let filter = &self.mel_filter[mel * bins..(mel + 1) * bins];
                let power = filter
                    .iter()
                    .zip(&buffer[..bins])
                    .map(|(&weight, value)| weight * value.norm_sqr())
                    .sum::<f32>();
                values[frame * self.config.features + mel] = (power + 2.0_f32.powi(-24)).ln();
            }
        }
        let valid_frames = samples.len() / self.hop_length;
        if valid_frames > 0 {
            normalize_per_feature(
                &mut values[..valid_frames * self.config.features],
                valid_frames,
                self.config.features,
            );
        }
        values[valid_frames * self.config.features..].fill(self.config.pad_value);
        Ok(MelOutput {
            values,
            frames,
            features: self.config.features,
        })
    }
}

impl FastConformer {
    fn load(config: EncoderConfig, tensors: &HashMap<String, Tensor>) -> Result<Self> {
        let pre_encode = Conv2dPreEncoder::load(tensors, config.subsampling_conv_channels)?;
        let mut layers = Vec::with_capacity(config.n_layers);
        for index in 0..config.n_layers {
            layers.push(ConformerLayer::load(index, &config, tensors)?);
        }
        Ok(Self {
            config,
            pre_encode,
            layers,
        })
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        values: &[f32],
        frames: usize,
        features: usize,
    ) -> Result<Vec<f32>> {
        let (mut current, seq) = self
            .pre_encode
            .forward(provider, values, frames, features)?;
        let positional = relative_positional_encoding(seq, self.config.d_model);
        for layer in &self.layers {
            current = layer.forward(provider, &current, &positional, seq, &self.config)?;
        }
        Ok(current)
    }
}

impl Conv2dPreEncoder {
    fn load(tensors: &HashMap<String, Tensor>, channels: usize) -> Result<Self> {
        Ok(Self {
            conv0_weight: required_tensor(
                tensors,
                "conformer.pre_encode.conv.0.weight",
                &[channels, 1, 3, 3],
            )?,
            conv0_bias: required_tensor(tensors, "conformer.pre_encode.conv.0.bias", &[channels])?,
            conv2_weight: required_tensor(
                tensors,
                "conformer.pre_encode.conv.2.weight",
                &[channels, 1, 3, 3],
            )?,
            conv2_bias: required_tensor(tensors, "conformer.pre_encode.conv.2.bias", &[channels])?,
            conv3: Linear::load(
                tensors,
                "conformer.pre_encode.conv.3",
                channels,
                channels,
                true,
            )?,
            conv5_weight: required_tensor(
                tensors,
                "conformer.pre_encode.conv.5.weight",
                &[channels, 1, 3, 3],
            )?,
            conv5_bias: required_tensor(tensors, "conformer.pre_encode.conv.5.bias", &[channels])?,
            conv6: Linear::load(
                tensors,
                "conformer.pre_encode.conv.6",
                channels,
                channels,
                true,
            )?,
            out: Linear::load(
                tensors,
                "conformer.pre_encode.out",
                channels * 16,
                512,
                true,
            )?,
            channels,
        })
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        values: &[f32],
        frames: usize,
        features: usize,
    ) -> Result<(Vec<f32>, usize)> {
        ensure!(features == 128, "FastConformer expects 128 mel features");
        ensure!(
            values.len() == frames * features,
            "mel tensor shape mismatch"
        );
        let (mut current, mut time, mut freq) = conv2d_first(
            values,
            frames,
            features,
            &self.conv0_weight.data,
            &self.conv0_bias.data,
            self.channels,
            2,
        );
        relu_in_place(&mut current);
        (current, time, freq) = depthwise_conv2d(
            &current,
            self.channels,
            time,
            freq,
            &self.conv2_weight.data,
            &self.conv2_bias.data,
            2,
        );
        current = pointwise_conv2d(provider, &current, self.channels, time, freq, &self.conv3)?;
        relu_in_place(&mut current);
        (current, time, freq) = depthwise_conv2d(
            &current,
            self.channels,
            time,
            freq,
            &self.conv5_weight.data,
            &self.conv5_bias.data,
            2,
        );
        current = pointwise_conv2d(provider, &current, self.channels, time, freq, &self.conv6)?;
        relu_in_place(&mut current);
        ensure!(
            freq == 16,
            "FastConformer subsampling produced {freq} frequency bins"
        );

        let mut flattened = vec![0.0; time * self.channels * freq];
        for t in 0..time {
            for channel in 0..self.channels {
                let source = (channel * time + t) * freq;
                let destination = (t * self.channels + channel) * freq;
                flattened[destination..destination + freq]
                    .copy_from_slice(&current[source..source + freq]);
            }
        }
        Ok((self.out.forward(provider, &flattened, time)?, time))
    }
}

impl ConformerLayer {
    fn load(
        index: usize,
        config: &EncoderConfig,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        let prefix = format!("conformer.layers.{index}");
        Ok(Self {
            norm_feed_forward1: LayerNorm::load(
                tensors,
                &format!("{prefix}.norm_feed_forward1"),
                config.d_model,
            )?,
            feed_forward1: FeedForward::load(tensors, &format!("{prefix}.feed_forward1"), config)?,
            norm_self_att: LayerNorm::load(
                tensors,
                &format!("{prefix}.norm_self_att"),
                config.d_model,
            )?,
            self_attn: RelativeAttention::load(tensors, &format!("{prefix}.self_attn"), config)?,
            norm_conv: LayerNorm::load(tensors, &format!("{prefix}.norm_conv"), config.d_model)?,
            conv: ConvModule::load(tensors, &format!("{prefix}.conv"), config)?,
            norm_feed_forward2: LayerNorm::load(
                tensors,
                &format!("{prefix}.norm_feed_forward2"),
                config.d_model,
            )?,
            feed_forward2: FeedForward::load(tensors, &format!("{prefix}.feed_forward2"), config)?,
            norm_out: LayerNorm::load(tensors, &format!("{prefix}.norm_out"), config.d_model)?,
        })
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        values: &[f32],
        positional: &[f32],
        seq: usize,
        config: &EncoderConfig,
    ) -> Result<Vec<f32>> {
        let width = config.d_model;
        let normalized = self.norm_feed_forward1.forward(values, seq)?;
        let update = self
            .feed_forward1
            .forward(provider, &normalized, seq, width)?;
        let mut current = residual_scaled(values, &update, 0.5)?;

        let normalized = self.norm_self_att.forward(&current, seq)?;
        let update = self
            .self_attn
            .forward(provider, &normalized, positional, seq, config)?;
        add_in_place(&mut current, &update)?;

        let normalized = self.norm_conv.forward(&current, seq)?;
        let update = self.conv.forward(provider, &normalized, seq, width)?;
        add_in_place(&mut current, &update)?;

        let normalized = self.norm_feed_forward2.forward(&current, seq)?;
        let update = self
            .feed_forward2
            .forward(provider, &normalized, seq, width)?;
        current = residual_scaled(&current, &update, 0.5)?;
        self.norm_out.forward(&current, seq)
    }
}

impl FeedForward {
    fn load(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        config: &EncoderConfig,
    ) -> Result<Self> {
        let expanded = config.d_model * config.ff_expansion_factor;
        Ok(Self {
            linear1: Linear::load(
                tensors,
                &format!("{prefix}.linear1"),
                config.d_model,
                expanded,
                true,
            )?,
            linear2: Linear::load(
                tensors,
                &format!("{prefix}.linear2"),
                expanded,
                config.d_model,
                true,
            )?,
        })
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        values: &[f32],
        rows: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        ensure!(
            values.len() == rows * width,
            "Conformer FFN input shape mismatch"
        );
        let mut hidden = self.linear1.forward(provider, values, rows)?;
        swish_in_place(&mut hidden);
        self.linear2.forward(provider, &hidden, rows)
    }
}

impl RelativeAttention {
    fn load(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        config: &EncoderConfig,
    ) -> Result<Self> {
        let width = config.d_model;
        let head_dim = width / config.n_heads;
        Ok(Self {
            linear_q: Linear::load(tensors, &format!("{prefix}.linear_q"), width, width, true)?,
            linear_k: Linear::load(tensors, &format!("{prefix}.linear_k"), width, width, true)?,
            linear_v: Linear::load(tensors, &format!("{prefix}.linear_v"), width, width, true)?,
            linear_out: Linear::load(tensors, &format!("{prefix}.linear_out"), width, width, true)?,
            linear_pos: Linear::load(
                tensors,
                &format!("{prefix}.linear_pos"),
                width,
                width,
                false,
            )?,
            pos_bias_u: required_tensor(
                tensors,
                &format!("{prefix}.pos_bias_u"),
                &[config.n_heads, head_dim],
            )?,
            pos_bias_v: required_tensor(
                tensors,
                &format!("{prefix}.pos_bias_v"),
                &[config.n_heads, head_dim],
            )?,
        })
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        values: &[f32],
        positional: &[f32],
        seq: usize,
        config: &EncoderConfig,
    ) -> Result<Vec<f32>> {
        let width = config.d_model;
        let heads = config.n_heads;
        let head_dim = width / heads;
        let q = self.linear_q.forward(provider, values, seq)?;
        let k = self.linear_k.forward(provider, values, seq)?;
        let v = self.linear_v.forward(provider, values, seq)?;
        let position_count = 2 * seq - 1;
        ensure!(
            positional.len() == position_count * width,
            "relative positional tensor shape mismatch"
        );
        let p = self
            .linear_pos
            .forward(provider, positional, position_count)?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut context = vec![0.0; seq * width];
        let mut scores = vec![0.0; seq];
        for head in 0..heads {
            let u = &self.pos_bias_u.data[head * head_dim..(head + 1) * head_dim];
            let rel_bias = &self.pos_bias_v.data[head * head_dim..(head + 1) * head_dim];
            for query in 0..seq {
                let q_offset = query * width + head * head_dim;
                let q_head = &q[q_offset..q_offset + head_dim];
                for (key, score) in scores.iter_mut().enumerate() {
                    let k_offset = key * width + head * head_dim;
                    let k_head = &k[k_offset..k_offset + head_dim];
                    let relative_index = seq - 1 - query + key;
                    let p_offset = relative_index * width + head * head_dim;
                    let p_head = &p[p_offset..p_offset + head_dim];
                    let mut content_score = 0.0;
                    let mut position_score = 0.0;
                    for dim in 0..head_dim {
                        content_score += (q_head[dim] + u[dim]) * k_head[dim];
                        position_score += (q_head[dim] + rel_bias[dim]) * p_head[dim];
                    }
                    *score = (content_score + position_score) * scale;
                }
                softmax_in_place(&mut scores);
                let destination = query * width + head * head_dim;
                for dim in 0..head_dim {
                    context[destination + dim] = scores
                        .iter()
                        .enumerate()
                        .map(|(key, &score)| score * v[key * width + head * head_dim + dim])
                        .sum();
                }
            }
        }
        self.linear_out.forward(provider, &context, seq)
    }
}

impl ConvModule {
    fn load(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        config: &EncoderConfig,
    ) -> Result<Self> {
        let width = config.d_model;
        Ok(Self {
            pointwise1: Linear::load(
                tensors,
                &format!("{prefix}.pointwise_conv1"),
                width,
                2 * width,
                true,
            )?,
            depthwise_weight: required_tensor(
                tensors,
                &format!("{prefix}.depthwise_conv.weight"),
                &[width, 1, config.conv_kernel_size],
            )?,
            depthwise_bias: required_tensor(
                tensors,
                &format!("{prefix}.depthwise_conv.bias"),
                &[width],
            )?,
            batch_norm_weight: required_tensor(
                tensors,
                &format!("{prefix}.batch_norm.weight"),
                &[width],
            )?,
            batch_norm_bias: required_tensor(
                tensors,
                &format!("{prefix}.batch_norm.bias"),
                &[width],
            )?,
            running_mean: required_tensor(
                tensors,
                &format!("{prefix}.batch_norm.running_mean"),
                &[width],
            )?,
            running_var: required_tensor(
                tensors,
                &format!("{prefix}.batch_norm.running_var"),
                &[width],
            )?,
            pointwise2: Linear::load(
                tensors,
                &format!("{prefix}.pointwise_conv2"),
                width,
                width,
                true,
            )?,
        })
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        values: &[f32],
        seq: usize,
        width: usize,
    ) -> Result<Vec<f32>> {
        let gated = self.pointwise1.forward(provider, values, seq)?;
        let mut glu = vec![0.0; seq * width];
        for row in 0..seq {
            for channel in 0..width {
                let first = gated[row * 2 * width + channel];
                let gate = gated[row * 2 * width + width + channel];
                glu[row * width + channel] = first * sigmoid(gate);
            }
        }
        let kernel = self.depthwise_weight.shape[2];
        let padding = kernel / 2;
        let mut convolved = vec![0.0; seq * width];
        for row in 0..seq {
            for channel in 0..width {
                let mut value = self.depthwise_bias.data[channel];
                for tap in 0..kernel {
                    let source = row as isize + tap as isize - padding as isize;
                    if (0..seq as isize).contains(&source) {
                        value += glu[source as usize * width + channel]
                            * self.depthwise_weight.data[channel * kernel + tap];
                    }
                }
                let normalized = (value - self.running_mean.data[channel])
                    / (self.running_var.data[channel] + 1.0e-5).sqrt();
                convolved[row * width + channel] = normalized
                    * self.batch_norm_weight.data[channel]
                    + self.batch_norm_bias.data[channel];
            }
        }
        swish_in_place(&mut convolved);
        self.pointwise2.forward(provider, &convolved, seq)
    }
}

impl AudioAdapter {
    fn load(tensors: &HashMap<String, Tensor>) -> Result<Self> {
        Ok(Self {
            norm: LayerNorm::load(tensors, "audio_adapter.model.0", 512)?,
            linear1: Linear::load(tensors, "audio_adapter.model.1", 512, 2048, true)?,
            linear2: Linear::load(tensors, "audio_adapter.model.3", 2048, 2048, true)?,
        })
    }

    fn forward(&self, provider: &mut dyn KernelProvider, values: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            values.len() % 512 == 0,
            "audio adapter input shape mismatch"
        );
        let rows = values.len() / 512;
        let normalized = self.norm.forward(values, rows)?;
        let mut hidden = self.linear1.forward(provider, &normalized, rows)?;
        gelu_in_place(&mut hidden);
        self.linear2.forward(provider, &hidden, rows)
    }
}

impl Linear {
    fn load(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        input: usize,
        output: usize,
        has_bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            weight: linear_weight(tensors, &format!("{prefix}.weight"), input, output)?,
            bias: has_bias
                .then(|| required_tensor(tensors, &format!("{prefix}.bias"), &[output]))
                .transpose()?,
            label: prefix.to_owned(),
        })
    }

    fn forward(
        &self,
        provider: &mut dyn KernelProvider,
        values: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let (output, input) = self.weight.matrix_shape()?;
        ensure!(
            values.len() == rows * input,
            "{} input shape mismatch",
            self.label
        );
        let mut result = vec![0.0; rows * output];
        provider.matmul_static_rhs(
            rows,
            output,
            input,
            values,
            &self.weight.data,
            BLayout::RowMajorNkTransposed,
            &mut result,
        )?;
        if let Some(bias) = &self.bias {
            for row in result.chunks_exact_mut(output) {
                for (value, &bias) in row.iter_mut().zip(&bias.data) {
                    *value += bias;
                }
            }
        }
        Ok(result)
    }
}

impl LayerNorm {
    fn load(tensors: &HashMap<String, Tensor>, prefix: &str, width: usize) -> Result<Self> {
        Ok(Self {
            weight: required_tensor(tensors, &format!("{prefix}.weight"), &[width])?,
            bias: required_tensor(tensors, &format!("{prefix}.bias"), &[width])?,
            eps: 1.0e-5,
        })
    }

    fn forward(&self, values: &[f32], rows: usize) -> Result<Vec<f32>> {
        let width = self.weight.data.len();
        ensure!(
            values.len() == rows * width,
            "layer norm input shape mismatch"
        );
        let mut result = values.to_vec();
        for row in result.chunks_exact_mut(width) {
            let mean = row.iter().sum::<f32>() / width as f32;
            let variance = row
                .iter()
                .map(|value| {
                    let delta = *value - mean;
                    delta * delta
                })
                .sum::<f32>()
                / width as f32;
            let inverse = 1.0 / (variance + self.eps).sqrt();
            for (index, value) in row.iter_mut().enumerate() {
                *value =
                    (*value - mean) * inverse * self.weight.data[index] + self.bias.data[index];
            }
        }
        Ok(result)
    }
}

impl ModelFamily for AudioFamily {
    fn family_name(&self) -> &'static str {
        "lfm2-audio"
    }

    fn token_length(&self, tokenizer: &Tokenizer, text: &str, max_length: usize) -> Result<usize> {
        ModelFamily::token_length(&self.backbone, tokenizer, text, max_length)
    }

    fn embed_batch(
        &self,
        provider: &mut dyn KernelProvider,
        tokenizer: &Tokenizer,
        texts: &[&str],
        max_length: usize,
        shape: Option<BatchShape>,
    ) -> Result<Vec<Vec<f32>>> {
        ModelFamily::embed_batch(
            &self.backbone,
            provider,
            tokenizer,
            texts,
            max_length,
            shape,
        )
    }

    fn default_label(&self, precision: Precision) -> String {
        format!("LFM2-Audio-1.5B@owned-rt-{}", precision.as_str())
    }

    fn notes(&self) -> String {
        format!(
            "LFM2-Audio ASR-capable family; {}",
            ModelFamily::notes(&self.backbone)
        )
    }
}

pub(super) fn detect_config(config: &serde_json::Value) -> bool {
    config.get("lfm").is_some()
        && config.get("encoder").is_some()
        && config.get("preprocessor").is_some()
        && config
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|architectures| {
                architectures.iter().any(|architecture| {
                    architecture.as_str().is_some_and(|name| {
                        name.to_ascii_lowercase().contains("lfm2audio")
                            || name.to_ascii_lowercase().contains("lfm2_audio")
                    })
                })
            })
}

pub(super) fn load_family(path: &Path, precision: Precision) -> Result<Box<dyn ModelFamily>> {
    Ok(Box::new(AudioFamily {
        backbone: lfm2::Model::load(path, precision)?,
    }))
}

fn append_text(
    tokenizer: &Tokenizer,
    text: &str,
    tokens: &mut Vec<u32>,
    modality: &mut Vec<bool>,
) -> Result<()> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|error| anyhow::anyhow!("tokenize ASR chat segment: {error}"))?;
    tokens.extend_from_slice(encoding.get_ids());
    modality.extend(std::iter::repeat_n(false, encoding.len()));
    Ok(())
}

fn validate_audio_config(config: &RawAudioConfig) -> Result<()> {
    let pre = &config.preprocessor;
    ensure!(
        pre.sample_rate == 16_000,
        "LFM2-Audio requires 16 kHz audio"
    );
    ensure!(
        pre.normalize == "per_feature",
        "unsupported audio normalization"
    );
    ensure!(pre.window == "hann", "unsupported audio analysis window");
    ensure!(
        pre.features == 128 && pre.n_fft == 512,
        "unsupported mel shape"
    );
    ensure!(pre.log, "LFM2-Audio requires log mel features");
    ensure!(pre.frame_splicing == 1, "frame splicing is not supported");
    ensure!(
        pre.pad_to == 0 && pre.pad_value == 0.0,
        "unexpected mel padding config"
    );
    ensure!(
        (pre.dither - 1.0e-5).abs() < f32::EPSILON,
        "unexpected training dither value"
    );

    let encoder = &config.encoder;
    ensure!(
        encoder.feat_in == 128
            && encoder.n_layers == 17
            && encoder.d_model == 512
            && encoder.n_heads == 8,
        "unsupported FastConformer dimensions"
    );
    ensure!(
        encoder.subsampling == "dw_striding",
        "unsupported subsampling"
    );
    ensure!(
        encoder.subsampling_factor == 8 && encoder.subsampling_conv_channels == 256,
        "unsupported FastConformer pre-encoder"
    );
    ensure!(
        !encoder.causal_downsampling,
        "ASR encoder must be noncausal"
    );
    ensure!(
        encoder.ff_expansion_factor == 4,
        "unsupported FFN expansion"
    );
    ensure!(
        encoder.self_attention_model == "rel_pos",
        "relative attention required"
    );
    ensure!(
        encoder.att_context_size == [-1, -1],
        "attention must be bidirectional"
    );
    ensure!(!encoder.xscaling, "unexpected positional xscaling");
    ensure!(
        encoder.untie_biases,
        "relative attention biases must be untied"
    );
    ensure!(
        encoder.pos_emb_max_len == 5000,
        "unexpected positional table size"
    );
    ensure!(
        encoder.conv_kernel_size == 9,
        "unsupported conformer kernel"
    );
    ensure!(
        encoder.conv_norm_type == "batch_norm",
        "batch norm running stats required"
    );
    Ok(())
}

fn required_tensor(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Tensor> {
    let tensor = get_tensor(tensors, name)?;
    ensure!(
        tensor.shape == expected_shape,
        "tensor {name} shape {:?}, expected {:?}",
        tensor.shape,
        expected_shape
    );
    Ok(tensor)
}

fn linear_weight(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    input: usize,
    output: usize,
) -> Result<Tensor> {
    let mut tensor = get_tensor(tensors, name)?;
    ensure!(
        tensor.shape.len() >= 2
            && tensor.shape[0] == output
            && tensor.shape[1] == input
            && tensor.shape[2..].iter().all(|&dimension| dimension == 1),
        "tensor {name} shape {:?}, expected [{output}, {input}] with optional singleton convolution axes",
        tensor.shape
    );
    tensor.shape = vec![output, input];
    tensor.strides = vec![input, 1];
    Ok(tensor)
}

fn read_mono_wav(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("open WAV {}", path.display()))?;
    let spec = reader.spec();
    ensure!(spec.channels == 1, "ASR input must be mono WAV");
    ensure!(
        spec.sample_rate == SAMPLE_RATE,
        "ASR input must be {SAMPLE_RATE} Hz, got {} Hz",
        spec.sample_rate
    );
    let values = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 2.0_f32.powi(spec.bits_per_sample as i32 - 1);
            if spec.bits_per_sample <= 16 {
                reader
                    .samples::<i16>()
                    .map(|sample| sample.map(|value| value as f32 / scale))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            } else {
                reader
                    .samples::<i32>()
                    .map(|sample| sample.map(|value| value as f32 / scale))
                    .collect::<std::result::Result<Vec<_>, _>>()?
            }
        }
    };
    ensure!(
        values.iter().all(|sample| sample.is_finite()),
        "WAV contains non-finite samples"
    );
    Ok(values)
}

fn slaney_mel_filter(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
) -> Vec<f32> {
    let bins = n_fft / 2 + 1;
    let min_mel = hz_to_slaney_mel(f_min);
    let max_mel = hz_to_slaney_mel(f_max);
    let mel_points = (0..n_mels + 2)
        .map(|index| min_mel + (max_mel - min_mel) * index as f32 / (n_mels + 1) as f32)
        .map(slaney_mel_to_hz)
        .collect::<Vec<_>>();
    let frequencies = (0..bins)
        .map(|index| index as f32 * sample_rate as f32 / n_fft as f32)
        .collect::<Vec<_>>();
    let mut filter = vec![0.0; n_mels * bins];
    for mel in 0..n_mels {
        let lower_width = mel_points[mel + 1] - mel_points[mel];
        let upper_width = mel_points[mel + 2] - mel_points[mel + 1];
        let normalization = 2.0 / (mel_points[mel + 2] - mel_points[mel]);
        for (bin, &frequency) in frequencies.iter().enumerate() {
            let lower = (frequency - mel_points[mel]) / lower_width;
            let upper = (mel_points[mel + 2] - frequency) / upper_width;
            filter[mel * bins + bin] = lower.min(upper).max(0.0) * normalization;
        }
    }
    filter
}

fn hz_to_slaney_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    const LOG_STEP: f32 = 0.068_751_78; // ln(6.4) / 27
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / LOG_STEP
    } else {
        hz / F_SP
    }
}

fn slaney_mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    const LOG_STEP: f32 = 0.068_751_78;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (LOG_STEP * (mel - MIN_LOG_MEL)).exp()
    } else {
        mel * F_SP
    }
}

fn normalize_per_feature(values: &mut [f32], frames: usize, features: usize) {
    for feature in 0..features {
        let mean = (0..frames)
            .map(|frame| values[frame * features + feature])
            .sum::<f32>()
            / frames as f32;
        let variance = (0..frames)
            .map(|frame| {
                let delta = values[frame * features + feature] - mean;
                delta * delta
            })
            .sum::<f32>()
            / frames.saturating_sub(1).max(1) as f32;
        let std = variance.sqrt();
        for frame in 0..frames {
            values[frame * features + feature] =
                (values[frame * features + feature] - mean) / (std + 1.0e-5);
        }
    }
}

fn conv2d_first(
    input: &[f32],
    input_time: usize,
    input_freq: usize,
    weight: &[f32],
    bias: &[f32],
    channels: usize,
    stride: usize,
) -> (Vec<f32>, usize, usize) {
    let output_time = input_time.div_ceil(stride);
    let output_freq = input_freq.div_ceil(stride);
    let mut output = vec![0.0; channels * output_time * output_freq];
    for channel in 0..channels {
        for time in 0..output_time {
            for freq in 0..output_freq {
                let mut value = bias[channel];
                for kt in 0..3 {
                    for kf in 0..3 {
                        let source_time = time as isize * stride as isize + kt - 1;
                        let source_freq = freq as isize * stride as isize + kf - 1;
                        if (0..input_time as isize).contains(&source_time)
                            && (0..input_freq as isize).contains(&source_freq)
                        {
                            value += input
                                [source_time as usize * input_freq + source_freq as usize]
                                * weight[channel * 9 + kt as usize * 3 + kf as usize];
                        }
                    }
                }
                output[(channel * output_time + time) * output_freq + freq] = value;
            }
        }
    }
    (output, output_time, output_freq)
}

fn depthwise_conv2d(
    input: &[f32],
    channels: usize,
    input_time: usize,
    input_freq: usize,
    weight: &[f32],
    bias: &[f32],
    stride: usize,
) -> (Vec<f32>, usize, usize) {
    let output_time = input_time.div_ceil(stride);
    let output_freq = input_freq.div_ceil(stride);
    let mut output = vec![0.0; channels * output_time * output_freq];
    for channel in 0..channels {
        for time in 0..output_time {
            for freq in 0..output_freq {
                let mut value = bias[channel];
                for kt in 0..3 {
                    for kf in 0..3 {
                        let source_time = time as isize * stride as isize + kt - 1;
                        let source_freq = freq as isize * stride as isize + kf - 1;
                        if (0..input_time as isize).contains(&source_time)
                            && (0..input_freq as isize).contains(&source_freq)
                        {
                            let source = (channel * input_time + source_time as usize) * input_freq
                                + source_freq as usize;
                            value +=
                                input[source] * weight[channel * 9 + kt as usize * 3 + kf as usize];
                        }
                    }
                }
                output[(channel * output_time + time) * output_freq + freq] = value;
            }
        }
    }
    (output, output_time, output_freq)
}

fn pointwise_conv2d(
    provider: &mut dyn KernelProvider,
    input: &[f32],
    channels: usize,
    time: usize,
    freq: usize,
    linear: &Linear,
) -> Result<Vec<f32>> {
    let rows = time * freq;
    let mut row_major = vec![0.0; rows * channels];
    for channel in 0..channels {
        for t in 0..time {
            for f in 0..freq {
                row_major[(t * freq + f) * channels + channel] =
                    input[(channel * time + t) * freq + f];
            }
        }
    }
    let transformed = linear.forward(provider, &row_major, rows)?;
    let mut output = vec![0.0; channels * rows];
    for channel in 0..channels {
        for t in 0..time {
            for f in 0..freq {
                output[(channel * time + t) * freq + f] =
                    transformed[(t * freq + f) * channels + channel];
            }
        }
    }
    Ok(output)
}

fn relative_positional_encoding(seq: usize, width: usize) -> Vec<f32> {
    let mut output = vec![0.0; (2 * seq - 1) * width];
    for row in 0..2 * seq - 1 {
        let position = (seq - 1) as isize - row as isize;
        for pair in 0..width / 2 {
            let frequency = 10_000.0_f32.powf(-((2 * pair) as f32) / width as f32);
            let angle = position as f32 * frequency;
            output[row * width + 2 * pair] = angle.sin();
            output[row * width + 2 * pair + 1] = angle.cos();
        }
    }
    output
}

fn residual_scaled(residual: &[f32], update: &[f32], scale: f32) -> Result<Vec<f32>> {
    ensure!(residual.len() == update.len(), "residual shape mismatch");
    Ok(residual
        .iter()
        .zip(update)
        .map(|(&residual, &update)| residual + scale * update)
        .collect())
}

fn add_in_place(values: &mut [f32], update: &[f32]) -> Result<()> {
    ensure!(values.len() == update.len(), "residual shape mismatch");
    for (value, &update) in values.iter_mut().zip(update) {
        *value += update;
    }
    Ok(())
}

fn relu_in_place(values: &mut [f32]) {
    for value in values {
        *value = value.max(0.0);
    }
}

fn swish_in_place(values: &mut [f32]) {
    for value in values {
        *value *= sigmoid(*value);
    }
}

fn gelu_in_place(values: &mut [f32]) {
    for value in values {
        *value *= 0.5 * (1.0 + libm::erff(*value / std::f32::consts::SQRT_2));
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softmax_in_place(values: &mut [f32]) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_family_detection_does_not_capture_plain_lfm2() {
        let audio = serde_json::json!({
            "architectures": ["Lfm2AudioForConditionalGeneration"],
            "preprocessor": {},
            "encoder": {},
            "lfm": {}
        });
        let plain = serde_json::json!({
            "architectures": ["Lfm2ForCausalLM"],
            "model_type": "lfm2"
        });
        assert!(detect_config(&audio));
        assert!(!detect_config(&plain));
        assert!(!crate::lfm2::detect_config(&audio));
        assert!(crate::lfm2::detect_config(&plain));
    }

    #[test]
    fn slaney_filter_has_nonzero_triangles() {
        let filter = slaney_mel_filter(16_000, 512, 128, 0.0, 8_000.0);
        assert_eq!(filter.len(), 128 * 257);
        assert!(filter
            .chunks_exact(257)
            .all(|mel| mel.iter().any(|&value| value > 0.0)));
        assert!(filter
            .iter()
            .all(|&value| value >= 0.0 && value.is_finite()));
    }

    #[test]
    fn per_feature_normalization_has_zero_mean_and_unit_sample_variance() {
        let mut values = vec![1.0, 4.0, 2.0, 6.0, 3.0, 8.0];
        normalize_per_feature(&mut values, 3, 2);
        for feature in 0..2 {
            let column = (0..3)
                .map(|row| values[row * 2 + feature])
                .collect::<Vec<_>>();
            let mean = column.iter().sum::<f32>() / 3.0;
            let variance = column
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f32>()
                / 2.0;
            assert!(mean.abs() < 1.0e-6);
            assert!((variance - 1.0).abs() < 3.0e-5);
        }
    }

    #[test]
    fn relative_positions_run_from_positive_to_negative() {
        let positions = relative_positional_encoding(3, 4);
        assert!((positions[0] - 2.0_f32.sin()).abs() < 1.0e-6);
        assert_eq!(&positions[8..12], &[0.0, 1.0, 0.0, 1.0]);
        assert!((positions[16] + 2.0_f32.sin()).abs() < 1.0e-6);
    }
}
