#![cfg(target_os = "macos")]

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};

use super::{
    DecodeKernel, DecodeKernelTimings, DecodeRuntime, DecodeStageTimings, MetalDecoder, Model,
};
use crate::quant::WeightQuantization;
use crate::{encode_f16_bits, Execution, MetalExecutionConfig, Precision};

#[repr(C)]
struct StepLayerParams {
    input_norm: *const c_void,
    post_attention_norm: *const c_void,
    q_weight: *const c_void,
    q_weight_q8: *const c_void,
    q_norm: *const c_void,
    k_weight: *const c_void,
    k_weight_q8: *const c_void,
    k_norm: *const c_void,
    v_weight: *const c_void,
    v_weight_q8: *const c_void,
    o_weight: *const c_void,
    o_weight_q8: *const c_void,
    gate_weight: *const c_void,
    gate_weight_q8: *const c_void,
    up_weight: *const c_void,
    up_weight_q8: *const c_void,
    down_weight: *const c_void,
    down_weight_q8: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct NativeTimings {
    feed_wall_s: f64,
    execute_wall_s: f64,
    logits_readback_wall_s: f64,
    kv_update_wall_s: f64,
    kernel_rmsnorm_s: f64,
    kernel_qkv_matvec_s: f64,
    kernel_qk_norm_rope_s: f64,
    kernel_attention_s: f64,
    kernel_o_proj_s: f64,
    kernel_residual_rmsnorm_s: f64,
    kernel_down_proj_s: f64,
    kernel_gate_up_swiglu_s: f64,
    kernel_lm_head_s: f64,
    kernel_samples: u64,
    step_calls: u64,
}

pub(crate) struct MetalStepKvCache {
    position: usize,
}

pub(crate) struct MetalStepDecoder<'a> {
    prefill: MetalDecoder<'a>,
    raw: NonNull<c_void>,
    model: &'a Model,
    bucket: usize,
    host_prepare_wall_s: f64,
    // Chained-decode span. k=1 preserves the fully instrumented per-token path
    // byte-for-byte; k>1 encodes k forward passes plus on-GPU argmax into one
    // command buffer with a single readback. Runtime-tunable via
    // SYNAPSE_METAL_STEP_CHAIN_K (default 1; chaining is opt-in).
    chain_k: usize,
    // Opt-in batched speculative verification (SYNAPSE_METAL_STEP_BATCHED_VERIFY).
    // When set, the verifier runs K draft tokens through one batched forward
    // (weights streamed once per layer) instead of K sequential single-token
    // steps. Default off: the speculative path keeps its prior sequential shape.
    batched_verify: bool,
}

impl<'a> MetalStepDecoder<'a> {
    pub(crate) fn new(
        model: &'a Model,
        precision: Precision,
        execution: &MetalExecutionConfig,
        bucket: usize,
        weight_quantization: WeightQuantization,
    ) -> Result<Self> {
        ensure!(
            matches!(precision, Precision::F16),
            "Metal step activations require --dtype f16"
        );
        ensure!(
            matches!(execution.execution, Execution::Explicit),
            "Metal step prefill requires explicit MPSGraph compilation"
        );
        ensure!(
            [512, 1024, 2048].contains(&bucket),
            "decode cache bucket must be 512, 1024, or 2048"
        );
        ensure!(
            model.weight_quantization == weight_quantization,
            "Metal step weight quantization does not match the loaded model"
        );

        let started = Instant::now();
        let prefill = MetalDecoder::new(model, precision, execution, bucket)?;
        let library_path = metal_step_library_path()?;
        let library = CString::new(library_path.to_string_lossy().as_bytes())?;
        let raw = NonNull::new(unsafe {
            synapse_qwen3_metal_step_context_new(
                bucket as u64,
                model.config.hidden_size as u64,
                model.config.num_attention_heads as u64,
                model.config.num_key_value_heads as u64,
                model.config.head_dim as u64,
                model.config.intermediate_size as u64,
                model.config.vocab_size as u64,
                model.config.rms_norm_eps,
                library.as_ptr(),
            )
        })
        .ok_or_else(last_error)?;
        let decoder = Self {
            prefill,
            raw,
            model,
            bucket,
            host_prepare_wall_s: started.elapsed().as_secs_f64(),
            chain_k: read_chain_k(),
            batched_verify: read_batched_verify(),
        };
        let params = decoder.layer_params()?;
        let final_norm = decoder.model.final_norm.weight.metal_f16_bits()?;
        let lm_head = decoder.model.lm_head()?.metal_f16_bits()?;
        // The chained-decode embedding gather reads this resident f16 table.
        // These are the same bits encode_f16_bits(embedding) produces on the
        // per-token host path, so the fed activation is byte-identical.
        let embeddings = decoder.model.embeddings.metal_f16_bits()?;
        let status = unsafe {
            synapse_qwen3_metal_step_prepare(
                decoder.raw.as_ptr(),
                params.len() as u64,
                u32::from(weight_quantization.is_quantized()),
                params.as_ptr(),
                final_norm.as_ptr().cast(),
                lm_head.as_ptr().cast(),
                decoder
                    .model
                    .lm_head_q8_0()
                    .map_or(std::ptr::null(), |weight| weight.as_bytes().as_ptr())
                    .cast(),
                embeddings.as_ptr().cast(),
            )
        };
        if status != 0 {
            bail!(
                "Qwen3 Metal step preparation failed with status {status}: {}",
                last_error()
            );
        }
        Ok(decoder)
    }

    fn layer_params(&self) -> Result<Vec<StepLayerParams>> {
        self.model
            .layers
            .iter()
            .map(|layer| {
                let q_weight = layer.q_proj.tensor.metal_f16_bits()?;
                let k_weight = layer.k_proj.tensor.metal_f16_bits()?;
                let v_weight = layer.v_proj.tensor.metal_f16_bits()?;
                let o_weight = layer.o_proj.tensor.metal_f16_bits()?;
                let gate_weight = layer.gate_proj.tensor.metal_f16_bits()?;
                let up_weight = layer.up_proj.tensor.metal_f16_bits()?;
                let down_weight = layer.down_proj.tensor.metal_f16_bits()?;
                Ok(StepLayerParams {
                    input_norm: layer.input_norm.weight.metal_f16_bits()?.as_ptr().cast(),
                    post_attention_norm: layer
                        .post_attention_norm
                        .weight
                        .metal_f16_bits()?
                        .as_ptr()
                        .cast(),
                    q_weight: q_weight.as_ptr().cast(),
                    q_weight_q8: layer
                        .q_proj
                        .q8_0
                        .as_ref()
                        .map_or(std::ptr::null(), |weight| weight.as_bytes().as_ptr())
                        .cast(),
                    q_norm: layer.q_norm.weight.metal_f16_bits()?.as_ptr().cast(),
                    k_weight: k_weight.as_ptr().cast(),
                    k_weight_q8: layer
                        .k_proj
                        .q8_0
                        .as_ref()
                        .map_or(std::ptr::null(), |weight| weight.as_bytes().as_ptr())
                        .cast(),
                    k_norm: layer.k_norm.weight.metal_f16_bits()?.as_ptr().cast(),
                    v_weight: v_weight.as_ptr().cast(),
                    v_weight_q8: layer
                        .v_proj
                        .q8_0
                        .as_ref()
                        .map_or(std::ptr::null(), |weight| weight.as_bytes().as_ptr())
                        .cast(),
                    o_weight: o_weight.as_ptr().cast(),
                    o_weight_q8: layer
                        .o_proj
                        .q8_0
                        .as_ref()
                        .map_or(std::ptr::null(), |weight| weight.as_bytes().as_ptr())
                        .cast(),
                    gate_weight: gate_weight.as_ptr().cast(),
                    gate_weight_q8: layer
                        .gate_proj
                        .q8_0
                        .as_ref()
                        .map_or(std::ptr::null(), |weight| weight.as_bytes().as_ptr())
                        .cast(),
                    up_weight: up_weight.as_ptr().cast(),
                    up_weight_q8: layer
                        .up_proj
                        .q8_0
                        .as_ref()
                        .map_or(std::ptr::null(), |weight| weight.as_bytes().as_ptr())
                        .cast(),
                    down_weight: down_weight.as_ptr().cast(),
                    down_weight_q8: layer
                        .down_proj
                        .q8_0
                        .as_ref()
                        .map_or(std::ptr::null(), |weight| weight.as_bytes().as_ptr())
                        .cast(),
                })
            })
            .collect()
    }

    fn embedding(&self, token: u32) -> Result<&[f32]> {
        let token = token as usize;
        ensure!(
            token < self.model.config.vocab_size,
            "token id {token} outside Qwen3 vocabulary"
        );
        let hidden = self.model.config.hidden_size;
        Ok(&self.model.embeddings.data[token * hidden..(token + 1) * hidden])
    }

    fn rope(&self, position: usize) -> (Vec<u16>, Vec<u16>) {
        let head_dim = self.model.config.head_dim;
        let mut cosine = Vec::with_capacity(head_dim);
        let mut sine = Vec::with_capacity(head_dim);
        for index in 0..head_dim {
            let rotary_index = index % (head_dim / 2);
            let frequency = 1.0
                / self
                    .model
                    .config
                    .rope_theta
                    .powf((2 * rotary_index) as f32 / head_dim as f32);
            let (sin, cos) = (position as f32 * frequency).sin_cos();
            cosine.push(cos);
            sine.push(sin);
        }
        (encode_f16_bits(&cosine), encode_f16_bits(&sine))
    }

    /// Concatenated rope tables for `steps` consecutive positions starting at
    /// `start`. Each position's head_dim block is produced by the exact same
    /// per-index formula as `rope`, so the chain's step S reads a block that is
    /// byte-identical to what the per-token path would compute for `start + S`.
    fn rope_chain(&self, start: usize, steps: usize) -> (Vec<u16>, Vec<u16>) {
        let head_dim = self.model.config.head_dim;
        let mut cosine = Vec::with_capacity(head_dim * steps);
        let mut sine = Vec::with_capacity(head_dim * steps);
        for step in 0..steps {
            let (cos_bits, sin_bits) = self.rope(start + step);
            cosine.extend_from_slice(&cos_bits);
            sine.extend_from_slice(&sin_bits);
        }
        (cosine, sine)
    }

    /// Verifies a proposed token span in the existing chained-step command
    /// buffer. Each proposal is gathered from the supplied device-side token
    /// list; the returned argmax follows that proposal. The session compares its
    /// pending argmax to the first proposal and shifts these results for later
    /// proposals, leaving the full span resident for either commit or rewind.
    pub(crate) fn verify_tokens(
        &mut self,
        cache: &mut MetalStepKvCache,
        tokens: &[u32],
    ) -> Result<Vec<u32>> {
        ensure!(
            !tokens.is_empty(),
            "verification requires at least one token"
        );
        ensure!(
            cache.position + tokens.len() <= self.bucket,
            "speculative verification exceeds cache capacity"
        );
        ensure!(
            tokens
                .iter()
                .all(|&token| (token as usize) < self.model.config.vocab_size),
            "speculative verification received a token outside the Qwen3 vocabulary"
        );
        let (rope_cos, rope_sin) = self.rope_chain(cache.position, tokens.len());
        let mut argmaxes = vec![0u32; tokens.len()];
        let status = unsafe {
            synapse_qwen3_metal_step_verify(
                self.raw.as_ptr(),
                cache.position as u64,
                tokens.as_ptr(),
                tokens.len() as u32,
                rope_cos.as_ptr(),
                rope_sin.as_ptr(),
                argmaxes.as_mut_ptr(),
                self.model.config.rms_norm_eps,
            )
        };
        if status != 0 {
            bail!(
                "Qwen3 Metal step verification failed with status {status}: {}",
                last_error()
            );
        }
        cache.position += tokens.len();
        Ok(argmaxes)
    }

    /// Verifies a draft span in ONE batched forward pass: all `tokens.len()`
    /// positions run through the transformer as a batch (mat-mat with K columns)
    /// so each layer's weights stream once instead of once per token. Returns the
    /// greedy argmax after each supplied token, aligned exactly as `verify_tokens`.
    /// By construction the per-position logits are bit-identical to K sequential
    /// single-token `advance` steps at the same positions; the batch only shares
    /// the weight read across positions, never reordering one dot's accumulation.
    pub(crate) fn verify_batch(
        &mut self,
        cache: &mut MetalStepKvCache,
        tokens: &[u32],
    ) -> Result<Vec<u32>> {
        self.verify_batch_inner(cache, tokens, None)
    }

    /// Batched verification that also reads back the full per-position f32 logits,
    /// flattened as `tokens.len()` contiguous `vocab_size` rows (row `i` is the
    /// logits after position `cache.position + i`). This is the byte-exact gate
    /// surface: row `i` must equal the logits from a sequential `advance` at that
    /// position. The serving path uses `verify_batch` (argmax-only readback).
    #[allow(dead_code)]
    pub(crate) fn verify_batch_logits(
        &mut self,
        cache: &mut MetalStepKvCache,
        tokens: &[u32],
    ) -> Result<Vec<f32>> {
        let mut logits = vec![0.0f32; tokens.len() * self.model.config.vocab_size];
        self.verify_batch_inner(cache, tokens, Some(&mut logits))?;
        Ok(logits)
    }

    fn verify_batch_inner(
        &mut self,
        cache: &mut MetalStepKvCache,
        tokens: &[u32],
        logits_out: Option<&mut [f32]>,
    ) -> Result<Vec<u32>> {
        ensure!(
            !tokens.is_empty(),
            "batched verification requires at least one token"
        );
        ensure!(
            tokens.len() <= 16,
            "batched verification supports at most 16 draft tokens, got {}",
            tokens.len()
        );
        ensure!(
            cache.position + tokens.len() <= self.bucket,
            "batched speculative verification exceeds cache capacity"
        );
        ensure!(
            tokens
                .iter()
                .all(|&token| (token as usize) < self.model.config.vocab_size),
            "batched speculative verification received a token outside the Qwen3 vocabulary"
        );
        if let Some(logits) = &logits_out {
            ensure!(
                logits.len() == tokens.len() * self.model.config.vocab_size,
                "batched verification logits output has the wrong length"
            );
        }
        let (rope_cos, rope_sin) = self.rope_chain(cache.position, tokens.len());
        let mut argmaxes = vec![0u32; tokens.len()];
        let status = unsafe {
            synapse_qwen3_metal_step_verify_batch(
                self.raw.as_ptr(),
                cache.position as u64,
                tokens.as_ptr(),
                tokens.len() as u32,
                rope_cos.as_ptr(),
                rope_sin.as_ptr(),
                argmaxes.as_mut_ptr(),
                logits_out.map_or(std::ptr::null_mut(), |logits| logits.as_mut_ptr()),
                self.model.config.rms_norm_eps,
            )
        };
        if status != 0 {
            bail!(
                "Qwen3 Metal step batched verification failed with status {status}: {}",
                last_error()
            );
        }
        cache.position += tokens.len();
        Ok(argmaxes)
    }

    /// Encode `steps` chained decode passes into one command buffer and return
    /// the `steps` argmax token ids. `seed` is the token whose embedding feeds
    /// step 0 (the last committed token); each later step gathers its input from
    /// the prior step's device-side argmax. Because the gather and argmax are
    /// byte-exact with the per-token path, the returned ids are identical to
    /// calling `advance` `steps` times and taking each host argmax.
    pub(crate) fn advance_chain(
        &mut self,
        cache: &mut MetalStepKvCache,
        seed: u32,
        steps: usize,
    ) -> Result<Vec<u32>> {
        ensure!(steps > 0, "chain step count must be positive");
        ensure!(
            cache.position + steps <= self.bucket,
            "chained decode exceeds cache capacity"
        );
        let (rope_cos, rope_sin) = self.rope_chain(cache.position, steps);
        let mut token_ids = vec![0u32; steps];
        let status = unsafe {
            synapse_qwen3_metal_step_chain(
                self.raw.as_ptr(),
                cache.position as u64,
                steps as u32,
                seed,
                rope_cos.as_ptr(),
                rope_sin.as_ptr(),
                token_ids.as_mut_ptr(),
                self.model.config.rms_norm_eps,
            )
        };
        if status != 0 {
            bail!(
                "Qwen3 Metal step chain failed with status {status}: {}",
                last_error()
            );
        }
        cache.position += steps;
        Ok(token_ids)
    }

    fn timings(&self) -> DecodeStageTimings {
        let mut native = NativeTimings::default();
        unsafe { synapse_qwen3_metal_step_timings(self.raw.as_ptr(), &mut native) };
        let prefill = self.prefill.stage_timings();
        DecodeStageTimings {
            graph_prepare_wall_s: prefill.graph_prepare_wall_s,
            host_prepare_wall_s: self.host_prepare_wall_s + prefill.host_prepare_wall_s,
            feed_wall_s: prefill.feed_wall_s + native.feed_wall_s,
            execute_wall_s: prefill.execute_wall_s + native.execute_wall_s,
            logits_readback_wall_s: prefill.logits_readback_wall_s + native.logits_readback_wall_s,
            kv_update_wall_s: prefill.kv_update_wall_s + native.kv_update_wall_s,
            kernel_gpu: DecodeKernelTimings {
                rmsnorm_s: native.kernel_rmsnorm_s,
                qkv_matvec_s: native.kernel_qkv_matvec_s,
                qk_norm_rope_s: native.kernel_qk_norm_rope_s,
                attention_s: native.kernel_attention_s,
                o_proj_s: native.kernel_o_proj_s,
                residual_rmsnorm_s: native.kernel_residual_rmsnorm_s,
                down_proj_s: native.kernel_down_proj_s,
                gate_up_swiglu_s: native.kernel_gate_up_swiglu_s,
                lm_head_s: native.kernel_lm_head_s,
                samples: native.kernel_samples,
            },
            prefill_calls: prefill.prefill_calls,
            step_calls: native.step_calls,
        }
    }
}

impl DecodeKernel for MetalStepDecoder<'_> {
    type Cache = MetalStepKvCache;

    fn capacity(&self) -> usize {
        self.bucket
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, Vec<f32>)> {
        ensure!(!tokens.is_empty(), "decode prompt must not be empty");
        ensure!(
            tokens.len() <= self.bucket,
            "decode prompt exceeds cache bucket"
        );
        let (mps_cache, logits) = self.prefill.prefill(tokens)?;
        let one_layer_elements =
            2 * self.model.config.num_key_value_heads * self.bucket * self.model.config.head_dim;
        let mut cache_bits = Vec::with_capacity(self.model.layers.len() * one_layer_elements);
        for layer in 0..self.model.layers.len() {
            cache_bits.extend(self.prefill.inspect_cache_bits(layer)?);
        }
        let status = unsafe {
            synapse_qwen3_metal_step_import_caches(
                self.raw.as_ptr(),
                cache_bits.as_ptr(),
                cache_bits.len() as u64,
            )
        };
        if status != 0 {
            bail!(
                "Qwen3 Metal step KV handoff failed with status {status}: {}",
                last_error()
            );
        }
        Ok((
            MetalStepKvCache {
                position: mps_cache.position,
            },
            logits,
        ))
    }

    fn advance(&mut self, cache: &mut Self::Cache, token: u32) -> Result<Vec<f32>> {
        ensure!(
            cache.position < self.bucket,
            "decode cache capacity exhausted"
        );
        let input = encode_f16_bits(self.embedding(token)?);
        let (rope_cos, rope_sin) = self.rope(cache.position);
        let mut logits = vec![0.0f32; self.model.config.vocab_size];
        let status = unsafe {
            synapse_qwen3_metal_step(
                self.raw.as_ptr(),
                cache.position as u64,
                input.as_ptr(),
                rope_cos.as_ptr(),
                rope_sin.as_ptr(),
                logits.as_mut_ptr(),
                self.model.config.rms_norm_eps,
            )
        };
        if status != 0 {
            bail!(
                "Qwen3 Metal step failed with status {status}: {}",
                last_error()
            );
        }
        cache.position += 1;
        Ok(logits)
    }

    fn cache_position(&self, cache: &Self::Cache) -> usize {
        cache.position
    }

    fn verify_tokens(&mut self, cache: &mut Self::Cache, tokens: &[u32]) -> Result<Vec<u32>> {
        if self.batched_verify {
            MetalStepDecoder::verify_batch(self, cache, tokens)
        } else {
            MetalStepDecoder::verify_tokens(self, cache, tokens)
        }
    }

    fn rewind(&mut self, cache: &mut Self::Cache, position: usize) -> Result<()> {
        ensure!(
            position <= cache.position,
            "cannot rewind Metal step cache forward from {} to {position}",
            cache.position
        );
        // Key/value data is addressed by [layer, head, position, dimension].
        // Attention reads only positions <= cache.position, RoPE is recomputed
        // from that position, and every activation/argmax scratch buffer is
        // overwritten by the next command buffer. No auxiliary decode state
        // advances with a chain, so changing this logical bound is sufficient.
        cache.position = position;
        Ok(())
    }

    fn chain_span(&self) -> usize {
        self.chain_k
    }

    fn advance_chain(
        &mut self,
        cache: &mut Self::Cache,
        seed: u32,
        steps: usize,
    ) -> Result<Vec<u32>> {
        MetalStepDecoder::advance_chain(self, cache, seed, steps)
    }

    fn inspect_cache_layer(&self, _cache: &Self::Cache, layer: usize) -> Result<Vec<f32>> {
        ensure!(
            layer < self.model.layers.len(),
            "KV cache layer {layer} out of range"
        );
        let elements =
            2 * self.model.config.num_key_value_heads * self.bucket * self.model.config.head_dim;
        let mut bits = vec![0u16; elements];
        let status = unsafe {
            synapse_qwen3_metal_step_cache_copy(
                self.raw.as_ptr(),
                layer as u64,
                bits.as_mut_ptr(),
                elements as u64,
            )
        };
        if status != 0 {
            bail!(
                "Qwen3 Metal step cache inspection failed with status {status}: {}",
                last_error()
            );
        }
        Ok(bits
            .into_iter()
            .map(half::f16::from_bits)
            .map(f32::from)
            .collect())
    }

    fn stage_timings(&self) -> DecodeStageTimings {
        self.timings()
    }
}

impl DecodeRuntime for MetalStepDecoder<'_> {
    fn lane(&self) -> &'static str {
        "owned-rt-metal-step"
    }

    fn kv_update_path(&self) -> &'static str {
        "metal-step-private-in-slot-f16-kv-cache"
    }

    fn weight_feed_path(&self) -> &'static str {
        match self.model.weight_quantization {
            WeightQuantization::None => "metal-step-persistent-f16-matvec",
            WeightQuantization::Q8_0 => "metal-step-persistent-q8_0-fused-dequant-matvec",
        }
    }

    fn optimization_level(&self) -> u8 {
        1
    }
}

impl Drop for MetalStepDecoder<'_> {
    fn drop(&mut self) {
        unsafe {
            synapse_qwen3_metal_step_context_free(self.raw.as_ptr());
        }
    }
}

fn metal_step_library_path() -> Result<PathBuf> {
    let executable = std::env::current_exe().context("locate spike executable")?;
    let beside_executable = executable
        .parent()
        .context("spike executable has no parent directory")?
        .join("qwen3_decode_metal_step.metallib");
    if beside_executable.is_file() {
        return Ok(beside_executable);
    }
    let build_path = Path::new(env!("SYNAPSE_UNIFIED_RT_METAL_STEP_LIB"));
    ensure!(
        build_path.is_file(),
        "Metal step metallib is missing beside {} and at {}",
        executable.display(),
        build_path.display()
    );
    Ok(build_path.to_owned())
}

/// Chain span from SYNAPSE_METAL_STEP_CHAIN_K (default 1: the fully
/// instrumented per-token path, byte-identical to the pre-wave-6 tree and to
/// the pinned campaign baseline). Chaining is opt-in because its measured M1
/// win (+2.58% Q8 at k=16) is below the 3% ship bar there, while faster
/// machines (M5 indicative +29.7%) benefit substantially: the right span is a
/// per-machine serving decision, not a build default. Values are clamped
/// to at least 1 so an out-of-range or unparseable setting degrades to the
/// fully instrumented per-token path rather than failing.
fn read_chain_k() -> usize {
    std::env::var("SYNAPSE_METAL_STEP_CHAIN_K")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|value| value.max(1))
        .unwrap_or(1)
}

/// Batched speculative verification is opt-in via
/// SYNAPSE_METAL_STEP_BATCHED_VERIFY=1. Off by default so the speculative path
/// keeps its prior sequential single-token verification shape; the batched path
/// is byte-identical (it only shares the per-layer weight read across the K
/// proposed positions), so enabling it changes latency, not output.
fn read_batched_verify() -> bool {
    std::env::var("SYNAPSE_METAL_STEP_BATCHED_VERIFY")
        .ok()
        .map(|value| value.trim() == "1")
        .unwrap_or(false)
}

fn last_error() -> anyhow::Error {
    unsafe {
        let raw = synapse_qwen3_metal_step_last_error();
        if raw.is_null() {
            anyhow::anyhow!("unknown Qwen3 Metal step error")
        } else {
            anyhow::anyhow!(CStr::from_ptr(raw).to_string_lossy().into_owned())
        }
    }
}

unsafe extern "C" {
    fn synapse_qwen3_metal_step_context_new(
        bucket: u64,
        hidden: u64,
        query_heads: u64,
        kv_heads: u64,
        head_dim: u64,
        intermediate: u64,
        vocab: u64,
        epsilon: f32,
        metallib_path: *const c_char,
    ) -> *mut c_void;
    fn synapse_qwen3_metal_step_context_free(context: *mut c_void);
    fn synapse_qwen3_metal_step_prepare(
        context: *mut c_void,
        layer_count: u64,
        quantized: u32,
        params: *const StepLayerParams,
        final_norm_weight: *const c_void,
        lm_head_weight: *const c_void,
        lm_head_q8: *const u8,
        embeddings: *const c_void,
    ) -> i32;
    fn synapse_qwen3_metal_step_verify(
        context: *mut c_void,
        position: u64,
        token_ids: *const u32,
        steps: u32,
        rope_cos: *const u16,
        rope_sin: *const u16,
        argmaxes_out: *mut u32,
        epsilon: f32,
    ) -> i32;
    fn synapse_qwen3_metal_step_verify_batch(
        context: *mut c_void,
        position: u64,
        token_ids: *const u32,
        steps: u32,
        rope_cos: *const u16,
        rope_sin: *const u16,
        argmaxes_out: *mut u32,
        logits_out: *mut f32,
        epsilon: f32,
    ) -> i32;
    fn synapse_qwen3_metal_step_chain(
        context: *mut c_void,
        position: u64,
        steps: u32,
        token_in_first: u32,
        rope_cos: *const u16,
        rope_sin: *const u16,
        token_ids_out: *mut u32,
        epsilon: f32,
    ) -> i32;
    fn synapse_qwen3_metal_step_import_caches(
        context: *mut c_void,
        cache_data: *const u16,
        cache_data_elements: u64,
    ) -> i32;
    fn synapse_qwen3_metal_step(
        context: *mut c_void,
        position: u64,
        input: *const u16,
        rope_cos: *const u16,
        rope_sin: *const u16,
        logits: *mut f32,
        epsilon: f32,
    ) -> i32;
    fn synapse_qwen3_metal_step_timings(context: *mut c_void, timings: *mut NativeTimings);
    fn synapse_qwen3_metal_step_cache_copy(
        context: *mut c_void,
        layer: u64,
        output: *mut u16,
        elements: u64,
    ) -> i32;
    fn synapse_qwen3_metal_step_last_error() -> *const c_char;
}

#[cfg(test)]
mod tests {
    //! Real-model gates for the batched verification path. These require a Metal
    //! GPU and the Qwen3-0.6B weights, so they are `#[ignore]` and run explicitly:
    //!
    //! ```text
    //! SYNAPSE_UNIFIED_RT_QWEN3_0_6B=<snapshot dir> \
    //!   cargo test -p spike-unified-rt --release batched_verify -- --ignored --nocapture
    //! ```
    //!
    //! The central invariant is machine-independent: batched verification must
    //! produce logits bit-identical to K sequential single-token steps, because
    //! batching only shares the per-layer weight read across the K positions and
    //! never reorders one dot product's accumulation. These tests prove that on
    //! whatever GPU runs them. Performance is measured separately on the project's
    //! M1 reference machine (the decode timing authority documented in
    //! METAL-STEP.md), which also settles the completion-06 fixture canary that
    //! drifts on other machines' Metal compilers.

    use super::{MetalStepDecoder, MetalStepKvCache};
    use crate::quant::WeightQuantization;
    use crate::qwen3_decode::DecodeKernel;
    use crate::{Execution, MetalExecutionConfig, Precision};

    const BUCKET: usize = 1024;

    fn model_path() -> std::path::PathBuf {
        std::path::PathBuf::from(
            std::env::var_os("SYNAPSE_UNIFIED_RT_QWEN3_0_6B")
                .expect("set SYNAPSE_UNIFIED_RT_QWEN3_0_6B to the Qwen3-0.6B snapshot directory"),
        )
    }

    /// Build a Metal step decoder over the real model and hand it to `body`.
    /// The model outlives the decoder borrow within the closure scope.
    fn with_decoder<R>(
        weight_quant: WeightQuantization,
        body: impl FnOnce(&crate::qwen3::Model, &mut MetalStepDecoder) -> R,
    ) -> R {
        let model =
            crate::qwen3::Model::load_with_quant(&model_path(), Precision::F16, weight_quant)
                .expect("load Qwen3-0.6B");
        let execution = MetalExecutionConfig {
            execution: Execution::Explicit,
            package_root: None,
        };
        let mut decoder =
            MetalStepDecoder::new(&model, Precision::F16, &execution, BUCKET, weight_quant)
                .expect("construct Metal step decoder");
        body(&model, &mut decoder)
    }

    /// Host greedy argmax matching the sampler rule: highest logit, lowest id on
    /// tie (replace only on strictly greater under total_cmp).
    fn greedy_argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for (id, value) in logits.iter().enumerate().skip(1) {
            if value.total_cmp(&logits[best]) == std::cmp::Ordering::Greater {
                best = id;
            }
        }
        best as u32
    }

    fn logits_bits(logits: &[f32]) -> Vec<u32> {
        logits.iter().map(|value| value.to_bits()).collect()
    }

    /// Deterministic synthetic prompt of a given length. The exact tokens do not
    /// matter because the batched and sequential paths must produce identical
    /// logits for ANY input; varied lengths exercise short, medium, and deep-
    /// context attention prefixes.
    fn synthetic_prompt(length: usize) -> Vec<u32> {
        (0..length)
            .map(|index| (1000 + index * 7919 % 5000) as u32)
            .collect()
    }

    /// Greedy continuation tokens and the per-position logits that produced them,
    /// via sequential single-token advances starting from `cache` (already
    /// prefilled). Returns (tokens, logits) where logits[i] is the logits after
    /// feeding tokens[i]; the seed logits (before any feed) are returned too.
    fn sequential_greedy(
        decoder: &mut MetalStepDecoder,
        cache: &mut MetalStepKvCache,
        seed_logits: &[f32],
        count: usize,
    ) -> (Vec<u32>, Vec<Vec<f32>>) {
        let mut tokens = Vec::with_capacity(count);
        let mut logits = Vec::with_capacity(count);
        let mut next = greedy_argmax(seed_logits);
        for _ in 0..count {
            tokens.push(next);
            let step_logits = decoder.advance(cache, next).expect("sequential advance");
            next = greedy_argmax(&step_logits);
            logits.push(step_logits);
        }
        (tokens, logits)
    }

    fn byte_identical_gate_for(weight_quant: WeightQuantization) {
        with_decoder(weight_quant, |_model, decoder| {
            for prompt_len in [1usize, 5, 33, 128, 469] {
                let prompt = synthetic_prompt(prompt_len);
                let (cache, seed_logits) = decoder.prefill(&prompt).expect("prefill");
                let mut cache = cache;
                let base_position = decoder.cache_position(&cache);
                assert_eq!(base_position, prompt_len);

                // Sequential reference: greedy draft tokens and their logits.
                let (draft, seq_logits) = sequential_greedy(decoder, &mut cache, &seed_logits, 16);

                for k in [1usize, 2, 4, 8, 16] {
                    let draft = &draft[..k];
                    // Rewind to the prefix and run one batched forward over K tokens.
                    decoder.rewind(&mut cache, base_position).expect("rewind");
                    let batch_logits = decoder
                        .verify_batch_logits(&mut cache, draft)
                        .expect("batched verify logits");
                    let vocab = seq_logits[0].len();
                    assert_eq!(batch_logits.len(), k * vocab);
                    for i in 0..k {
                        let batch_row = &batch_logits[i * vocab..(i + 1) * vocab];
                        assert_eq!(
                            logits_bits(batch_row),
                            logits_bits(&seq_logits[i]),
                            "batched logits diverge from sequential at prompt_len={prompt_len} k={k} position {i} ({weight_quant:?})"
                        );
                    }
                    // Argmax surface agrees too (this is what the session consumes).
                    decoder.rewind(&mut cache, base_position).expect("rewind");
                    let batch_argmaxes = decoder
                        .verify_batch(&mut cache, draft)
                        .expect("batched verify argmaxes");
                    for i in 0..k {
                        assert_eq!(
                            batch_argmaxes[i],
                            greedy_argmax(&seq_logits[i]),
                            "batched argmax diverges at prompt_len={prompt_len} k={k} position {i}"
                        );
                    }
                }
            }
        });
    }

    #[test]
    #[ignore]
    fn batched_verify_logits_are_byte_identical_to_sequential_f16() {
        byte_identical_gate_for(WeightQuantization::None);
    }

    #[test]
    #[ignore]
    fn batched_verify_logits_are_byte_identical_to_sequential_q8() {
        byte_identical_gate_for(WeightQuantization::Q8_0);
    }

    fn determinism_gate_for(weight_quant: WeightQuantization) {
        with_decoder(weight_quant, |_model, decoder| {
            let prompt = synthetic_prompt(64);
            let (cache, seed_logits) = decoder.prefill(&prompt).expect("prefill");
            let mut cache = cache;
            let base_position = decoder.cache_position(&cache);
            let (draft, _) = sequential_greedy(decoder, &mut cache, &seed_logits, 8);

            decoder.rewind(&mut cache, base_position).expect("rewind");
            let first = decoder
                .verify_batch_logits(&mut cache, &draft)
                .expect("first batched run");
            decoder.rewind(&mut cache, base_position).expect("rewind");
            let second = decoder
                .verify_batch_logits(&mut cache, &draft)
                .expect("second batched run");
            assert_eq!(
                logits_bits(&first),
                logits_bits(&second),
                "batched verification is not deterministic ({weight_quant:?})"
            );
        });
    }

    #[test]
    #[ignore]
    fn batched_verify_is_deterministic_f16() {
        determinism_gate_for(WeightQuantization::None);
    }

    #[test]
    #[ignore]
    fn batched_verify_is_deterministic_q8() {
        determinism_gate_for(WeightQuantization::Q8_0);
    }

    /// Forced-rejection rollback: verify a K-token draft whose token at index
    /// `wrong` is corrupted, accept the correct prefix, rewind to it, and confirm
    /// the greedy continuation is byte-exact with the target-only stream. Run for
    /// every rejection position so each KV slot in the batch window is exercised
    /// as the rollback boundary.
    fn forced_rejection_gate_for(weight_quant: WeightQuantization) {
        with_decoder(weight_quant, |_model, decoder| {
            let prompt = synthetic_prompt(48);
            let (cache, seed_logits) = decoder.prefill(&prompt).expect("prefill");
            let mut cache = cache;
            let base_position = decoder.cache_position(&cache);
            // Target-only greedy reference, long enough to cover the continuation.
            let (target, _) = sequential_greedy(decoder, &mut cache, &seed_logits, 32);
            let vocab = {
                decoder.rewind(&mut cache, base_position).expect("rewind");
                let probe = decoder.advance(&mut cache, target[0]).expect("probe");
                decoder.rewind(&mut cache, base_position).expect("rewind");
                probe.len()
            };

            for k in [4usize, 8] {
                for wrong in 0..k {
                    // Corrupt one draft token; the prefix before it stays correct.
                    let mut draft = target[..k].to_vec();
                    draft[wrong] = (target[wrong] + 1) % vocab as u32;

                    decoder.rewind(&mut cache, base_position).expect("rewind");
                    decoder
                        .verify_batch(&mut cache, &draft)
                        .expect("batched verify");
                    // Accept the `wrong` correct tokens, discard the rest.
                    decoder
                        .rewind(&mut cache, base_position + wrong)
                        .expect("rewind to acceptance boundary");
                    // Re-advance the correct token and follow greedy; every step
                    // must match the target-only continuation.
                    let mut next = target[wrong];
                    for step in 0..8 {
                        let logits = decoder.advance(&mut cache, next).expect("continue");
                        let argmax = greedy_argmax(&logits);
                        assert_eq!(
                            argmax,
                            target[wrong + 1 + step],
                            "continuation diverged after rejection: k={k} wrong={wrong} step={step} ({weight_quant:?})"
                        );
                        next = argmax;
                    }
                }
            }
        });
    }

    #[test]
    #[ignore]
    fn batched_verify_forced_rejection_preserves_continuation_f16() {
        forced_rejection_gate_for(WeightQuantization::None);
    }

    #[test]
    #[ignore]
    fn batched_verify_forced_rejection_preserves_continuation_q8() {
        forced_rejection_gate_for(WeightQuantization::Q8_0);
    }

    /// Per-token batched-verify cost curve. Prints the median wall time per
    /// verify_batch(K) call and the per-token (wall/K) figure for K in
    /// {1,2,4,8,16}. The authoritative numbers are taken on the project's M1
    /// reference machine (the decode timing authority; see METAL-STEP.md and
    /// BATCHED-VERIFY.md); on any other GPU this still works as a functional
    /// timing harness. Select weights with SYNAPSE_METAL_STEP_BATCHED_PROBE_QUANT
    /// =f16|q8 (default q8).
    fn timing_probe(weight_quant: WeightQuantization) {
        with_decoder(weight_quant, |_model, decoder| {
            let prompt = synthetic_prompt(64);
            let (cache, seed_logits) = decoder.prefill(&prompt).expect("prefill");
            let mut cache = cache;
            let base_position = decoder.cache_position(&cache);
            let (draft, _) = sequential_greedy(decoder, &mut cache, &seed_logits, 16);

            // Warmup so GPU clocks and pipeline caches are steady before timing.
            for _ in 0..5 {
                decoder.rewind(&mut cache, base_position).expect("rewind");
                decoder
                    .verify_batch(&mut cache, &draft[..8])
                    .expect("warmup");
            }

            println!("BATCHED_VERIFY_PROBE quant={weight_quant:?} prompt_len={base_position}");
            // Single-token reference: sequential greedy `advance` (the unchanged
            // per-token decode path). Running it in this same harness and build
            // gives a direct baseline beside the batched numbers and confirms the
            // additive batched path does not perturb the existing per-token path
            // (it reproduces the documented f16 decode rate; see BATCHED-VERIFY.md).
            {
                let steps = 64;
                let iterations = 8;
                let mut samples = Vec::with_capacity(iterations);
                for _ in 0..iterations {
                    decoder.rewind(&mut cache, base_position).expect("rewind");
                    let mut next = greedy_argmax(&seed_logits);
                    let started = std::time::Instant::now();
                    for _ in 0..steps {
                        let logits = decoder.advance(&mut cache, next).expect("advance");
                        next = greedy_argmax(&logits);
                    }
                    samples.push(started.elapsed().as_secs_f64() / steps as f64);
                }
                samples.sort_by(|a, b| a.total_cmp(b));
                let median = samples[iterations / 2];
                println!(
                    "SINGLE_TOKEN_REFERENCE per_token_ms={:.4} decode_tok_per_s={:.2}",
                    median * 1e3,
                    1.0 / median
                );
            }
            for k in [1usize, 2, 4, 8, 16] {
                let draft = &draft[..k];
                let iterations = 40;
                let mut samples = Vec::with_capacity(iterations);
                for _ in 0..iterations {
                    decoder.rewind(&mut cache, base_position).expect("rewind");
                    let started = std::time::Instant::now();
                    decoder
                        .verify_batch(&mut cache, draft)
                        .expect("verify_batch");
                    samples.push(started.elapsed().as_secs_f64());
                }
                samples.sort_by(|a, b| a.total_cmp(b));
                let median = samples[iterations / 2];
                println!(
                    "BATCHED_VERIFY_PROBE k={k:>2} median_call_ms={:.4} per_token_ms={:.4} verify_tok_per_s={:.2}",
                    median * 1e3,
                    median * 1e3 / k as f64,
                    k as f64 / median
                );
            }
        });
    }

    #[test]
    #[ignore]
    fn batched_verify_timing_probe() {
        let quant = match std::env::var("SYNAPSE_METAL_STEP_BATCHED_PROBE_QUANT")
            .ok()
            .as_deref()
        {
            Some("f16") => WeightQuantization::None,
            _ => WeightQuantization::Q8_0,
        };
        timing_probe(quant);
    }
}
