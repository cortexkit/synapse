//! Qwen3-0.6B Metal step decode engine — production-owned port.
//!
//! Ported from the proven `bench/spikes/unified-rt/src/qwen3_decode_metal_step.rs`
//! spike engine. The spike used an MPSGraph prefill decoder (`MetalDecoder`) to
//! prime the KV cache, then handed off to the Metal step kernels for token
//! stepping. The production port replaces that with device-resident causal
//! prefill via the step engine's own verify path (token-by-token on device),
//! eliminating the deprecated MPSGraph decode dependency entirely. The spec
//! requires: "Per-token decode does not use MPSGraph. Production owns causal
//! prefill and Metal token stepping."
//!
//! The Metal kernels (`.metal`), the Objective-C driver (`.m`), and the FFI
//! binding are byte-identical to the spike so the pinned fixture batteries
//! reproduce exactly. The only change is the prefill strategy: the step
//! engine's `verify` path feeds prompt tokens one-by-one through the same
//! device-resident forward pass, producing the same KV cache state and the
//! same greedy argmax after the final prompt token.

#![cfg(target_os = "macos")]

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::PathBuf;
use std::ptr::NonNull;

use anyhow::{bail, ensure, Context, Result};

use super::decode_kernel::{DecodeKernel, DecodeRuntime};
use super::quant::WeightQuantization;
use super::qwen3_decode_model::Model;
use crate::runtime::{encode_f16_bits, Precision};

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

pub struct MetalStepKvCache {
    pub position: usize,
}

/// Production-owned Qwen3 Metal step decoder.
///
/// Drives causal prefill and greedy token stepping through the device-resident
/// Metal step kernels. Prefill uses the step engine's verify path
/// (token-by-token on device) instead of the deprecated MPSGraph prefill.
pub struct MetalStepDecoder<'a> {
    raw: NonNull<c_void>,
    model: &'a Model,
    bucket: usize,
    // Chained-decode span. k=1 preserves the fully instrumented per-token path
    // byte-for-byte; k>1 encodes k forward passes plus on-GPU argmax into one
    // command buffer with a single readback. Production baseline is K=1
    // (chain_span=1); chaining is opt-in and OFF for the certified baseline.
    chain_k: usize,
}

impl<'a> MetalStepDecoder<'a> {
    pub fn new(
        model: &'a Model,
        precision: Precision,
        bucket: usize,
        weight_quantization: WeightQuantization,
    ) -> Result<Self> {
        ensure!(
            matches!(precision, Precision::F16),
            "Metal step activations require f16"
        );
        ensure!(
            [512, 1024, 2048].contains(&bucket),
            "decode cache bucket must be 512, 1024, or 2048"
        );
        ensure!(
            model.weight_quantization == weight_quantization,
            "Metal step weight quantization does not match the loaded model"
        );

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
            raw,
            model,
            bucket,
            chain_k: read_chain_k(),
        };
        let params = decoder.layer_params()?;
        let final_norm = decoder.model.final_norm.weight.metal_f16_bits()?;
        let lm_head = decoder.model.lm_head()?.metal_f16_bits()?;
        // The chained-decode embedding gather reads this resident f16 table.
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

    pub fn import_caches(&self, cache_bits: &[u16]) -> Result<()> {
        let status = unsafe {
            synapse_qwen3_metal_step_import_caches(
                self.raw.as_ptr(),
                cache_bits.as_ptr(),
                cache_bits.len() as u64,
            )
        };
        if status != 0 {
            bail!(
                "Qwen3 Metal step KV import failed with status {status}: {}",
                last_error()
            );
        }
        Ok(())
    }

    pub fn inspect_cache_bits(&self, layer: usize) -> Result<Vec<u16>> {
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
                "Qwen3 Metal step cache inspection failed: status {status}: {}",
                last_error()
            );
        }
        Ok(bits)
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

    /// Verifies a proposed token span on device, returning the greedy argmax
    /// after each token. This is the device-resident causal prefill path: prompt
    /// tokens are fed one-by-one through the same forward pass, advancing the KV
    /// cache to the prompt length and producing the first generated token's
    /// argmax after the final prompt token.
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

    /// Encode `steps` chained decode passes into one command buffer and return
    /// the `steps` argmax token ids. `seed` is the token whose embedding feeds
    /// step 0; each later step gathers its input from the prior step's
    /// device-side argmax.
    pub fn advance_chain(
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
}

impl DecodeKernel for MetalStepDecoder<'_> {
    type Cache = MetalStepKvCache;

    fn capacity(&self) -> usize {
        self.bucket
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, u32)> {
        ensure!(!tokens.is_empty(), "decode prompt must not be empty");
        ensure!(
            tokens.len() <= self.bucket,
            "decode prompt exceeds cache bucket"
        );
        // Device-resident causal prefill: feed prompt tokens through the step
        // engine's verify path, which advances the KV cache on device and
        // returns the greedy argmax after each token. The argmax after the
        // final prompt token is the first generated token. The verify path
        // computes argmaxes on device and never materializes a host-visible
        // logits vector, so the contract returns the token id directly.
        let mut cache = MetalStepKvCache { position: 0 };
        let argmaxes = self.verify_tokens(&mut cache, tokens)?;
        let first_token = *argmaxes.last().expect("non-empty prompt");
        Ok((cache, first_token))
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
        MetalStepDecoder::verify_tokens(self, cache, tokens)
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
                "Qwen3 Metal step cache inspection failed: status {status}: {}",
                last_error()
            );
        }
        Ok(bits
            .into_iter()
            .map(half::f16::from_bits)
            .map(f32::from)
            .collect())
    }
}

impl DecodeRuntime for MetalStepDecoder<'_> {
    fn lane(&self) -> &'static str {
        "owned-metal-decode"
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
    let executable = std::env::current_exe().context("locate engine executable")?;
    let beside_executable = executable
        .parent()
        .context("engine executable has no parent directory")?
        .join("qwen3_decode_metal_step.metallib");
    if beside_executable.is_file() {
        return Ok(beside_executable);
    }
    let build_path = PathBuf::from(env!("SYNAPSE_OWNED_DECODE_QWEN3_STEP_LIB"));
    ensure!(
        build_path.is_file(),
        "Metal step metallib is missing beside {} and at {}",
        executable.display(),
        build_path.display()
    );
    Ok(build_path)
}

/// Chain span from SYNAPSE_METAL_STEP_CHAIN_K (default 1: the fully
/// instrumented per-token path, byte-identical to the pinned campaign
/// baseline). Chaining is opt-in; production baseline is K=1.
fn read_chain_k() -> usize {
    std::env::var("SYNAPSE_METAL_STEP_CHAIN_K")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .map(|value| value.max(1))
        .unwrap_or(1)
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
    fn synapse_qwen3_metal_step_cache_copy(
        context: *mut c_void,
        layer: u64,
        output: *mut u16,
        elements: u64,
    ) -> i32;
    fn synapse_qwen3_metal_step_last_error() -> *const c_char;
}
