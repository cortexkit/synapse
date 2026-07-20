#![cfg(target_os = "macos")]

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};

use super::{DecodeKernel, DecodeRuntime, DecodeStageTimings, MetalDecoder, Model};
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
        };
        let params = decoder.layer_params()?;
        let final_norm = decoder.model.final_norm.weight.metal_f16_bits()?;
        let lm_head = decoder.model.lm_head()?.metal_f16_bits()?;
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
