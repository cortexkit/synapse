//! LFM2 Metal decode-step driver.
//!
//! This module binds the LFM2-specific step kernels in
//! `lfm2_decode_metal_step.metal` (compiled to a metallib by `build.rs`) and
//! drives them from safe Rust. It is the LFM2 analogue of
//! `qwen3_decode_metal_step.rs`: the attention layers will reuse the proven
//! Qwen3 step kernels, while the short-convolution layers use the new
//! device-resident conv-cache step kernel exposed here.
//!
//! Scope of this increment: the convolution step kernel and its rolling-cache
//! model, proven bit-exact and deterministic against the `lfm2.rs` CPU reference
//! (`decode_conv`) on real LFM2-1.2B dimensions. End-to-end hybrid orchestration
//! (chaining the conv step with the reused attention/matvec/RMSNorm kernels into
//! a full token-exact decoder), the Q8 path, the 20x64 sha256 fixtures, and the
//! authoritative M1 timing are tracked as follow-ups in LFM2-METAL-STEP.md.
#![cfg(target_os = "macos")]
// The convolution step engine is not yet wired into the binary decode path
// (end-to-end hybrid orchestration is a follow-up); its only consumer right now
// is the exactness gate below. Silence dead-code in non-test builds so the FFI
// still compiles and links without warnings until the integration lands.
#![cfg_attr(not(test), allow(dead_code))]

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use anyhow::{bail, Context, Result};

/// Locate the compiled LFM2 step metallib. Prefer the copy that `build.rs`
/// places beside the executable (so a relocatable run does not depend on the
/// build-script path), falling back to the absolute path baked in at compile
/// time. Mirrors the Qwen3 step library lookup.
fn metal_step_library_path() -> Result<PathBuf> {
    let executable = std::env::current_exe().context("locate spike executable")?;
    let beside_executable = executable
        .parent()
        .context("spike executable has no parent directory")?
        .join("lfm2_decode_metal_step.metallib");
    if beside_executable.is_file() {
        return Ok(beside_executable);
    }
    let build_path = Path::new(env!("SYNAPSE_UNIFIED_RT_LFM2_STEP_LIB"));
    anyhow::ensure!(
        build_path.is_file(),
        "LFM2 Metal step metallib is missing beside {} and at {}",
        executable.display(),
        build_path.display()
    );
    Ok(build_path.to_owned())
}

fn last_error() -> anyhow::Error {
    unsafe {
        let raw = synapse_lfm2_metal_step_last_error();
        if raw.is_null() {
            anyhow::anyhow!("unknown LFM2 Metal step error")
        } else {
            anyhow::anyhow!(CStr::from_ptr(raw).to_string_lossy().into_owned())
        }
    }
}

/// Safe owner of the native LFM2 step context and its per-layer conv caches.
///
/// Each convolution layer keeps a device-resident rolling cache that a step
/// advances in place; the caches live for the lifetime of this engine, exactly
/// like the KV caches of the attention step.
pub(crate) struct Lfm2ConvStepEngine {
    raw: NonNull<c_void>,
    hidden: usize,
    kernel_size: usize,
}

impl Lfm2ConvStepEngine {
    /// Create the Metal context and upload one static depthwise conv-weight
    /// table per convolution layer. `conv_weights[layer]` holds
    /// `hidden * kernel_size` f32 taps in channel-major order
    /// (index = channel * kernel_size + tap), matching `lfm2.rs` storage.
    pub(crate) fn new(hidden: usize, kernel_size: usize, conv_weights: &[&[f32]]) -> Result<Self> {
        let library_path = metal_step_library_path()?;
        let library = CString::new(library_path.to_string_lossy().as_bytes())?;
        let raw = NonNull::new(unsafe {
            synapse_lfm2_metal_step_context_new(hidden as u64, kernel_size as u64, library.as_ptr())
        })
        .ok_or_else(last_error)?;
        let mut engine = Self {
            raw,
            hidden,
            kernel_size,
        };
        let weight_pointers: Vec<*const f32> =
            conv_weights.iter().map(|weight| weight.as_ptr()).collect();
        let status = unsafe {
            synapse_lfm2_metal_step_prepare(
                engine.raw.as_ptr(),
                weight_pointers.len() as u64,
                weight_pointers.as_ptr(),
            )
        };
        if status != 0 {
            let error = last_error();
            engine.release();
            return Err(error)
                .with_context(|| format!("LFM2 Metal step prepare failed ({status})"));
        }
        Ok(engine)
    }

    /// Run one convolution decode step for `layer`: advance its device-resident
    /// cache in place and return `out[hidden] = gate * conv(newest position)`.
    pub(crate) fn step(&mut self, layer: usize, product: &[f32], gate: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(product.len() == self.hidden, "product width mismatch");
        anyhow::ensure!(gate.len() == self.hidden, "gate width mismatch");
        let mut out = vec![0.0f32; self.hidden];
        let status = unsafe {
            synapse_lfm2_conv_step(
                self.raw.as_ptr(),
                layer as u64,
                product.as_ptr(),
                gate.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        if status != 0 {
            bail!("LFM2 conv step failed ({status}): {}", last_error());
        }
        Ok(out)
    }

    /// Read a layer's current device-resident rolling cache back to the host.
    pub(crate) fn read_cache(&self, layer: usize) -> Result<Vec<f32>> {
        let mut host = vec![0.0f32; self.kernel_size * self.hidden];
        let status = unsafe {
            synapse_lfm2_conv_cache_read(self.raw.as_ptr(), layer as u64, host.as_mut_ptr())
        };
        if status != 0 {
            bail!("LFM2 conv cache read failed ({status}): {}", last_error());
        }
        Ok(host)
    }

    /// Overwrite a layer's device-resident rolling cache from the host. Used to
    /// seed tests and as the hook a future rewind/rollback would use.
    pub(crate) fn write_cache(&mut self, layer: usize, host: &[f32]) -> Result<()> {
        anyhow::ensure!(
            host.len() == self.kernel_size * self.hidden,
            "conv cache length mismatch"
        );
        let status = unsafe {
            synapse_lfm2_conv_cache_write(self.raw.as_ptr(), layer as u64, host.as_ptr())
        };
        if status != 0 {
            bail!("LFM2 conv cache write failed ({status}): {}", last_error());
        }
        Ok(())
    }

    fn release(&mut self) {
        unsafe { synapse_lfm2_metal_step_context_free(self.raw.as_ptr()) };
    }
}

impl Drop for Lfm2ConvStepEngine {
    fn drop(&mut self) {
        self.release();
    }
}

unsafe extern "C" {
    fn synapse_lfm2_metal_step_context_new(
        hidden: u64,
        kernel_size: u64,
        metallib_path: *const c_char,
    ) -> *mut c_void;
    fn synapse_lfm2_metal_step_prepare(
        context: *mut c_void,
        conv_layer_count: u64,
        conv_weights: *const *const f32,
    ) -> i32;
    fn synapse_lfm2_conv_step(
        context: *mut c_void,
        layer: u64,
        product: *const f32,
        gate: *const f32,
        out: *mut f32,
    ) -> i32;
    fn synapse_lfm2_conv_cache_read(context: *mut c_void, layer: u64, host: *mut f32) -> i32;
    fn synapse_lfm2_conv_cache_write(context: *mut c_void, layer: u64, host: *const f32) -> i32;
    fn synapse_lfm2_metal_step_context_free(context: *mut c_void);
    fn synapse_lfm2_metal_step_last_error() -> *const c_char;
}

// ===========================================================================
// Hybrid decode-step engine (stage C).
//
// Safe Rust owner of the hybrid native context. It extracts the f16 weights
// (and the f32 conv taps) from a loaded `lfm2.rs` model, uploads them once, and
// drives the device-resident hybrid forward: token-by-token prefill via the
// explicit-token verify path, then fast chained greedy decode with on-GPU
// argmax. RoPE tables use LFM2's rope_theta (1e6), regenerated per position --
// not Qwen3's theta. This is the engine the 20x64 token-exactness gate drives.
// ===========================================================================

use crate::encode_f16_bits;
use crate::lfm2::{Mixer, Model, Weight};
use crate::quant::Q8_0Tensor;

/// The Q8_0 block bytes of a weight as a raw pointer, or null when the weight
/// is not quantized. The bytes are owned by the loaded model and stay alive
/// across the synchronous native prepare upload.
fn q8_ptr(weight: &Weight) -> *const c_void {
    weight
        .q8_0
        .as_ref()
        .map_or(std::ptr::null(), |quantized| quantized.as_bytes().as_ptr())
        .cast()
}

#[repr(C)]
struct Lfm2HybridLayerParams {
    operator_norm: *const c_void,
    ffn_norm: *const c_void,
    gate_weight: *const c_void,
    gate_weight_q8: *const c_void,
    up_weight: *const c_void,
    up_weight_q8: *const c_void,
    down_weight: *const c_void,
    down_weight_q8: *const c_void,
    in_proj_weight: *const c_void,
    in_proj_weight_q8: *const c_void,
    conv_weight: *const c_void,
    out_proj_weight: *const c_void,
    out_proj_weight_q8: *const c_void,
    q_weight: *const c_void,
    q_weight_q8: *const c_void,
    k_weight: *const c_void,
    k_weight_q8: *const c_void,
    v_weight: *const c_void,
    v_weight_q8: *const c_void,
    o_weight: *const c_void,
    o_weight_q8: *const c_void,
    q_norm: *const c_void,
    k_norm: *const c_void,
    is_attention: u64,
}

unsafe extern "C" {
    fn synapse_lfm2_hybrid_step_context_new(
        bucket: u64,
        hidden: u64,
        query_heads: u64,
        kv_heads: u64,
        head_dim: u64,
        intermediate: u64,
        vocab: u64,
        kernel_size: u64,
        epsilon: f32,
        metallib_path: *const c_char,
    ) -> *mut c_void;
    fn synapse_lfm2_hybrid_step_prepare(
        context: *mut c_void,
        layer_count: u64,
        quantized: u32,
        params: *const Lfm2HybridLayerParams,
        final_norm_weight: *const c_void,
        lm_head_weight: *const c_void,
        lm_head_q8: *const c_void,
        embeddings: *const c_void,
    ) -> i32;
    fn synapse_lfm2_hybrid_step_chain(
        context: *mut c_void,
        position: u64,
        steps: u32,
        token_in_first: u32,
        rope_cos: *const u16,
        rope_sin: *const u16,
        token_ids_out: *mut u32,
        epsilon: f32,
    ) -> i32;
    fn synapse_lfm2_hybrid_step_verify(
        context: *mut c_void,
        position: u64,
        token_ids: *const u32,
        steps: u32,
        rope_cos: *const u16,
        rope_sin: *const u16,
        argmaxes_out: *mut u32,
        epsilon: f32,
    ) -> i32;
    fn synapse_lfm2_hybrid_step(
        context: *mut c_void,
        position: u64,
        input: *const u16,
        rope_cos: *const u16,
        rope_sin: *const u16,
        logits: *mut f32,
        epsilon: f32,
    ) -> i32;
    fn synapse_lfm2_hybrid_step_reset(context: *mut c_void) -> i32;
    fn synapse_lfm2_hybrid_step_context_free(context: *mut c_void);
}

/// Owned f16 mirrors of one layer's weights, kept alive across the synchronous
/// GPU upload in [`Lfm2HybridStepEngine::new`]. Pointers handed to the native
/// prepare call reference these buffers; the call copies them into private GPU
/// storage and waits, so the mirrors can drop once prepare returns.
struct HybridLayerWeights {
    operator_norm: Vec<u16>,
    ffn_norm: Vec<u16>,
    gate: Vec<u16>,
    up: Vec<u16>,
    down: Vec<u16>,
    in_proj: Vec<u16>,
    out_proj: Vec<u16>,
    q: Vec<u16>,
    k: Vec<u16>,
    v: Vec<u16>,
    o: Vec<u16>,
    q_norm: Vec<u16>,
    k_norm: Vec<u16>,
}

pub(crate) struct Lfm2HybridStepEngine {
    raw: NonNull<c_void>,
    hidden: usize,
    head_dim: usize,
    vocab: usize,
    bucket: usize,
    rope_theta: f32,
    epsilon: f32,
}

impl Lfm2HybridStepEngine {
    pub(crate) fn new(model: &Model, bucket: usize, quantized: bool) -> Result<Self> {
        let config = &model.config;
        let hidden = config.hidden_size;
        let head_dim = config.head_dim;
        let library_path = metal_step_library_path()?;
        let library = CString::new(library_path.to_string_lossy().as_bytes())?;
        let raw = NonNull::new(unsafe {
            synapse_lfm2_hybrid_step_context_new(
                bucket as u64,
                hidden as u64,
                config.num_attention_heads as u64,
                config.num_key_value_heads as u64,
                head_dim as u64,
                config.intermediate_size as u64,
                config.vocab_size as u64,
                config.conv_kernel_size as u64,
                config.rms_norm_eps,
                library.as_ptr(),
            )
        })
        .ok_or_else(last_error)?;
        let mut engine = Self {
            raw,
            hidden,
            head_dim,
            vocab: config.vocab_size,
            bucket,
            rope_theta: config.rope_theta,
            epsilon: config.rms_norm_eps,
        };

        // Build owned f16 mirrors for the weights the engine stores as f16 in
        // BOTH modes (the two per-layer norms and, for attention layers, the q/k
        // head norms), plus -- only in f16 mode -- the matmul weights. In Q8 mode
        // the matmul weights are handed to the native prepare as the model's
        // Q8_0 block bytes (referenced directly below), so their f16 mirrors stay
        // empty. Unused per-layer-type fields stay null/empty; the native prepare
        // only dereferences the fields matching each layer's mixer.
        let mut weights: Vec<HybridLayerWeights> = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let mut holder = HybridLayerWeights {
                operator_norm: encode_f16_bits(&layer.operator_norm.weight.data),
                ffn_norm: encode_f16_bits(&layer.ffn_norm.weight.data),
                gate: Vec::new(),
                up: Vec::new(),
                down: Vec::new(),
                in_proj: Vec::new(),
                out_proj: Vec::new(),
                q: Vec::new(),
                k: Vec::new(),
                v: Vec::new(),
                o: Vec::new(),
                q_norm: Vec::new(),
                k_norm: Vec::new(),
            };
            if !quantized {
                holder.gate = encode_f16_bits(&layer.w1.tensor.data);
                holder.up = encode_f16_bits(&layer.w3.tensor.data);
                holder.down = encode_f16_bits(&layer.w2.tensor.data);
            }
            match &layer.mixer {
                Mixer::Conv(conv) => {
                    if !quantized {
                        holder.in_proj = encode_f16_bits(&conv.in_proj.tensor.data);
                        holder.out_proj = encode_f16_bits(&conv.out_proj.tensor.data);
                    }
                }
                Mixer::Attention(attn) => {
                    holder.q_norm = encode_f16_bits(&attn.q_norm.weight.data);
                    holder.k_norm = encode_f16_bits(&attn.k_norm.weight.data);
                    if !quantized {
                        holder.q = encode_f16_bits(&attn.q_proj.tensor.data);
                        holder.k = encode_f16_bits(&attn.k_proj.tensor.data);
                        holder.v = encode_f16_bits(&attn.v_proj.tensor.data);
                        holder.o = encode_f16_bits(&attn.out_proj.tensor.data);
                    }
                }
            }
            weights.push(holder);
        }
        let null = std::ptr::null();
        let params: Vec<Lfm2HybridLayerParams> = model
            .layers
            .iter()
            .zip(&weights)
            .map(|(layer, holder)| {
                let is_attention = matches!(layer.mixer, Mixer::Attention(_));
                // Conv projections (conv layers) or nulls (attention layers). The
                // f16 pointer is the owned mirror (f16 mode) and the q8 pointer is
                // the model's block bytes (Q8 mode); the unused slot is null/empty.
                let (in_proj_f16, in_proj_q8, conv_weight, out_proj_f16, out_proj_q8) =
                    match &layer.mixer {
                        Mixer::Conv(conv) => (
                            holder.in_proj.as_ptr().cast(),
                            q8_ptr(&conv.in_proj),
                            conv.conv_weight.data.as_ptr().cast(),
                            holder.out_proj.as_ptr().cast(),
                            q8_ptr(&conv.out_proj),
                        ),
                        Mixer::Attention(_) => (null, null, null, null, null),
                    };
                let (q_f16, q_q8, k_f16, k_q8, v_f16, v_q8, o_f16, o_q8, q_norm, k_norm) =
                    match &layer.mixer {
                        Mixer::Attention(attn) => (
                            holder.q.as_ptr().cast(),
                            q8_ptr(&attn.q_proj),
                            holder.k.as_ptr().cast(),
                            q8_ptr(&attn.k_proj),
                            holder.v.as_ptr().cast(),
                            q8_ptr(&attn.v_proj),
                            holder.o.as_ptr().cast(),
                            q8_ptr(&attn.out_proj),
                            holder.q_norm.as_ptr().cast(),
                            holder.k_norm.as_ptr().cast(),
                        ),
                        Mixer::Conv(_) => (null, null, null, null, null, null, null, null, null, null),
                    };
                Lfm2HybridLayerParams {
                    operator_norm: holder.operator_norm.as_ptr().cast(),
                    ffn_norm: holder.ffn_norm.as_ptr().cast(),
                    gate_weight: holder.gate.as_ptr().cast(),
                    gate_weight_q8: q8_ptr(&layer.w1),
                    up_weight: holder.up.as_ptr().cast(),
                    up_weight_q8: q8_ptr(&layer.w3),
                    down_weight: holder.down.as_ptr().cast(),
                    down_weight_q8: q8_ptr(&layer.w2),
                    in_proj_weight: in_proj_f16,
                    in_proj_weight_q8: in_proj_q8,
                    conv_weight,
                    out_proj_weight: out_proj_f16,
                    out_proj_weight_q8: out_proj_q8,
                    q_weight: q_f16,
                    q_weight_q8: q_q8,
                    k_weight: k_f16,
                    k_weight_q8: k_q8,
                    v_weight: v_f16,
                    v_weight_q8: v_q8,
                    o_weight: o_f16,
                    o_weight_q8: o_q8,
                    q_norm,
                    k_norm,
                    is_attention: u64::from(is_attention),
                }
            })
            .collect();
        let final_norm = encode_f16_bits(&model.final_norm.weight.data);
        // Tied embeddings: when there is no separate LM head the head weight is
        // the embedding table itself (LFM2-1.2B ties them). The f16 engine feeds
        // the f16 embedding bits as the head; the Q8 engine quantizes the table
        // separately for head use while the gather table stays f16. Mirrors
        // Model::lm_head / lm_head_q8_0, which are private to lfm2.rs.
        let lm_head_data = match &model.lm_head {
            Some(head) => &head.tensor.data,
            None => &model.embeddings.data,
        };
        let embeddings = encode_f16_bits(&model.embeddings.data);
        let lm_head_f16;
        let mut tied_lm_head_q8: Option<Q8_0Tensor> = None;
        let (lm_head_fp16_ptr, lm_head_q8_ptr): (*const c_void, *const c_void) = if quantized {
            let q8: *const c_void = match &model.lm_head {
                Some(head) => q8_ptr(head),
                None => {
                    let quantized_head = Q8_0Tensor::quantize(lm_head_data, hidden)?;
                    let ptr = quantized_head.as_bytes().as_ptr().cast();
                    // Moving the tensor into the owner keeps its heap buffer (and
                    // therefore `ptr`) alive across the synchronous upload below.
                    tied_lm_head_q8 = Some(quantized_head);
                    ptr
                }
            };
            (null, q8)
        } else {
            lm_head_f16 = encode_f16_bits(lm_head_data);
            (lm_head_f16.as_ptr().cast(), null)
        };
        let status = unsafe {
            synapse_lfm2_hybrid_step_prepare(
                engine.raw.as_ptr(),
                params.len() as u64,
                u32::from(quantized),
                params.as_ptr(),
                final_norm.as_ptr().cast(),
                lm_head_fp16_ptr,
                lm_head_q8_ptr,
                embeddings.as_ptr().cast(),
            )
        };
        if status != 0 {
            let error = last_error();
            engine.release();
            return Err(error)
                .with_context(|| format!("LFM2 hybrid step prepare failed ({status})"));
        }
        // Keep the mirrors and the tied Q8 head alive until after the synchronous
        // upload above.
        drop(weights);
        drop(tied_lm_head_q8);
        Ok(engine)
    }

    /// RoPE cos/sin for one position, encoded to f16 bits. Uses LFM2's
    /// rope_theta and the half-split pair layout the qk_norm_rope kernel reads
    /// (index i in [0, head_dim), the kernel consumes [0, head_dim/2)).
    fn rope(&self, position: usize) -> (Vec<u16>, Vec<u16>) {
        let head_dim = self.head_dim;
        let mut cosine = Vec::with_capacity(head_dim);
        let mut sine = Vec::with_capacity(head_dim);
        for index in 0..head_dim {
            let rotary_index = index % (head_dim / 2);
            let frequency = 1.0
                / self
                    .rope_theta
                    .powf((2 * rotary_index) as f32 / head_dim as f32);
            let (sin, cos) = (position as f32 * frequency).sin_cos();
            cosine.push(cos);
            sine.push(sin);
        }
        (encode_f16_bits(&cosine), encode_f16_bits(&sine))
    }

    /// Concatenated rope tables for `steps` consecutive positions from `start`.
    fn rope_chain(&self, start: usize, steps: usize) -> (Vec<u16>, Vec<u16>) {
        let head_dim = self.head_dim;
        let mut cosine = Vec::with_capacity(head_dim * steps);
        let mut sine = Vec::with_capacity(head_dim * steps);
        for step in 0..steps {
            let (cos_bits, sin_bits) = self.rope(start + step);
            cosine.extend_from_slice(&cos_bits);
            sine.extend_from_slice(&sin_bits);
        }
        (cosine, sine)
    }

    /// Zero every layer's conv and KV caches (start-of-sequence state).
    pub(crate) fn reset(&mut self) -> Result<()> {
        let status = unsafe { synapse_lfm2_hybrid_step_reset(self.raw.as_ptr()) };
        if status != 0 {
            bail!("LFM2 hybrid step reset failed ({status}): {}", last_error());
        }
        Ok(())
    }

    /// Prefill a prompt token-by-token on device, returning the greedy argmax
    /// after the final prompt token (the first generated token). Advances all
    /// caches to `prompt.len()`. Mirrors lfm2.rs::decode_token over the prompt.
    pub(crate) fn prefill(&mut self, prompt: &[u32]) -> Result<u32> {
        anyhow::ensure!(!prompt.is_empty(), "LFM2 hybrid prefill needs a prompt");
        anyhow::ensure!(
            prompt.len() <= self.bucket,
            "prompt longer than the decode bucket"
        );
        let (cos, sin) = self.rope_chain(0, prompt.len());
        let mut argmaxes = vec![0u32; prompt.len()];
        let status = unsafe {
            synapse_lfm2_hybrid_step_verify(
                self.raw.as_ptr(),
                0,
                prompt.as_ptr(),
                prompt.len() as u32,
                cos.as_ptr(),
                sin.as_ptr(),
                argmaxes.as_mut_ptr(),
                self.epsilon,
            )
        };
        if status != 0 {
            bail!("LFM2 hybrid prefill failed ({status}): {}", last_error());
        }
        Ok(*argmaxes.last().expect("non-empty prompt"))
    }

    /// Chained greedy generation: feed `first_token` and decode `steps` tokens
    /// on device with on-GPU argmax, starting at `position`. Returns the `steps`
    /// tokens produced AFTER `first_token` (i.e. token k is the argmax obtained
    /// once token k-1 has been decoded).
    pub(crate) fn chain(
        &mut self,
        position: usize,
        steps: usize,
        first_token: u32,
    ) -> Result<Vec<u32>> {
        if steps == 0 {
            return Ok(Vec::new());
        }
        anyhow::ensure!(
            position + steps <= self.bucket,
            "decode would overrun the bucket"
        );
        let (cos, sin) = self.rope_chain(position, steps);
        let mut tokens = vec![0u32; steps];
        let status = unsafe {
            synapse_lfm2_hybrid_step_chain(
                self.raw.as_ptr(),
                position as u64,
                steps as u32,
                first_token,
                cos.as_ptr(),
                sin.as_ptr(),
                tokens.as_mut_ptr(),
                self.epsilon,
            )
        };
        if status != 0 {
            bail!("LFM2 hybrid chain failed ({status}): {}", last_error());
        }
        Ok(tokens)
    }

    /// Single host-fed forward pass returning full f32 logits (layer-parity
    /// probe building block). `input` is the f16 embedding row for the token.
    #[allow(dead_code)]
    pub(crate) fn step_logits(&mut self, position: usize, input: &[u16]) -> Result<Vec<f32>> {
        anyhow::ensure!(input.len() == self.hidden, "input width mismatch");
        let (cos, sin) = self.rope(position);
        let mut logits = vec![0.0f32; self.vocab];
        let status = unsafe {
            synapse_lfm2_hybrid_step(
                self.raw.as_ptr(),
                position as u64,
                input.as_ptr(),
                cos.as_ptr(),
                sin.as_ptr(),
                logits.as_mut_ptr(),
                self.epsilon,
            )
        };
        if status != 0 {
            bail!("LFM2 hybrid step failed ({status}): {}", last_error());
        }
        Ok(logits)
    }

    fn release(&mut self) {
        unsafe { synapse_lfm2_hybrid_step_context_free(self.raw.as_ptr()) };
    }
}

impl Drop for Lfm2HybridStepEngine {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::Lfm2ConvStepEngine;
    use super::Lfm2HybridStepEngine;
    use crate::{CpuProvider, KernelProvider};

    /// Real LFM2-1.2B convolution dimensions: hidden_size = 2048 and
    /// conv_L_cache = 3 (see config.json). The exactness gate runs at these dims
    /// so the proven kernel covers the production shapes, not a toy.
    const HIDDEN: usize = 2048;
    const KERNEL_SIZE: usize = 3;

    /// Deterministic LCG so the gate is reproducible without a fixture file.
    /// Values stay in a modest range so the 3-tap convolution cannot overflow
    /// f32 or produce non-finite results.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            // Numerical Recipes LCG constants; full 64-bit period.
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((self.0 >> 33) as f32) / ((1u64 << 31) as f32); // [0, 1)
            (unit - 0.5) * 4.0 // [-2, 2)
        }
        fn fill(&mut self, count: usize) -> Vec<f32> {
            (0..count).map(|_| self.next_f32()).collect()
        }
    }

    /// CPU reference for one convolution decode step, mirroring `lfm2.rs`
    /// `decode_conv` exactly: advance the rolling state, run the CPU provider's
    /// depthwise causal conv over the whole window, then gate the newest row.
    /// The heavy convolution math is the real `CpuProvider` reduction; only the
    /// three cache-management lines are reproduced verbatim from `decode_conv`.
    fn cpu_conv_step(
        cpu: &mut CpuProvider,
        state: &mut [f32],
        product: &[f32],
        gate: &[f32],
        conv_weight: &[f32],
    ) -> Vec<f32> {
        // decode_conv: state.copy_within(hidden.., 0)
        state.copy_within(HIDDEN.., 0);
        // decode_conv: state[(kernel_size - 1) * hidden..].copy_from_slice(&product)
        state[(KERNEL_SIZE - 1) * HIDDEN..].copy_from_slice(product);
        let mut convolved = vec![0.0f32; state.len()];
        cpu.depthwise_causal_conv1d(
            state,
            1,
            KERNEL_SIZE,
            HIDDEN,
            conv_weight,
            KERNEL_SIZE,
            &mut convolved,
        )
        .expect("cpu depthwise conv");
        let last = (KERNEL_SIZE - 1) * HIDDEN;
        let mut out = vec![0.0f32; HIDDEN];
        // decode_conv: gate[channel] *= convolved_state[last + channel]
        for channel in 0..HIDDEN {
            out[channel] = gate[channel] * convolved[last + channel];
        }
        out
    }

    fn f32_bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// Exactness contract: the Metal conv-step kernel must be
    /// bit-identical to the `lfm2.rs` CPU reference at every step, and the
    /// device-resident rolling cache must match the CPU state after the run.
    #[test]
    fn conv_step_kernel_is_bit_exact_vs_cpu_reference() {
        let mut rng = Rng(0x5eed_1f2a);
        let conv_weight = rng.fill(HIDDEN * KERNEL_SIZE);
        let mut cpu = CpuProvider::platform_for_test();
        let mut engine = Lfm2ConvStepEngine::new(HIDDEN, KERNEL_SIZE, &[conv_weight.as_slice()])
            .expect("engine");

        let mut cpu_state = vec![0.0f32; KERNEL_SIZE * HIDDEN];
        let steps = 16; // well past the 3-row window so rolling is exercised.
        for step in 0..steps {
            let product = rng.fill(HIDDEN);
            let gate = rng.fill(HIDDEN);
            let expected = cpu_conv_step(&mut cpu, &mut cpu_state, &product, &gate, &conv_weight);
            let actual = engine.step(0, &product, &gate).expect("metal step");
            assert_eq!(
                f32_bits(&actual),
                f32_bits(&expected),
                "conv step {step} diverged from the CPU reference"
            );
        }
        let metal_cache = engine.read_cache(0).expect("read cache");
        assert_eq!(
            f32_bits(&metal_cache),
            f32_bits(&cpu_state),
            "device-resident conv cache diverged from the CPU rolling state"
        );
    }

    /// Determinism gate: two independent runs over the same inputs must produce
    /// byte-identical output streams (and caches). This is the two-runs-identical
    /// requirement applied to the conv step.
    #[test]
    fn conv_step_kernel_is_deterministic() {
        let mut setup = Rng(0xcafe_f00d);
        let conv_weight = setup.fill(HIDDEN * KERNEL_SIZE);
        // Pre-generate the shared input stream so both runs see identical data.
        let steps = 12;
        let mut products = Vec::with_capacity(steps);
        let mut gates = Vec::with_capacity(steps);
        for _ in 0..steps {
            products.push(setup.fill(HIDDEN));
            gates.push(setup.fill(HIDDEN));
        }

        let run = || {
            let mut engine =
                Lfm2ConvStepEngine::new(HIDDEN, KERNEL_SIZE, &[conv_weight.as_slice()])
                    .expect("engine");
            let mut outs = Vec::with_capacity(steps);
            for step in 0..steps {
                outs.push(
                    engine
                        .step(0, &products[step], &gates[step])
                        .expect("metal step"),
                );
            }
            let cache = engine.read_cache(0).expect("read cache");
            (outs, cache)
        };

        let (first_outs, first_cache) = run();
        let (second_outs, second_cache) = run();
        for step in 0..steps {
            assert_eq!(
                f32_bits(&first_outs[step]),
                f32_bits(&second_outs[step]),
                "conv step {step} is not deterministic across runs"
            );
        }
        assert_eq!(
            f32_bits(&first_cache),
            f32_bits(&second_cache),
            "conv cache is not deterministic across runs"
        );
    }

    /// The cache-write hook seeds a window correctly: writing a known state and
    /// reading it back must round-trip bit-exactly. This is the primitive a
    /// future rewind/rollback builds on.
    #[test]
    fn conv_cache_write_round_trips() {
        let mut rng = Rng(0x1234_5678);
        let conv_weight = rng.fill(HIDDEN * KERNEL_SIZE);
        let mut engine = Lfm2ConvStepEngine::new(HIDDEN, KERNEL_SIZE, &[conv_weight.as_slice()])
            .expect("engine");
        let seeded = rng.fill(KERNEL_SIZE * HIDDEN);
        engine.write_cache(0, &seeded).expect("write cache");
        let read_back = engine.read_cache(0).expect("read cache");
        assert_eq!(
            f32_bits(&read_back),
            f32_bits(&seeded),
            "conv cache round-trip mismatch"
        );
    }

    /// Real-checkpoint gate: the actual LFM2-1.2B convolution weights flow through
    /// the step kernel bit-exactly. It loads the production snapshot, takes a real
    /// conv layer's depthwise weights, and runs the kernel against the lfm2.rs CPU
    /// reference with deterministic synthetic activations. This validates the
    /// real-weight path (and the model-load plumbing) that the end-to-end decoder
    /// builds on. Gated on the checkpoint like the Qwen3 real-model gates, so the
    /// default suite stays checkpoint-free.
    #[test]
    #[ignore]
    fn conv_step_matches_cpu_on_real_lfm2_1_2b_conv_weights() {
        use crate::lfm2::{Mixer, Model};
        use crate::Precision;
        let path = std::path::PathBuf::from(
            std::env::var_os("SYNAPSE_UNIFIED_RT_LFM2_1_2B")
                .expect("set SYNAPSE_UNIFIED_RT_LFM2_1_2B to the LFM2-1.2B snapshot directory"),
        );
        let model = Model::load(&path, Precision::F16).expect("load LFM2-1.2B");
        assert_eq!(
            model.config.hidden_size, HIDDEN,
            "real LFM2-1.2B hidden size"
        );
        let conv = model
            .layers
            .iter()
            .find_map(|layer| match &layer.mixer {
                Mixer::Conv(conv) => Some(conv),
                _ => None,
            })
            .expect("LFM2-1.2B has a convolution layer");
        assert_eq!(
            conv.kernel_size, KERNEL_SIZE,
            "real LFM2-1.2B conv kernel size"
        );
        let conv_weight = conv.conv_weight.data.clone();
        assert_eq!(
            conv_weight.len(),
            HIDDEN * KERNEL_SIZE,
            "real conv weight length"
        );

        let mut rng = Rng(0x9e37_79b9);
        let mut cpu = CpuProvider::platform_for_test();
        let mut engine = Lfm2ConvStepEngine::new(HIDDEN, KERNEL_SIZE, &[conv_weight.as_slice()])
            .expect("engine");
        let mut cpu_state = vec![0.0f32; KERNEL_SIZE * HIDDEN];
        let steps = 8;
        for step in 0..steps {
            let product = rng.fill(HIDDEN);
            let gate = rng.fill(HIDDEN);
            let expected = cpu_conv_step(&mut cpu, &mut cpu_state, &product, &gate, &conv_weight);
            let actual = engine.step(0, &product, &gate).expect("metal step");
            assert_eq!(
                f32_bits(&actual),
                f32_bits(&expected),
                "real-weight conv step {step} diverged from the CPU reference"
            );
        }
        let metal_cache = engine.read_cache(0).expect("read cache");
        assert_eq!(
            f32_bits(&metal_cache),
            f32_bits(&cpu_state),
            "real-weight conv cache diverged from the CPU rolling state"
        );
    }

    // ---------------------------------------------------------------------
    // f16 rounding-policy probe (stage B).
    //
    // The Metal step engine stores every weight as IEEE f16 bits
    // (`encode_f16_bits`), while the `lfm2.rs` CPU reference decodes with the
    // checkpoint weights loaded as bf16->f32 (no f16 rounding; `load_with_quant`
    // ignores the precision argument). Before assembling the hybrid engine we
    // must settle which oracle the f16 engine is expected to match: does rounding
    // the weights to f16 change the greedy token sequence at all?
    //
    // This probe runs the CPU reference greedy decode twice over the pinned
    // twenty-prompt set -- once with the native loaded weights, once with every
    // weight replaced by its f16 round-trip -- and reports per-prompt
    // token-exactness. If the two agree on every prompt, the literal CPU
    // reference is a valid 20/20 oracle for the f16 engine; if they diverge, the
    // f16 engine's oracle must be the f16-weight CPU reference and the divergent
    // prompts are listed. The result is printed (run with `--nocapture`) and
    // recorded in LFM2-METAL-STEP.md; it is the f16 policy the end-to-end gate
    // builds on.
    // ---------------------------------------------------------------------

    use crate::lfm2::{Mixer, Model};
    use crate::qwen3_decode::top_logits;
    use crate::{decode_f16_bits, encode_f16_bits, Precision};
    use sha2::{Digest, Sha256};
    use std::collections::HashSet;

    /// Round one f32 buffer to IEEE f16 and back, in place. After this every
    /// value equals `decode_f16_bits(encode_f16_bits(value))`, i.e. the exact f32
    /// value the Metal engine obtains from its stored f16 weight bits.
    fn round_to_f16_in_place(data: &mut [f32]) {
        let bits = encode_f16_bits(data);
        let rounded = decode_f16_bits(&bits);
        data.copy_from_slice(&rounded);
    }

    /// Replace every weight in the model with its f16 round-trip so the CPU
    /// reference decodes from the same weight bits the Metal engine uses. Walks
    /// the full hybrid layout: embeddings (tied LM head), per-layer norms, the
    /// SwiGLU FFN weights, and the conv- or attention-mixer weights.
    fn round_model_weights_to_f16(model: &mut Model) {
        round_to_f16_in_place(&mut model.embeddings.data);
        round_to_f16_in_place(&mut model.final_norm.weight.data);
        if let Some(head) = model.lm_head.as_mut() {
            round_to_f16_in_place(&mut head.tensor.data);
        }
        for layer in &mut model.layers {
            round_to_f16_in_place(&mut layer.operator_norm.weight.data);
            round_to_f16_in_place(&mut layer.ffn_norm.weight.data);
            round_to_f16_in_place(&mut layer.w1.tensor.data);
            round_to_f16_in_place(&mut layer.w2.tensor.data);
            round_to_f16_in_place(&mut layer.w3.tensor.data);
            match &mut layer.mixer {
                Mixer::Conv(conv) => {
                    round_to_f16_in_place(&mut conv.in_proj.tensor.data);
                    round_to_f16_in_place(&mut conv.conv_weight.data);
                    round_to_f16_in_place(&mut conv.out_proj.tensor.data);
                }
                Mixer::Attention(attn) => {
                    round_to_f16_in_place(&mut attn.q_proj.tensor.data);
                    round_to_f16_in_place(&mut attn.q_norm.weight.data);
                    round_to_f16_in_place(&mut attn.k_proj.tensor.data);
                    round_to_f16_in_place(&mut attn.k_norm.weight.data);
                    round_to_f16_in_place(&mut attn.v_proj.tensor.data);
                    round_to_f16_in_place(&mut attn.out_proj.tensor.data);
                }
            }
        }
    }

    /// Greedy decode through the CPU reference (`Model::decode_token`), the
    /// decode contract the Metal step engine must match. Prefills the prompt
    /// token-by-token
    /// into a fresh cache, then emits up to `max_tokens` argmax tokens (highest
    /// logit, lowest id on tie via `top_logits`), stopping after a stop token is
    /// emitted -- the same rule as `lfm2_decode.rs::Decoder`.
    fn greedy_decode_cpu(
        model: &Model,
        provider: &mut dyn KernelProvider,
        prompt: &[u32],
        max_tokens: usize,
        stop_tokens: &HashSet<u32>,
        f16_activations: bool,
    ) -> Vec<u32> {
        let mut cache = model.empty_decode_cache(prompt.len() + max_tokens);
        let mut logits = Vec::new();
        for &token in prompt {
            let (_, next_logits) = if f16_activations {
                model
                    .decode_token_f16_activations(provider, &mut cache, token)
                    .expect("cpu prefill step (f16 activations)")
            } else {
                model
                    .decode_token(provider, &mut cache, token)
                    .expect("cpu prefill step")
            };
            logits = next_logits;
        }
        let mut generated = Vec::with_capacity(max_tokens);
        let mut next = top_logits(&logits, 1)[0].token_id;
        for _ in 0..max_tokens {
            generated.push(next);
            if stop_tokens.contains(&next) {
                break;
            }
            let (_, next_logits) = if f16_activations {
                model
                    .decode_token_f16_activations(provider, &mut cache, next)
                    .expect("cpu decode step (f16 activations)")
            } else {
                model
                    .decode_token(provider, &mut cache, next)
                    .expect("cpu decode step")
            };
            logits = next_logits;
            next = top_logits(&logits, 1)[0].token_id;
        }
        generated
    }

    /// The pinned twenty-prompt decode set (`decode-prompts.jsonl`), parsed to
    /// (id, prompt text) pairs in file order.
    fn decode_prompt_set() -> Vec<(String, String)> {
        let raw = include_str!("../decode-prompts.jsonl");
        raw.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let value: serde_json::Value =
                    serde_json::from_str(line).expect("decode-prompts.jsonl row parses");
                let id = value["id"].as_str().expect("prompt id").to_string();
                let prompt = value["prompt"].as_str().expect("prompt text").to_string();
                (id, prompt)
            })
            .collect()
    }

    /// Stable sha256 over a token fixture so a generated oracle can be pinned and
    /// later compared byte-for-byte. Serialises each prompt's tokens as a line of
    /// space-separated ids; the digest covers the whole set in file order.
    fn fixture_sha256(rows: &[(String, Vec<u32>)]) -> String {
        let mut digest = Sha256::new();
        for (id, tokens) in rows {
            digest.update(id.as_bytes());
            digest.update(b"\n");
            for token in tokens {
                digest.update(token.to_le_bytes());
            }
            digest.update(b"\n");
        }
        format!("{:x}", digest.finalize())
    }

    /// Pinned sha256 of the 20-prompt x 64-token greedy fixture generated from
    /// the native `lfm2.rs` CPU reference (bf16->f32 weights, one-thread
    /// deterministic gemm) on the LFM2-1.2B snapshot
    /// `933cee00d754fb3bfe06c644c0cb95453f2d8bb2`. The f16-weight CPU reference
    /// produces the byte-identical fixture (see the gate below), so this single
    /// digest is the oracle the Metal step engine must reproduce 20/20.
    const PINNED_DECODE_FIXTURE_SHA256: &str =
        "49ee80e8ba5d4940854fdbcd044406f5f3af4d5f6d35456eb247cfd506bd307b";

    /// Pinned sha256 of the 20-prompt x 64-token greedy fixture generated from
    /// the Q8-dequantized `lfm2.rs` CPU reference (the Q8 oracle) on the LFM2-1.2B
    /// snapshot `933cee00d754fb3bfe06c644c0cb95453f2d8bb2`. This -- not the f16
    /// fixture -- is the oracle the Q8 Metal step engine must reproduce within the
    /// certified near-tie band. The Q8 oracle differs from the f16 oracle on 7/20
    /// prompts (median match depth 61), the expected Q8 quantization drift.
    const PINNED_Q8_DECODE_FIXTURE_SHA256: &str =
        "1b0918c70a3b173275106a439aaf2dda3e74746d081a94e6e9b5c39e040a3cbf";

    /// f16 rounding-policy probe on the real LFM2-1.2B checkpoint. See the block
    /// comment above for what it settles. Run with:
    ///
    /// ```text
    /// SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    ///     f16_weight_rounding_policy -- --ignored --nocapture
    /// ```
    ///
    /// `LFM2_F16_PROBE_LIMIT` / `LFM2_F16_PROBE_MAX_TOKENS` cap the prompt count
    /// and tokens-per-prompt for a fast calibration run; unset, they default to
    /// the full 20x64 fixture set.
    #[test]
    #[ignore]
    fn f16_weight_rounding_policy_probe() {
        use tokenizers::Tokenizer;

        let path = std::path::PathBuf::from(
            std::env::var_os("SYNAPSE_UNIFIED_RT_LFM2_1_2B")
                .expect("set SYNAPSE_UNIFIED_RT_LFM2_1_2B to the LFM2-1.2B snapshot directory"),
        );
        let limit: usize = std::env::var("LFM2_F16_PROBE_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(usize::MAX);
        let max_tokens: usize = std::env::var("LFM2_F16_PROBE_MAX_TOKENS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(64);

        let mut tokenizer =
            Tokenizer::from_file(path.join("tokenizer.json")).expect("load tokenizer");
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).expect("disable truncation");

        let prompts = decode_prompt_set();
        let prompts: Vec<_> = prompts.into_iter().take(limit).collect();
        assert!(!prompts.is_empty(), "decode prompt set is empty");

        // Native CPU reference (bf16->f32 weights, the literal lfm2.rs contract).
        let native_model = Model::load(&path, Precision::F16).expect("load LFM2-1.2B");
        let stop_tokens: HashSet<u32> =
            native_model.generation_stop_ids().iter().copied().collect();
        let mut native_provider = CpuProvider::platform_for_test();
        let mut native_rows = Vec::new();
        let native_started = std::time::Instant::now();
        for (id, prompt) in &prompts {
            let prompt_ids = native_model
                .encode_generation(&tokenizer, prompt, 2048)
                .expect("encode prompt");
            let tokens = greedy_decode_cpu(
                &native_model,
                &mut native_provider,
                &prompt_ids,
                max_tokens,
                &stop_tokens,
                false,
            );
            println!("[native] {id}: {} tokens", tokens.len());
            native_rows.push((id.clone(), tokens));
        }
        let native_secs = native_started.elapsed().as_secs_f64();

        // f16-weight CPU reference (every weight replaced by its f16 round-trip).
        let mut f16_model = Model::load(&path, Precision::F16).expect("load LFM2-1.2B");
        round_model_weights_to_f16(&mut f16_model);
        let mut f16_provider = CpuProvider::platform_for_test();
        let mut f16_rows = Vec::new();
        let f16_started = std::time::Instant::now();
        for (id, prompt) in &prompts {
            let prompt_ids = f16_model
                .encode_generation(&tokenizer, prompt, 2048)
                .expect("encode prompt");
            let tokens = greedy_decode_cpu(
                &f16_model,
                &mut f16_provider,
                &prompt_ids,
                max_tokens,
                &stop_tokens,
                false,
            );
            f16_rows.push((id.clone(), tokens));
        }
        let f16_secs = f16_started.elapsed().as_secs_f64();

        // f16-activation CPU reference (native weights, activations rounded to f16
        // at every layer boundary). This emulates a step engine that reuses the
        // f16-activation Qwen3 kernels and measures whether that reuse can stay
        // token-exact against the f32 CPU reference.
        let mut f16act_provider = CpuProvider::platform_for_test();
        let mut f16act_rows = Vec::new();
        let f16act_started = std::time::Instant::now();
        for (id, prompt) in &prompts {
            let prompt_ids = native_model
                .encode_generation(&tokenizer, prompt, 2048)
                .expect("encode prompt");
            let tokens = greedy_decode_cpu(
                &native_model,
                &mut f16act_provider,
                &prompt_ids,
                max_tokens,
                &stop_tokens,
                true,
            );
            f16act_rows.push((id.clone(), tokens));
        }
        let f16act_secs = f16act_started.elapsed().as_secs_f64();

        // Compare the two CPU references prompt-by-prompt.
        let mut identical = 0usize;
        for ((id, native_tokens), (_, f16_tokens)) in native_rows.iter().zip(&f16_rows) {
            if native_tokens == f16_tokens {
                identical += 1;
            } else {
                let shared = native_tokens.len().min(f16_tokens.len());
                let first_diff = (0..shared).find(|&i| native_tokens[i] != f16_tokens[i]);
                println!(
                    "[policy] DIVERGENCE {id}: native {} tok, f16 {} tok, first diff at step {:?}",
                    native_tokens.len(),
                    f16_tokens.len(),
                    first_diff
                );
            }
        }

        println!("=== LFM2 f16 rounding-policy probe ===");
        println!("prompts: {}, max_tokens: {}", prompts.len(), max_tokens);
        println!(
            "native cpu decode: {:.1}s, f16-weight cpu decode: {:.1}s, f16-activation cpu decode: {:.1}s",
            native_secs, f16_secs, f16act_secs
        );
        let native_sha = fixture_sha256(&native_rows);
        let f16_sha = fixture_sha256(&f16_rows);
        let f16act_sha = fixture_sha256(&f16act_rows);
        println!("native fixture sha256: {native_sha}");
        println!("f16-weight fixture sha256: {f16_sha}");
        println!("f16-activation fixture sha256: {f16act_sha}");
        println!(
            "POLICY (weights): f16-weight CPU reference is token-identical to the native CPU reference on {}/{} prompts",
            identical,
            prompts.len()
        );
        let mut act_identical = 0usize;
        for ((id, native_tokens), (_, act_tokens)) in native_rows.iter().zip(&f16act_rows) {
            if native_tokens == act_tokens {
                act_identical += 1;
            } else {
                let shared = native_tokens.len().min(act_tokens.len());
                let first_diff = (0..shared).find(|&i| native_tokens[i] != act_tokens[i]);
                println!(
                    "[policy] ACTIVATION DIVERGENCE {id}: native {} tok, f16-act {} tok, first diff at step {:?}",
                    native_tokens.len(),
                    act_tokens.len(),
                    first_diff
                );
            }
        }
        println!(
            "POLICY (activations): f16-activation CPU reference is token-identical to the native CPU reference on {}/{} prompts",
            act_identical,
            prompts.len()
        );

        // Optional fixture cut: write the native CPU-reference tokens as JSONL so
        // the Metal step gate can load a pinned oracle instead of re-running the
        // (slow) CPU decode. Enable with LFM2_F16_FIXTURE_OUT=<path>.
        if let Some(out) = std::env::var_os("LFM2_F16_FIXTURE_OUT") {
            let mut body = String::new();
            for (id, tokens) in &native_rows {
                let tokens_json = serde_json::to_string(tokens).expect("serialize tokens");
                body.push_str(&format!(
                    "{{\"id\":{},\"tokens\":{tokens_json}}}\n",
                    serde_json::to_string(id).expect("serialize id")
                ));
            }
            std::fs::write(&out, body).expect("write fixture");
            println!("wrote fixture to {}", out.display());
        }

        // Gate assertions (only meaningful on the full 20x64 set; a calibration run
        // with a reduced limit/tokens skips the pin so it cannot false-fail).
        if limit >= prompts.len() && max_tokens == 64 {
            assert_eq!(
                identical,
                prompts.len(),
                "f16 weight rounding changed the greedy token sequence"
            );
            assert_eq!(
                native_sha, PINNED_DECODE_FIXTURE_SHA256,
                "native CPU-reference decode fixture drifted from the pinned oracle"
            );
            assert_eq!(
                f16_sha, PINNED_DECODE_FIXTURE_SHA256,
                "f16-weight CPU-reference decode fixture differs from the native oracle"
            );
        }
    }

    // ---------------------------------------------------------------------
    // Stage D: Q8_0 oracle and certification.
    //
    // The Q8 engine stores every dense decode matmul (attention Q/K/V/O, the
    // SwiGLU gate/up/down projections, the conv in_proj/out_proj, and the tied
    // LM head) as GGUF Q8_0 blocks, reusing the Qwen3 pack-4 GEMV kernels; the
    // norms, the f32 depthwise conv taps, and the f16 embedding gather table are
    // unchanged. Its oracle is the lfm2.rs CPU reference running the
    // Q8-dequantized weights -- NOT the f16 fixture -- cut with the same 20x64
    // protocol. Because the engine and the oracle share the exact same quantized
    // bytes, the large Q8 quantization error cancels in their comparison; what
    // remains is the f16-activation / matmul-rounding / transcendental gap the
    // f16 engine already certified at ~0.03 vocab-wide (stage C). The oracle
    // mirrors the engine's weight representation so that gap stays tight:
    // matmul weights are Q8-dequantized from the model's Q8_0 blocks, the norms
    // and the input embedding table are f16-rounded (the engine stores them f16),
    // and the tied LM head is Q8-dequantized separately from the f16 input table
    // (the engine gathers f16 input but multiplies a Q8 head).
    // ---------------------------------------------------------------------

    use crate::lfm2::Weight;
    use crate::quant::{Q8_0Tensor, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, WeightQuantization};
    use half::f16;

    /// Dequantize GGUF Q8_0 block bytes back to f32 (each 34-byte block is an f16
    /// scale followed by 32 i8 quants; value = quant * scale). This is the inverse
    /// of `Q8_0Tensor::quantize` and reproduces, on the CPU, the exact weight
    /// values the Metal Q8 GEMV kernels multiply against.
    fn dequantize_q8_0(bytes: &[u8]) -> Vec<f32> {
        let mut values =
            Vec::with_capacity(bytes.len() / Q8_0_BLOCK_BYTES * Q8_0_BLOCK_ELEMENTS);
        for block in bytes.chunks_exact(Q8_0_BLOCK_BYTES) {
            let scale = f32::from(f16::from_bits(u16::from_le_bytes([block[0], block[1]])));
            for quant in &block[2..] {
                values.push((*quant as i8) as f32 * scale);
            }
        }
        values
    }

    /// Replace a weight's f32 tensor data with the dequantized values of its Q8_0
    /// blocks, so the CPU reference matmul reads exactly the quantized weight the
    /// engine uses. No-op when the weight is not quantized.
    fn dequantize_weight_q8_in_place(weight: &mut Weight) {
        if let Some(quantized) = &weight.q8_0 {
            let values = dequantize_q8_0(quantized.as_bytes());
            weight.tensor.data.copy_from_slice(&values);
        }
    }

    /// Turn a Q8-loaded model into the Q8 CPU-reference oracle in place, returning
    /// the f16-rounded embedding table the decode uses for INPUT lookup. After
    /// this call every dense matmul weight holds its Q8-dequantized values, the
    /// norms are f16-rounded, and `model.embeddings.data` holds the Q8-dequantized
    /// tied LM head (what `Model::lm_head` reads for the head matmul). The
    /// returned input table is kept separate so the input gather stays f16 while
    /// the head is Q8 -- the same separation the engine makes.
    fn prepare_q8_oracle_model(model: &mut Model) -> Vec<f32> {
        let hidden = model.config.hidden_size;
        // f16 input embedding table (matches the engine's f16 gather table).
        let input_embeddings = decode_f16_bits(&encode_f16_bits(&model.embeddings.data));
        // Tied LM head: quantize the native embedding table to the same Q8_0 bytes
        // the engine uploads for the head, then dequantize them back into
        // embeddings.data for the CPU head matmul.
        let native_embeddings = model.embeddings.data.clone();
        let head_q8 =
            Q8_0Tensor::quantize(&native_embeddings, hidden).expect("quantize tied LM head");
        model
            .embeddings
            .data
            .copy_from_slice(&dequantize_q8_0(head_q8.as_bytes()));
        round_to_f16_in_place(&mut model.final_norm.weight.data);
        for layer in &mut model.layers {
            round_to_f16_in_place(&mut layer.operator_norm.weight.data);
            round_to_f16_in_place(&mut layer.ffn_norm.weight.data);
            dequantize_weight_q8_in_place(&mut layer.w1);
            dequantize_weight_q8_in_place(&mut layer.w2);
            dequantize_weight_q8_in_place(&mut layer.w3);
            match &mut layer.mixer {
                Mixer::Conv(conv) => {
                    dequantize_weight_q8_in_place(&mut conv.in_proj);
                    // conv_weight stays native f32 (the engine keeps the depthwise
                    // taps in f32, never quantized).
                    dequantize_weight_q8_in_place(&mut conv.out_proj);
                }
                Mixer::Attention(attn) => {
                    round_to_f16_in_place(&mut attn.q_norm.weight.data);
                    round_to_f16_in_place(&mut attn.k_norm.weight.data);
                    dequantize_weight_q8_in_place(&mut attn.q_proj);
                    dequantize_weight_q8_in_place(&mut attn.k_proj);
                    dequantize_weight_q8_in_place(&mut attn.v_proj);
                    dequantize_weight_q8_in_place(&mut attn.out_proj);
                }
            }
        }
        input_embeddings
    }

    /// Greedy decode through the Q8 CPU reference (`Model::decode_embedding` fed
    /// the f16 input table), the contract the Q8 Metal engine must match. Mirrors
    /// `greedy_decode_cpu` but looks the input embedding up in the separate f16
    /// table (the model's embeddings.data now holds the Q8 tied head).
    fn greedy_decode_q8_cpu(
        model: &Model,
        provider: &mut dyn KernelProvider,
        input_embeddings: &[f32],
        prompt: &[u32],
        max_tokens: usize,
        stop_tokens: &HashSet<u32>,
    ) -> Vec<u32> {
        let hidden = model.config.hidden_size;
        let mut cache = model.empty_decode_cache(prompt.len() + max_tokens);
        let mut logits = Vec::new();
        for &token in prompt {
            let start = token as usize * hidden;
            let (_, next_logits) = model
                .decode_embedding(provider, &mut cache, &input_embeddings[start..start + hidden])
                .expect("q8 cpu prefill step");
            logits = next_logits;
        }
        let mut generated = Vec::with_capacity(max_tokens);
        let mut next = top_logits(&logits, 1)[0].token_id;
        for _ in 0..max_tokens {
            generated.push(next);
            if stop_tokens.contains(&next) {
                break;
            }
            let start = next as usize * hidden;
            let (_, next_logits) = model
                .decode_embedding(provider, &mut cache, &input_embeddings[start..start + hidden])
                .expect("q8 cpu decode step");
            logits = next_logits;
            next = top_logits(&logits, 1)[0].token_id;
        }
        generated
    }

    /// Q8 oracle fixture probe on the real LFM2-1.2B checkpoint. Cuts the
    //  Q8-dequantized CPU-reference greedy fixture (20x64) and pins its sha256;
    /// this is the oracle the Q8 Metal engine is certified against. Run with:
    ///
    /// ```text
    /// SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    ///     q8_weight_oracle_probe -- --ignored --nocapture
    /// ```
    ///
    /// Write the fixture with `LFM2_Q8_FIXTURE_OUT=fixtures/lfm2-q8-step-reference.jsonl`.
    #[test]
    #[ignore]
    fn q8_weight_oracle_probe() {
        use tokenizers::Tokenizer;

        let path = std::path::PathBuf::from(
            std::env::var_os("SYNAPSE_UNIFIED_RT_LFM2_1_2B")
                .expect("set SYNAPSE_UNIFIED_RT_LFM2_1_2B to the LFM2-1.2B snapshot directory"),
        );
        let limit: usize = std::env::var("LFM2_Q8_PROBE_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(usize::MAX);
        let max_tokens: usize = std::env::var("LFM2_Q8_PROBE_MAX_TOKENS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(64);

        let mut tokenizer =
            Tokenizer::from_file(path.join("tokenizer.json")).expect("load tokenizer");
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).expect("disable truncation");

        let prompts = decode_prompt_set();
        let prompts: Vec<_> = prompts.into_iter().take(limit).collect();
        assert!(!prompts.is_empty(), "decode prompt set is empty");

        let mut model = Model::load_with_quant(&path, Precision::F16, WeightQuantization::Q8_0)
            .expect("load LFM2-1.2B (Q8_0)");
        let stop_tokens: HashSet<u32> = model.generation_stop_ids().iter().copied().collect();
        let input_embeddings = prepare_q8_oracle_model(&mut model);
        let mut provider = CpuProvider::platform_for_test();
        let mut rows = Vec::new();
        let started = std::time::Instant::now();
        for (id, prompt) in &prompts {
            let prompt_ids = model
                .encode_generation(&tokenizer, prompt, 2048)
                .expect("encode prompt");
            let tokens = greedy_decode_q8_cpu(
                &model,
                &mut provider,
                &input_embeddings,
                &prompt_ids,
                max_tokens,
                &stop_tokens,
            );
            println!("[q8-oracle] {id}: {} tokens", tokens.len());
            rows.push((id.clone(), tokens));
        }
        let secs = started.elapsed().as_secs_f64();
        let q8_sha = fixture_sha256(&rows);
        println!("=== LFM2 Q8 oracle probe ===");
        println!("prompts: {}, max_tokens: {}", prompts.len(), max_tokens);
        println!("q8 cpu decode: {secs:.1}s");
        println!("q8 fixture sha256: {q8_sha}");

        // Match-depth context vs the native f16 fixture (descriptive only; the Q8
        // engine's oracle is the Q8 fixture, not this).
        let native = pinned_fixture_rows();
        if native.len() == rows.len() {
            let mut exact = 0usize;
            let mut depths = Vec::new();
            for ((id, tokens), (_, native_tokens)) in rows.iter().zip(&native) {
                let shared = tokens.len().min(native_tokens.len());
                let depth = (0..shared)
                    .find(|&i| tokens[i] != native_tokens[i])
                    .unwrap_or(shared);
                let full = depth == tokens.len() && tokens.len() == native_tokens.len();
                if full {
                    exact += 1;
                }
                depths.push(depth);
                println!(
                    "[q8-vs-f16] {id}: depth {depth}/{}{}",
                    tokens.len(),
                    if full { "*" } else { "" }
                );
            }
            depths.sort_unstable();
            println!(
                "[q8-vs-f16] {exact}/{} exact prompts, median depth {}",
                rows.len(),
                depths[depths.len() / 2]
            );
        }

        if let Some(out) = std::env::var_os("LFM2_Q8_FIXTURE_OUT") {
            let mut body = String::new();
            for (id, tokens) in &rows {
                let tokens_json = serde_json::to_string(tokens).expect("serialize tokens");
                body.push_str(&format!(
                    "{{\"id\":{},\"tokens\":{tokens_json}}}\n",
                    serde_json::to_string(id).expect("serialize id")
                ));
            }
            std::fs::write(&out, body).expect("write fixture");
            println!("wrote fixture to {}", out.display());
        }

        // Pin the oracle (only on the full 20x64 set; a calibration run with a
        // reduced limit/tokens skips the pin so it cannot false-fail).
        if limit >= prompts.len() && max_tokens == 64 {
            assert_eq!(
                q8_sha, PINNED_Q8_DECODE_FIXTURE_SHA256,
                "Q8 CPU-reference decode fixture drifted from the pinned oracle"
            );
        }
    }

    /// Measure the vocab-wide logit error between the Q8 Metal engine and the Q8
    /// CPU-reference oracle, feeding both the identical token stream. This is the
    /// quantity the certified near-tie band is derived from (stage C measured the
    /// f16 engine at ~0.03 and set the band to 0.05). Because the engine and the
    /// oracle share the same quantized weight bytes, the Q8 quantization error
    /// cancels here; what is measured is the residual f16-activation / matmul /
    /// transcendental gap. Run with:
    ///
    /// ```text
    /// SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    ///     q8_hybrid_step_logit_error_probe -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn q8_hybrid_step_logit_error_probe() {
        use tokenizers::Tokenizer;

        let path = std::path::PathBuf::from(
            std::env::var_os("SYNAPSE_UNIFIED_RT_LFM2_1_2B")
                .expect("set SYNAPSE_UNIFIED_RT_LFM2_1_2B to the LFM2-1.2B snapshot directory"),
        );
        let limit: usize = std::env::var("LFM2_Q8_BAND_LIMIT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(3);

        let mut tokenizer =
            Tokenizer::from_file(path.join("tokenizer.json")).expect("load tokenizer");
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).expect("disable truncation");

        // Engine: a Q8-loaded model drives the quantized hybrid step engine.
        let engine_model =
            Model::load_with_quant(&path, Precision::F16, WeightQuantization::Q8_0)
                .expect("load LFM2-1.2B (Q8_0) for the engine");
        let hidden = engine_model.config.hidden_size;
        let stop_tokens: HashSet<u32> =
            engine_model.generation_stop_ids().iter().copied().collect();
        let mut engine =
            Lfm2HybridStepEngine::new(&engine_model, 2048, true).expect("q8 hybrid engine");

        // Oracle: a separately loaded Q8 model, dequantized in place.
        let mut oracle_model =
            Model::load_with_quant(&path, Precision::F16, WeightQuantization::Q8_0)
                .expect("load LFM2-1.2B (Q8_0) for the oracle");
        let input_embeddings = prepare_q8_oracle_model(&mut oracle_model);
        let mut provider = CpuProvider::platform_for_test();

        let prompts = decode_prompt_set();
        let mut global_max = 0.0f32;
        for (id, prompt) in prompts.iter().take(limit) {
            let prompt_ids = oracle_model
                .encode_generation(&tokenizer, prompt, 2048)
                .expect("encode prompt");
            // Token stream = prompt then the oracle's own greedy continuation, so
            // both sides are compared on identical positions.
            let generated = greedy_decode_q8_cpu(
                &oracle_model,
                &mut provider,
                &input_embeddings,
                &prompt_ids,
                64,
                &stop_tokens,
            );
            let stream: Vec<u32> = prompt_ids
                .iter()
                .copied()
                .chain(generated.iter().copied())
                .collect();

            engine.reset().expect("reset engine caches");
            let mut oracle_cache = oracle_model.empty_decode_cache(stream.len() + 1);
            let mut prompt_max = 0.0f32;
            for (position, &token) in stream.iter().enumerate() {
                let start = token as usize * hidden;
                let input_row = &input_embeddings[start..start + hidden];
                let f16_row = encode_f16_bits(input_row);
                let engine_logits = engine
                    .step_logits(position, &f16_row)
                    .expect("engine step logits");
                let (_, oracle_logits) = oracle_model
                    .decode_embedding(&mut provider, &mut oracle_cache, input_row)
                    .expect("oracle decode step");
                let max_delta = engine_logits
                    .iter()
                    .zip(&oracle_logits)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                prompt_max = prompt_max.max(max_delta);
                global_max = global_max.max(max_delta);
            }
            println!("[q8-band] {id}: max|dlogit| over {} positions = {prompt_max:.6}", stream.len());
        }
        println!("[q8-band] global max|dlogit| over the full vocabulary = {global_max:.6}");
    }

    /// Load the pinned Q8 fixture rows ({"id","tokens"} per line) cut from the
    /// Q8-dequantized CPU reference, for row-by-row diagnostics alongside the sha.
    fn q8_pinned_fixture_rows() -> Vec<(String, Vec<u32>)> {
        let raw = include_str!("../fixtures/lfm2-q8-step-reference.jsonl");
        raw.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let value: serde_json::Value =
                    serde_json::from_str(line).expect("q8 fixture row parses");
                let id = value["id"].as_str().expect("q8 fixture id").to_string();
                let tokens = value["tokens"]
                    .as_array()
                    .expect("q8 fixture tokens")
                    .iter()
                    .map(|token| token.as_u64().expect("q8 fixture token id") as u32)
                    .collect();
                (id, tokens)
            })
            .collect()
    }

    /// Run the Q8 CPU reference for one prompt, feeding the prompt then the first
    /// `step` pinned generated tokens, and return the logits that predict generated
    /// token `step`. Used to certify a fork is a near-tie: the Q8-oracle top-2 there
    /// must be the two fork tokens separated by less than the band.
    fn q8_cpu_logits_predicting_step(
        model: &Model,
        input_embeddings: &[f32],
        prompt_ids: &[u32],
        pinned_tokens: &[u32],
        step: usize,
    ) -> Vec<f32> {
        let hidden = model.config.hidden_size;
        let mut provider = CpuProvider::platform_for_test();
        let mut cache = model.empty_decode_cache(prompt_ids.len() + step + 1);
        let mut logits = Vec::new();
        for &token in prompt_ids {
            let start = token as usize * hidden;
            let (_, next) = model
                .decode_embedding(&mut provider, &mut cache, &input_embeddings[start..start + hidden])
                .expect("q8 cpu prefill step");
            logits = next;
        }
        for &token in &pinned_tokens[..step] {
            let start = token as usize * hidden;
            let (_, next) = model
                .decode_embedding(&mut provider, &mut cache, &input_embeddings[start..start + hidden])
                .expect("q8 cpu decode step");
            logits = next;
        }
        logits
    }

    /// Structural-invariant ceiling on the Q8-oracle top-2 logit gap at a certified
    /// fork. Derived from the measured ~0.051 vocab-wide Q8 engine-vs-oracle logit
    /// error (see q8_hybrid_step_logit_error_probe and LFM2-METAL-STEP.md stage D):
    /// a fork whose oracle top-2 gap is below this band is a rounding coin-flip
    /// (the engine's per-logit error of <= ~0.051 on each of the top two can flip
    /// the order, so flips are possible up to ~2x0.051 ~= 0.10), not a real
    /// divergence. 0.12 leaves margin above that flip threshold.
    const NEAR_TIE_BAND_Q8: f32 = 0.12;

    /// Structural-invariant ceiling on the number of divergent prompts (same bound
    /// as f16: the engine-oracle gap is a small rounding effect, so forks are rare).
    const MAX_CERTIFIED_FORKS_Q8: usize = 2;

    /// M1 authority exact Q8 fork signature (the primary gate). PENDING the locked
    /// M1 run: these are sentinel values; on the M1 the gate records the observed
    /// fork to pin rather than asserting a not-yet-measured signature (see the gate
    /// body). Fill these from the M1 transcript, exactly as stage C pinned its
    /// M1_FORK_* constants from the M1 run.
    const Q8_M1_FORK_PROMPT: &str = "";
    const Q8_M1_FORK_STEP: usize = 0;
    const Q8_M1_FORK_CPU_TOKEN: u32 = 0;
    const Q8_M1_FORK_ENGINE_TOKEN: u32 = 0;

    /// Q8 token-exactness gate, two-tier (mirrors the f16 certification model).
    ///
    /// STRUCTURAL INVARIANT (asserted on every machine): at most
    /// MAX_CERTIFIED_FORKS_Q8 prompts diverge from the Q8 CPU-reference oracle,
    /// and each divergence is a top-2 swap whose Q8-oracle top-2 logit gap is below
    /// NEAR_TIE_BAND_Q8 -- a rounding coin-flip inside the engine's measured ~0.051
    /// Q8 error band, not a real divergence. A length mismatch is never certified.
    ///
    /// PRIMARY GATE (M1 authority only): the exact M1 Q8 fork signature, once
    /// measured, is pinned and asserted. Until the Q8_M1_FORK_* constants are
    /// filled from the locked M1 run, the M1 branch records the observed fork to
    /// pin (sentinel detection) instead of asserting a placeholder. Run with:
    ///
    /// ```text
    /// SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    ///     q8_hybrid_step_engine_matches_pinned_fixture -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn q8_hybrid_step_engine_matches_pinned_fixture_within_certified_near_tie() {
        let path = std::path::PathBuf::from(
            std::env::var_os("SYNAPSE_UNIFIED_RT_LFM2_1_2B")
                .expect("set SYNAPSE_UNIFIED_RT_LFM2_1_2B to the LFM2-1.2B snapshot directory"),
        );
        let mut tokenizer =
            Tokenizer::from_file(path.join("tokenizer.json")).expect("load tokenizer");
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).expect("disable truncation");

        // Engine: Q8-loaded model driving the quantized hybrid step engine.
        let engine_model =
            Model::load_with_quant(&path, Precision::F16, WeightQuantization::Q8_0)
                .expect("load LFM2-1.2B (Q8_0)");
        let stop_tokens: HashSet<u32> =
            engine_model.generation_stop_ids().iter().copied().collect();
        let mut engine =
            Lfm2HybridStepEngine::new(&engine_model, 2048, true).expect("q8 hybrid engine");

        let prompts = decode_prompt_set();
        assert_eq!(prompts.len(), 20, "expected the twenty-prompt decode set");
        let rows = run_metal_fixture(
            &mut engine,
            &engine_model,
            &tokenizer,
            &prompts,
            &stop_tokens,
            64,
        );

        // Compare every prompt to the pinned Q8 oracle, collecting divergences.
        let pinned = q8_pinned_fixture_rows();
        let mut divergent: Vec<(String, usize, u32, u32)> = Vec::new(); // (id, step, engine, cpu)
        for ((id, tokens), (pinned_id, pinned_tokens)) in rows.iter().zip(&pinned) {
            assert_eq!(id, pinned_id, "q8 fixture prompt order mismatch");
            let shared = tokens.len().min(pinned_tokens.len());
            let first_diff = (0..shared).find(|&i| tokens[i] != pinned_tokens[i]);
            match first_diff {
                Some(step) if tokens.len() == pinned_tokens.len() => {
                    println!(
                        "[q8] DIVERGENCE {id}: first diff at step {step}: engine {} vs oracle {}",
                        tokens[step], pinned_tokens[step]
                    );
                    divergent.push((id.clone(), step, tokens[step], pinned_tokens[step]));
                }
                None if tokens.len() == pinned_tokens.len() => {
                    println!("[q8] {id}: {} tokens, byte-exact vs Q8 oracle", tokens.len());
                }
                _ => {
                    panic!(
                        "uncertified Q8 divergence on {id}: engine {} tok vs oracle {} tok (first diff {first_diff:?})",
                        tokens.len(),
                        pinned_tokens.len()
                    );
                }
            }
        }

        // STRUCTURAL INVARIANT: bound the fork count, and certify each fork is a
        // top-2 swap within the band by running the Q8 CPU reference to the fork.
        assert!(
            divergent.len() <= MAX_CERTIFIED_FORKS_Q8,
            "too many Q8 divergent prompts ({}) vs the certified-fork ceiling {MAX_CERTIFIED_FORKS_Q8}: {divergent:?}",
            divergent.len()
        );
        // Oracle model for fork certification (dequantized weights + f16 input).
        let mut oracle_model =
            Model::load_with_quant(&path, Precision::F16, WeightQuantization::Q8_0)
                .expect("load LFM2-1.2B (Q8_0) for oracle");
        let input_embeddings = prepare_q8_oracle_model(&mut oracle_model);
        for (id, step, engine_token, cpu_token) in &divergent {
            let prompt_text = prompts
                .iter()
                .find(|(prompt_id, _)| prompt_id == id)
                .map(|(_, text)| text.clone())
                .expect("divergent prompt text");
            let prompt_ids = oracle_model
                .encode_generation(&tokenizer, &prompt_text, 2048)
                .expect("encode prompt");
            let pinned_tokens = pinned
                .iter()
                .find(|(pinned_id, _)| pinned_id == id)
                .map(|(_, tokens)| tokens.clone())
                .expect("pinned tokens");
            let fork_logits = q8_cpu_logits_predicting_step(
                &oracle_model,
                &input_embeddings,
                &prompt_ids,
                &pinned_tokens,
                *step,
            );
            let (best, second, gap) = top2_gap(&fork_logits);
            println!(
                "[q8] fork {id} step {step}: Q8-oracle top-2 = ({best}, {second}), gap {gap:.6} (band {NEAR_TIE_BAND_Q8})"
            );
            assert_eq!(
                best, *cpu_token,
                "Q8 oracle top-1 at the fork on {id} is not the pinned token (not a top-2 swap)"
            );
            assert_eq!(
                second, *engine_token,
                "the Q8 engine's fork token on {id} is not the oracle's runner-up (not a top-2 swap)"
            );
            assert!(
                gap < NEAR_TIE_BAND_Q8,
                "fork Q8-oracle top-2 gap {gap} on {id} exceeds band {NEAR_TIE_BAND_Q8}: a real divergence, not a certified near-tie"
            );
        }

        if is_m1_authority() {
            if Q8_M1_FORK_PROMPT.is_empty() {
                // M1 pin pending: record the observed fork signature to pin.
                println!(
                    "[q8] M1 AUTHORITY: Q8 fork signature to pin (fill Q8_M1_FORK_* constants): {divergent:?}"
                );
            } else {
                assert_eq!(
                    divergent.len(),
                    1,
                    "M1 authority: expected exactly the one pinned Q8 fork, got {divergent:?}"
                );
                let (id, step, engine_token, cpu_token) = &divergent[0];
                assert_eq!(id, Q8_M1_FORK_PROMPT, "M1 authority: Q8 fork prompt drifted");
                assert_eq!(*step, Q8_M1_FORK_STEP, "M1 authority: Q8 fork step drifted");
                assert_eq!(
                    *cpu_token, Q8_M1_FORK_CPU_TOKEN,
                    "M1 authority: Q8 oracle token at the fork drifted"
                );
                assert_eq!(
                    *engine_token, Q8_M1_FORK_ENGINE_TOKEN,
                    "M1 authority: Q8 engine token at the fork drifted"
                );
                println!(
                    "[q8] M1 AUTHORITY: pinned Q8 fork signature confirmed ({Q8_M1_FORK_PROMPT} step {Q8_M1_FORK_STEP}, engine {Q8_M1_FORK_ENGINE_TOKEN} vs oracle {Q8_M1_FORK_CPU_TOKEN})"
                );
            }
        } else {
            println!(
                "[q8] advisory (non-M1): {} Q8 fork(s) within band; observed {divergent:?}",
                divergent.len()
            );
        }
    }

    /// Q8 determinism gate: two full twenty-prompt Q8 decodes through the engine
    /// must be byte-identical (same token rows, same sha).
    #[test]
    #[ignore]
    fn q8_hybrid_step_engine_is_deterministic() {
        let path = std::path::PathBuf::from(
            std::env::var_os("SYNAPSE_UNIFIED_RT_LFM2_1_2B")
                .expect("set SYNAPSE_UNIFIED_RT_LFM2_1_2B to the LFM2-1.2B snapshot directory"),
        );
        let mut tokenizer =
            Tokenizer::from_file(path.join("tokenizer.json")).expect("load tokenizer");
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).expect("disable truncation");
        let model = Model::load_with_quant(&path, Precision::F16, WeightQuantization::Q8_0)
            .expect("load LFM2-1.2B (Q8_0)");
        let stop_tokens: HashSet<u32> = model.generation_stop_ids().iter().copied().collect();
        let prompts = decode_prompt_set();
        let mut engine = Lfm2HybridStepEngine::new(&model, 2048, true).expect("q8 hybrid engine");
        let first = run_metal_fixture(&mut engine, &model, &tokenizer, &prompts, &stop_tokens, 64);
        let second = run_metal_fixture(&mut engine, &model, &tokenizer, &prompts, &stop_tokens, 64);
        for ((id, first_tokens), (_, second_tokens)) in first.iter().zip(&second) {
            assert_eq!(
                first_tokens, second_tokens,
                "Q8 hybrid step decode is not deterministic for {id}"
            );
        }
        assert_eq!(
            fixture_sha256(&first),
            fixture_sha256(&second),
            "Q8 hybrid step decode fixture sha is not deterministic across runs"
        );
        println!(
            "q8 determinism: two runs byte-identical, sha {}",
            fixture_sha256(&first)
        );
    }

    /// Q8 single-stream decode throughput cell (run in `--release`). Mirrors the f16
    /// timing probe but on the quantized engine: prefill (untimed) then time one
    /// chained 64-token greedy decode per prompt, median of 20 prompts x 2 repeats.
    /// Authoritative only on the locked M1; advisory on the build host. Run with:
    ///
    /// ```text
    /// SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    ///     --release q8_hybrid_step_timing_probe -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn q8_hybrid_step_timing_probe() {
        let path = std::path::PathBuf::from(
            std::env::var_os("SYNAPSE_UNIFIED_RT_LFM2_1_2B")
                .expect("set SYNAPSE_UNIFIED_RT_LFM2_1_2B to the LFM2-1.2B snapshot directory"),
        );
        let mut tokenizer =
            Tokenizer::from_file(path.join("tokenizer.json")).expect("load tokenizer");
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).expect("disable truncation");
        let model = Model::load_with_quant(&path, Precision::F16, WeightQuantization::Q8_0)
            .expect("load LFM2-1.2B (Q8_0)");
        let prompts = decode_prompt_set();
        let mut engine = Lfm2HybridStepEngine::new(&model, 2048, true).expect("q8 hybrid engine");
        const MAX_TOKENS: usize = 64;
        const REPEATS: usize = 2;

        let mut sample = |prompt: &str| -> f64 {
            engine.reset().expect("reset");
            let prompt_ids = model
                .encode_generation(&tokenizer, prompt, 2048)
                .expect("encode prompt");
            let first = engine.prefill(&prompt_ids).expect("prefill");
            let position = prompt_ids.len();
            let started = std::time::Instant::now();
            let tokens = engine.chain(position, MAX_TOKENS, first).expect("chain");
            let elapsed = started.elapsed().as_secs_f64();
            assert_eq!(tokens.len(), MAX_TOKENS, "chain produced a partial decode");
            MAX_TOKENS as f64 / elapsed
        };

        let _ = sample(&prompts[0].1);

        let mut rates = Vec::with_capacity(prompts.len() * REPEATS);
        for repeat in 0..REPEATS {
            for (id, prompt) in &prompts {
                let rate = sample(prompt);
                println!("[q8-timing] repeat {repeat} {id}: {rate:.2} tok/s");
                rates.push(rate);
            }
        }
        rates.sort_by(|a, b| a.partial_cmp(b).expect("finite rate"));
        let median = rates[rates.len() / 2];
        let min = rates[0];
        let max = rates[rates.len() - 1];
        println!(
            "[q8-timing] LFM2-1.2B owned Metal step Q8_0: median {median:.2} tok/s (min {min:.2}, max {max:.2}), {:.2} ms/token warm, {} samples",
            1000.0 / median,
            rates.len()
        );
    }

    // ---------------------------------------------------------------------
    // Stage C: end-to-end hybrid step engine certification.
    //
    // The hybrid engine walks the full LFM2 backbone on device (ten conv layers
    // + six attention layers + shared SwiGLU/head tail) and must reproduce the
    // pinned native CPU-reference greedy fixture 20/20. This is the gate that
    // certifies the finer in-kernel f16 rounding the stage-B probe left open
    // (stage B rounded activations only at layer boundaries; the assembled
    // engine also rounds inside the conv path and the attention kernels).
    // ---------------------------------------------------------------------

    use tokenizers::Tokenizer;

    /// Load the pinned fixture rows ({"id","tokens"} per line) cut from the
    /// native CPU reference, for row-by-row diagnostics alongside the sha gate.
    fn pinned_fixture_rows() -> Vec<(String, Vec<u32>)> {
        let raw = include_str!("../fixtures/lfm2-f16-step-reference.jsonl");
        raw.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                let value: serde_json::Value =
                    serde_json::from_str(line).expect("fixture row parses");
                let id = value["id"].as_str().expect("fixture id").to_string();
                let tokens = value["tokens"]
                    .as_array()
                    .expect("fixture tokens")
                    .iter()
                    .map(|token| token.as_u64().expect("fixture token id") as u32)
                    .collect();
                (id, tokens)
            })
            .collect()
    }

    /// Greedy decode through the hybrid Metal step engine, mirroring
    /// `greedy_decode_cpu`: prefill the prompt token-by-token (verify path),
    /// then chain greedy generation with on-GPU argmax, truncating after the
    /// first stop token (inclusive) exactly like lfm2_decode.rs::Decoder.
    fn greedy_decode_metal(
        engine: &mut Lfm2HybridStepEngine,
        model: &Model,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_tokens: usize,
        stop_tokens: &HashSet<u32>,
    ) -> Vec<u32> {
        engine.reset().expect("reset caches");
        let prompt_ids = model
            .encode_generation(tokenizer, prompt, 2048)
            .expect("encode prompt");
        let first = engine.prefill(&prompt_ids).expect("metal prefill");
        let mut generated = Vec::with_capacity(max_tokens);
        generated.push(first);
        if stop_tokens.contains(&first) || max_tokens <= 1 {
            return generated;
        }
        let position = prompt_ids.len();
        let rest = engine
            .chain(position, max_tokens - 1, first)
            .expect("metal chain");
        for token in rest {
            generated.push(token);
            if stop_tokens.contains(&token) {
                break;
            }
        }
        generated
    }

    /// Run the full twenty-prompt x 64-token decode through the engine.
    fn run_metal_fixture(
        engine: &mut Lfm2HybridStepEngine,
        model: &Model,
        tokenizer: &Tokenizer,
        prompts: &[(String, String)],
        stop_tokens: &HashSet<u32>,
        max_tokens: usize,
    ) -> Vec<(String, Vec<u32>)> {
        prompts
            .iter()
            .map(|(id, prompt)| {
                let tokens =
                    greedy_decode_metal(engine, model, tokenizer, prompt, max_tokens, stop_tokens);
                (id.clone(), tokens)
            })
            .collect()
    }

    fn load_lfm2_checkpoint() -> (std::path::PathBuf, Model, Tokenizer) {
        let path = std::path::PathBuf::from(
            std::env::var_os("SYNAPSE_UNIFIED_RT_LFM2_1_2B")
                .expect("set SYNAPSE_UNIFIED_RT_LFM2_1_2B to the LFM2-1.2B snapshot directory"),
        );
        let mut tokenizer =
            Tokenizer::from_file(path.join("tokenizer.json")).expect("load tokenizer");
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).expect("disable truncation");
        let model = Model::load(&path, Precision::F16).expect("load LFM2-1.2B");
        (path, model, tokenizer)
    }

    // -- f16 near-tie certification model ------------------------------------
    //
    // The f16 hybrid engine matches the f32 CPU-reference oracle to ~0.03
    // vocab-wide logit precision (measured max|dlogit| on both the M5 build host
    // and the M1 authority). Greedy tokens therefore agree with the oracle
    // everywhere except at near-ties whose CPU top-2 gap falls inside that error
    // band, where the f16 rounding tips the coin-flip. WHICH near-tie flips is
    // GPU-architecture-dependent: the reused kernels' transcendentals (exp in the
    // attention softmax, rsqrt in rmsnorm) round differently on different Apple
    // GPUs even compiled IEEE-strict. Observed: the M5 build host forks
    // completion-15 / step 17 (CPU top-2 gap 0.0004); the M1 authority forks
    // completion-05 / step 8 (gap 0.0073). This mirrors the documented Qwen3 f16
    // precedent (METAL-STEP.md: the completion-06 near-tie drifts on the M5 Metal
    // compiler; the M1 is the fixture authority). The oracle (the pinned CPU
    // fixture) is machine-independent and untouched; only the engine's coin-flip
    // resolution is machine-dependent, bounded by the band invariant below.
    //
    // Two tiers:
    //   * STRUCTURAL INVARIANT (every machine): at most MAX_CERTIFIED_FORKS
    //     prompts diverge, and each divergence is a top-2 SWAP whose CPU top-2
    //     logit gap is below NEAR_TIE_BAND. A real regression -- a wrong token at
    //     a decisive gap, or many forks -- cannot hide inside this.
    //   * PRIMARY GATE (M1 authority only): the exact M1 fork signature is pinned;
    //     any deviation on the M1 fails. Other machines run the structural
    //     invariant as an advisory canary and record their observed fork.

    /// Structural-invariant ceiling on the CPU top-2 logit gap at a certified
    /// fork. Justified by the measured ~0.03 vocab-wide f16 logit error vs the f32
    /// oracle (see LFM2-METAL-STEP.md stage C): a fork whose CPU top-2 gap is
    /// below this band is a rounding coin-flip, not a real divergence. 0.05 leaves
    /// margin over the observed ~0.03 error.
    const NEAR_TIE_BAND: f32 = 0.05;

    /// Structural-invariant ceiling on the number of divergent prompts. Observed
    /// one fork per machine; the bound catches a regression that forks widely.
    const MAX_CERTIFIED_FORKS: usize = 2;

    /// M1 authority exact fork signature (the primary gate). completion-05,
    /// generated step 8: the engine emits 7693 where the CPU oracle emits 1827, a
    /// certified near-tie (CPU top-2 gap 0.0073 < NEAR_TIE_BAND).
    const M1_FORK_PROMPT: &str = "completion-05";
    const M1_FORK_STEP: usize = 8;
    const M1_FORK_CPU_TOKEN: u32 = 1827;
    const M1_FORK_ENGINE_TOKEN: u32 = 7693;

    /// The M5 build host's observed canary fork (advisory; recorded, not
    /// asserted): completion-15 / step 17, engine 523 vs CPU 518, gap 0.0004.
    const M5_CANARY_PROMPT: &str = "completion-15";
    const M5_CANARY_STEP: usize = 17;

    /// Whether this process runs on the M1 fixture/timing authority
    /// (LFM2-DECODE-BASELINES.md rig: [bench-host], Apple M1 Max,
    /// MacBookPro18,2). The f16 near-tie coin-flips resolve differently on other
    /// Apple GPUs, so the exact fork signature is pinned only here; elsewhere the
    /// structural band invariant is the gate. An explicit env override
    /// (SYNAPSE_LFM2_STEP_AUTHORITY=m1) covers a relocated bench.
    fn is_m1_authority() -> bool {
        if let Ok(value) = std::env::var("SYNAPSE_LFM2_STEP_AUTHORITY") {
            return value == "m1" || value == "1";
        }
        match std::process::Command::new("sysctl")
            .arg("-n")
            .arg("hw.model")
            .output()
        {
            Ok(output) if output.status.success() => {
                String::from_utf8_lossy(&output.stdout).contains("MacBookPro18,2")
            }
            _ => false,
        }
    }

    /// Run the lfm2.rs CPU reference for one prompt, feeding the prompt then the
    /// first `step` pinned generated tokens, and return the logits that predict
    /// generated token `step`. Used to certify the fork is a near-tie: the CPU
    /// top-2 there must be the two fork tokens separated by less than the epsilon.
    fn cpu_logits_predicting_step(
        model: &Model,
        prompt_ids: &[u32],
        pinned_tokens: &[u32],
        step: usize,
    ) -> Vec<f32> {
        let mut provider = CpuProvider::platform_for_test();
        let mut cache = model.empty_decode_cache(prompt_ids.len() + step + 1);
        let mut logits = Vec::new();
        for &token in prompt_ids {
            let (_, next) = model
                .decode_token(&mut provider, &mut cache, token)
                .expect("cpu prefill step");
            logits = next;
        }
        for &token in &pinned_tokens[..step] {
            let (_, next) = model
                .decode_token(&mut provider, &mut cache, token)
                .expect("cpu decode step");
            logits = next;
        }
        logits
    }

    /// (best_id, second_id, best_minus_second) for a logit vector, ties broken
    /// toward the lower id to match the greedy sampler.
    fn top2_gap(logits: &[f32]) -> (u32, u32, f32) {
        let mut best = 0u32;
        let mut second = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        let mut second_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            let i = i as u32;
            if v > best_val || (v == best_val && i < best) {
                second = best;
                second_val = best_val;
                best = i;
                best_val = v;
            } else if v > second_val || (v == second_val && i < second) {
                second = i;
                second_val = v;
            }
        }
        (best, second, best_val - second_val)
    }

    /// Token-exactness gate, two-tier (see the certification-model comment above).
    ///
    /// STRUCTURAL INVARIANT (asserted on every machine): at most
    /// MAX_CERTIFIED_FORKS prompts diverge from the f32 CPU-reference oracle, and
    /// each divergence is a top-2 swap whose CPU top-2 logit gap is below
    /// NEAR_TIE_BAND -- i.e. a rounding coin-flip inside the engine's measured
    /// ~0.03 f16 error band, not a real divergence.
    ///
    /// PRIMARY GATE (asserted only on the M1 authority): the exact M1 fork
    /// signature (completion-05 / step 8 / engine 7693 vs CPU 1827) is pinned; any
    /// deviation on the M1 fails. On any other machine (e.g. the M5 build host)
    /// the structural invariant is the gate and the observed fork is printed as an
    /// advisory canary note (the M5 fork is completion-15 / step 17). Run with:
    ///
    /// ```text
    /// SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    ///     hybrid_step_engine_matches_pinned_fixture -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn hybrid_step_engine_matches_pinned_fixture_within_certified_near_tie() {
        let (_path, model, tokenizer) = load_lfm2_checkpoint();
        let stop_tokens: HashSet<u32> = model.generation_stop_ids().iter().copied().collect();
        let prompts = decode_prompt_set();
        assert_eq!(prompts.len(), 20, "expected the twenty-prompt decode set");
        let mut engine = Lfm2HybridStepEngine::new(&model, 2048, false).expect("hybrid engine");
        let rows = run_metal_fixture(&mut engine, &model, &tokenizer, &prompts, &stop_tokens, 64);

        // Compare every prompt to the pinned f32 oracle, collecting divergences.
        let pinned = pinned_fixture_rows();
        let mut divergent: Vec<(String, usize, u32, u32)> = Vec::new(); // (id, step, engine, cpu)
        for ((id, tokens), (pinned_id, pinned_tokens)) in rows.iter().zip(&pinned) {
            assert_eq!(id, pinned_id, "fixture prompt order mismatch");
            let shared = tokens.len().min(pinned_tokens.len());
            let first_diff = (0..shared).find(|&i| tokens[i] != pinned_tokens[i]);
            match first_diff {
                Some(step) if tokens.len() == pinned_tokens.len() => {
                    println!(
                        "[metal] DIVERGENCE {id}: first diff at step {step}: engine {} vs oracle {}",
                        tokens[step], pinned_tokens[step]
                    );
                    divergent.push((id.clone(), step, tokens[step], pinned_tokens[step]));
                }
                None if tokens.len() == pinned_tokens.len() => {
                    println!(
                        "[metal] {id}: {} tokens, byte-exact vs oracle",
                        tokens.len()
                    );
                }
                _ => {
                    // A length mismatch is never a certified near-tie.
                    panic!(
                        "uncertified divergence on {id}: engine {} tok vs oracle {} tok (first diff {first_diff:?})",
                        tokens.len(),
                        pinned_tokens.len()
                    );
                }
            }
        }

        // STRUCTURAL INVARIANT: bound the fork count, and certify each fork is a
        // top-2 swap within the band by running the CPU reference to the fork.
        assert!(
            divergent.len() <= MAX_CERTIFIED_FORKS,
            "too many divergent prompts ({}) vs the certified-fork ceiling {MAX_CERTIFIED_FORKS}: {divergent:?}",
            divergent.len()
        );
        for (id, step, engine_token, cpu_token) in &divergent {
            let prompt_text = prompts
                .iter()
                .find(|(prompt_id, _)| prompt_id == id)
                .map(|(_, text)| text.clone())
                .expect("divergent prompt text");
            let prompt_ids = model
                .encode_generation(&tokenizer, &prompt_text, 2048)
                .expect("encode prompt");
            let pinned_tokens = pinned
                .iter()
                .find(|(pinned_id, _)| pinned_id == id)
                .map(|(_, tokens)| tokens.clone())
                .expect("pinned tokens");
            let fork_logits =
                cpu_logits_predicting_step(&model, &prompt_ids, &pinned_tokens, *step);
            let (best, second, gap) = top2_gap(&fork_logits);
            println!(
                "[metal] fork {id} step {step}: CPU top-2 = ({best}, {second}), gap {gap:.6} (band {NEAR_TIE_BAND})"
            );
            assert_eq!(
                best, *cpu_token,
                "oracle top-1 at the fork on {id} is not the pinned token (not a top-2 swap)"
            );
            assert_eq!(
                second, *engine_token,
                "the engine's fork token on {id} is not the oracle's runner-up (not a top-2 swap)"
            );
            assert!(
                gap < NEAR_TIE_BAND,
                "fork CPU top-2 gap {gap} on {id} exceeds band {NEAR_TIE_BAND}: a real divergence, not a certified near-tie"
            );
        }

        if is_m1_authority() {
            // PRIMARY GATE: the M1 fork signature is pinned exactly.
            assert_eq!(
                divergent.len(),
                1,
                "M1 authority: expected exactly the one pinned fork, got {divergent:?}"
            );
            let (id, step, engine_token, cpu_token) = &divergent[0];
            assert_eq!(id, M1_FORK_PROMPT, "M1 authority: fork prompt drifted");
            assert_eq!(*step, M1_FORK_STEP, "M1 authority: fork step drifted");
            assert_eq!(
                *cpu_token, M1_FORK_CPU_TOKEN,
                "M1 authority: oracle token at the fork drifted"
            );
            assert_eq!(
                *engine_token, M1_FORK_ENGINE_TOKEN,
                "M1 authority: engine token at the fork drifted"
            );
            println!(
                "[metal] M1 AUTHORITY: pinned fork signature confirmed ({M1_FORK_PROMPT} step {M1_FORK_STEP}, engine {M1_FORK_ENGINE_TOKEN} vs oracle {M1_FORK_CPU_TOKEN})"
            );
        } else {
            // Advisory canary on non-authority machines: record the observed fork.
            println!(
                "[metal] advisory (non-M1): {} fork(s) within band; M5 canary reference is {M5_CANARY_PROMPT} step {M5_CANARY_STEP}; observed {divergent:?}",
                divergent.len()
            );
        }
    }

    /// Determinism gate: two full twenty-prompt decodes through the engine must
    /// be byte-identical (same token rows, same sha). Run with the same env var
    /// as the exactness gate.
    #[test]
    #[ignore]
    fn hybrid_step_engine_is_deterministic() {
        let (_path, model, tokenizer) = load_lfm2_checkpoint();
        let stop_tokens: HashSet<u32> = model.generation_stop_ids().iter().copied().collect();
        let prompts = decode_prompt_set();
        let mut engine = Lfm2HybridStepEngine::new(&model, 2048, false).expect("hybrid engine");
        let first = run_metal_fixture(&mut engine, &model, &tokenizer, &prompts, &stop_tokens, 64);
        let second = run_metal_fixture(&mut engine, &model, &tokenizer, &prompts, &stop_tokens, 64);
        for ((id, first_tokens), (_, second_tokens)) in first.iter().zip(&second) {
            assert_eq!(
                first_tokens, second_tokens,
                "hybrid step decode is not deterministic for {id}"
            );
        }
        assert_eq!(
            fixture_sha256(&first),
            fixture_sha256(&second),
            "hybrid step decode fixture sha is not deterministic across runs"
        );
        println!(
            "determinism: two runs byte-identical, sha {}",
            fixture_sha256(&first)
        );
    }

    /// Single-stream decode throughput cell (run in `--release`). For each prompt
    /// it prefills (untimed) then times one chained 64-token greedy decode -- the
    /// sustained step-engine decode rate, comparable to the `Decode tok/s` column
    /// of LFM2-DECODE-BASELINES.md (owned MPSGraph f16 6.345, llama.cpp Metal F16
    /// 130.74, Q8_0 203.65 on this rig). Reports the median over the twenty
    /// prompts x two repeats plus min/max and warm ms/token. This is the
    /// authoritative number only on the locked M1 (see LFM2-METAL-STEP.md stage
    /// C); on the M5 build host it is advisory. Run with:
    ///
    /// ```text
    /// SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    ///     --release hybrid_step_timing_probe -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn hybrid_step_timing_probe() {
        let (_path, model, tokenizer) = load_lfm2_checkpoint();
        let prompts = decode_prompt_set();
        let mut engine = Lfm2HybridStepEngine::new(&model, 2048, false).expect("hybrid engine");
        const MAX_TOKENS: usize = 64;
        const REPEATS: usize = 2;

        // One timed sample: reset caches, prefill the prompt (untimed), then time
        // a chained 64-token greedy decode. The rate is the chained-step decode
        // rate (MAX_TOKENS decode steps), excluding prefill, matching the
        // baselines' decode column.
        let mut sample = |prompt: &str| -> f64 {
            engine.reset().expect("reset");
            let prompt_ids = model
                .encode_generation(&tokenizer, prompt, 2048)
                .expect("encode prompt");
            let first = engine.prefill(&prompt_ids).expect("prefill");
            let position = prompt_ids.len();
            let started = std::time::Instant::now();
            let tokens = engine.chain(position, MAX_TOKENS, first).expect("chain");
            let elapsed = started.elapsed().as_secs_f64();
            assert_eq!(tokens.len(), MAX_TOKENS, "chain produced a partial decode");
            MAX_TOKENS as f64 / elapsed
        };

        // Uncounted warmup so pipeline setup and thermals are steady before timing.
        let _ = sample(&prompts[0].1);

        let mut rates = Vec::with_capacity(prompts.len() * REPEATS);
        for repeat in 0..REPEATS {
            for (id, prompt) in &prompts {
                let rate = sample(prompt);
                println!("[timing] repeat {repeat} {id}: {rate:.2} tok/s");
                rates.push(rate);
            }
        }
        rates.sort_by(|a, b| a.partial_cmp(b).expect("finite rate"));
        let median = rates[rates.len() / 2];
        let min = rates[0];
        let max = rates[rates.len() - 1];
        println!(
            "[timing] LFM2-1.2B owned Metal step f16: median {median:.2} tok/s (min {min:.2}, max {max:.2}), {:.2} ms/token warm, {} samples",
            1000.0 / median,
            rates.len()
        );
    }

    /// Bisection probe for a divergent prompt: feed BOTH the Metal engine and the
    /// lfm2.rs CPU reference the SAME token sequence (the CPU greedy sequence)
    /// position by position, and compare the per-position logits/argmax. The
    /// first position whose argmax differs localizes the fork; printing the top
    /// logits there distinguishes a genuine near-tie f16 flip (top-2 within f16
    /// epsilon) from a structural bug (large logit disagreement). Run with:
    ///
    /// ```text
    /// SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    ///     hybrid_step_localize_divergence -- --ignored --nocapture
    /// ```
    ///
    /// `LFM2_PROBE_PROMPT` selects the prompt id (default completion-15).
    #[test]
    #[ignore]
    fn hybrid_step_localize_divergence() {
        let (_path, model, tokenizer) = load_lfm2_checkpoint();
        let stop_tokens: HashSet<u32> = model.generation_stop_ids().iter().copied().collect();
        let want_id = std::env::var("LFM2_PROBE_PROMPT").unwrap_or_else(|_| "completion-15".into());
        let prompts = decode_prompt_set();
        let (_id, prompt) = prompts
            .iter()
            .find(|(id, _)| id == &want_id)
            .expect("probe prompt id present");

        let prompt_ids = model
            .encode_generation(&tokenizer, prompt, 2048)
            .expect("encode prompt");
        let n = prompt_ids.len();

        // CPU greedy sequence (prompt ++ generated), the shared token stream.
        let mut provider = CpuProvider::platform_for_test();
        let generated =
            greedy_decode_cpu(&model, &mut provider, &prompt_ids, 64, &stop_tokens, false);
        let mut seq = prompt_ids.clone();
        seq.extend_from_slice(&generated);
        println!(
            "prompt {want_id}: prompt_len {n}, generated {}",
            generated.len()
        );

        fn argmax(logits: &[f32]) -> u32 {
            let mut best = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best = i as u32;
                }
            }
            best
        }
        fn top3(logits: &[f32]) -> Vec<(u32, f32)> {
            let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
            idx.sort_by(|&a, &b| {
                logits[b as usize]
                    .total_cmp(&logits[a as usize])
                    .then(a.cmp(&b))
            });
            idx.iter()
                .take(3)
                .map(|&i| (i, logits[i as usize]))
                .collect()
        }

        // CPU per-position logits feeding the shared sequence.
        let mut cpu_cache = model.empty_decode_cache(seq.len() + 1);
        let mut cpu_provider = CpuProvider::platform_for_test();
        let mut cpu_argmaxes = Vec::with_capacity(seq.len());
        let mut cpu_top = Vec::with_capacity(seq.len());
        for &tok in &seq {
            let (_, logits) = model
                .decode_token(&mut cpu_provider, &mut cpu_cache, tok)
                .expect("cpu decode");
            cpu_argmaxes.push(argmax(&logits));
            cpu_top.push(top3(&logits));
        }

        // Metal per-position logits feeding the SAME sequence.
        let mut engine = Lfm2HybridStepEngine::new(&model, 2048, false).expect("hybrid engine");
        engine.reset().expect("reset");
        let mut metal_argmaxes = Vec::with_capacity(seq.len());
        let mut metal_top = Vec::with_capacity(seq.len());
        let mut metal_logits_at: Vec<Option<Vec<f32>>> = Vec::new();
        for (t, &tok) in seq.iter().enumerate() {
            let embed = model.token_embedding(tok).expect("embedding");
            let embed_f16 = encode_f16_bits(embed);
            let logits = engine.step_logits(t, &embed_f16).expect("metal step");
            metal_argmaxes.push(argmax(&logits));
            metal_top.push(top3(&logits));
            metal_logits_at.push(Some(logits));
        }

        // First fork (argmax differs given the identical prefix).
        let fork = (0..seq.len()).find(|&t| cpu_argmaxes[t] != metal_argmaxes[t]);
        println!("first argmax fork at sequence position {fork:?}");
        let center = fork.unwrap_or(seq.len().saturating_sub(1));
        let lo = center.saturating_sub(2);
        let hi = (center + 3).min(seq.len());
        for t in lo..hi {
            let cpu_logits = {
                // Recompute CPU logits at t for the max-diff metric.
                let mut c = model.empty_decode_cache(t + 2);
                let mut p = CpuProvider::platform_for_test();
                let mut last = Vec::new();
                for s in 0..=t {
                    let (_, l) = model.decode_token(&mut p, &mut c, seq[s]).expect("cpu");
                    last = l;
                }
                last
            };
            let metal_logits = metal_logits_at[t].as_ref().expect("metal logits");
            let max_diff = cpu_logits
                .iter()
                .zip(metal_logits.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!(
                "pos {t} (gen {}): cpu argmax {} metal argmax {} | max|dlogit| {max_diff:.6}",
                t as isize - n as isize,
                cpu_argmaxes[t],
                metal_argmaxes[t]
            );
            println!("   cpu top3:   {:?}", cpu_top[t]);
            println!("   metal top3: {:?}", metal_top[t]);
        }
    }
}
