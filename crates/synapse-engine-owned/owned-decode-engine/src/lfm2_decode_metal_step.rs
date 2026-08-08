//! LFM2-1.2B Metal hybrid step decode engine — production-owned port.
//!
//! Ported from the proven `bench/spikes/unified-rt/src/lfm2_decode_metal_step.rs`
//! spike engine. The hybrid engine drives the LFM2 backbone's short-convolution
//! layers (via the device-resident conv-cache step kernel) and attention layers
//! (via the reused Qwen3 step kernels compiled IEEE-strict into the LFM2
//! metallib) in one device-resident forward pass. Causal prefill and greedy
//! token stepping both run on device; no MPSGraph is used.
//!
//! The Metal kernels (`.metal`), the Objective-C driver (`.m`), and the FFI
//! binding are byte-identical to the spike so the pinned fixture batteries
//! reproduce exactly. The conv step kernel is bit-exact vs the CPU reference;
//! the attention layers reuse the proven Qwen3 kernels under IEEE-strict math.

#![cfg(target_os = "macos")]

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::PathBuf;
use std::ptr::NonNull;

use anyhow::{bail, ensure, Context, Result};

use super::decode_kernel::{DecodeKernel, DecodeRuntime};
use super::lfm2_decode_model::{Mixer, Model, Weight};
use super::quant::{Q8_0Tensor, WeightQuantization};
use crate::runtime::{encode_f16_bits, Precision};

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

pub struct Lfm2HybridStepCache {
    /// Number of prompt and generated tokens committed to the resident cache.
    /// The supervised worker constructs the zero-position cache when constrained
    /// prefill needs logits for the first permitted token.
    pub position: usize,
}

/// Production-owned LFM2 hybrid Metal step engine.
///
/// Safe Rust owner of the hybrid native context. It extracts the f16 weights
/// (and the f32 conv taps) from a loaded LFM2 model, uploads them once, and
/// drives the device-resident hybrid forward: token-by-token prefill via the
/// explicit-token verify path, then fast chained greedy decode with on-GPU
/// argmax. RoPE tables use LFM2's rope_theta (1e6), regenerated per position.
pub struct Lfm2HybridStepEngine {
    raw: NonNull<c_void>,
    hidden: usize,
    head_dim: usize,
    vocab: usize,
    bucket: usize,
    rope_theta: f32,
    epsilon: f32,
    weight_quantization: WeightQuantization,
    /// Request-selected chained decode span. K=1 is the certified baseline
    /// because it is the shape validated for production parity.
    chain_k: usize,
    /// Host copy of the f16 embedding gather table uploaded at prepare time.
    /// Kept so `DecodeKernel::advance` can gather a token's embedding row on
    /// the host and feed the exact same f16 bits the device-resident chain and
    /// verify kernels gather internally (vocab x hidden x 2 bytes).
    embedding_table: Vec<u16>,
}

impl Lfm2HybridStepEngine {
    pub fn new(
        model: &Model,
        precision: Precision,
        bucket: usize,
        weight_quantization: WeightQuantization,
    ) -> Result<Self> {
        ensure!(
            matches!(precision, Precision::F16),
            "LFM2 hybrid step activations require f16"
        );
        ensure!(
            [512, 1024, 2048].contains(&bucket),
            "decode cache bucket must be 512, 1024, or 2048"
        );
        ensure!(
            model.weight_quantization == weight_quantization,
            "LFM2 hybrid step weight quantization does not match the loaded model"
        );
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
        // The f16 embedding gather table is uploaded by prepare and also kept
        // on the host so single-token `advance` steps can gather the token's
        // embedding row (the native step FFI takes an f16 row, not a token id).
        let embedding_table = encode_f16_bits(&model.embeddings.data);
        let engine = Self {
            raw,
            hidden,
            head_dim,
            vocab: config.vocab_size,
            bucket,
            rope_theta: config.rope_theta,
            epsilon: config.rms_norm_eps,
            weight_quantization,
            chain_k: 1,
            embedding_table,
        };

        let quantized = weight_quantization.is_quantized();
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
                        Mixer::Conv(_) => {
                            (null, null, null, null, null, null, null, null, null, null)
                        }
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
        // separately for head use while the gather table stays f16.
        let lm_head_data = match &model.lm_head {
            Some(head) => &head.tensor.data,
            None => &model.embeddings.data,
        };
        let lm_head_f16;
        let mut tied_lm_head_q8: Option<Q8_0Tensor> = None;
        let (lm_head_fp16_ptr, lm_head_q8_ptr): (*const c_void, *const c_void) = if quantized {
            let q8: *const c_void = match &model.lm_head {
                Some(head) => q8_ptr(head),
                None => {
                    let quantized_head = Q8_0Tensor::quantize(lm_head_data, hidden)?;
                    let ptr = quantized_head.as_bytes().as_ptr().cast();
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
                engine.embedding_table.as_ptr().cast(),
            )
        };
        if status != 0 {
            // Release ownership belongs to Drop exclusively: returning Err drops
            // `engine`, whose Drop frees the native context exactly once. A
            // manual release here would free it a second time (double-free).
            // This matches the Qwen3 step engine's prepare-failure path.
            return Err(last_error())
                .with_context(|| format!("LFM2 hybrid step prepare failed ({status})"));
        }
        // Keep the mirrors and the tied Q8 head alive until after the synchronous
        // upload above.
        drop(weights);
        drop(tied_lm_head_q8);
        Ok(engine)
    }

    /// RoPE cos/sin for one position, encoded to f16 bits. Uses LFM2's
    /// rope_theta and the half-split pair layout the qk_norm_rope kernel reads.
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
    pub fn reset(&mut self) -> Result<()> {
        let status = unsafe { synapse_lfm2_hybrid_step_reset(self.raw.as_ptr()) };
        if status != 0 {
            bail!("LFM2 hybrid step reset failed ({status}): {}", last_error());
        }
        Ok(())
    }

    /// Prefill a prompt token-by-token on device, returning the greedy argmax
    /// after the final prompt token (the first generated token). Advances all
    /// caches to `prompt.len()`.
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<u32> {
        ensure!(!prompt.is_empty(), "LFM2 hybrid prefill needs a prompt");
        ensure!(
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
    /// tokens produced AFTER `first_token`.
    pub fn chain(&mut self, position: usize, steps: usize, first_token: u32) -> Result<Vec<u32>> {
        if steps == 0 {
            return Ok(Vec::new());
        }
        ensure!(
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

    /// Single host-fed forward pass returning full f32 logits.
    pub(crate) fn step_logits(&mut self, position: usize, input: &[u16]) -> Result<Vec<f32>> {
        ensure!(input.len() == self.hidden, "input width mismatch");
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

impl DecodeKernel for Lfm2HybridStepEngine {
    type Cache = Lfm2HybridStepCache;

    fn capacity(&self) -> usize {
        self.bucket
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<(Self::Cache, u32)> {
        ensure!(!tokens.is_empty(), "decode prompt must not be empty");
        ensure!(
            tokens.len() <= self.bucket,
            "decode prompt exceeds cache bucket"
        );
        // Device-resident causal prefill via the verify path, which computes
        // the argmax on device and returns only the argmax (no host-visible
        // logits vector exists to hand back). Continue generation with
        // `advance` (full logits per token) or `chain` (fused greedy stepping).
        let first_token = Lfm2HybridStepEngine::prefill(self, tokens)?;
        let cache = Lfm2HybridStepCache {
            position: tokens.len(),
        };
        Ok((cache, first_token))
    }

    fn advance(&mut self, cache: &mut Self::Cache, token: u32) -> Result<Vec<f32>> {
        ensure!(
            cache.position < self.bucket,
            "decode cache capacity exhausted"
        );
        let token_idx = token as usize;
        ensure!(
            token_idx < self.vocab,
            "token id {token} outside LFM2 vocabulary"
        );
        // Gather the token's embedding row from the host copy of the f16 table
        // uploaded at prepare. These are the exact bits the device-resident
        // chain/verify kernels gather for the same token, so a single-token
        // advance is bit-identical to one chained step. The row is copied out
        // of the table so the gather does not hold a borrow across the step.
        let row_start = token_idx * self.hidden;
        let input = self.embedding_table[row_start..row_start + self.hidden].to_vec();
        let logits = self.step_logits(cache.position, &input)?;
        cache.position += 1;
        Ok(logits)
    }

    fn cache_position(&self, cache: &Self::Cache) -> usize {
        cache.position
    }

    fn verify_tokens(&mut self, cache: &mut Self::Cache, tokens: &[u32]) -> Result<Vec<u32>> {
        ensure!(
            !tokens.is_empty(),
            "verification requires at least one token"
        );
        ensure!(
            cache.position + tokens.len() <= self.bucket,
            "verification exceeds cache capacity"
        );
        let (cos, sin) = self.rope_chain(cache.position, tokens.len());
        let mut argmaxes = vec![0u32; tokens.len()];
        let status = unsafe {
            synapse_lfm2_hybrid_step_verify(
                self.raw.as_ptr(),
                cache.position as u64,
                tokens.as_ptr(),
                tokens.len() as u32,
                cos.as_ptr(),
                sin.as_ptr(),
                argmaxes.as_mut_ptr(),
                self.epsilon,
            )
        };
        if status != 0 {
            bail!("LFM2 hybrid verify failed ({status}): {}", last_error());
        }
        cache.position += tokens.len();
        Ok(argmaxes)
    }

    fn rewind(&mut self, cache: &mut Self::Cache, position: usize) -> Result<()> {
        ensure!(
            position <= cache.position,
            "cannot rewind LFM2 cache forward from {} to {position}",
            cache.position
        );
        cache.position = position;
        Ok(())
    }

    fn chain_span(&self) -> usize {
        self.chain_k
    }

    fn set_chain_span(&mut self, span: usize) -> Result<()> {
        ensure!(
            (1..=16).contains(&span),
            "chain span must be between 1 and 16"
        );
        self.chain_k = span;
        Ok(())
    }

    fn advance_chain(
        &mut self,
        cache: &mut Self::Cache,
        seed: u32,
        steps: usize,
    ) -> Result<Vec<u32>> {
        ensure!(steps > 0, "chain step count must be positive");
        ensure!(
            cache.position + steps <= self.bucket,
            "chained decode exceeds cache capacity"
        );
        let (cos, sin) = self.rope_chain(cache.position, steps);
        let mut token_ids = vec![0u32; steps];
        let status = unsafe {
            synapse_lfm2_hybrid_step_chain(
                self.raw.as_ptr(),
                cache.position as u64,
                steps as u32,
                seed,
                cos.as_ptr(),
                sin.as_ptr(),
                token_ids.as_mut_ptr(),
                self.epsilon,
            )
        };
        if status != 0 {
            bail!("LFM2 hybrid chain failed ({status}): {}", last_error());
        }
        cache.position += steps;
        Ok(token_ids)
    }

    fn inspect_cache_layer(&self, _cache: &Self::Cache, _layer: usize) -> Result<Vec<f32>> {
        // Cache inspection is a spike-only diagnostic; the production engine
        // does not expose it. Return an empty vector.
        Ok(Vec::new())
    }
}

impl DecodeRuntime for Lfm2HybridStepEngine {
    fn lane(&self) -> &'static str {
        "owned-metal-decode"
    }

    fn kv_update_path(&self) -> &'static str {
        "metal-step-private-in-slot-hybrid-cache"
    }

    fn weight_feed_path(&self) -> &'static str {
        match self.weight_quantization {
            WeightQuantization::None => "metal-step-persistent-f16-hybrid-matvec",
            WeightQuantization::Q8_0 => "metal-step-persistent-q8_0-fused-dequant-hybrid-matvec",
        }
    }

    fn optimization_level(&self) -> u8 {
        1
    }
}

impl Drop for Lfm2HybridStepEngine {
    fn drop(&mut self) {
        self.release();
    }
}

fn metal_step_library_path() -> Result<PathBuf> {
    let executable = std::env::current_exe().context("locate engine executable")?;
    let beside_executable = executable
        .parent()
        .context("engine executable has no parent directory")?
        .join("lfm2_decode_metal_step.metallib");
    if beside_executable.is_file() {
        return Ok(beside_executable);
    }
    let build_path = PathBuf::from(env!("SYNAPSE_OWNED_DECODE_LFM2_STEP_LIB"));
    ensure!(
        build_path.is_file(),
        "LFM2 Metal step metallib is missing beside {} and at {}",
        executable.display(),
        build_path.display()
    );
    Ok(build_path)
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

unsafe extern "C" {
    fn synapse_lfm2_metal_step_last_error() -> *const c_char;
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

#[cfg(test)]
mod tests {
    use super::super::lfm2_decode_model::{Config, RmsNorm};
    use super::*;
    use crate::runtime::{Tensor, TensorDType};

    fn tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
        let strides = shape
            .iter()
            .rev()
            .scan(1usize, |stride, &dim| {
                let current = *stride;
                *stride *= dim;
                Some(current)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        Tensor {
            dtype: TensorDType::F32,
            shape,
            strides,
            data,
            metal_f16_bits: None,
        }
    }

    /// A minimal LFM2 model with zero layers. It passes the Rust-side checks
    /// in `Lfm2HybridStepEngine::new` (precision, bucket, quantization match)
    /// and creates a native context, but the native prepare call rejects a
    /// zero layer count — exactly the prepare-failure error path that used to
    /// double-free the context.
    fn layerless_model() -> Model {
        let hidden = 64usize;
        let vocab = 128usize;
        Model {
            config: Config {
                hidden_size: hidden,
                intermediate_size: 128,
                serialized_intermediate_size: 128,
                num_attention_heads: 2,
                num_hidden_layers: 0,
                num_key_value_heads: 1,
                head_dim: 32,
                rms_norm_eps: 1e-6,
                rope_theta: 1_000_000.0,
                vocab_size: vocab,
                layer_types: Vec::new(),
                conv_kernel_size: 3,
                tie_word_embeddings: true,
                bos_token_id: None,
                eos_token_id: 0,
                pad_token_id: None,
            },
            embeddings: tensor(vec![vocab, hidden], vec![0.0; vocab * hidden]),
            layers: Vec::new(),
            final_norm: RmsNorm {
                weight: tensor(vec![hidden], vec![1.0; hidden]),
                eps: 1e-6,
            },
            lm_head: None,
            tied_lm_head_q8_0: None,
            weight_quantization: WeightQuantization::None,
            generation_stop_ids: vec![0],
        }
    }

    /// Fault site `lfm2-hybrid-step-prepare-failure` in
    /// `decode-ownership-manifest-v1`: when `prepare` fails after the native
    /// context was created, the context must be released exactly once. The
    /// historical defect released it in the error path AND again in `Drop`.
    /// Under a normal `cargo test` run this exercises the full error path
    /// (context creation, failed prepare, error return, `Drop`); under an
    /// AddressSanitizer build a regression to the double release aborts the
    /// process, which is what makes this the ASan regression gate for the
    /// fault site. Requires a Metal device (macOS engine crate contract).
    #[test]
    fn prepare_failure_releases_context_exactly_once() {
        let model = layerless_model();
        let result =
            Lfm2HybridStepEngine::new(&model, Precision::F16, 512, WeightQuantization::None);
        let error = match result {
            Ok(_) => panic!("prepare must reject a model with zero layers"),
            Err(error) => error,
        };
        let message = format!("{error:#}");
        assert!(
            message.contains("prepare failed"),
            "expected the prepare-failure path (context created, prepare rejected), got: {message}"
        );
        // `engine` was dropped on the error return above. If the context were
        // released twice, an ASan build would have aborted before reaching
        // this assertion.
    }
}
