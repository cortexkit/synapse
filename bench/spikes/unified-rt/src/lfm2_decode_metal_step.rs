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

#[cfg(test)]
mod tests {
    use super::Lfm2ConvStepEngine;
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
}
