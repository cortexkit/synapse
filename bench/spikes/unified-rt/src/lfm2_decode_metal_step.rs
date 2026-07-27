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
}
