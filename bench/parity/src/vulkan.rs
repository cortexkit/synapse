//! Vulkan memory floors.
//!
//! `vulkan_floors` is the single sizing function the manifest generator and
//! the Vulkan worker both call. It lays out every buffer the worker allocates
//! for one model (`buffer_plan`) and derives:
//!
//! - `vulkan_min_storage_buffer_range`: the largest single buffer the worker
//!   binds, weights or activations, which `maxStorageBufferRange` must cover;
//! - `vulkan_min_device_local_bytes`: the allocator's peak when processing
//!   `sequences` sequences of `context_tokens` tokens in sub-batches of at
//!   most `sub_batch_max_tokens` tokens.
//!
//! The allocation model: every weight tensor is its own buffer in the
//! profile's storage dtype; activation buffers are sized for one full
//! sub-batch, allocated once and reused by every sub-batch; the result buffer
//! holds every sequence's output. Attention is computed in key tiles with
//! running softmax statistics, so no buffer is quadratic in sequence length.
//! A sequence is never split across sub-batches, so `sub_batch_max_tokens`
//! must be at least `context_tokens`.

use crate::arch::expected_tensors;
use crate::manifest::{DType, Family, Fp32Tensors, Model, Operation};
use crate::{perr, Result};

/// Sequences the device-local floor is sized for.
pub const FLOOR_SEQUENCES: u64 = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub name: String,
    pub bytes: u64,
    /// True for activation buffers reused across sub-batches.
    pub per_sub_batch: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanFloors {
    pub min_storage_buffer_range: u64,
    pub min_device_local_bytes: u64,
}

const F32: u64 = 4;
const I32: u64 = 4;

/// Storage buffers share a device-local arena with 256-byte-aligned offsets.
/// Round every slice, including the final slice, so the arena has no unaccounted tail.
pub const BUFFER_ALIGNMENT: u64 = 256;

pub fn aligned_bytes(bytes: u64) -> u64 {
    bytes.div_ceil(BUFFER_ALIGNMENT) * BUFFER_ALIGNMENT
}

pub fn arena_bytes(plan: &[Buffer]) -> u64 {
    plan.iter().map(|buffer| aligned_bytes(buffer.bytes)).sum()
}

/// Every buffer the Vulkan worker allocates for `model`.
pub fn buffer_plan(
    model: &Model,
    storage: DType,
    fp32: &Fp32Tensors,
    context_tokens: u64,
    sequences: u64,
    sub_batch_max_tokens: u64,
) -> Result<Vec<Buffer>> {
    if sub_batch_max_tokens < context_tokens {
        return Err(perr!(
            "vulkan_sub_batch_max_tokens {sub_batch_max_tokens} is below the {context_tokens}-token context; a sequence would have to be split"
        ));
    }
    let arch = &model.architecture;
    let hidden = arch.int("hidden_size")?;
    let heads = arch.int("num_attention_heads")?;
    let intermediate = arch.int("intermediate_size")?;
    let (q_dim, kv_dim) = match arch.family {
        Family::Modernbert => (hidden, hidden),
        Family::Qwen3 => {
            let head_dim = arch.int("head_dim")?;
            (
                heads * head_dim,
                arch.int("num_key_value_heads")? * head_dim,
            )
        }
    };
    let storage_size = match storage {
        DType::F16 => 2,
        DType::F32 => 4,
    };
    let mut plan = Vec::new();
    for (name, shape) in expected_tensors(model)? {
        let elements: u64 = shape.iter().product();
        let element_size = match fp32 {
            Fp32Tensors::All(_) => 4,
            Fp32Tensors::List(kept) if kept.contains(&name) => 4,
            Fp32Tensors::List(_) => storage_size,
        };
        plan.push(Buffer {
            name: format!("weight:{name}"),
            bytes: elements * element_size,
            per_sub_batch: false,
        });
    }
    let t = sub_batch_max_tokens;
    let sequences_per_sub_batch = sub_batch_max_tokens / context_tokens;
    let mut activation = |name: &str, bytes: u64| {
        plan.push(Buffer {
            name: format!("activation:{name}"),
            bytes,
            per_sub_batch: true,
        });
    };
    activation("token_ids", t * I32);
    activation("positions", t * I32);
    activation("sequence_lengths", sequences_per_sub_batch * I32);
    activation("hidden", t * hidden * F32);
    activation("normed", t * hidden * F32);
    activation("qkv", t * (q_dim + 2 * kv_dim) * F32);
    activation("attention_out", t * q_dim * F32);
    activation("attention_stats", t * heads * 2 * F32);
    // Both families' MLPs are gated: the input projection produces two
    // intermediate-width halves, the gated product one.
    activation("mlp_in", t * 2 * intermediate * F32);
    activation("mlp_act", t * intermediate * F32);
    activation("pooled", sequences_per_sub_batch * hidden * F32);
    if model.operation == Operation::Rerank {
        activation("head_scratch", sequences_per_sub_batch * hidden * F32);
    }
    plan.push(Buffer {
        name: "result".to_string(),
        bytes: sequences * u64::from(model.output.dimension) * F32,
        per_sub_batch: false,
    });
    Ok(plan)
}

/// `vulkan_min_storage_buffer_range` and `vulkan_min_device_local_bytes` for
/// one model and profile; `models.json` records these on each Vulkan profile
/// and `validate` fails unless they equal this function's result.
pub fn vulkan_floors(
    model: &Model,
    storage: DType,
    fp32: &Fp32Tensors,
    context_tokens: u64,
    sequences: u64,
    sub_batch_max_tokens: u64,
) -> Result<VulkanFloors> {
    let plan = buffer_plan(
        model,
        storage,
        fp32,
        context_tokens,
        sequences,
        sub_batch_max_tokens,
    )?;
    Ok(VulkanFloors {
        min_storage_buffer_range: plan.iter().map(|b| b.bytes).max().unwrap_or(0),
        min_device_local_bytes: peak_bytes(&plan, context_tokens, sequences, sub_batch_max_tokens),
    })
}

/// Walk the sequences sub-batch by sub-batch, tracking live bytes, and
/// return the peak. Weights and the result buffer stay resident for the whole
/// run; activation buffers are allocated before the first sub-batch and reused.
pub fn peak_bytes(
    plan: &[Buffer],
    context_tokens: u64,
    sequences: u64,
    sub_batch_max_tokens: u64,
) -> u64 {
    let resident: u64 = plan
        .iter()
        .filter(|b| !b.per_sub_batch)
        .map(|b| aligned_bytes(b.bytes))
        .sum();
    let activations: u64 = plan
        .iter()
        .filter(|b| b.per_sub_batch)
        .map(|b| aligned_bytes(b.bytes))
        .sum();
    let per_sub_batch = (sub_batch_max_tokens / context_tokens).max(1);
    let mut peak = resident;
    let mut remaining = sequences;
    while remaining > 0 {
        let take = remaining.min(per_sub_batch);
        peak = peak.max(resident + activations);
        remaining -= take;
    }
    peak
}

#[cfg(test)]
mod alignment_tests {
    use super::*;

    #[test]
    fn arena_rounds_each_buffer_to_256_bytes() {
        let plan = vec![
            Buffer {
                name: "weight".into(),
                bytes: 257,
                per_sub_batch: false,
            },
            Buffer {
                name: "activation".into(),
                bytes: 1,
                per_sub_batch: true,
            },
            Buffer {
                name: "result".into(),
                bytes: 256,
                per_sub_batch: false,
            },
        ];
        assert_eq!(arena_bytes(&plan), 512 + 256 + 256);
        assert_eq!(peak_bytes(&plan, 8192, 256, 8192), 1024);
        assert_eq!(arena_bytes(&plan) % BUFFER_ALIGNMENT, 0);
    }
}
