# Qwen3-Embedding-0.6B owned-runtime graph

## Scope and architecture

This spike adds `Qwen/Qwen3-Embedding-0.6B` as the accelerated 600M model family. The runtime auto-detects Qwen3 from `config.json`; MiniLM behavior and its provider hook remain unchanged.

The graph implements the Hugging Face Qwen3 layout:

- 28 decoder layers, hidden size 1024.
- 16 query heads, 8 key/value heads, and grouped-query attention.
- Pre-attention and pre-MLP RMSNorm.
- Per-head query and key RMSNorm after projection. These Qwen3-specific norms are required for parity.
- RoPE with the model-configured `rope_theta` (1,000,000 in the tested snapshot).
- Causal attention plus padding-key masking.
- Bias-free Q/K/V/O and MLP projections.
- SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
- Final RMSNorm, last valid token pooling, and L2 normalization.
- Raw document text with no instruction prefix. Tokenizer padding is stripped before restoring exactly one terminal EOS token, matching the validated MLX path and the ORT tokenizer output.

Safetensor F32/F16/BF16 storage is converted to owned `f32` tensors at load. Both providers execute Qwen3 in fp32. The existing experimental f16 MiniLM path is unchanged; Qwen3 f16 is explicitly rejected.

## Provider graph

The CPU correctness path delegates every dense and attention product to the existing provider matmul hooks. On macOS this is Accelerate SGEMM; RMSNorm, RoPE, causal masking, softmax, SwiGLU, pooling, and normalization are owned Rust operations.

An optional Qwen3-specific block hook lets the Metal provider execute the family graph as a device-resident block instead of issuing one provider call per matrix operation.

`qwen3_mpsgraph.m` builds all 28 layers and the final RMSNorm as one MPSGraph for each batch shape. Hidden states are uploaded after CPU embedding lookup and read back after the final norm. Q/K/V, RoPE, GQA, attention, residuals, norms, and SwiGLU remain device-resident between those boundaries. Static parameters are cached in Metal buffers. KV heads are reshaped to `[batch, kv_heads, 1, seq, head_dim]`, broadcast over the query-per-KV group dimension, and reshaped to query-head layout; no CPU repeat is materialized.

Required gates are mean cosine at least 0.9999 and mean top-10 neighbor overlap at least 0.995, with every shared vector used as a rank query. The CLI enforces both thresholds whenever `--reference` is supplied, and a unit test fixes the certification boundaries.
