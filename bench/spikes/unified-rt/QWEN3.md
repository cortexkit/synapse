# Qwen3-Embedding-0.6B owned-runtime graph

## Scope and architecture

This spike adds `Qwen/Qwen3-Embedding-0.6B` as the accelerated 600M model family. The runtime auto-detects Qwen3 from `config.json`; MiniLM behavior and its provider hook remain unchanged.

The implementation follows the parity-validated `bench/lanes/mlx` graph and cross-checks the Hugging Face Qwen3 layout:

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

The Metal path uses a Qwen3-specific additive block hook. Model-family block hooks are intentional in this evidence spike: a general graph abstraction will be extracted after MiniLM, ModernBERT, and Qwen3 provide three concrete graph shapes.

`qwen3_mpsgraph.m` builds all 28 layers and the final RMSNorm as one MPSGraph for each batch shape. Hidden states are uploaded after CPU embedding lookup and read back after the final norm. Q/K/V, RoPE, GQA, attention, residuals, norms, and SwiGLU remain device-resident between those boundaries. Static parameters are cached in Metal buffers. KV heads are reshaped to `[batch, kv_heads, 1, seq, head_dim]`, broadcast over the query-per-KV group dimension, and reshaped to query-head layout; no CPU repeat is materialized.

## 400-chunk parity gate

Corpus: first 400 records of `bench/data/corpus-v2.jsonl` (46,716 post-tokenization input tokens), max length 512, length-sorted greedy batching with a 4,000,000 attention-unit budget, no document prefix.

Reference: `bench/lanes/ort-embed`, `onnx-community/Qwen3-Embedding-0.6B-ONNX` fp32, `--pooling last --max-length 512`, no `--prefix-document`.

| Runtime | Mean cosine vs ORT fp32 | Mean top-10 overlap | Gate |
|---|---:|---:|---|
| Owned CPU / Accelerate fp32 | 0.9999999999957123 | 1.000000 | pass |
| Owned Metal / resident MPSGraph fp32 | 0.9999999999932960 | 1.000000 | pass |

Required gates are mean cosine at least 0.9999 and mean top-10 neighbor overlap at least 0.995, with every shared vector used as a rank query. The CLI enforces both thresholds whenever `--reference` is supplied, and a unit test fixes the certification boundaries.

## M1 throughput

Timed on `[bench-host-alias]` under `[bench-user-home]/bench.lock`, using the same 400 chunks, max length 512, fp32, and length-sorted batching. The first row used a newly staged binary and the second immediately repeated the same command after Metal's process-independent shader caches had been populated.

| Runtime | First run tok/s | Warm tok/s | Precision |
|---|---:|---:|---|
| Owned Metal / resident MPSGraph | 3,935.1 | 4,473.3 | fp32 |
| mlx-rs Qwen3 bench lane | — | 7,700 | bf16 |

The owned fp32 graph reaches about 51% of the published mlx-rs bf16 throughput on the first run and 58% warm. This is a credible fp32 block-resident result, not an attempt to beat a bf16 implementation. The Metal graph's warm result is about 1.14x the first run. Qwen3 f16 remains out of scope.

For context only, the local (non-M1) correctness runs measured 715.3 tok/s on owned CPU and 5,389.6 tok/s on Metal. Those numbers are not used as M1 throughput evidence.
