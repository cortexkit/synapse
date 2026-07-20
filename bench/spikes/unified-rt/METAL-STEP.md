# Metal custom decode step

Status: **spike implementation; correctness and locked-M1 measurement gates are pending**.
The default Qwen3 decode backend remains `mpsgraph`. Select the custom path with
`--decode-backend metal-step`.

## Scope and handoff

This path is limited to Qwen3 causal decode in `bench/spikes/unified-rt`.
Embedding/prefill remains on the existing MPSGraph graph. At the first decode
step, the MPSGraph context exports each layer's `[kv_heads, bucket, head_dim]`
key and value arrays as f16 bits. The custom context imports those arrays once
with a blit into private, persistent Metal buffers. Every later token uses only
our command buffer and Metal functions; no MPSGraph executable is called by the
step path.

The custom context keeps activations, logits, weights, and KV handles alive
across calls. It submits one command buffer per token, waits once, and reads the
shared logits buffer for the existing CPU greedy argmax and hook loop. Cache
inspection uses the same layer/concatenated-KV interface as the MPSGraph path,
so pause, splice, token taps, and addressable weight regions remain host-side
protocols rather than backend-specific behavior.

## Kernel list

The build script compiles `src/qwen3_decode_metal_step.metal` with `xcrun metal`
and links the resulting `qwen3_decode_metal_step.metallib` beside the Cargo
executable. Runtime loading first resolves that executable-relative file, with
the build output as a development fallback; the timed step never compiles
Metal source.

The per-token command buffer encodes these kernels for each of the 28 layers:

1. `metal_step_rmsnorm` — pre-attention RMSNorm.
2. `metal_step_qkv_matvec` — one grid for Q, K, and V. V writes directly to
   the layer's `kv_heads * bucket * head_dim` cache slot.
3. `metal_step_qk_norm_rope` — Q/K head RMSNorm and Qwen rotary embedding;
   normalized K writes directly into the addressed KV cache slot.
4. `metal_step_attention` — stable two-pass softmax and P/V accumulation
   over the resident cache (the current correctness candidate uses one query-head
   thread; simdgroup reduction work remains gated behind parity).
5. `metal_step_matvec_residual` — O projection.
6. `metal_step_residual_rmsnorm` — fused attention residual add and pre-MLP RMSNorm.
7. `metal_step_gate_up_swiglu` — fused gate/up matvec and SiLU product.
8. `metal_step_matvec_residual` — down projection plus MLP residual.
9. A final `metal_step_rmsnorm` and `metal_step_lm_head` produce f32 logits.

The same matvec kernels select either resident f16 weights or GGUF-compatible
Q8_0 blocks. Q8_0 uses the existing 34-byte layout: little-endian f16 scale
followed by 32 signed int8 values. Dequantization occurs inside the dot product;
no dequantized matrix is materialized.

## Correctness gate transcripts

The Xcode toolchain was explicitly selected after the CLT-only build failed:

```text
$ DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun -sdk macosx --find metal
/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.yr6fBk/Metal.xctoolchain/usr/bin/metal
$ DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo build -p spike-unified-rt --release --bin spike-unified-rt
Finished `release` profile [optimized]
$ stat target/release/qwen3_decode_metal_step.metallib
40914 bytes; no cargo:warning emitted
```

The ordered model and hook gates are green:

```text
Gate 1 f16:
  20/20 exact prompts, 1,280/1,280 tokens, 0 near-tie exemptions
Gate 2 Q8_0:
  15/20 exact prompts, median match depth 64.0, 0 near-tie exemptions
  quantized_weight_sha256=4c774c188ec089ac1e9b30e9797b364993d5c2445d3e479d5d1b94f6d10969d0
Gate 3 hooks:
  cargo test -p spike-unified-rt qwen3_decode — 8 passed, 0 failed
  token tap, pause/resume, splice, addressable regions, and deterministic ties passed
Gate 4 prefill handoff:
  all-MPSGraph and metal-step outputs equal for completion-01, 64/64 tokens
```

The parity failure was localized by dumping every layer's new cache slot: all
pre-existing slots were byte-identical, while the new slot differed only in the
K/V values produced by the broken state ping-pong. The down projection wrote into
the old current buffer, but the host loop then advanced `current` to the O
projection buffer, dropping the MLP output before layer 1. Removing that swap
made the current-state buffer persistent and produced the gate results above.

## Measurement table

Timed on `[bench-host]` (M1 Max), AC power, while holding and then
promptly releasing `[bench-user-home]/bench.lock`. Each cell used 12 distinct prompts
from the fixed stride-seven schedule, repeated twice (24 fresh processes total).
The metallib was transferred beside the executable and loaded executable-relative.

| Backend | Weights | Prompts / repeats | Decode tok/s | Encode / GPU / host per token | Status |
| --- | --- | --- | ---: | --- | --- |
| MPSGraph reference | f16 | 12 x 2 | 84.32 baseline | pending fresh breakdown | prior baseline |
| Metal step | f16 | 12 x 2 | **5.8080 median** | 0.1025 ms / 171.8693 ms / 0.0146 ms | gates green |
| Metal step | Q8_0 | 12 x 2 | **5.0405 median** | 0.1021 ms / 198.0879 ms / 0.0141 ms | gates green |
| llama.cpp Metal | Q8_0 | 12 x 2 | not run | not run | `llama-cli` unavailable on locked M1 |

The owned-step host column excludes GPU command-buffer wait, logits readback,
and sampler time; median readback was 0.033 ms/token and median sampling was
0.158 ms/token. A fresh llama.cpp comparison could not be made because neither
`llama-cli` nor `llama-server` is installed on the locked host; no substitute or
stale server result is relabeled as the requested control.

## Campaign targets

The first correctness-green implementation is intentionally a parity-first
candidate, not a throughput winner: the single-thread attention kernel and
28-layer dispatch count make GPU execution the dominant stage. The next campaign
should restore the 32-wide simdgroup score reduction only after preserving the
20/20 gate, then fuse or batch the remaining per-layer dispatches. Q8_0 currently
has lower throughput than f16 on this implementation despite its lower weight
traffic, so its dequant matvec and launch overhead need a separate profile before
claiming a bandwidth win.
