# Metal custom decode step

Status: **wave 2 kernels are correctness-gated and measured; the custom path remains a spike and is not the default backend**.
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
across calls. Dense weights are uploaded once through a shared staging buffer
into private Metal buffers; the step path never reads model-owned host memory.
It submits one command buffer per token, waits once, and reads the shared logits
buffer for the existing CPU greedy argmax and hook loop. Cache inspection uses
the same layer/concatenated-KV interface as the MPSGraph path, so pause, splice,
token taps, and addressable weight regions remain host-side protocols rather
than backend-specific behavior.

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
   over the resident cache. One 32-wide simdgroup owns each query head: lanes
   split independent KV positions while keeping each QK dot serial, lane zero
   performs the reference-order softmax reductions, and all lanes cooperate
   over the independent value dimensions.
5. `metal_step_matvec_residual` — O projection.
6. `metal_step_residual_rmsnorm` — fused attention residual add and pre-MLP RMSNorm.
7. `metal_step_gate_up_swiglu` — fused gate/up matvec and SiLU product.
8. `metal_step_matvec_residual` — down projection plus MLP residual.
9. A final `metal_step_rmsnorm` and `metal_step_lm_head` produce f32 logits.

The same matvec kernels select either resident f16 weights or GGUF-compatible
Q8_0 blocks. F16 rows use one lane per output row, with two adjacent half4
loads unrolled while the f32 accumulator remains serial. Q8_0 uses the existing
34-byte layout: little-endian f16 scale followed by 32 signed int8 values. Q8_0
rows use one simdgroup per output row, loading one quant from each 32-element
block per lane; four block iterations are unrolled without changing per-lane
accumulation order. Dequantization occurs inside the dot product and no
dequantized matrix is materialized.

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

Wave 2 refresh (Xcode Metal toolchain, local M5):

```text
$ DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun -sdk macosx --find metal
/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.yr6fBk/Metal.xctoolchain/usr/bin/metal
$ DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo build -p spike-unified-rt --locked --release --bin spike-unified-rt
Finished `release` profile [optimized]; no cargo:warning emitted
$ stat target/release/qwen3_decode_metal_step.metallib
48866 bytes; loaded executable-relative beside spike-unified-rt
Gate 1 f16: 20/20 exact prompts, 1,280/1,280 tokens, 0 near-tie exemptions
Gate 2 Q8_0: 14/20 exact prompts, median match depth 64.0, 0 near-tie exemptions
  quantized_weight_sha256=4c774c188ec089ac1e9b30e9797b364993d5c2445d3e479d5d1b94f6d10969d0
Gate 3 hooks: cargo test -p spike-unified-rt qwen3_decode — 7 passed, 0 failed
  token tap, pause/resume, splice, addressable regions, ties, constraints, and Q8 regions passed
Gate 4 prefill handoff: included in the 20-prompt metal-step f16 run; no mismatch
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
| Metal step baseline | f16 | 12 x 2 | **5.8080 median** | 0.1025 ms / 171.8693 ms / 0.0146 ms | parity-first baseline |
| Metal step baseline | Q8_0 | 12 x 2 | **5.0405 median** | 0.1021 ms / 198.0879 ms / 0.0141 ms | parity-first baseline |
| Metal step wave 1 | f16 | 12 x 2 | **17.7992 median** | 0.1021 ms / 55.8754 ms / 0.0340 ms | 20/20 exact; AC |
| Metal step wave 1 | Q8_0 | 12 x 2 | **29.2656 median** | 0.1018 ms / 33.8624 ms / 0.0327 ms | 14/20 exact; median depth 64.0; AC |
| Metal step wave 2 | f16 | 12 x 2 | **32.3296 median** | 0.1022 ms / 30.6257 ms / 0.0324 ms | 20/20 exact; AC; +81.6% vs wave 1 |
| Metal step wave 2 | Q8_0 | 12 x 2 | **49.9470 median** | 0.1022 ms / 19.7155 ms / 0.0328 ms | 14/20 exact; median depth 64.0; AC; +70.6% vs wave 1 |
| llama.cpp Metal | Q8_0 | 12 x 2 | not run | not run | `llama-cli` unavailable on locked M1 |

The owned-step host column excludes GPU command-buffer wait, logits readback,
and sampler time; median readback was 0.033 ms/token and median sampling was
0.158 ms/token. A fresh llama.cpp comparison could not be made because neither
`llama-cli` nor `llama-server` is installed on the locked host; no substitute or
stale server result is relabeled as the requested control.

## Campaign targets

Wave 1 keeps the MPSGraph backend as the default. On the locked M1, the
correctness-preserving attention P/V parallelism and f16 dot unrolling lifted
f16 from 5.8080 to 17.7992 tok/s; Q8_0 cooperative dequant matvec lifted Q8_0
from 5.0405 to 29.2656 tok/s, making Q8_0 faster than f16. GPU execution still
accounts for 55.8754 ms/token f16 and 33.8624 ms/token Q8_0, while encode/feed
is 0.1021/0.1018 ms/token, so dispatch consolidation is not yet the limiting
stage. The remaining gap to the 84.32 tok/s MPSGraph baseline is a kernel-level
profile problem, not host dispatch overhead.

Wave 2 keeps MPSGraph as the default and applies only order-preserving
parallelism. Attention lanes now split KV positions rather than splitting a
single QK reduction; f16 rows remain independent serial reductions; and Q8
keeps its one-simdgroup-per-row reduction while exposing four block iterations
per lane. On the locked M1, f16 reached 32.3296 tok/s and Q8_0 reached 49.9470
tok/s, with GPU execution falling to 30.6257 ms and 19.7155 ms per token.
Both cells beat their wave-1 references; the measured GPU stage remains well
above the roughly 10 ms dispatch-fusion threshold.


## Wave 1 progression log

The local M5 runs below were correctness/performance probes, not substitutes for
locked-M1 results. Every kernel probe was followed by the f16 20-prompt gate;
Q8_0 rows were reported only after a fresh Q8_0 quality gate.

| Change | f16 gate | Local probe | Locked-M1 result | Decision |
| --- | --- | --- | --- | --- |
| 32-wide QK reduction and fully cooperative attention prototype | 19/20, 0 near ties; completion-06 diverged at step 7 | 33.9162 tok/s | not timed | Reverted: reduction order changed logits |
| Lane-zero reference-order scores plus 32-wide P/V attention | 20/20, 0 near ties | 28.2297 tok/s first passing run | included in final | Kept: exact score order, parallel value dimensions |
| Cooperative f16 matvec with `simd_sum` | 19/20, 0 near ties; completion-06 diverged at step 25 | 68.6494 tok/s combined probe | not timed | Reverted: f16 parity is a hard gate |
| Four-product f16 dot unroll, then half4 loads | 20/20, 0 near ties after each probe | 43.6843 then 43.9977 tok/s | **17.7992 tok/s** | Kept: same f32 accumulation order, materially faster than baseline |
| Fuse f16 QK-norm+RoPE into QKV epilogue | 20/20, 0 near ties | 43.5677 tok/s | not timed | Reverted: command fusion was slower than the unfused half4 candidate |
| Q8_0 block-row cooperative matvec | 20/20 f16 gate | final Q8_0: 14/20 exact, median depth 64.0, 55.2564 tok/s | **29.2656 tok/s** after 14/20, median depth 64.0 gate | Kept: Q8_0 is now faster than f16 |
| Shared-to-private weight upload | 20/20 f16 gate; Q8_0 gate remained 14/20, median depth 64.0 | no material isolated delta on M5; residency verified | included in final | Kept: private residency removes repeated host-visible weight access |

The locked-M1 final f16 sweep used AC power, `[bench-user-home]/bench.lock`, 12
varied prompts selected by the house stride-seven schedule, two fresh-process
repeats, and 24 samples total. The median stage breakdown was 55.8754 ms GPU
execution, 0.1021 ms feed/encode, and 0.0340 ms logits readback per token.
The locked-M1 Q8_0 quality gate was 14/20 exact with median match depth 64.0
and zero near-tie exemptions; its 12 x 2 timed median was 29.2656 tok/s with
33.8624 ms GPU execution, 0.1018 ms feed/encode, and 0.0327 ms readback per
token. The local final f16 gate was 20/20 exact and the local final Q8_0 gate
was 14/20 exact with median depth 64.0.

## Wave 2 progression log

Every wave-2 kernel change passed the local f16 exactness gate before the
locked-M1 timing cells; the Q8 row was reported only after a fresh Q8 quality
gate. The M1 cells used AC power (100%, charged), `[bench-user-home]/bench.lock`, no
`Runner.Worker`, the fixed stride-seven 12-prompt schedule, two fresh-process
repeats, and the executable-relative 48,866-byte metallib.

| Change | f16 gate | Local probe | Locked-M1 result | Decision |
| --- | --- | --- | --- | --- |
| KV-position-parallel QK dots with lane-zero reference-order softmax | 20/20 exact, 0 near ties | 67.8324 tok/s gate run | **32.3296 tok/s**, 30.6257 ms GPU | Kept: 20/20 exact; independent scores removed the serial attention bottleneck |
| F16 row-parallel path with two-half4 dot unroll | 20/20 exact, 0 near ties | included above; 161.73 GB/s effective weight rate | included above | Kept: serial per-row accumulation preserved exactness |
| Q8 one-simdgroup-per-row four-block unroll | 20/20 f16 gate; Q8 14/20, depth 64.0 | 92.3741 tok/s quality run | **49.9470 tok/s**, 19.7155 ms GPU | Kept: Q8 quality floor passed; +70.6% vs wave 1 |

The wave-2 f16 cell is +81.6% over 17.7992 tok/s, and the Q8_0 cell is +70.6%
over 29.2656 tok/s. Neither crosses the 84.32 tok/s MPSGraph reference; the
remaining gap is GPU execution rather than dispatch (feed stayed 0.1022 ms/token).

A fresh llama.cpp control was not available on the locked host: neither
`llama-cli` nor `llama-server` was installed, and no source checkout or archived
MANIFEST was present under `[bench-user-home]/bench-tools`. The historical Q8_0
control remains 190.36--203.45 tok/s from `DECODE-WAVE1.md`; it is not relabeled
as a fresh wave-1 measurement.
