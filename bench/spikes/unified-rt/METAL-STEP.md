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
5. `metal_step_matvec_residual` — O projection plus attention residual.
6. `metal_step_rmsnorm` — pre-MLP RMSNorm.
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
41282 bytes
```

Gate 1 was run against the committed 20-prompt, 64-token fixture and stopped on
failure. The first implementation exposed NaN logits; the in-slot key-write and
residual-add fixes removed the NaNs, but the parity gate still diverges at the
second generated token:

```text
$ target/release/spike-unified-rt ... --decode-backend metal-step ... --max-new-tokens 64
owned-rt-metal-step: exact prompts 0/20
completion-01 step 1: owned token 11, reference token 13
completion-02 step 1: owned token 220, reference token 23
...
Error: token-exact fp32/f16 decode gate failed
```

The MPSGraph control produces `completion-01` tokens `12095, 13`; the custom
step produces `12095, 11`. The first generated token is correct on all 20 rows,
but Gate 1 is **not green**, so Q8_0, hook/prefill gates, and all timing cells
were not run. No threshold or near-tie exemption was applied.

```text
Gate 1: FAIL (0/20 exact; stopped here)
Gate 2: NOT RUN (ordered gate stop)
Gate 3: NOT RUN (ordered gate stop)
Gate 4: NOT RUN (ordered gate stop)
Timing: NOT RUN (ordered gate stop)
```

## Measurement table

Numbers belong here only after gates 1–4 pass on the locked M1 Max using
`mkdir [bench-user-home]/bench.lock` for timed cells and varied prompt text for every
iteration.

| Backend | Weights | Prompts / repeats | Decode tok/s | Encode / GPU / host per token | Status |
| --- | --- | --- | ---: | --- | --- |
| MPSGraph reference | f16 | 12 x 2 | 84.32 baseline | pending fresh breakdown | prior baseline |
| Metal step | f16 | 20 x 1 gate | not reported | not reported | **Gate 1 failed** |
| Metal step | Q8_0 | — | not run | not run | ordered gate stop |
| llama.cpp Metal | Q8_0 | — | not run | not run | ordered gate stop |
Do not use prefix-cache replay for this table. Report encode, GPU command-buffer
wait, logits readback/host, and total per-token wall time separately.

## Campaign targets

After certification, the next campaign should first measure whether shared-mode
persistent weights are limiting bandwidth; if so, copy the static feeds into
private buffers during preparation. The attention kernel currently favors a
simple parity-safe two-pass softmax. A simdgroup score/reduction variant is the
next controlled experiment, keeping the current kernel as the hard parity
fallback. Only after the f16 and Q8_0 gates remain green should command-buffer
pipelining or online softmax be evaluated.
