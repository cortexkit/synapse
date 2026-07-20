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
3. `metal_step_qk_norm_rope` — Q/K head RMSNorm and Qwen rotary embedding.
4. `metal_step_attention` — one 32-wide simdgroup per query head,
   `simd_sum`/`simd_max` reductions, stable two-pass softmax, and P/V
   accumulation over the resident cache.
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

The local source/type gate passed:

```text
$ cargo fmt --all -- --check
$ cargo check -p spike-unified-rt
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.90s
```

The local machine has Metal framework headers but no `metal` developer-tool
utility, so build.rs reports the following and defers metallib generation to a
full Xcode toolchain:

```text
cargo:warning=Metal developer tools unavailable; Metal step metallib will be built by a macOS toolchain
```

The model-backed gates below have not been claimed without the locked checkpoint
and fixture run. They must be run in order before any throughput number is
reported:

```text
# 1. f16: 20/20 prompts x 64 tokens, token exact vs the MPSGraph reference
cargo run -p spike-unified-rt --release -- \
  --model "$QWEN3" --tokenizer "$TOKENIZER" \
  --generate-prompts bench/spikes/unified-rt/decode-prompts.jsonl \
  --decode-reference "$DECODE_REFERENCE" --dtype f16 --device metal \
  --decode-backend metal-step --max-new-tokens 64 --out /tmp/metal-step-f16.json

# 2. Q8_0: established quant protocol; no near-tie exemptions
cargo run -p spike-unified-rt --release -- \
  --model "$QWEN3" --tokenizer "$TOKENIZER" \
  --generate-prompts bench/spikes/unified-rt/decode-prompts.jsonl \
  --decode-reference "$DECODE_REFERENCE" --dtype f16 --weight-quant q8-0 \
  --device metal --decode-backend metal-step --max-new-tokens 64 \
  --out /tmp/metal-step-q8.json

# 3. CPU hook suite and 4. real-prefill handoff fixture
cargo test -p spike-unified-rt qwen3_decode
```

Expected gate records are **20/20 exact** for f16; for Q8_0, at least 10/20
exact prompts and median match depth at least 59.0. The Q8 gate has no near-tie
exception. A failed gate stops the campaign; thresholds are not relaxed.

## Measurement table

Numbers belong here only after gates 1–4 pass on the locked M1 Max using
`mkdir [bench-user-home]/bench.lock` for timed cells and varied prompt text for every
iteration.

| Backend | Weights | Prompts / repeats | Decode tok/s | Encode / GPU / host per token | Status |
| --- | --- | --- | ---: | --- | --- |
| MPSGraph reference | f16 | 12 x 2 | 84.32 baseline | pending fresh breakdown | recorded baseline |
| Metal step | f16 | 12 x 2 | pending | pending | gate pending |
| Metal step | Q8_0 | 12 x 2 | pending | pending | gate pending |
| llama.cpp Metal | Q8_0 | 12 x 2 | pending fresh reference | pending | re-measure required |

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
