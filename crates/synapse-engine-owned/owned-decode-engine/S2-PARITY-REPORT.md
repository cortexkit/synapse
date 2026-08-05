# S2 Engine Port — Delivery Report Addendum

## Four-lane A/B parity results (M5, this machine)

| Lane | Family | Format | Byte-identical vs spike | Forks | Notes |
|------|--------|--------|--------------------------|-------|-------|
| qwen3-f16 | Qwen3-0.6B | f16 | 20/20 | 0 | Byte-identical vs spike CLI output |
| qwen3-q8 | Qwen3-0.6B | Q8_0 | 20/20 | 0 | Byte-identical vs spike CLI output (f16 prefill + Q8 stepping) |
| lfm2-f16 | LFM2-1.2B | f16 | 19/20 vs CPU oracle | 1 | Fork at completion-15/step17: prod 523 vs oracle 518 (gap 0.0004). The spike's Metal step engine produces the exact same fork on this machine. Production engine is byte-identical to spike Metal step engine. |
| lfm2-q8 | LFM2-1.2B | Q8_0 | 20/20 | 0 | Byte-identical vs spike reference fixture |

## Verification protocol

1. **Spike harness run**: the spike binary (`bench/spikes/unified-rt`) was built with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` and run on the 20-prompt x 64-token fixture (`decode-prompts.jsonl`) for all four lanes. Qwen3 lanes used the CLI (`--decode-backend metal-step`); LFM2 lanes used the `#[ignore]` test harness (`hybrid_step_engine_matches_pinned_fixture_within_certified_near_tie` and `q8_hybrid_step_engine_matches_pinned_fixture_within_certified_near_tie`).

2. **Production engine run**: the production owned-decode-engine (`crates/synapse-engine-owned/owned-decode-engine/`) was run on the same 20-prompt x 64-token fixture through the `owned_decode_parity.rs` integration test. The test loads the same model checkpoints, runs the same greedy-top-1 decode protocol, and compares token streams byte-for-byte.

3. **Comparison**: token streams are compared element-by-element. A divergence at step N means the production engine emitted a different token than the spike at that step. The LFM2 f16 fork at completion-15/step17 is a certified near-tie (CPU top-2 gap 0.0004 < 0.05 band) that the spike's own Metal step engine also produces on this machine — the production engine is byte-identical to the spike's Metal step engine.

## Q8 prefill strategy

The spike's Qwen3 Metal step engine uses the MPSGraph `MetalDecoder` for prefill (f16 weights) and imports the KV cache into the step engine for Q8 stepping. The production port replicates this by:
1. Loading an f16 model and creating an f16 Metal step decoder for prefill
2. Running prefill with the f16 engine (verify path, no MPSGraph)
3. Exporting the KV cache bits from the f16 engine
4. Importing them into the Q8 engine via `synapse_qwen3_metal_step_import_caches`
5. Stepping with the Q8 engine

This matches the spike's f16-prefill + Q8-stepping approach byte-for-byte without using MPSGraph.

## Metallib compilation

Both metallibs were compiled with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`:
- `qwen3_decode_metal_step.metallib` (160,825 bytes)
- `lfm2_decode_metal_step.metallib` (165,980 bytes, compiled with `-fno-fast-math -ffp-contract=off`)

The LFM2 metallib includes a reused IEEE-strict copy of the Qwen3 step kernels for the attention layers, compiled with the same IEEE-strict flags.

## M5 advisory fork signature (LFM2 f16)

The M5 build host forks at completion-15/step17: engine emits 523 where the CPU oracle emits 518 (CPU top-2 gap 0.000362). This is the documented M5 canary fork (LFM2-METAL-STEP.md). The M1 authority's fork signature is pinned at completion-05/step8 (different near-tie, different machine). This M5 run is advisory; M1 constants are not re-pinned.

## Addendum: batched-verify Rust exposure for quantum-bounded prefill

The S2 port shipped the `.metal` and `.m` files byte-identical to the spike,
including the mat-mat batched-verify kernels and the
`synapse_qwen3_metal_step_verify_batch` entry point; the Rust path initially
exposed only the per-token verify. The prefill quantum-bounding work added
the missing Rust exposure — `MetalStepDecoder::verify_tokens_batch` (argmax
readback) and `verify_tokens_batch_logits` (full-logits gate surface) plus
the FFI declaration — without touching the byte-identical kernel surface.
This is additive: the default `prefill`/`verify_tokens` routing and every
pinned fixture battery are unchanged (re-verified: four-lane parity battery
20/20 on both Qwen3 lanes, certified near-tie fork on LFM2 f16).

The batched path's arithmetic identity is re-gated in production by
`tests/owned_decode_prefill_chunking.rs` (port of the spike campaign's
byte-identical/determinism/forced-rejection battery: f16 and Q8_0, K in
{1,2,4,8,16}, prompt depths {1,5,33,128,469}), plus the chunked-prefill
bit-exactness gate (per-token spans 8/16/32 and batched spans 16 must leave
the KV cache byte-identical and the first-token argmax unchanged versus the
uninterrupted single-command-buffer prefill). The serving prefill uses the
batched spans by default on the strength of that fixture evidence.