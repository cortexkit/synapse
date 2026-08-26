# Batched speculative verification on the Metal step path

Status: **implemented, gated, and measured on the locked M1. The batched verify
path produces logits byte-identical to K sequential single-token steps for every
K in {1,2,4,8,16} (f16 and Q8_0), is additive behind an opt-in entry point, and
turns the target-side verifier from K weight-streaming steps into ONE prefill-
style forward. On the M1 it verifies at up to 200.8 tok/s-equivalent (Q8, K=8) —
1.34x the 149.40 single-token decode baseline — which is the lever that makes a
4B-class target servable fast locally once any viable drafter exists.**

This is the "true batched verification" lever identified in
`../ane-spec-decode/PHASE-B.md`: phase B proved the acceptance loop (byte-identical
composition, logical KV rewind, 67% real acceptance) but verified draft tokens
SEQUENTIALLY via chain-K — K dependent single-token steps, re-streaming every
layer's weights K times. True speculative decoding's bandwidth win comes from
verifying K tokens in one batched forward so the weights stream through the GPU
once for all K positions. This document records that path, its exactness proof,
and the per-token verify cost curve that the whole 4B-serving program forks on.

The drafter question is CLOSED-dead (all three ANE shapes measured out in
PHASE-B/PHASE-C); this path builds NO drafter machinery. The verification path is
exercised by self-draft / mock-draft sources only.

## Where it lives

- `src/qwen3_decode_metal_step.metal` — four mat-mat projection kernels
  (`metal_step_{qkv_matvec,matvec_residual,gate_up_swiglu,lm_head}_batch`),
  templated on a compile-time column count.
- `src/qwen3_decode_metal_step.m` — `synapse_qwen3_metal_step_verify_batch`
  entry, lazy batch buffers, and `encode_forward_batch` (one command buffer for
  the whole K-token forward).
- `src/qwen3_decode_metal_step.rs` — `MetalStepDecoder::verify_batch` (argmax-only
  readback, the serving path) and `verify_batch_logits` (full logits readback, the
  gate surface), plus opt-in routing of the speculative verifier via
  `SYNAPSE_METAL_STEP_BATCHED_VERIFY=1`.

## Mechanism

One command buffer runs all K draft positions through the transformer as a batch:

1. **Embedding gather** (reused single-token kernel, per-column offset) writes
   batch_input row k from proposal token k.
2. **Per layer**: RMSNorm, QK-norm+RoPE, attention, residual-RMSNorm, and argmax
   reuse the EXISTING single-token kernels dispatched per column through buffer
   offsets, so each column is bit-identical to a standalone single-token step.
   The four heavy projections (Q/K/V, O, gate/up, down, LM head) use the new
   mat-mat kernels that stream each weight row ONCE and apply it to all K column
   activations.
3. **KV write**: each column writes its key/value into cache slots
   `base_position..base_position+K-1` as usual. All K slots are written before any
   column's attention runs, so column k's causal prefix (positions
   `<= base_position+k`) is fully resident and identical to the sequential path —
   this is the causal masking within the batch window.
4. **Readback**: K argmax ids (4*K bytes) for the serving path; optionally the
   full K*vocab f32 logits for the byte-exact gate.

### The exactness law (and why the kernels are templated)

Batching parallelizes ACROSS positions (independent reductions); it never
reorders the accumulation WITHIN one dot product. Each (output row, position) dot
walks the weight in the same ascending column/block order and adds products in the
same order as the single-token kernels, so every position's logits are bit-identical
to a sequential single-token step at that position.

This is enforced by construction, and the construction is subtle: the mat-mat
kernels hold K running accumulators and update each in the single-token serial
order. The column count is a **compile-time template parameter** (instantiated for
N in {1,2,4,8,16}; the entry rounds `config.batch` up to the next power of two and
writes only the first `batch` columns). With N constant, the column loop unrolls
and the N accumulators stay in registers indexed by constants, so each column's
`sums[k] += p0; sums[k] += p1; ...` chain is preserved exactly.

The first implementation used a runtime column count with a `thread float sums[16]`
array. That spilled the accumulators to memory and let the Metal compiler fold the
eight products of a column group before adding them to the running sum — a
reordering that changed the rounding and broke the byte-exact gate by ~0.05%
(compounding over 28 layers). The fix is the template instantiation: constant-
indexed register accumulators keep the single-token order. The gate below is what
caught it and now locks it.

### Rollback compatibility

On rejection at position j, the session rewinds the logical cache length to
`base_position + j` (the existing phase-B `rewind`); KV slots j..K-1 are excluded
because attention reads only positions <= the logical length, and the next forward
overwrites them. The forced-rejection gate below exercises every rejection position
for K=4 and K=8 and checks the continuation is byte-exact with target-only greedy.

## Gates (all green)

Byte-identical logits, determinism, and forced-rejection continuation are
machine-independent (they compare the batched path against the sequential path on
the same GPU), so they pass on any Apple GPU. They were run on both the local M5
and the locked M1 authority.

```text
SYNAPSE_UNIFIED_RT_QWEN3_0_6B=<Qwen3-0.6B snapshot> \
  cargo test -p spike-unified-rt --release -- --ignored \
  batched_verify_logits_are_byte_identical_to_sequential_f16 \
  batched_verify_logits_are_byte_identical_to_sequential_q8 \
  batched_verify_is_deterministic_f16 batched_verify_is_deterministic_q8 \
  batched_verify_forced_rejection_preserves_continuation_f16 \
  batched_verify_forced_rejection_preserves_continuation_q8
=> 6 passed, 0 failed   (M5 and M1)
```

- **Byte-identical logits**: for prompt lengths {1, 5, 33, 128, 469} (short through
  deep context) and K in {1,2,4,8,16}, every position's full f32 logit vector from
  `verify_batch_logits` is bit-for-bit equal to the logits from a sequential
  `advance` at the same position, for both f16 and Q8_0. The argmax surface agrees
  too. The 469-token case is the depth fixture (bucket 1024).
- **Determinism**: two batched runs over the same draft produce bit-identical
  logits (f16 and Q8).
- **Forced-rejection continuation**: for K=4 and K=8, corrupt the draft at every
  position in the window, accept the correct prefix, rewind to it, and confirm the
  greedy continuation matches the target-only stream token-for-token (f16 and Q8).

**Completion-06 canary (M1 authority)**: completion-06 is the documented M5-only
cross-machine near-tie that resolves correctly only on the M1 fixture authority.
The batched path inherits it by transitivity: the M1 byte-identical gate proves
batched logits == sequential logits for arbitrary inputs at every K, and the
single-token path is the unchanged, documented M1 fixture authority (20/20 exact,
completion-06 included). The batched path therefore produces the fixture-correct
completion-06 output on the M1; there is no batched-specific code path that could
diverge from the single-token logits it is proven equal to.

**Baseline reproduction (no perturbation)**: the batched path is additive — the
single-token kernels, `encode_forward`, the single-token `synapse_qwen3_metal_step`
entry, and the default `verify_tokens` routing are byte-for-byte unchanged (the
diff adds new kernels/functions and an opt-in env route; the batch buffers are
allocated lazily on first `verify_batch`, never on the default path). Measured in
the same harness/build on the M1, the single-token reference reproduces the pinned
f16 baseline (19.44 ms/token = 51.44 tok/s vs pinned 51.25, +0.4%), and batched
K=1 costs the same as single-token (Q8 8.35 vs 8.12 ms/token), confirming the
existing step path is unperturbed.

## Measurement (locked M1 authority)

Locked M1 Max (`<bench-host>`), AC power, exclusive `$SYNAPSE_BENCH_ROOT/bench.lock`
held then released, no `Runner.Worker`, 1-minute load 0.8-3.4 (< 8 bar). Built with
the M1's own cargo (1.97.1) and Xcode Metal toolchain, Qwen3-0.6B snapshot
`c1899de289a04d12100db370d81485cdf75e47ca`. Each K timed as the median of 40
`verify_batch(K)` calls (rewound to a fixed 64-token prefix between calls), two
independent runs agreeing to < 0.2%.

This is a NEW protocol (single-prompt, bucket 1024, verify-only wall time) reported
separately from the campaign baseline. The campaign single-token baseline stays
149.3952 tok/s (Q8, 12-prompt bucket-512 protocol); nothing about it is changed.

### Q8_0 per-token verify cost vs K

| K | call wall (ms) | per-token verify (ms) | verify tok/s-equiv | vs single-token (same harness, 8.12 ms/tok) | vs 149.40 baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.35 | 8.35 | 119.8 | 0.97x | 0.80x |
| 2 | 12.58 | 6.29 | 159.0 | 1.29x | 1.06x |
| 4 | 21.33 | 5.33 | 187.5 | 1.52x | 1.26x |
| 8 | 39.83 | **4.98** | **200.8** | **1.63x** | **1.34x** |
| 16 | 120.58 | 7.54 | 132.7 | 1.08x | 0.89x |

### f16 per-token verify cost vs K

| K | call wall (ms) | per-token verify (ms) | verify tok/s-equiv | vs single-token (same harness, 19.44 ms/tok) | vs 51.25 baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.82 | 18.82 | 53.1 | 1.03x | 1.04x |
| 2 | 22.75 | 11.38 | 87.9 | 1.71x | 1.72x |
| 4 | 31.15 | 7.79 | 128.4 | 2.50x | 2.51x |
| 8 | 48.72 | 6.09 | 164.2 | 3.19x | 3.20x |
| 16 | 88.93 | **5.56** | **179.9** | **3.50x** | **3.51x** |

**Reading the curves.** Q8 amortizes the weight stream best at K=8 (4.98 ms/token,
200.8 tok/s-equiv, +34% over the 149.40 single-token baseline), then regresses at
K=16: the N=16 template instantiation's 16 register accumulators per output row
drop occupancy enough that the per-token cost climbs back to 7.54 ms. f16 — less
register pressure (no Q8 partial arrays) — improves monotonically through K=16
(5.56 ms/token, +251% over the 51.25 baseline). The weight-streaming win is real
and large; the practical sweet spot is K=8 for Q8 and K=16 (or larger) for f16.

The K=1 batched point is NOT the single-token baseline: it carries the batched
buffer layout and template-switch overhead and does not use the campaign tree's
pack-four-rows tuning, so it sits slightly above the optimized single-token path.
The win is in K >= 2, where one weight stream serves multiple positions.

## Break-even: 0.6B-drafts-for-4B-target

Phase B computed a 4B break-even acceptance rate of **1.91 (impossible)** using
SEQUENTIAL verify (K=4 chain = 40.5 ms on M5) plus the contention-heavy in-loop
ANE draft (109.1 ms per 4-token proposal). Recomputing with the measured M1
batched-verify numbers and the standard Leviathan expected-acceptance model
(`E[alpha,K] = (1 - alpha^(K+1)) / (1 - alpha)` committed tokens per round; break-
even when `(T_draft + T_verify) / E[alpha,K] = T_target_step`):

Break-even acceptance alpha vs the 4B target single-token step cost (Q8, K=4):

| verify | draft cost / proposal | T4B=15ms | T4B=20ms | T4B=30ms | T4B=45ms |
| --- | --- | ---: | ---: | ---: | ---: |
| sequential | phase-B in-loop K4 (109.1ms) | impossible | impossible | 0.97 | 0.77 |
| **batched** | phase-B in-loop K4 (109.1ms) | impossible | impossible | 0.93 | 0.72 |
| sequential | phase-C unroll K4 (31.6ms) | 0.92 | 0.78 | 0.56 | 0.30 |
| **batched** | phase-C unroll K4 (31.6ms) | 0.82 | 0.68 | 0.44 | **0.15** |
| sequential | SPIKE-A isolated K4 (8.6ms) | 0.69 | 0.53 | 0.27 | 0.00 |
| **batched** | SPIKE-A isolated K4 (8.6ms) | 0.52 | 0.33 | **0.00** | **0.00** |

(`0.00` = spec decode is cheaper than the target even at zero acceptance, because
one batched verify of K tokens plus the draft costs less than one 4B target step;
`impossible` = even perfect acceptance cannot beat the target.)

Affordable draft budget per round (ms) at a given acceptance rate, batched verify
(what a drafter may spend and still break even):

| alpha | K | T4B=15ms | T4B=20ms | T4B=30ms | T4B=45ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 4 | 7.7 | 17.4 | 36.8 | 65.9 |
| 0.67 | 4 | 18.0 | 31.1 | 57.3 | 96.6 |
| 0.67 | 8 | 4.4 | 19.1 | 48.6 | 92.8 |
| 0.80 | 4 | 29.1 | 45.9 | 79.5 | 129.9 |
| 0.80 | 8 | 25.1 | 46.7 | 90.0 | 155.0 |

**Verdict (the number the program forks on).** Batched verification removes the
TARGET-SIDE verifier as the bottleneck: with a realistic unrolled-K4 draft
(31.6 ms), the 4B pairing breaks even at alpha = 0.15-0.82 across plausible 4B
target step costs (45-15 ms) — all well below the 67% acceptance phase B measured
for real. With a cheap draft (SPIKE-A isolated 8.6 ms) it breaks even at alpha <=
0.52 everywhere and is unconditionally cheaper for a >= 30 ms 4B target. The 4B
target step cost is UNMEASURED in this program (no 4B model was run); the table is
the sensitivity over that unknown, scaled from the measured 0.6B Q8 step. The
binding constraint is now squarely the DRAFTER, which phase B/C closed as dead at
the stateless ANE shapes — batched verify makes the verifier cheap enough that any
drafter under ~30-60 ms per K=4 proposal (at 67% acceptance, 30 ms 4B target) flips
the 4B pairing positive.

## Reproducing

```text
# Build (M1 or M5; Xcode Metal toolchain required)
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cargo build -p spike-unified-rt --release --bin spike-unified-rt

# Correctness gates (any Apple GPU; need the Qwen3-0.6B snapshot)
SYNAPSE_UNIFIED_RT_QWEN3_0_6B=<snapshot> \
  cargo test -p spike-unified-rt --release -- --ignored batched_verify

# Timed curve (run on the locked M1; Q8 default, f16 via the env)
SYNAPSE_UNIFIED_RT_QWEN3_0_6B=<snapshot> \
  cargo test -p spike-unified-rt --release -- --ignored --nocapture batched_verify_timing_probe
SYNAPSE_METAL_STEP_BATCHED_PROBE_QUANT=f16 ... (same)

# Use the batched verifier in the speculative session (opt-in; default unchanged)
SYNAPSE_METAL_STEP_BATCHED_VERIFY=1 spike-unified-rt ... --decode-backend metal-step --speculative-draft ane
```
