# What does verifying K tokens cost on the owned Metal decode engine? (Apple M5 Max)

**Status: harness built and smoke-tested. The full measurement has not been run yet.**
Every number on this page comes from a smoke pass. The smoke passes ran while the
machine was under a very heavy load (1-minute loadavg 78–97). They prove the
harness works. They are not a result.

## Why

Speculative decoding guesses K tokens cheaply, then checks all K with the full model
in one forward pass. It only pays if that pass costs much less than K single-token
steps. A published Metal engine for a 4B model reports that verifying 8 tokens costs
1.5x one token. They got down from 2.3x by splitting one long serial loop across the
GPU's threadgroups. Our only comparable number is old: about 6x at K=8 for Q8 on an
M1 with Qwen3-0.6B. That was a different chip and a different model size. This
harness measures our own curve on this M5 Max. With it we can decide whether the
threadgroup split is worth porting.

## Which lanes support batched verify (from source)

Both Qwen3-0.6B weight lanes, **f16 and Q8_0**, support batched verify. Activations
are always f16. K can be 1 to 16.

- `crates/synapse-engine-owned/owned-decode-engine/src/qwen3_decode_metal_step.rs`
  - `:100-103`: `MetalStepDecoder::new` requires `Precision::F16` activations. Weight
    quantization is `None` (f16) or `Q8_0`, and it is forwarded to the driver as
    `quantized` (`:152`).
  - `:339`: `MAX_BATCH_VERIFY_TOKENS = 16`. `:399-475`: `verify_tokens_batch` →
    `verify_tokens_batch_inner` checks 1 ≤ K ≤ 16 and cache capacity. It does not
    check quantization.
- `…/qwen3_decode_metal_step.m`
  - `:15`: `METAL_STEP_MAX_BATCH_K 16`.
  - `:1178-1258`: `synapse_qwen3_metal_step_verify_batch` validates only
    steps/capacity (`:1194-1199`). `ensure_batch_resources` (`:781-835`) has no
    quantization gate either.
  - The four batched projection encoders pass `context->quantized` and choose the
    dispatch shape from it: QKV `:1021-1028`, O/down `:1050-1055`, gate/up
    `:1072-1077`, LM head `:1091-1096`.
- `…/qwen3_decode_metal_step.metal`: each batched kernel has an f16 branch
  (`config.quantized == 0`) and a Q8_0 branch. QKV `:867`/`:891`, matvec+residual
  `:967`/`:978`, gate/up `:1033`/`:1047`, LM head `:1106`/`:1115`. They are templated
  on K rounded up to 1/2/4/8/16 (`:942-953`).
- `crates/synapse-engine-owned/tests/owned_decode_prefill_chunking.rs:296-303` checks
  batched logits against sequential single-token logits, byte for byte, for both
  f16 and Q8_0.

## Finding: the engine's kernel profiler can't see batched verify

`SYNAPSE_METAL_STEP_PROFILE=1` does not cover the verify paths.

- The variable is read into `context->profile_kernels` (`qwen3_decode_metal_step.m:315-316`).
  Only `synapse_qwen3_metal_step` (`:1538-1593`) reads that flag. That function is the
  host-fed single-token `advance` path. There, each kernel class is run in its own
  command buffer and the GPU start/end times are summed per class.
  `synapse_qwen3_metal_step_verify_batch` (`:1178`) and `synapse_qwen3_metal_step_verify`
  (`:1303`) never check it. Both record only wall-clock totals.
- The per-kernel totals are stored in a C struct. The only way to read them is
  `synapse_qwen3_metal_step_timings` (`:1651-1654`). The Rust FFI block
  (`qwen3_decode_metal_step.rs:704-779`) does not bind that function, and
  `MetalStepDecoder`'s context pointer is private. So code outside the engine can't
  read the profiler numbers even for K=1.

Getting per-kernel batched timings from inside the engine would take an engine
change. That change was ruled out: this driver belongs to a certified lane, and a
profiling hook would change the compiled binary. This harness measures from outside
instead (see the next two sections).

## Method (`crates/synapse-engine-owned/examples/verify_k_cost.rs`)

- **Model**: `SYNAPSE_OWNED_DECODE_QWEN3_0_6B` points at the snapshot directory. This is
  the same variable the other owned-decode harnesses use. Smoke passes used
  `Qwen/Qwen3-0.6B` (the causal LM, `c1899de2…`). Context bucket is 2048, the
  production workload bucket.
- **Depths**: the prompt text is tokenized and repeated until 32 or 470 tokens sit in
  the KV cache. It is prefilled in batched 16-token chunks.
- **Correctness guard (runs before any timing)**: at each depth, 16 sequential
  single-token `advance` calls produce the model's greedy continuation. That
  continuation becomes the draft. For every measured K:
  - `verify_tokens_batch_logits(draft[..K])` must be bit-identical to the sequential
    logits;
  - `verify_tokens_batch(draft[..K])` must return the same greedy tokens.
  If either check fails, the run stops. Every timed call also checks the tokens it
  returns. Because the draft is the greedy stream, every verify is a full accept.
- **Arms**:
  - `batch` is `verify_tokens_batch`, the thing being measured.
  - `sequential` is `DecodeKernel::verify_tokens`, which runs K ordinary single-token
    forward passes in one command buffer. It shows what K single-token steps cost
    with no host round trips, for context only.
- **Timing**: wall time of one verify call. Before each call the cache is rewound to
  the depth, so every call rewrites the same KV slots with the same values. The
  rewind itself is not timed.
- **Order**: warm-up first (20 calls per arm and K). Then 300 rounds. Each round runs
  every arm and K once, and the order rotates by one each round. Ambient load
  therefore hits all K alike.
- **Statistics**: median, p10, p90, min, max (nearest rank). Two ratios:
  - ratio of medians, median(K) / median(batch K=1);
  - paired ratio, t(K) / t(batch K=1) within one round, summarized the same way.
    This one is more robust to load drift.
- **Load**: the 1/5/15-minute load averages and the `uptime` line are recorded at the
  start and end of every 50-round block, and in the JSON report.
- **Depth split**: attention is the only kernel whose work grows with the number of
  cached positions (`metal_step_attention` loops over `position <= config.position`).
  For each K, the harness reports:
  - `extra_short = median(K) − median(1)` at depth 32;
  - `extra_long`, the same at depth 470;
  - `depth_dependent = extra_long − extra_short`.
  - **What this can separate**: whether K's extra cost depends on depth (attention) or
    is flat (weight GEMV/GEMM plus fixed dispatch and launch cost).
    `depth_dependent` is a lower bound on attention's share at depth 470, because
    depth 32 already includes 32 positions of attention.
  - **What it cannot separate**: individual kernels inside the flat part (QKV vs O vs
    gate/up vs down vs LM head vs norms and argmax), or scan cost vs dispatch cost
    inside attention. Only per-kernel GPU timing can do that, and nothing available
    here provides it (see "Per-kernel attribution" below).

## Full-run command (run only when the machine is free)

```sh
cd <synapse checkout>
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export SYNAPSE_OWNED_DECODE_QWEN3_0_6B=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
cargo build -p synapse-engine-owned --release --example verify_k_cost
uptime   # expect a low loadavg; nothing else should be using the GPU
SYNAPSE_VERIFY_K_OUT=docs/evidence/verify-k-cost-m5/full.json \
  ./target/release/examples/verify_k_cost | tee docs/evidence/verify-k-cost-m5/full.log
```

It covers both lanes, both depths, K ∈ {1,2,4,8,16}, both arms, 20 warm-up rounds and
300 timed rounds.

**Expected duration**: about 10 minutes at the smoke-pass speed. That estimate
extrapolates the smoke medians, taken at loadavg 80–100, to 320 rounds × 2 lanes ×
2 depths. The `sequential` arm is about 60% of it. A quiet machine should be faster.
`SYNAPSE_VERIFY_K_ROUNDS=<n>` changes the round count, and
`SYNAPSE_VERIFY_K_LANES=f16` or `=q8` runs a single lane. The first build compiles
the workspace dependencies, which took 15 minutes on the loaded machine.

What to read: the `SUMMARY` table printed before the JSON. The main figure is the
`batch` rows' `ratio` and `paired` columns. The `lane … K=… extra over K=1` lines
give the depth split.

## GPU time per verify call (Metal System Trace)

Wall time includes host encoding and waiting on other GPU clients. The trace gives
GPU-busy time per verify call instead. In trace mode each verify call is exactly one
command buffer, and each (depth, K) phase is 200 back-to-back batched calls with two
idle seconds before and after.

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export SYNAPSE_OWNED_DECODE_QWEN3_0_6B=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
BIN=$PWD/target/release/examples/verify_k_cost
for LANE in q8 f16; do
  xcrun xctrace record --template 'Metal System Trace' \
    --output /tmp/verify_k_$LANE.trace --target-stdout - \
    --env SYNAPSE_VERIFY_K_MODE=trace --env SYNAPSE_VERIFY_K_LANES=$LANE \
    --env SYNAPSE_OWNED_DECODE_QWEN3_0_6B=$SYNAPSE_OWNED_DECODE_QWEN3_0_6B \
    --launch -- "$BIN" | tee docs/evidence/verify-k-cost-m5/trace_$LANE.log
  python3 docs/evidence/verify-k-cost-m5/trace_gpu_time.py /tmp/verify_k_$LANE.trace \
    | tee docs/evidence/verify-k-cost-m5/trace_$LANE.gpu.txt
done
```

How to read it. No Instruments GUI is needed.

1. The program log has four `TRACE PHASE lane=… depth=… K=… calls=200 …` lines per
   lane, in the order depth 32 K=1, depth 32 K=8, depth 470 K=1, depth 470 K=8.
2. `trace_gpu_time.py` prints one row per cluster of command buffers. Each phase is
   one cluster with exactly `calls` (200) buffers. The clusters with other counts are
   setup: model upload, prefill, correctness guard, warm-up. Their order matches the
   log. Read `median_ms` for each phase cluster. GPU-time cost(8)/cost(1) is
   median(K=8 cluster) / median(K=1 cluster) at the same depth.
3. The script keeps only the `verify_k_cost` process's `Compute` intervals from the
   trace's `metal-gpu-intervals` table and sums them per command buffer
   (`cmdbuffer-id`). When other processes share the GPU, one command buffer appears
   as several pieces. Summing the pieces leaves out the gaps where other work ran.

Cost: each recording took about 2.5 minutes of wall time and made about 300 MB of
trace for a few seconds of our GPU work. The trace records every process. Delete the
`.trace` files afterwards.

### Per-kernel attribution: not available from the command line (verified)

Metal System Trace on this Xcode merges all compute encoders of a command buffer into
one `Compute Command 0` interval. That was checked on the smoke trace: the intervals
of each command buffer have one encoder id and the label
`Command Buffer 0:Compute Command 0`. So the trace gives GPU time **per verify call,
not per kernel**. The trace also has a per-shader table, `metal-shader-profiler-intervals`
(columns Shader Name, Pipeline, Sample Duration), but it was empty with the stock
template. Filling it needs the Metal System Trace recording option "Shader Timeline".
That option is set in the Instruments GUI and saved as a custom template, which is
then passed to `xctrace record --template <file>.tracetemplate`. **That route has not
been tested here.** Until it is, per-kernel attribution comes from the depth split
above plus the source reading below.

## What the source already says about where K=8's extra cost goes

`encode_forward_batch` (`qwen3_decode_metal_step.m:1108-1170`) batches only the
projections. Everything else is dispatched once per draft column, and each dispatch
is its own compute encoder:

- Per layer, per column: input `rmsnorm` (`:1130-1134`), `qk_norm_rope`
  (`:1136-1140`), `attention` (`:1141-1145`), `residual_rmsnorm` (`:1148-1152`). Per
  layer, batched: QKV, O, gate/up, down.
- After the layers: final `rmsnorm` × K, one batched LM head, then argmax partial +
  final × K (`:1158-1169`). Embedding gather × K (`:1223-1227`).
- Encoder count per verify call: K=1 → 1 + 28×8 + 1 + 1 + 2 = **229**; K=8 →
  8 + 28×36 + 8 + 1 + 16 = **1041** (4.5×). The per-column encoders read and write
  shared buffers, so under Metal's default hazard tracking they run one after
  another.

The attention kernel (`qwen3_decode_metal_step.metal:345-430`) is the "long serial
loop" pattern that the published threadgroup split targets:

- It launches one 32-lane simdgroup per query head: `query_heads × 32` = 512 threads,
  which is 16 simdgroups for the whole GPU (`.m:919`).
- Lane 0 alone runs the softmax max and denominator over every cached position
  (`.metal:393-400`).
- Each lane then walks every position again for the value sum (`:408-415`).
- In batched verify this runs once per column, one column after another. K=8 at
  depth 470 is 8 serial runs of a 16-simdgroup kernel, each scanning about 470
  positions on one lane.

The smoke pass points the same way. Treat it as indicative only, because of the load.

- Q8_0: 74% of K=8's extra wall time at depth 470 depends on depth.
- f16: 59%.
- Trace smoke (Q8_0): GPU time per call rose from 4.3 ms at depth 32 to 9.3 ms at
  depth 470 at K=1, and to 42.4 ms at depth 470 at K=8.

The flat part also has a candidate. The f16 batched projections run one thread per
output row, and that thread computes a serial dot product for all N columns
(`.metal:736-785`, dispatched as `output_width` threads: 1024 threads for O and down).
So f16 per-thread work grows N× with little extra parallelism. Q8_0 spreads each row
over 8 sub-lanes (`:891-926`, `:978-996`).

## Smoke-pass numbers (SMOKE PASS, NOT A RESULT)

Run on 2026-09-23 at 16:18 local time, Apple M5 Max 128 GiB, Qwen3-0.6B, K ∈ {1,8},
3 warm-up rounds, 12 timed rounds (2 blocks of 6). The whole process took 69 s of
wall time, model loads included. The correctness guard passed for both lanes at both
depths: batched logits were bit-identical to sequential steps and the greedy tokens
matched.

**Load was extreme**: 1-minute loadavg 78.4 → 97.1 during the samples (`uptime` at
start: `load averages: 44.85 36.13 36.76`, climbing). Other processes were using the
GPU heavily; the trace shows other GPU clients interleaved with ours. The absolute
times are several times what a quiet machine should give, and the spreads are wide.

Wall ms per verify call. ratio = median(K)/median(batch K=1). paired = median of the
per-round t(K)/t(batch K=1).

| lane | depth | arm | K | median | p10 | p90 | min | max | ratio | paired |
|---|---|---|---|---|---|---|---|---|---|---|
| f16 | 32 | batch | 1 | 10.30 | 9.22 | 19.70 | 8.93 | 20.92 | 1.00 | 1.00 |
| f16 | 32 | batch | 8 | 24.29 | 22.91 | 56.37 | 21.28 | 88.42 | 2.36 | 2.31 |
| f16 | 32 | sequential | 8 | 73.59 | 71.42 | 123.23 | 68.49 | 125.99 | 7.14 | 6.91 |
| f16 | 470 | batch | 1 | 18.65 | 16.42 | 23.28 | 15.45 | 25.37 | 1.00 | 1.00 |
| f16 | 470 | batch | 8 | 52.76 | 49.76 | 62.79 | 48.18 | 70.34 | 2.83 | 2.82 |
| f16 | 470 | sequential | 8 | 118.61 | 117.11 | 123.15 | 116.05 | 141.01 | 6.36 | 6.32 |
| q8_0 | 32 | batch | 1 | 8.24 | 4.75 | 11.78 | 4.61 | 12.79 | 1.00 | 1.00 |
| q8_0 | 32 | batch | 8 | 17.26 | 15.83 | 18.91 | 15.75 | 22.44 | 2.09 | 2.21 |
| q8_0 | 32 | sequential | 8 | 35.14 | 32.57 | 43.57 | 32.40 | 67.46 | 4.26 | 4.11 |
| q8_0 | 470 | batch | 1 | 10.49 | 9.86 | 13.96 | 9.74 | 22.03 | 1.00 | 1.00 |
| q8_0 | 470 | batch | 8 | 44.53 | 42.75 | 47.50 | 42.75 | 49.65 | 4.25 | 4.20 |
| q8_0 | 470 | sequential | 8 | 78.60 | 77.49 | 82.32 | 75.14 | 86.19 | 7.49 | 7.49 |

The sequential K=1 rows matched batch K=1 within noise (f16 ratio 0.95/1.00; Q8_0
0.65/1.03, with the depth-32 Q8_0 batch K=1 samples spread 4.6–12.8 ms). They are
left out of the table.

Depth split (smoke):

| lane | K | extra over K=1 at depth 32 | at depth 470 | depth-dependent | share of depth-470 extra |
|---|---|---|---|---|---|
| f16 | 8 | 13.98 ms | 34.12 ms | 20.13 ms | 59% |
| q8_0 | 8 | 9.01 ms | 34.04 ms | 25.03 ms | 74% |

Trace smoke (Q8_0 only, 5 calls per phase, loaded machine). GPU-busy ms per verify
call from `trace_gpu_time.py`: depth 32 K=1 4.28; depth 470 K=1 9.32; depth 470
K=8 42.37 (GPU-time ratio 4.5). In this run the depth-32 K=8 phase merged into the
next depth's setup cluster: the binary used for it did not yet have the idle tail
after the last phase, which has since been added. That merge is why no depth-32 K=8
number appears here.
