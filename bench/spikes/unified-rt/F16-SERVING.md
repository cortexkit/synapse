# f16 serving path assembly

## Status

The runtime now accepts `--dtype f16` for MiniLM, gte-modernbert, and Qwen3-Embedding-0.6B. Metal uses explicit `MPSGraphExecutable` compilation with optimization level 0, synchronous compilation, and explicit specialization by default. `--execution lazy` keeps the old `runWithFeeds` path for A/B checks.

`--package-cache DIR` stores one `.mpsgraphpackage` per `(batch, seq)` shape. The cache directory key includes model-family, graph revision, a hash of the canonical checkpoint path, dtype, and macOS build; the filename supplies the shape. Packages are never appended. A cache hit uses `initWithMPSGraphPackageAtURL:compilationDescriptor:` and a miss compiles, specializes, and serializes before the first execution of that shape.

This assembly is certified against ORT fp32 for all three families. MiniLM was re-certified on the locked M1 against the frozen evidence pack's original reference; ModernBERT and Qwen3 also pass the full 400-vector gate. The locked-machine matrix and reference provenance are recorded in [M1-SERVING-MATRIX.md](M1-SERVING-MATRIX.md).

## Embedding campaign baseline

The first embedding-throughput lane is registered as
`gte-modernbert-f16-metal-embed` on the locked M1. It measures the owned
`Alibaba-NLP/gte-modernbert-base` snapshot at revision
`e7f32e3c00f91d699e8c43b53106206bcc72bb22`, f16 weights, Metal, bucket policy v1,
`max_length=512`, and a 4,000,000 attention-unit budget. The objective is the
median steady-state batch-embedding rate in tok/s. Each of six folds runs two
fresh processes; each process performs seven passes, discards pass one as the
load/warmup pass, and the fold scores the worse of the two remaining steady
observations. The campaign controller also requires finite, nonzero 768-wide
vectors and a same-process repeated-input determinism probe.

The fixture is a deterministic 2,000-row stratified selection from
`bench/data/corpus-v2.jsonl`: seed `metal-embed-corpus-v1:20260723`, rows grouped
by the first path component, each group and row ordered by SHA-256 of the seed
plus its label, then selected round-robin until 2,000 rows. The committed fixture
contains only `{id,text}` rows and is pinned by corpus SHA-256
`25d1d54427030d94c882dd96a5f5d26bfda426d902028e75aa8c3d527e34a7a7`.
Reference vectors are the current master's own owned-runtime output on exactly
that slice (source `056487f508693ff3539c71c32874bf32fad7aa00`, CLS pooling plus
L2 normalization), encoded as pinned float32 rows with SHA-256
`d55221d41098aa293507c734ebedbf2df7f095c5e7c767943167403bbb520afd`. The hard
quality floor is mean cosine `>= 0.9999` and worst-decile top-10 rank overlap
`>= 0.97`; any parity, dimensionality, finite/nonzero, or determinism failure
rejects the speed result.

The original fresh locked-M1 campaign baseline (the pre-winner bar; see the
campaign #1 result below for the current baseline) is **18,500.3485 tok/s**
(median of the
36 worse-of-two steady samples), recorded in
`results/metal-embed-campaign/m1-baseline.json`. The run used
`<bench-host>` (Apple M1 Max), acquired and released
`$SYNAPSE_BENCH_ROOT/bench.lock`, and rejected an active `Runner.Worker`. Its minimum
mean cosine was `0.999999022` and minimum worst-decile top-10 overlap was
`0.97400`. The fleet's ordinary coalesced observation remains about 14--15k
tok/s, and the earlier
local M5 serving note recorded 19,740.1 tok/s; neither substitutes for this
locked-M1 campaign bar. The same-session MLX Python cell was skipped because
`/tmp/synapse-mlx-minilm-venv/bin/python` was absent on the M1. The optional
llama-server cell was also skipped because no lane-specific command/model was
configured.

### Campaign #1 result: fused scaled-dot-product attention (integrated)

Embedding campaign #1 (`release-embedding-benchmark-1`, harness
`gte-modernbert-f16-metal-embed`) hit `time_budget_exhausted` mid-round-1, but
its one fully-measured proposal was a clean gated WIN, promoted inside the round
before the clock ran out. The winner is **proposal_bcc9cf06** "Fused
scaled-dot-product attention for ModernBERT Metal encoder prefill"
(slot0:anthropic/claude-fable-5).

Mechanism: the encoder's materialized-score attention chain (transpose -> matmul
-> cast f32 -> scale -> mask add -> softmax -> cast back -> PV matmul) is
replaced with MPSGraph's fused `scaledDotProductAttentionWithQueryTensor:...`
op. The fused path is `@available(macOS 15.0, *)`-guarded; Q, K, and V are cast
to f32 for the fused op and the context is cast back to the graph dtype. The
additive mask (full or local sliding-window) feeds the fused op directly, so
ModernBERT's alternating global/local attention is preserved. The pre-macOS-15
materialized-score chain is retained verbatim in the `else` branch as the safety
net, and both branches compile against the current SDK.

In-block measurement on the locked M1 (full 6x2 interleaved block, harness
quality battery run per sample side): candidate median **19,677.8 tok/s** vs
paired control median **18,533.1 tok/s** = **+6.2%** (bootstrap CI
[+6.12%, +6.32%], `target_met: true`, `promoted: true`).

M1 confirmation for this integration (this branch, harness
`17dd612a...`-pinned, fixtures digest-pinned, AC power, no `Runner.Worker`,
one-minute load < 2.5): baseline tree at `a2d4fdf` median **18,502.5549 tok/s**
vs patched tree median **19,683.0005 tok/s** = **+6.38%** (36 worse-of-two steady
samples per side). The full quality battery passed on the patched tree: mean
cosine floor `0.999999071` (>= 0.9999), worst-decile top-10 rank overlap floor
`0.97500` (>= 0.97), byte-identical same-process determinism repeat, and
finite/nonzero 768-wide vectors. The baseline tree reproduced the frozen
18,500.3485 tok/s bar to 0.01%, evidencing a drift-free box. The fused path is
confirmed executing at runtime (macOS 26.5 on the M1) by the reproduced speedup.

The new locked-M1 campaign baseline is therefore **19,683.000545618823 tok/s**
(the M1-confirmed patched median). The registration baseline in
`.cortexkit/campaign-lab.jsonc` is bumped to this value, and the harness
`BASELINE_TOK_S` constant plus its `harness_sha256` pin move together with it
(harness `41e1ba34e7bc02860b39cff1d68241244367a7d6406f41a7f5fd5d90ff3dc118`),
keeping the harness's pinned-baseline consistency check satisfied.

The campaign's other four proposals died unmeasured when the budget exhausted;
they are the seed material for the next embedding campaign:

1. **Compile encoder MPSGraph executables at optimization level 1 instead of O0**
   (slot1:anthropic/claude-opus-4-8) — raise the explicit-executable compile
   level so MPSGraph fuses/reduces the 22-layer graph; watch the package-cache
   reuse branch that can silently keep an O0 package.
2. **Compile explicit MPSGraph executables at O1 with a separate cache identity**
   (slot4:alibaba-token-plan/qwen3.8-max-preview) — the same O1 lever with a
   distinct package suffix to avoid stale O0 executables (forces a one-time
   recompile).
3. **Cache bucket-owned RoPE feeds and accelerate additive mask construction**
   (slot2:openai/gpt-5.6-sol) — reuse the lifetime-stable bucket RoPE storage
   through the static Metal-buffer cache and stop reallocating/copying the four
   immutable RoPE tables per batch; mask/RoPE bytes unchanged (low parity risk).
4. **Broadcast reformulation of the attention masks**
   (slot3:kimi-for-coding/k3) — kill the O(batch*seq^2) CPU mask build and the
   16.8MB/batch mask feeds by deriving the key-padding mask from key position
   alone; estimated 2-6%, needs a package-cache clear / GRAPH_REVISION bump for
   the changed placeholder shapes.

### First-campaign steering inventory

The first campaign should spend its search budget on the following untried or
unmeasured levers, without changing the vector fingerprint:

1. **Fused/flash-style encoder prefill attention.** TEI's CUDA advantage points
to fused attention plus dynamic batching. The owned Metal encoder has never had
that fused/flash-style prefill pass; this is the primary Metal hypothesis.
2. **Bucket-policy v3 shapes.** Policy v1 is the serving baseline and v2 is
retained as a rejected A/B experiment. A v3 shape/row ladder must be a new
policy identity and must re-run the padding and parity gates rather than reuse
v1 packages.
3. **Norm fusion.** Norm fusion is proven on the CUDA decode path but has not
been tried on the owned ModernBERT encoder. It is valid only if the pinned
fingerprint remains inside both quality thresholds.
4. **Batched-layer dispatch consolidation.** The embedding path's per-batch
Metal dispatch count is not measured in the unified runner. Production inline
hooks (`SYNAPSE_EMBED_PROFILE=1`) already attribute scheduler dispatch, engine
mutex, bucket/executable selection, MPSGraph execution, and readback; they do
not yet expose the unified runner's layer-dispatch table. The first profiling
ask is therefore a per-batch attribution table for those layer dispatches, with
an explicit `unmeasured` cell until the hook exists.

## f16 parity

Certification thresholds are mean cosine `>= 0.9999` and mean top-10 overlap `>= 0.995` against ORT fp32.

| Family | Rows | f16 mean cosine | Top-10 overlap | Result | Topology |
|---|---:|---:|---:|---|---|
| MiniLM | 400 | `0.9999993186` | `0.999250` | **PASS** | native f16 matmuls and inter-block activations; fp32 layernorm, GELU, and softmax islands |
| gte-modernbert | 400 | `0.9999990763` | `0.998500` | **PASS** | f16 storage/inter-block activations; fp32 norm scales, matmuls, RoPE, layernorm, additive masks, and softmax |
| Qwen3-Embedding-0.6B | 400 | `0.9999987315` | `0.998500` | **PASS** | native f16 weights/matmuls/inter-block activations; fp32 RMSNorm and masked softmax |

The rejected regenerated MiniLM ONNX reference scored `0.99920633` against the unchanged fp32 runtime and `0.99920582` against f16, so it was not equivalent to the existing parity oracle. The locked-M1 rerun used the frozen evidence pack's original reference and passed both gates for fp32 and f16. `cargo test` still covers f16-vs-fp32 behavior for the tiny resident encoder block.

### ModernBERT differential

A lazy diagnostic run set `SYNAPSE_MODERNBERT_DUMP_DIR` for matching one-row fp32 and f16 executions. The first layer stays aligned through global-theta RoPE and masked attention, then diverges at the first MLP layer normalization:

| Layer-0 stage | cosine vs fp32 | relative L2 |
|---|---:|---:|
| QKV projection | `0.999999965` | `0.000263` |
| query after RoPE | `0.999999941` | `0.000343` |
| softmax | `0.999999828` | `0.000608` |
| attention context | `0.999999681` | `0.000803` |
| attention residual | `0.999999952` | `0.000309` |
| MLP normalization | `0.853556339` | `2.029007` |
| MLP projection | `0.855836552` | `1.593074` |
| MLP output | `0.773072914` | `5.301206` |

The earlier all-matmul-fp32 fallback already diverged at layer 0 and compounded across the stack:

| Layer | cosine vs fp32 | relative L2 |
|---:|---:|---:|
| 0 | `0.667685672` | `5.268641` |
| 1 | `0.649719347` | `5.128571` |
| 4 | `0.467170140` | `2.245389` |
| 11 | `0.032200779` | `1.002839` |
| 21 | `0.474928131` | `0.941781` |

This killed the initial “global theta 160,000 RoPE is the first divergence” hypothesis. The root cause was the pointer-keyed static Metal buffer cache receiving per-call f16 norm-scale temporaries. Allocator address reuse let later calls bind a cached buffer for a different layer's gamma. Feeding all norm scales as fp32 from model-owned storage removed that lifetime violation. The repaired layer-0 stages remain aligned through the MLP:

| Repaired layer-0 stage | cosine vs fp32 | relative L2 |
|---|---:|---:|
| MLP normalization | `0.999999936` | `0.000357` |
| MLP projection | `0.999999922` | `0.000396` |
| MLP output | `0.999999944` | `0.000334` |

## Contended local throughput

These are Apple M5 Max, macOS 26.5.2 build 25F84 measurements. They are explicitly **not** the requested locked M1 first/warm/steady protocol and must not replace it. Each shown run used explicit O0 and per-shape packages; cache state is noted.

| Family / 400-row corpus | dtype | tokens | cache state | tok/s | f16/fp32 |
|---|---|---:|---|---:|---:|
| MiniLM parity corpus | fp32 | 158,310 | miss/compile | `71,950.2` | — |
| MiniLM parity corpus | f16 | 158,310 | hit/load | `86,957.7` | `1.21x` |
| gte-modernbert standard corpus | fp32 | 62,838 | hit/load | `23,183.6` | — |
| gte-modernbert standard corpus | f16 | 62,838 | hit/load | `17,063.7` | `0.74x` (fixed parity) |
| Qwen3 standard corpus | fp32 | 46,716 | mixed cache | `6,551.5` | — |
| Qwen3 standard corpus | f16 | 46,716 | mixed cache | `6,813.0` | `1.04x` |

The locked-M1 comparison is now complete under the benchmark-lock and `Runner.Worker` protocol; see [M1-SERVING-MATRIX.md](M1-SERVING-MATRIX.md). These M5 rows remain useful only as the contended-host comparison captured before that run.

## Magic-context corpus payoff

The requested `/tmp/mc-corpus.jsonl` run completed on the contended M5 host:

| Path | chunks | tokens | wall | tok/s | versus 342 s fp32 baseline |
|---|---:|---:|---:|---:|---:|
| gte-modernbert f16 + explicit O0 (invalid norm feeds) | 11,293 | 4,172,183 | `214.032 s` | `19,493.3` | `1.60x` / 37.4% less wall time |
| gte-modernbert f16 + explicit O0 (fixed norm feeds) | 11,293 | 4,172,183 | `211.356 s` | `19,740.1` | `1.62x` / 38.2% less wall time |

The fixed payoff run passed the separate 400-row parity gate and was labeled `gte-modernbert-f16-o0-contended-fixed-parity`. Both payoff measurements are contended local results, not locked-machine serving numbers.

## Package cache behavior

A fresh-process MiniLM cache-hit run loaded 19 shapes in `0.123996 s` total (`6.526 ms/shape`). Package sizes on this SDK were:

| Family/dtype | package count | per-package size | total |
|---|---:|---:|---:|
| MiniLM f16 | 19 | `54.585–54.603 KiB` | `1,037.37 KiB` |
| MiniLM fp32 | 19 | `49.991–50.005 KiB` | `950.04 KiB` |
| Qwen3 f16 | 5 | `261.389–261.535 KiB` | `1,307.39 KiB` |
| Qwen3 fp32 | 5 | `203.681–203.795 KiB` | `1,018.77 KiB` |
| ModernBERT f16 MC corpus | 162 | `189.494–189.670 KiB` | `30,720.88 KiB` |

The 162-package MC result reflects exact dynamic `(batch, seq)` shapes. A production bucketing policy should reduce that set before graduation. Package compatibility is separated by OS build and an explicit graph revision, so placeholder or topology changes cannot load stale packages. A checkpoint replaced in-place at the same canonical snapshot path would retain the same model key; immutable snapshot paths are assumed for this spike.

## Commands executed

Build and gates:

```sh
cargo fmt --manifest-path bench/spikes/unified-rt/Cargo.toml
cargo test --manifest-path bench/spikes/unified-rt/Cargo.toml
cargo build --release --manifest-path bench/spikes/unified-rt/Cargo.toml

BIN=target/release/spike-unified-rt
QWEN=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
$BIN --model "$QWEN" --tokenizer "$QWEN/tokenizer.json" \
  --corpus /tmp/qwen3-corpus-400.jsonl --reference /tmp/qwen3-ort-400-vectors.jsonl \
  --out target/qwen3-f16-400.json --dtype f16 --device metal \
  --package-cache target/qwen-packages
$BIN --model "$QWEN" --tokenizer "$QWEN/tokenizer.json" \
  --corpus /tmp/qwen3-corpus-400.jsonl --reference /tmp/qwen3-ort-400-vectors.jsonl \
  --out target/qwen3-f32-400.json --dtype f32 --device metal \
  --package-cache target/qwen-f32-packages

GTE=$HOME/.cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots/e7f32e3c00f91d699e8c43b53106206bcc72bb22
$BIN --model "$GTE" --tokenizer "$GTE/tokenizer.json" \
  --corpus /tmp/modernbert-corpus-400.jsonl --reference /tmp/modernbert-ort-400-vectors.jsonl \
  --out target/modernbert-f16-400-parity.json --dtype f16 --device metal \
  --package-cache target/mb-f16-packages --min-parity 0 --min-rank-overlap 0
$BIN --model "$GTE" --tokenizer "$GTE/tokenizer.json" \
  --corpus /tmp/modernbert-corpus-400.jsonl --reference /tmp/modernbert-ort-400-vectors.jsonl \
  --out target/modernbert-f32-400.json --dtype f32 --device metal \
  --package-cache target/mb-f32-packages
$BIN --model "$GTE" --tokenizer "$GTE/tokenizer.json" \
  --corpus /tmp/mc-corpus.jsonl --out target/modernbert-f16-mc.json \
  --dtype f16 --device metal --package-cache target/mb-f16-packages \
  --model-label gte-modernbert-f16-o0-contended-invalid-parity

MINILM=$HOME/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
$BIN --model "$MINILM" --tokenizer "$MINILM/tokenizer.json" \
  --corpus bench/data/corpus-parity.jsonl --limit 400 \
  --out target/minilm-f16-400.json --dtype f16 --device metal \
  --package-cache target/minilm-packages
$BIN --model "$MINILM" --tokenizer "$MINILM/tokenizer.json" \
  --corpus bench/data/corpus-parity.jsonl --limit 400 \
  --out target/minilm-f32-400.json --dtype f32 --device metal \
  --package-cache target/minilm-f32-packages
```

Layer differential commands used the same one-row ModernBERT command twice, changing dtype and dump directory:

```sh
SYNAPSE_MODERNBERT_DUMP_DIR=target/mb-dump-f32 $BIN ... --limit 1 --dtype f32 --execution lazy
SYNAPSE_MODERNBERT_DUMP_DIR=target/mb-dump-f16 $BIN ... --limit 1 --dtype f16 --execution lazy
```

## Bucket-policy follow-up

The three serving-shape open items are implemented by [BUCKET-POLICY.md](BUCKET-POLICY.md): the main harness supports in-process `--passes`, accelerator bucket shapes are pre-discovered during cold load, and bucket policy v1 replaces corpus-dependent shapes by default. `--shapes exact` remains available for A/B measurements. The locked-M1 rerun described there is still pending.

### O1 steady-state measurement + dispatch attribution

Two open questions from the embedding campaigns are measured directly:

1. **O1 steady-state throughput** — the O0 compile choice was made on cold-compile cost
   (10.97s/shape at O1 vs 88ms at O0, see below), but O1's steady-state throughput
   delta had never been measured. MPSGraph O1 optimization produces no meaningful
   speedup for the 22-layer ModernBERT encoder graph.

2. **Dispatch/boundary attribution** — a coarse time split of the embed batch path
   was missing. Attribution instrumentation now exists (env-gated, default-off) and
   the first measurement is recorded below.

Both measurements used the locked M1 (same `<bench-host>`, AC power, no
`Runner.Worker`, load < 2.5), the harness's own 2,000-chunk fixture (corpus SHA-256
`25d1d544...`, reference vectors `d55221d4...`), and `max_length=512`, f16, bucket
policy v1, explicit execution.

#### O1 steady-state throughput

| Variant | Median tok/s | Mean tok/s | N | Deterministic | Mean cosine | Worst cosine |
|---------|-------------:|-----------:|---:|:-------------:|------------:|------------:|
| O0 (baseline) | 5,088.2 | 5,084.8 | 7 | yes | 0.999999 | 0.999999 |
| O1 | 5,088.5 | 5,089.7 | 7 | yes | 0.999999 | 0.999999 |
| **Delta** | **+0.01%** | | | | | |

O1 does **not** clear +3%. The O1 steady-state speedup is effectively zero
(+0.01%); MPSGraph's optimizer does not fuse or reduce the 22-layer graph in a
way that translates to wall-clock improvement. The O1 path is deterministic and
parity-identical to O0 (byte-identical vectors on same-input determinism probe,
mean cosine 0.999999 against the pinned reference).

#### Cold-compile cost

| Variant | Cold compile + first embed | Compile overhead vs O0 |
|---------|--------------------------:|----------------------:|
| O0 | 0.670s | — |
| O1 | 2.389s | +1.719s |

The O1 compile overhead is +1.72s per shape vs O0, confirming the original serving
policy rationale for choosing O0: the compile penalty is substantial and the
steady-state gain is nil.

#### Dispatch attribution

Attribution for a single 8×512 batch (SYNAPSE_EMBED_ATTRIBUTION=1,
SYNAPSE_EMBED_PROFILE=1):

| Stage | Time (ms) | % of forward | Notes |
|-------|----------:|-------------:|-------|
| Tokenize | 1.04 | 0.36% | HuggingFace tokenizers encode_batch |
| Mask build | 0.003 | ~0% | CPU padding + attention mask construction |
| MPSGraph plan | 37.55 | — | Graph topology lookup (cached after first call) |
| Executable select (cold) | 10.05 | — | First-call package load; 0 on cache hit |
| Buffer feed | 23.28 | 11.6% | Host→GPU memcpy for inputs, masks, RoPE tables |
| Graph execute | 200.04 | 70.1% | MPSGraph kernel execution on GPU |
| Readback | 0.55 | 0.19% | GPU→host result copy |
| Pool (L2 norm) | 0.01 | ~0% | CLS extraction + L2 normalize |
| **Total forward** | **285.3** | **100%** | Rust-side wall time for `MetalContext::forward` |

The dispatch attribution shows that **GPU execution dominates** (70% of forward wall
time), with buffer feed at 12% and all other stages negligible. This confirms the
host-side seam (tokenize + mask + pool) is already thin; future embed campaigns
should focus on GPU kernel efficiency, not host-side optimization.

#### Implementation details

- **`SYNAPSE_MPS_COMPILE_O1=1`** — env switch to compile MPSGraph executables at
  optimization level 1. Adds `-o1` suffix to the package cache directory identity
  (the `GRAPH_REVISION` constant is not changed). When unset, the default is O0
  (byte-identical to the previous behavior).
- **`SYNAPSE_EMBED_ATTRIBUTION=1`** — env switch to emit per-stage timing to
  stderr (tokenize, mask_build, forward, pool from Rust; buffer_feed, execute,
  readback from Obj-C). When unset, no timing output is produced (zero overhead,
  no `Instant::now` calls on the default path).
- **`SYNAPSE_EMBED_PROFILE=1`** — existing Obj-C profiling hook; extended with
  attribution output for buffer_feed, execute, and readback stages.

Both env switches are default-off, zero-effect when unset, and additive-only
(no code path changes when disabled).