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

The fresh locked-M1 campaign baseline is **18,500.3485 tok/s** (median of the
36 worse-of-two steady samples), recorded in
`results/metal-embed-campaign/m1-baseline.json`. The run used
`[bench-host]` (Apple M1 Max), acquired and released
`[bench-user-home]/bench.lock`, and rejected an active `Runner.Worker`. Its minimum
mean cosine was `0.999999022` and minimum worst-decile top-10 overlap was
`0.97400`. The fleet's ordinary coalesced observation remains about 14--15k
tok/s, and the earlier
local M5 serving note recorded 19,740.1 tok/s; neither substitutes for this
locked-M1 campaign bar. The same-session MLX Python cell was skipped because
`/tmp/synapse-mlx-minilm-venv/bin/python` was absent on the M1. The optional
llama-server cell was also skipped because no lane-specific command/model was
configured.

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
