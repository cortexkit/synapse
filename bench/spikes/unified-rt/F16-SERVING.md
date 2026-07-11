# f16 serving path assembly

## Status

The runtime now accepts `--dtype f16` for MiniLM, gte-modernbert, and Qwen3-Embedding-0.6B. Metal uses explicit `MPSGraphExecutable` compilation with optimization level 0, synchronous compilation, and explicit specialization by default. `--execution lazy` keeps the old `runWithFeeds` path for A/B checks.

`--package-cache DIR` stores one `.mpsgraphpackage` per `(batch, seq)` shape. The cache directory key includes model-family, graph revision, a hash of the canonical checkpoint path, dtype, and macOS build; the filename supplies the shape. Packages are never appended. A cache hit uses `initWithMPSGraphPackageAtURL:compilationDescriptor:` and a miss compiles, specializes, and serializes before the first execution of that shape.

This assembly remains partially certified because the official 400-row MiniLM ORT file used by the evidence run was not available on this host; the locally regenerated ONNX reference also failed the existing fp32 path and therefore is not a valid certification oracle. ModernBERT and Qwen3 pass the full 400-vector gate.

## f16 parity

Certification thresholds are mean cosine `>= 0.9999` and mean top-10 overlap `>= 0.995` against ORT fp32.

| Family | Rows | f16 mean cosine | Top-10 overlap | Result | Topology |
|---|---:|---:|---:|---|---|
| MiniLM | 400 | official rerun unavailable | official rerun unavailable | not re-certified | native f16 matmuls and inter-block activations; fp32 layernorm, GELU, and softmax islands (the frozen evidence run measured cosine `0.9999993186`) |
| gte-modernbert | 400 | `0.9999990763` | `0.998500` | **PASS** | f16 storage/inter-block activations; fp32 norm scales, matmuls, RoPE, layernorm, additive masks, and softmax |
| Qwen3-Embedding-0.6B | 400 | `0.9999987315` | `0.998500` | **PASS** | native f16 weights/matmuls/inter-block activations; fp32 RMSNorm and masked softmax |

The MiniLM reference check cannot be treated as a product failure: the generated ONNX file scored `0.99920633` against the unchanged fp32 runtime and `0.99920582` against f16, so it is not equivalent to the existing parity oracle. `cargo test` still covers f16-vs-fp32 behavior for the tiny resident encoder block.

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

The locked M1 comparison was not run from this M5 worktree. The benchmark-lock and `Runner.Worker` protocol remains required before these numbers can graduate.

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

## Open items

1. Re-run MiniLM against the exact official ORT fp32 reference used by the frozen evidence pack.
2. Run first/warm/steady fp32-vs-f16 for all families under the locked M1 `bench.lock` plus `Runner.Worker` check protocol.
3. Pre-discover serving buckets so every shape is compiled/loaded before inference timing; this spike currently prepares each shape synchronously immediately before its first execution.
4. Replace exact corpus shapes with a bounded serving bucket policy, especially for the 162-shape MC workload.
