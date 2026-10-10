# Where gte-modernbert's Metal time goes

## Answer

The premise, a fixed per-call or per-row cost, is **wrong**. For both models,
about 85% of engine time is GPU work that grows with the tokens each pass
computes. The fixed part per pass is 4-6 ms.

gte-modernbert is no faster than Qwen3 because **its MPSGraph does every
projection matmul in f32**. `modernbert_matmul` in `modernbert_mpsgraph.m` casts
both f16 operands to f32, multiplies, and casts the product back to f16. Qwen3's
graph multiplies in f16. Per computed token, gte's GPU time is 0.0189 ms and
Qwen3's is 0.0212 ms, even though gte does about a quarter of Qwen3's FLOPs per
token. That works out to about 10 TFLOPS for gte against about 34 for Qwen3.

An ablation that only multiplies in f16 (`ablation-f16-matmul.patch`) cut gte's
GPU run time from 24.1 s to 9.5 s. Replay wall time fell from 28.9 s to 13.8 s,
2.1x faster. The vectors are **not byte-identical**: min cosine 0.99995261,
median 0.99999887, max absolute component difference 0.00237 over all 6,341
rows. The fix is proposed below. It is not committed and waits for a decision.

Rows in a 64-row call do **not** run as one GPU pass. The module splits each
call into engine calls of at most 8 rows and 3,072 tokens, with rows sorted by
length. The engine runs each engine call as one synchronous GPU pass. That gives
826 passes for 127 calls, about 6.5 per call. Each pass waits for its own GPU
work, but rows inside a pass share it; nothing waits per row.

## Method

- **Probe.** `crates/synapse-engine-owned/examples/metal_fixed_cost.rs`. It
  replays the engram export (6,341 rows, 127 calls, `input_sha256`
  `1db27b7f…`, `meta_sha256` `bfcef3e0…`) straight into
  `OwnedMetalEmbedEngine`. Each call is tokenized with the module's
  `SanitizedTokenizer`, then split the same way as the module's bulk path
  (`plan_embedding_engine_batches`). The engine runs the parts serially, as its
  model mutex forces in the module too. Settings match the catalog profiles: f16,
  explicit execution, `max_tokens` 8,192, `attention_units` 8,192².
  One untimed warm-up replay compiles every shape. The measured replays must
  give back the warm-up vectors bit for bit, and they did.
- **Stage lines.** `SYNAPSE_EMBED_PROFILE=1` now prints, for each pass, from
  both families:
  - `pass_embed_in`: CPU token-embedding gather; for gte also the embedding
    LayerNorm.
  - `pass_metal`: CPU masks, RoPE tables and f16 packing (`prep`); the native
    call; f16 output decode.
  - `pass_native`: plan and executable lookup (`select`), Metal buffer and feed
    setup (`upload`), the synchronous executable run (`run`), and output
    `readback`.
  - `pass_host`: rows, real tokens, padding, CLS or last-token pooling with L2
    normalization (`pool`).
- **CPU encode vs GPU time.** `SYNAPSE_EMBED_PROFILE_GPU=1` encodes the same
  compiled executable into an `MPSCommandBuffer`, then commits it and waits.
  That splits `run` into CPU encode time and the GPU span, from the command
  buffers' `GPUStartTime` to `GPUEndTime`. Both variables unset leaves the
  serving path unchanged. Vectors under the GPU-split path matched the normal
  path's hash exactly.
- **Summaries.** `summarize.py` turns the probe JSON and stderr into
  `summary.txt`. Raw outputs, 3.6 MB stderr per run plus a 19 MB vector dump
  each, are kept outside git in `~/Backups/synapse-cert-dev/metal-fixed-cost/`
  (`run-1`, `run-gpu`, `ablation`).
- **Host.** Apple M5 Max, shared with production Synapse, which served Metal
  and the Neural Engine throughout. One-minute load average was 6-18. This run
  used only Metal. The binary ran as `ckdev-metal-fixed-cost` and did not start
  a daemon or touch production state.
- **Not run: the daemon head-to-head.** `aft_embed_headtohead` always preloads
  the Neural Engine lane next to Metal, which this task does not allow. The
  probe's engine time, 28.1 s for gte and 26.6 s for Qwen3, accounts for 87-89%
  of the head-to-head's 32.4 s and 30.5 s walls. The rest fits the module's
  measured ~8 ms per call (`docs/evidence/embed-batch-module-cost/`) plus
  daemon transport.

## Results

Serial replay, mean of two measured replays (`run-1`). Times are totals over
the replay. Per-pass values are the total divided by 826 passes.

| Stage | gte total | gte / pass | Qwen3 total | Qwen3 / pass | Scales with |
|---|---:|---:|---:|---:|---|
| Replay wall | 28.88 s | | 27.57 s | | |
| Tokenize (127 calls) | 0.71 s | 5.6 ms / call | 0.85 s | 6.7 ms / call | real tokens |
| Pad ids and mask (gte) | 0.001 s | 0.00 ms | in embed-in | | padded tokens |
| Embed-in (gather, gte embedding LayerNorm, on CPU) | 1.51 s | 1.83 ms | 0.20 s | 0.25 ms | padded tokens |
| Host prep (masks, RoPE, f16 pack) | 0.43 s | 0.52 ms | 0.71 s | 0.86 ms | padded tokens |
| Graph and executable lookup | 0.006 s | 0.01 ms | 0.000 s | 0.00 ms | fixed, no misses |
| Metal buffers and feeds | 0.64 s | 0.78 ms | 0.83 s | 1.00 ms | mostly fixed |
| **Synchronous executable run** | **24.06 s** | **29.13 ms** | **23.50 s** | **28.45 ms** | **computed tokens** |
| · CPU encode (`run-gpu`) | 4.96 s | 6.01 ms | 10.38 s | 12.57 ms | fixed (op count) |
| · GPU span (`run-gpu`) | 24.42 s | 29.56 ms | 22.55 s | 27.30 ms | computed tokens |
| Readback | 0.16 s | 0.19 ms | 0.22 s | 0.27 ms | padded tokens |
| f16 decode to f32 (CPU) | 0.38 s | 0.46 ms | 0.50 s | 0.61 ms | padded tokens |
| Pool and normalize | 0.007 s | 0.01 ms | 0.014 s | 0.02 ms | rows |
| Engine bookkeeping outside passes | 0.22 s | 0.27 ms | 0.19 s | 0.23 ms | fixed |

`run-gpu` is a separate single replay with the encode/GPU split on: gte took
29.25 s and Qwen3 26.69 s. Encode runs inside the GPU span, because MPSGraph
commits partial command buffers while it encodes, so the encode and GPU rows
overlap and do not add up.

### Fixed vs token-scaled run time

A least-squares fit of each pass's `run_ms` against computed tokens
(batch capacity × sequence bucket, padding included):

| | gte f32 matmul (today) | Qwen3 | gte f16 matmul (ablation) |
|---|---:|---:|---:|
| Fixed per pass | 4.24 ms | 6.18 ms | 1.50 ms |
| Per computed token | 0.01894 ms | 0.02124 ms | 0.00760 ms |
| r² | 0.988 | 0.836 | 0.919 |
| Fixed share of run (826 passes) | 3.5 s of 24.1 s | 5.1 s of 23.5 s | 1.2 s of 9.5 s |

Median run time by shape:

| Shape | gte today | gte f16 matmul | Qwen3 |
|---|---:|---:|---:|
| 1×192 / 1×128 | 8.8 ms | 4.3 ms | 11.0 ms (1×128) |
| 8×64 | 13.7 ms | 5.5 ms | 15.8 ms |
| 8×96 | 19.0 ms | 7.5 ms | 21.9 ms |
| 8×128 | 23.2 ms | 9.0 ms | 27.2 ms |
| 8×160 | 28.9 ms | 10.8 ms | 32.6 ms |
| 8×192 | 33.2 ms | 12.9 ms | 39.6 ms |
| 8×256 | 42.9 ms | 17.2 ms | 54.6 ms |

At 8×128, gte today runs 1,024 tokens in 23.2 ms. At about 0.23 GFLOP per token
(22 layers of 768/1,152-wide GeGLU projections plus attention), that is about
10 TFLOPS. Qwen3 runs about 0.91 GFLOP per token at 27.2 ms, about 34 TFLOPS.

### Padding and buckets

| | gte | Qwen3 |
|---|---:|---:|
| Real tokens | 855,848 | 679,090 |
| Computed tokens (capacity × bucket) | 1,085,568 | 866,240 |
| Computed / real | 1.268 | 1.276 |
| Empty row slots (partial last pass of a call) | 239 | 239 |
| Passes by shape | 8×64: 26, 8×96: 59, 8×128: 167, 8×160: 286, 8×192: 182, 8×256: 101, 8×320: 1, 1×192: 2, 1×256: 2 | 8×64: 40, 8×96: 131, 8×128: 397, 8×160: 212, 8×192: 38, 8×256: 4, 1×128: 1, 1×160: 2, 1×256: 1 |

The gte tokenizer gives 26% more tokens for the same text. The two models pad
by the same ratio.

### Suspects

- **Bucket padding waste: ruled out as the cause.** Both models compute 27%
  more tokens than they have, from one shared ladder. Removing every pad token
  would save at most 21% of the token-scaled time for either model. It would
  not explain why a model a quarter the size is no faster.
- **Graph recompiles or shape misses: ruled out.** Every measured pass hit a
  cached plan and executable. Plan and executable lookup totals 6 ms per replay.
- **Per-row command-buffer waits: ruled out.** The wait is per pass, one
  synchronous run per up-to-8-row engine call. At 826 waits, the measured fixed
  cost per pass is 4.2 ms for gte, 3.5 s per replay.
- **CPU work that does not overlap the GPU: real, second largest.** For gte,
  about 3.8 ms per pass runs on the CPU before or after the GPU, with nothing
  overlapping it: embed-in 1.83, prep 0.52, buffers 0.78, readback 0.19,
  decode 0.46. That is 3.1 s, 11% of the replay. Most of it is gte's CPU
  embedding LayerNorm over padded rows. After the matmul fix it becomes about a
  quarter of the time.
- **f32 projection matmuls in gte: confirmed, dominant.** See the ablation.

## Proposed fix (not applied)

In `modernbert_matmul`, multiply in the graph dtype, as Qwen3 does, instead of
casting both operands to f32:

```objc
primary = modernbert_cast(graph, primary, data_type);
secondary = modernbert_cast(graph, secondary, data_type);
return [graph matrixMultiplicationWithPrimaryTensor:primary secondaryTensor:secondary name:nil];
```

The patch is `ablation-f16-matmul.patch`. With f32 dtype nothing changes, since
the casts are no-ops. The gte reranker's catalog profile is f32, so it is
untouched. On macOS before 15, the non-fused attention fallback also calls
`modernbert_matmul` for its score and context products, so the patch would move
those to f16 too. That path was not measured here, because this host runs the
fused attention path.

**Output impact.** The vectors are not byte-identical. They differ by more
than a reordering of the sum would explain. Over the 6,341 replay rows: min
cosine 0.99995261, median 0.99999887, max absolute difference 0.00237. For
comparison, the fused-SDPA change that went in as graph revision 3→4 measured
F16 cosine 0.9999992. This one is a numeric change, so it must not ship under
the current fingerprint.

**Fingerprint.** The engine identity's build flags carry `graph_revision`
(currently 4) and `bucket_policy` (`v2`), and the provenance fingerprint hashes
those flags. The serialized-executable package cache key also uses
`graph_revision`, so without a bump a cached old executable would keep loading.
Shipping this therefore needs a graph revision bump, which changes the
fingerprint and makes gte vectors re-embed. `GRAPH_REVISION` is one constant
shared by every owned-metal family, so bumping it also moves Qwen3's
fingerprint, even though Qwen3's graph does not change, unless the revision
becomes per family. The numeric profile does not encode bucket boundaries or
padding. It encodes only the `bucket_policy` version label, so a ladder change
without a version bump would not change the fingerprint.

## Not measured

- The production daemon path, with two requests in flight through the module.
  The engine mutex serializes engine calls, so only tokenization and the reply
  codec can overlap GPU work there.
- Quality against the fp32 reference (`bench/parity`) for the f16-matmul
  variant. Only cosine against today's vectors was measured.
- A quiet host. Production Synapse used the same GPU during every run, which
  adds noise to absolute times. Both models and the ablation ran back to back
  under similar load (6-18).

## Reproduce

From the checkout root:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
env -u TMPDIR cargo build --release --locked -p synapse-engine-owned --example metal_fixed_cost
ln -f target/release/examples/metal_fixed_cost target/release/examples/ckdev-metal-fixed-cost
IN=~/.local/share/cortexkit/synapse/aft-headtohead/engram.jsonl
GTE=~/.cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots/e7f32e3c00f91d699e8c43b53106206bcc72bb22
QWEN=~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
OUT=~/Backups/synapse-cert-dev/metal-fixed-cost/run-N; mkdir -p "$OUT"
env -u TMPDIR SYNAPSE_EMBED_PROFILE=1 target/release/examples/ckdev-metal-fixed-cost \
  gte-modernbert "$GTE" "$IN" "$OUT/gte-modernbert.json" target/mfc-cache/gte 2 2> "$OUT/gte-modernbert.stderr"
env -u TMPDIR SYNAPSE_EMBED_PROFILE=1 target/release/examples/ckdev-metal-fixed-cost \
  qwen3 "$QWEN" "$IN" "$OUT/qwen3-0.6b.json" target/mfc-cache/qwen 2 2> "$OUT/qwen3-0.6b.stderr"
python3 docs/evidence/metal-fixed-cost/summarize.py "$OUT"
```

Add `SYNAPSE_EMBED_PROFILE_GPU=1` for the encode/GPU split. For the ablation,
apply `ablation-f16-matmul.patch`, rebuild, and give it its own package cache
directory. The serialized executable is keyed only by shape and graph revision,
so it must not share a cache with the unpatched build.
