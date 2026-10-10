# Where gte-modernbert's Metal time goes, and the fix

## Summary

The question's premise, that some fixed cost per call or per row dominates, is
**wrong**. For both models, about 85% of engine time is GPU work that grows with
the tokens each pass computes. The fixed part per pass is 4-6 ms.

gte-modernbert was no faster than Qwen3 because **its f16 MPSGraph did every
weight projection in f32**. `modernbert_linear` in `modernbert_mpsgraph.m` cast
both f16 operands to f32, multiplied, and cast the product back to f16. Qwen3's
graph multiplies in f16. Per computed token, gte's GPU time was 0.0189 ms and
Qwen3's was 0.0212 ms, even though gte does about a quarter of Qwen3's FLOPs per
token. That is about 10 TFLOPS for gte against about 34 for Qwen3.

**Fix (applied):** the f16 ModernBERT graph now runs its QKV, attention-output
and MLP projections in f16, as **graph revision 5**. The engram replay drops
from 30.1 s / 33.3 s to 14.8 s / 14.7 s, 2.0-2.3x faster in interleaved runs
(`run-final`, below).

- **Vectors are not byte-identical.** They are as close to the fp32 reference
  as before. Against the 17 CPU fp32 fixture cases, min cosine is 0.99999399
  new vs 0.99999136 old. On the 6,341-row replay, neighbour ranking matches the
  old graph's within the spread across query samples.
- **What changes and what does not.** The fingerprint moves for gte f16 lanes
  only. Qwen3's vectors, identity, fingerprint and cache keys are unchanged, and
  so are the f32 gte reranker's.

Rows in a 64-row call do **not** run as one GPU pass. The module splits each
call into engine calls of at most 8 rows and 3,072 tokens, with rows sorted by
length. The engine runs each engine call as one synchronous GPU pass. That gives
826 passes for 127 calls, about 6.5 per call. Each pass waits for its own GPU
work, but rows inside a pass share it; nothing waits per row.

## Method

- **Probe.** `crates/synapse-engine-owned/examples/metal_fixed_cost.rs`. It
  replays the engram export (6,341 rows, 127 calls, `input_sha256` `1db27b7f…`,
  `meta_sha256` `bfcef3e0…`) straight into `OwnedMetalEmbedEngine`. Each call is
  tokenized with the module's `SanitizedTokenizer`, then split the same way as
  the module's bulk path (`plan_embedding_engine_batches`). The engine runs the
  parts serially, as its model mutex forces in the module too. Settings match
  the catalog profiles: explicit execution, `max_tokens` 8,192,
  `attention_units` 8,192², f16 unless an f32 reference is asked for. One untimed
  warm-up replay compiles every shape. Measured replays must give back the
  warm-up vectors bit for bit, and they did.
- **Stage lines.** `SYNAPSE_EMBED_PROFILE=1` prints, for each pass, from both
  families:
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
  That splits `run` into CPU encode time and the GPU span. With both variables
  unset the serving path is unchanged, and the GPU-split path's vectors matched
  the normal path's hash exactly.
- **Summaries.** `summarize.py` turns the probe JSON and stderr into
  `summary.txt`. Raw outputs, about 3.6 MB of stderr and a 19 MB vector dump per
  run, are kept outside git in `~/Backups/synapse-cert-dev/metal-fixed-cost/`.
- **Host.** Apple M5 Max, macOS 27.0.1 (26A434), shared with production Synapse,
  which served Metal and the Neural Engine throughout. One-minute load average
  was 6-20 for the runs reported here; a burst to 120 voided one set of A/B
  runs, which are not used. Only Metal ran. Binaries ran under `ckdev-` names
  and did not start a daemon or touch production state.
- **Not run: the daemon head-to-head.** `aft_embed_headtohead` always preloads
  the Neural Engine lane next to Metal, which this task does not allow. The
  probe's engine time before the fix (28.1 s gte, 26.6 s Qwen3) accounts for
  87-89% of the head-to-head walls (32.4 s, 30.5 s). The rest fits the module's
  measured ~8 ms per call (`docs/evidence/embed-batch-module-cost/`) plus daemon
  transport.

## Where the time went (before the fix)

Serial replay, mean of two measured replays (`run-1`). Per-pass values are the
total divided by 826 passes.

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
29.25 s and Qwen3 26.69 s. MPSGraph commits partial command buffers while it
encodes, so encode runs inside the GPU span and the two rows overlap.

A least-squares fit of each pass's `run_ms` against computed tokens (batch
capacity × sequence bucket, padding included):

| | gte f32 matmuls (before) | Qwen3 | gte f16 matmuls (after, `run-final` round 1) |
|---|---:|---:|---:|
| Fixed per pass | 4.24 ms | 6.18 ms | 1.98 ms |
| Per computed token | 0.01894 ms | 0.02124 ms | 0.00777 ms |
| r² | 0.988 | 0.836 | 0.948 |

At 8×128, gte before the fix ran 1,024 tokens in 23.2 ms. At about 0.23 GFLOP
per token (22 layers of 768/1,152-wide GeGLU projections plus attention), that
is about 10 TFLOPS. Qwen3 runs about 0.91 GFLOP per token in 27.2 ms, about 34
TFLOPS.

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
  more tokens than they have, from one shared ladder.
- **Graph recompiles or shape misses: ruled out.** Every measured pass hit a
  cached plan and executable.
- **Per-row command-buffer waits: ruled out.** The wait is per pass, one
  synchronous run per up-to-8-row engine call.
- **CPU work that does not overlap the GPU: real, second largest.** For gte,
  about 3.8 ms per pass runs on the CPU with nothing overlapping it, mostly the
  CPU embedding LayerNorm over padded rows. That is 3.1 s, 11% of the replay
  before the fix and about 22% after. Not changed here.
- **f32 projection matmuls in gte: confirmed, dominant.** Fixed.

## The fix

`modernbert_linear` now multiplies in the graph dtype. It feeds the QKV,
attention-output and MLP input/output projections, and the reranker head:

```objc
MPSGraphTensor *transposed = [graph transposeTensor:weight dimension:0 withDimension:1 name:nil];
return [graph matrixMultiplicationWithPrimaryTensor:modernbert_cast(graph, input, data_type)
                                    secondaryTensor:modernbert_cast(graph, transposed, data_type)
                                               name:nil];
```

- **What changed.** In f16 graphs, all four projections per layer for 22
  layers (88 matmuls). The reranker head passes `MPSDataTypeFloat32` explicitly,
  so it is unchanged.
- **What did not.** `modernbert_matmul` keeps the f32 score and context products
  of the non-fused attention fallback (macOS before 15). That path is not taken
  on this host, so it was not measured. In f32 graphs the casts were and remain
  no-ops, so f32 graphs, including the f32 gte reranker profile, are
  structurally identical.
- **Graph revision per family and dtype.** `graph_revision(family, dtype)` in
  `synapse-engine-owned` returns 5 for f16 ModernBERT only. Every other graph
  stays on the base `GRAPH_REVISION` 4. The engine identity's `graph_revision`
  build flag, the package cache key and stale-key pruning all use it. The
  module's catalog numeric profile uses the matching
  `metal_kernel_revision(family, dtype)`, which is `owned-metal-graph-5-bucket-2`
  for f16 ModernBERT and `owned-metal-graph-4-bucket-2` for everything else.

### Why it was f32

The casts came in with `b7db9889` ("assemble f16 serving path", in
`bench/spikes/unified-rt`), during a hunt for a layer-0 f16 divergence, and were
carried into production by `ba535b34`. No commit message or comment states a
reason. `bench/spikes/unified-rt/F16-SERVING.md` records that "the earlier
all-matmul-fp32 fallback already diverged at layer 0 and compounded across the
stack". The root cause turned out to be the pointer-keyed static buffer cache
binding per-call f16 norm-scale temporaries, which was fixed separately. The f32
matmuls stayed. The same document lists gte f16 at 0.74x of fp32 throughput,
which is the cost of these casts.

I re-ran exactly that case: one-row lazy f32 and f16 executions with
`SYNAPSE_MODERNBERT_DUMP_DIR`, for fixture cases short-2 (66 tokens) and
boundary-512 (`parity/stage-dump-vs-f32.txt`). Against the f32 graph, every
layer-0 stage of the new graph matches the old graph's agreement to within
1e-8 in cosine. The final layer reaches cosine 0.9999995 (short-2) and 0.9999885
(boundary-512) new, against 0.9999999 and 0.9999852 old. All values are finite.
The residual stream peaks around 53,300 in both f16 graphs and in f32, under the
f16 limit of 65,504. That peak is a ModernBERT property both graphs share, not
something the matmul change touches.

## Quality

**fp32 reference fixtures** (`bench/parity`, all 17 gte cases, batches of 8 as
in the hardware test; `parity/*.jsonl`, written by `fixture_cosines`):

| Case | Tokens | Old f16 | New f16 | f32 graph |
|---|---:|---:|---:|---:|
| short-0 | 5 | 0.9999913637 | 0.9999939857 | 1.0000000000 |
| short-1 | 20 | 0.9999993929 | 0.9999987934 | 1.0000000000 |
| short-2 | 66 | 0.9999996032 | 0.9999996296 | 1.0000000000 |
| short-3 | 26 | 0.9999996024 | 0.9999995914 | 1.0000000000 |
| boundary-127 | 127 | 0.9999991992 | 0.9999992640 | 1.0000000000 |
| boundary-128 | 128 | 0.9999993527 | 0.9999993409 | 1.0000000000 |
| boundary-129 | 129 | 0.9999992366 | 0.9999992498 | 1.0000000000 |
| boundary-511 | 511 | 0.9999994971 | 0.9999992754 | 1.0000000000 |
| boundary-512 | 512 | 0.9999993321 | 0.9999994694 | 1.0000000000 |
| boundary-513 | 513 | 0.9999990578 | 0.9999990280 | 1.0000000000 |
| batch-0 | 18 | 0.9999995723 | 0.9999995309 | 1.0000000000 |
| batch-1 | 20 | 0.9999993929 | 0.9999987934 | 1.0000000000 |
| batch-2 | 17 | 0.9999991221 | 0.9999993402 | 1.0000000000 |
| batch-3 | 40 | 0.9999994849 | 0.9999993694 | 1.0000000000 |
| batch-4 | 128 | 0.9999992981 | 0.9999988418 | 1.0000000000 |
| batch-5 | 200 | 0.9999993516 | 0.9999993499 | 1.0000000000 |
| long-8192 | 8192 | 0.9999996492 | 0.9999996552 | 1.0000000000 |
| **min** | | **0.9999913637** | **0.9999939857** | 1 − 2.6e-11 |

- **Worst cases.** The worst case for both graphs is short-0 (5 tokens), not
  the longest sequence. long-8192 is among the best.
- **Catalog bar.** The catalog self-check bar is 0.999, so the margin is
  roughly 170x on 1 − cosine.
- **Hardware test.** The committed hardware test
  (`catalog_real_weights_pass_committed_evaluator`,
  `METAL_PARITY_MODEL=gte-modernbert-base`) passes every evaluator gate on the
  new graph, with min cosine 0.9999939857 (committed old-graph evidence:
  0.9999913583).

**Replay against the f32 Metal graph.** The f32 graph matches the fp32
fixtures to 1 − cosine ≤ 2.6e-11, so it serves as the reference for all 6,341
rows:

| | Old f16 | New f16 |
|---|---:|---:|
| Min cosine | 0.99994857 | 0.99994091 |
| p0.1 | 0.99998655 | 0.99998178 |
| Median | 0.999999210 | 0.999999109 |
| Rows below 0.9999 | 0 | 0 |

**Neighbour ranking on the full replay** (`quality_check.py`,
`quality-results.txt`). Setup:

- The reference is the f32 graph. 500 query rows are drawn with a fixed seed;
  each query is ranked against all 6,341 rows with itself excluded, and k = 10.
- A *swap* is a pair of items, within the union of both top-10 sets, that the
  reference orders by more than the stated margin and the lane reverses.
- The engine is deterministic: old vectors are byte-identical across four
  separate processes (including a base-source build and a package-loaded run),
  and new vectors across three. Run-to-run noise is therefore zero, so the
  spread across query samples is the noise yardstick.

| Seed 20261010 (primary) | Top-10 mean | Top-10 p5 | Swaps > 1e-4 | Swaps > 5e-4 | Mean abs sim error |
|---|---:|---:|---:|---:|---:|
| Old f16, repeat 1 | 0.997400 | 1.000 | 23 | 0 | 1.977e-4 |
| Old f16, repeat 2 | 0.997400 | 1.000 | 23 | 0 | 1.977e-4 |
| New f16 | 0.997400 | 1.000 | 26 | 1 | 1.997e-4 |

On the primary seed, the new graph has slightly more swaps above both margins
(26 vs 23, and 1 vs 0) and 1% more similarity error. **That difference does not
hold across query samples.** Over seeds 101-120 (same vectors, different 500
queries):

| Mean over 20 seeds | Old f16 | New f16 | New worse / better |
|---|---:|---:|---|
| Top-10 mean | 0.99763 | 0.99808 | worse in 5, better in 12 |
| Top-10 p5 | 1.000 | 1.000 | equal in all |
| Swaps > 1e-4 | 25.9 | 25.8 | worse in 12, better in 6 |
| Swaps > 5e-4 | 0.85 | 0.60 | worse in 5, better in 6 |
| Mean abs sim error | 1.9788e-4 | 1.9833e-4 (+0.2%) | worse in 12, better in 8 |

### Probe rank overlap: near-tie tolerant

`skeleton_e2e::probe_owned_gte_modernbert_certifies_against_family_reference`
asserts `rank_overlap >= 0.999` for the legacy `gte-modernbert-base-f16` lane.
The probe's `rank_overlap` compares each lane with the ORT CPU fp32 reference
vectors in `probe_corpus_gte_modernbert_ort_fp32.json` (`probe_evidence` →
`probe_evidence_between(vectors, reference)` in `crates/synapse-module/src/lib.rs`),
with k = 6 over 64 items. With the exact-membership metric, the new graph scored
0.99740 against 1.0 for the old one: a single swap.

Query p10, neighbours 9 and 11 at the top-6 edge, similarity sim(10,9) against
sim(10,11):

| Graph | sim(10,9) | sim(10,11) | Order | Margin |
|---|---:|---:|---|---:|
| ORT fp32 reference | 0.57470527 | 0.57448757 | 9 above 11 | 2.18e-4 |
| Old f16 | 0.57454132 | 0.57422220 | 9 above 11 | 3.19e-4 |
| New f16 | 0.57443545 | 0.57446238 | 11 above 9 | 2.69e-5 |
| Metal f32 | 0.57470517 | 0.57448766 | 9 above 11 | 2.18e-4 |

The reference agrees with the old graph here. The margin is, however, about
the size of either graph's per-pair similarity error against the reference: over
all 2,016 corpus pairs, old mean 1.54e-4 / p99 4.94e-4, new mean 1.66e-4 / p99
5.19e-4. The old graph's own error on sim(10,11) was 2.65e-4.

The bar was not loosened. Instead, the probe's rank overlap is now
**near-tie tolerant for every lane** and for the probe's alias comparisons:
`rank_overlap_metrics` counts a neighbour the lane ranks into the top k in place
of a reference neighbour as a hit when the reference separates the two by less
than `RANK_OVERLAP_NEAR_TIE_MARGIN` = 5e-4. That margin is about 2.5x the old
graph's measured mean similarity error of 1.98e-4 over 6,341 rows × 500 queries.
The e2e bar stays 0.999, and the production thresholds (mean cosine ≥ 0.999,
worst decile ≥ 0.9) are unchanged.

`rank_overlap_ignores_near_ties_but_catches_clear_swaps` plants both cases:

- a swap with a reference margin of at most 1.8e-4 keeps full overlap;
- a swap with a margin of at least 2.1e-3 drops the overlap to 0.984 and fails
  the 0.999 bar.

This e2e test runs a **debug** build. CPU-side Rust float paths in debug and
release builds give owned-Metal vectors that differ by up to 1e-3 per component;
release vectors of the new graph score exact-membership rank overlap 1.0 on this
corpus. That debug/release divergence predates this change and is not addressed
here.

## What did not change

- **Qwen3.** Replay vectors are byte-identical at base and on the final build:
  `vectors_sha256` `f0d3fc36…` for both. `only_the_f16_modernbert_graph_moved_to_revision_5`
  pins Qwen3's full engine identity. Its catalog Metal pin `62b20d3a…` and its
  profile pins are unchanged, and its package key stays `qwen3-0.6b-graph-v4-…`.
- **f32 gte reranker.** `preload_gte_raw_logits_match_baseline` logits are
  byte-identical between base and the final build: SHA-256 `7d6eff7f…`, the
  same in two separate new runs. The committed `preload-baseline.f32le`
  (`34c95ae0…`) no longer matches on this host **even at base**: 123 of 125
  logits differ, by at most 2.2e-5. That failure predates this change and is
  not touched here. The reranker catalog Metal pin `6da1fa7f…` and the legacy
  reranker fingerprint `2fa5f24c…` are unchanged.

## Fingerprint fallout

| Lane | Before | After |
|---|---|---|
| Catalog `gte-modernbert-base` Metal (`models.json`) | `b904dd7b…` | `9ca42893d0c355b12a569ccc4c752e705df72db9de5387906656625c9f536904` |
| Profile `gte-modernbert-base.owned-metal` | `3a0b0261…` | `00cc3677488368ec06008388fcbaa60952faed7442a9d973d5e01f8af577b16f` |
| Legacy preload `gte-modernbert-base-f16`, after redeploy | `24cc5271…` | `050c8db66c5f7fd5e33a8b5872a5b28512674e3e8e57ed673d88d4e6fa4b72cf` |

- **Recomputing the pins.** The catalog pin was recomputed with the module's
  own catalog fingerprint path (`catalog_fingerprints.rs`).
  `only_the_graph_revision_moved_the_gte_modernbert_metal_fingerprint` shows
  that recomputing the lane with graph revision 4 still gives `b904dd7b…`, so
  the graph revision is the only input that moved.
- **Captured preload record.** `bench/parity/preload/gte-modernbert-base-f16.json`
  is a production capture from 2026-10-02 and is left as captured. It will stop
  describing the live lane once the module is redeployed and should be
  re-captured then. `preload_fingerprints_rebuild_from_committed_inputs` still
  rebuilds both captured fingerprints from the captured identities. It also
  shows that the current build differs from the gte capture only in
  `graph_revision`.
- **Re-certification.** The catalog lane's serving approvals and self-checks
  are keyed by the catalog fingerprint, so the gte Metal lane needs a fresh
  self-check and approval under `9ca42893…`.

**Where `equivalent_to` comes from.** It is not declared in the catalog or in
code. It comes from alias rows in the module store, each holding a fingerprint
pair and its evidence (`declare_alias_pair`, `crates/synapse-module/src/store.rs`).
Two paths write them:

- `probe.start` (`execute_probe_job`), after measuring two certified lanes'
  probe vectors against each other with mean cosine ≥ 0.999 and worst decile
  ≥ 0.9;
- the admin `alias.declare` (requires `alias_admin_enabled`).

`equivalent_fingerprints_at` (`crates/synapse-core/src/fingerprint.rs`) matches
either side by fingerprint string; nothing is keyed by lane or model. Once the
legacy Metal lane serves `050c8db6…`:

- the existing row `24cc5271… ↔ 5a2374bc…` stays and still matches vectors
  stored under the old fingerprint;
- the new fingerprint has no alias until a `probe.start` covering both lanes
  (or an `alias.declare`) records one;
- a request pinning `24cc5271…` or `5a2374bc…` is refused by the new Metal lane
  as `substitution_rejected` until then.

The production store was not read.

## Speed after the fix

Interleaved old/new replays (`run-final`, one measured replay each, after an
untimed warm-up, load 13-25):

| Round | Old wall | New wall | Old engine | New engine | Speed-up |
|---|---:|---:|---:|---:|---:|
| 1 | 30.13 s | 14.76 s | 29.30 s | 13.92 s | 2.04x |
| 2 | 33.30 s | 14.71 s | 32.16 s | 13.89 s | 2.26x |

| Shape (median run) | Old | New |
|---|---:|---:|
| 8×64 | 13.9 ms | 7.9 ms |
| 8×128 | 23.6 ms | 9.6 ms |
| 8×160 | 29.3 ms | 11.6 ms |
| 8×192 | 33.8 ms | 13.7 ms |
| 8×256 | 43.6 ms | 18.4 ms |

After the fix, gte's GPU time per computed token is 0.0078 ms against Qwen3's
0.0212 ms. Serial CPU work (about 3.2 s) is now about 22% of the replay and is
the next largest cost.

## Not measured

- The production daemon path with two requests in flight. The engine mutex
  serializes engine calls, so only tokenization and the reply codec can overlap
  GPU work there.
- The non-fused attention fallback (macOS before 15). It keeps its f32
  products, but runs inside a graph whose projections are now f16.
- A quiet host. Production Synapse shared the GPU during every run; old and new
  were interleaved so both saw the same conditions.

## Reproduce

From the checkout root:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
env -u TMPDIR cargo build --release --locked -p synapse-engine-owned \
  --example metal_fixed_cost --example fixture_cosines --example probe_corpus_vectors
ln -f target/release/examples/metal_fixed_cost target/release/examples/ckdev-metal-fixed-cost
IN=~/.local/share/cortexkit/synapse/aft-headtohead/engram.jsonl
GTE=~/.cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots/e7f32e3c00f91d699e8c43b53106206bcc72bb22
QWEN=~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
OUT=~/Backups/synapse-cert-dev/metal-fixed-cost/run-N; mkdir -p "$OUT"
# Stage profile; add SYNAPSE_EMBED_PROFILE_GPU=1 for the encode/GPU split.
env -u TMPDIR SYNAPSE_EMBED_PROFILE=1 target/release/examples/ckdev-metal-fixed-cost \
  gte-modernbert "$GTE" "$IN" "$OUT/gte-modernbert.json" target/mfc-cache/gte 2 2> "$OUT/gte-modernbert.stderr"
env -u TMPDIR SYNAPSE_EMBED_PROFILE=1 target/release/examples/ckdev-metal-fixed-cost \
  qwen3 "$QWEN" "$IN" "$OUT/qwen3-0.6b.json" target/mfc-cache/qwen 2 2> "$OUT/qwen3-0.6b.stderr"
python3 docs/evidence/metal-fixed-cost/summarize.py "$OUT"
# f32 reference vectors, then the ranking comparison.
env -u TMPDIR target/release/examples/ckdev-metal-fixed-cost \
  gte-modernbert "$GTE" "$IN" "$OUT/f32.json" target/mfc-cache/gte-f32 1 f32
python3 docs/evidence/metal-fixed-cost/quality_check.py "$OUT/f32.json.vectors.f32" \
  "new=$OUT/gte-modernbert.json.vectors.f32"
# Per-case fp32 fixture cosines.
env -u TMPDIR target/release/examples/fixture_cosines gte-modernbert "$GTE" \
  bench/parity/fixtures/gte-modernbert-base/gte-modernbert-base.ref-v1.transformers-5.16.1.seed-0.json \
  target/mfc-cache/fixtures f16 explicit
```

To get old-graph vectors, check out `modernbert_mpsgraph.m` from `23006699`,
rebuild, run with its own package cache directory, then restore the file. Do not
share a package cache between old-graph and new-graph builds: a temporarily
reverted graph still writes under the revision-5 key.
`ablation-f16-matmul.patch` is the first, unconditional ablation (it also moved
the fallback products to f16) and is kept for the record.
