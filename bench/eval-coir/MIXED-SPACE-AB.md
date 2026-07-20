# Mixed-space retrieval A/B: GPU and ANE on one CosQA index

## Verdict

The two lanes can be mixed with a very small average retrieval delta, but the
strict tail-gated equivalence bar is **not met for the product direction**.
`M-A` (Metal corpus, ANE queries) has a mean NDCG@10 delta of `-0.000930`
and a worst-decile mean delta of `-0.009298` versus `M-M`, which is just inside
the `0.01` tail-loss bar. However, one query loses its only hit at both @10 and
@50: `q20229` (`how to flip a matrix in python`) goes from Recall `1` to `0`
and NDCG@10 `0.386853` to `0`. That is a query-level catastrophic break.

`A-M` (ANE corpus, Metal queries) is cleaner: mean NDCG@10 delta `-0.000139`,
worst-decile mean delta `-0.001386`, and no query loses a positive Recall@10
result. Thus the observed result supports, at most, a **directional A-M
compatibility declaration**. It does not support a bidirectional alias and
does not support the desired “embed once on Metal, then query on ANE” (`M-A`)
serving shape.

The vector fingerprints must remain distinct. A general symmetric
`equivalent_to` alias would hide the one-query failure and is not recommended.
If the alias machinery can express directional read compatibility, key it by
`(corpus_fingerprint, query_fingerprint, input_type, task/evaluation policy)`
and record the evidence and gate version; the only candidate direction here is
`ANE corpus <- Metal query` (`A-M`).

## Setup and lane identity

- **Host:** local production box, Apple M5 Max, `aarch64`, OS build `25F84`.
  No M1, rental, daemon reload, redeploy, or probe run was used.
- **Daemon:** production subc consumer at
  `/Users/[owner]/.local/share/cortexkit/run/subc-connection.json`;
  module generation `9`; machine profile hash
  `42a76cdd8dc2e5798629522c63dcfff1e5833ee1bf3c1f8bdb66dc2bbc04500d`.
- **Dataset:** prepared CoIR CosQA, 20,604 corpus documents, 500 test queries,
  and 500 qrels. Rows were sorted by the existing preparation path.
- **Corpus digest:**
  `a730b7ab09f86449a39780538220f9de0f2fa8ba0e8b4efce3e225e68b0fb098`.
- **Query digest:**
  `086bd4f24bc6078c6aa92d36189a0febbedd967c2e40b1350b293a9d45956d6e`.
- **Qrels digest:**
  `d3b9138da2302994d9664b1251e9a7f9995eb9ea52d4f216e6d5a56a6d28a757`.
- **Wire path:** `embed.batch` over subc with the existing
  `inline_embed_throughput.rs` consumer/identity and durable-page polling
  pattern (adapted into a read-only batch runner). Every request used batch 8,
  `accept_declared: true`, a unique `request_key`, and item IDs. No
  synapse-module source was changed.
- **Scoring:** `bench/eval-coir/coir_eval.py` exact NumPy brute-force cosine
  search and `pytrec_eval`. NDCG/MRR/Recall@10 used an independently produced
  top-10 run; Recall@50 used a top-50 run.

### Certified lanes

`models.list` at admission showed both entries `ready`, with no alias rows. The
final `models.list` and `admission.status` remained ready/certified,
non-stale, and queue-empty after all embeddings.

| lane | engine | fingerprint | numeric profile | certification evidence |
|---|---|---|---|---|
| Metal (`gte-modernbert-base-f16`) | `owned-metal` | `54a62ef80c4f28f6ba765854d81b9ab5e52d4864142cdd81662812465d3003b5` | `596b0e13a5b0bc7a4f162743bcb1f05be7bed4e7f0412649cb964ad47a29cfda` | mean cosine `0.9999994018`, rank overlap `0.9973958`, worst decile `0.9761905` |
| ANE (`gte-modernbert-base-ane-fp16`) | `ane-coreml-worker` | `5a2374bcb587ae22cd7ca93404ee7e89e9889527d15f8671feb0a226625278d8` | `3fcf855174fcf54b6390c5bd9492d9f2d050c951b8bb85bcc0a74f8e7535e75a4` | mean cosine `0.9998887586`, rank overlap `0.9921875`, worst decile `0.9285714`, ANE placement `0.9894242` |

The certified `models.list` rows had no `equivalent_to`/alias entry for this
pair.

## Timing projection and admission evidence

A 100-document sample (the first 100 sorted corpus rows) was run before the
full run, with batch 8 on each lane:

| sample lane | elapsed | projected 20,604-document corpus |
|---|---:|---:|
| Metal | `0.894 s` | `3.07 min` |
| ANE | `1.395 s` | `4.79 min` |

Both projections were far below the 90-minute cutoff, so the full 20,604-row
corpus was used for both lanes and all four cells.

The quality measurand is deterministic embedding output plus offline ranking;
host load is therefore irrelevant to the quality comparison. Queue cleanliness
was enforced instead: admission was sampled before and after every batch, and
all samples reported `execution_waiters=0` and zero lane waiters. The observed
1-minute host-load range during full embedding was `8.74`–`20.17`; it is
reported for provenance, not interpreted as a quality variable. The final
admission snapshot also reported `execution_waiters=0`, both lanes certified,
and `meeting_deadlines=true`.

Full-run execution provenance (including client/admission-call overhead):

| artifact | rows | batches | elapsed |
|---|---:|---:|---:|
| Metal corpus | 20,604 | 2,576 × 8 | `418.5 s` |
| ANE corpus | 20,604 | 2,576 × 8 | `307.8 s` |
| Metal queries | 500 | 63 × 8 | `3.0 s` |
| ANE queries | 500 | 63 × 8 | `4.5 s` |

### ANE truncation disclosure

The ANE corpus response disclosures reported **33/20,604 documents** truncated
(`0.160%`), each to effective 512 tokens. Their submitted lengths ranged up
to 1,957 tokens. No query was truncated, and the corpus truncation rate was
below the 5% sensitivity threshold; no truncation-exclusion re-score was
required. The 33 long documents are present in both `A-A` and `A-M`, so their
shared corpus-side effect is held constant in the A-M comparison.

## Four-cell retrieval result

All metrics are means over the 500 test queries. Deltas in the last column are
against the matching same-corpus control.

| cell | corpus | query | NDCG@10 | MRR@10 | Recall@10 | Recall@50 | control delta (NDCG@10) |
|---|---|---|---:|---:|---:|---:|---:|
| `M-M` | Metal | Metal | 0.367451 | 0.261478 | 0.638 | 0.930 | control |
| `A-A` | ANE | ANE | 0.358219 | 0.257843 | 0.626 | 0.924 | control |
| `M-A` | Metal | ANE | 0.366521 | 0.260525 | 0.636 | 0.928 | -0.000930 vs M-M |
| `A-M` | ANE | Metal | 0.358081 | 0.257676 | 0.626 | 0.924 | -0.000139 vs A-A |

Other mixed-cell aggregate deltas:

| mixed cell vs control | NDCG@10 | MRR@10 | Recall@10 | Recall@50 |
|---|---:|---:|---:|---:|
| `M-A` vs `M-M` | -0.000930 | -0.000952 | -0.002 | -0.002 |
| `A-M` vs `A-A` | -0.000139 | -0.000167 | 0.000 | 0.000 |

## Per-query delta distributions

Deltas are mixed minus the matching-corpus control. `p10` is the 10th
percentile boundary; because almost all queries tie, the more useful gate
statistic is also shown as the mean of the 50 most-negative query deltas
(`worst-decile mean`). Tie means absolute delta `<= 1e-12`.

| cell / metric | mean | p10 | worst-decile mean | min | tie fraction | negative queries |
|---|---:|---:|---:|---:|---:|---:|
| `M-A` NDCG@10 | -0.000930 | 0.000000 | -0.009298 | -0.386853 | 99.4% (497/500) | 3 |
| `M-A` MRR@10 | -0.000952 | 0.000000 | -0.009524 | -0.333333 | 99.4% (497/500) | 3 |
| `M-A` Recall@10 | -0.002000 | 0.000000 | -0.020000 | -1.000000 | 99.8% (499/500) | 1 |
| `M-A` Recall@50 | -0.002000 | 0.000000 | -0.020000 | -1.000000 | 99.8% (499/500) | 1 |
| `A-M` NDCG@10 | -0.000139 | 0.000000 | -0.001386 | -0.069323 | 99.8% (499/500) | 1 |
| `A-M` MRR@10 | -0.000167 | 0.000000 | -0.001667 | -0.083333 | 99.8% (499/500) | 1 |
| `A-M` Recall@10 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 100.0% (500/500) | 0 |
| `A-M` Recall@50 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 100.0% (500/500) | 0 |

### Worst queries

| cell | query | control NDCG@10 → mixed | control MRR@10 → mixed | Recall change |
|---|---|---:|---:|---:|
| `M-A` | `q20229`: “how to flip a matrix in python” | `0.386853 → 0.000000` | `0.333333 → 0.000000` | `1 → 0` at both @10 and @50 |
| `M-A` | `q20166`: “python normal distribution p values” | `0.356207 → 0.301030` | `0.250000 → 0.125000` | unchanged |
| `M-A` | `q20268`: “token to id python” | `0.356207 → 0.333333` | `0.142857 → 0.125000` | unchanged |
| `A-M` | `q20305`: “python keep processpool open until tasks complete” | `0.500000 → 0.430677` | `0.333333 → 0.250000` | unchanged |

## Vector-level cross-check

Using seed `20260720`, 1,000 identical corpus texts were sampled from the
20,604 rows and their Metal/ANE vector cosine was computed after L2
normalization:

| statistic | cosine |
|---|---:|
| mean | `0.999541402` |
| minimum | `0.998549461` |
| p01 | `0.998794048` |
| p10 | `0.999225265` |
| median | `0.999588519` |
| p95 | `0.999846459` |
| maximum | `0.999995470` |

This reproduces the known approximately-`0.9993` distinct-space relationship
at vector level and rules out a lane mix-up as the explanation for the task
result.

## Rerank-rescue arm

**Skipped.** The final certified `models.list` catalog contained only the two
embedding lanes; no certified reranker was ready and reachable through the
existing production path. No reranker was loaded, no daemon plumbing was added,
and no new rerank path was constructed. The worst mixed cell by mean NDCG
loss was `M-A`.

## Reproducibility artifact digests

The vector files are local ignored work artifacts, not committed to the
repository. Their SHA-256 digests are:

| artifact | rows | SHA-256 |
|---|---:|---|
| Metal corpus vectors | 20,604 | `ca29b6e95b4c9c38fe65de6b269bce4e878ddf717b6508559100f7e8a7890b86` |
| ANE corpus vectors | 20,604 | `b0a4ea097df38104f9ac56af4d87831bd624f44637a52a51cd19ba852805ec9e` |
| Metal query vectors | 500 | `844456347e590ec3f2908ed268fbbe71cd8bf80485fb2bb72847f160cd266a22` |
| ANE query vectors | 500 | `c164550aee120e7c78aee0a1b6a6fa0cc1d0105210902c77050cd9fd922cd8fe` |

No alias was declared by this A/B. The safe product conclusion is: retain
separate fingerprints and index provenance; do not enable bidirectional mixed-
space serving. A one-way `A-M` read-compatibility record may be considered only
if the declaration is directional and retains the task, cutoff, fingerprints,
and tail/catastrophe evidence. The desired `M-A` energy-saving path remains
blocked by the single catastrophic query in this 500-query evaluation.
