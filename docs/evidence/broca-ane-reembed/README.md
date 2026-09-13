# Broca ANE re-embed and retrieval-quality rig

## Union-judged decisive run

### Verdict: publishable, with a poor residual-noise control

All three pure-cosine retrieval arms now read one shared set of fresh labels. The
recipe change reverses the none-band result for production: arm 2 raises
production relevance from **8.5% to 31.4%**, while the unchanged ANE ranking is
exactly flat at **21.6%**. The isolated recipe lever is therefore **+22.9
percentage points** for production. Under the guidance change, arm 3 moves
production by **+2.1 points** and ANE by **+2.5 points** against arm 2; the gap
changes by only **+0.4 point**, so this run does not resolve a guidance effect.
Arm 3 minus arm 1 is not used because that comparison changes both recipe and
guidance.

**The order-sensitivity control is poor and bounds every result below.** Rejudging
a deterministic random 10% of split union batches (101 of 1,009) after
permuting row order gave only **59.6% exact-grade agreement and 87.1%
binary-relevance agreement** across 690 candidate labels. Put differently, 12.9%
of binary labels moved under order alone. The recipe delta is larger than that
raw disagreement rate and has an unchanged-ANE control, which supports its
direction inside this fixed instrument. The control is not a confidence interval,
so the exact effect size remains uncertain. The 2–3 point
guidance movements are much smaller than this noise floor and are not
interpretable as an effect.

### Inputs and one shared instrument

Both arm-3 inputs passed their predeclared SHA-256 checks:

- `broca-ctx-search-queries-arm3-rephrased.json`:
  `4c4a2aa7b8fbacc4426d22c15a8b6d2bb12e7e720c59fbd5b9b38794830a0688`
- `broca-ctx-search-query-vectors-arm3-rephrased-instructed.json`:
  `b802ceed305e8a16e499b63c00ffe46d11c2ce44ad19cb0f898fa6f545eabafe`

All 298 rephrasings were embedded through the existing 1,024-token ANE package.
Every rephrasing fit that bucket; every 4,096-dimensional production vector and
768-dimensional ANE vector passed its normalization and identity checks. Arm 3
then admitted the same 275 retrieval-eligible queries and produced 4,425 rows in
its two-system pool. Arm 1 and arm 2 rankings were reused byte-for-byte. No old
label transferred.

For each original query identity, the six top-10 lists were deduplicated by
stable row identity and sorted by that identity. The sorted union was split into
contiguous groups of at most eight candidate rows. This fixed rule produced
7,107 unique per-query candidate rows in 1,009 judge batches; every candidate
was judged exactly once. The split is a pure function of union membership, not
of arm provenance or incidental iteration order.

The judge saw the **original keyword bag** for every arm. Arm 3 retrieval alone
used the rephrasing. Across all 298 pairs, the rephrasing drops no original term
and adds no content term; it adds only grammatical connectives and word order,
so either form has the same content vocabulary. The original bag was chosen
because it is the request the agent actually issued, while the rephrasing is a
synthetic treatment rather than ground truth. This choice costs clarity: bags
are harder to judge than grammatical questions. That difficulty is constant
across arms, and the poor order control above measures its residual effect. The
unchanged lexical-overlap probe also uses the original bag for all three arms.

The absolute known-good relevance gate is retired. Its item restated the query
and therefore measured lexical credulity rather than answer recognition. For
liveness only, the restatement remained as a relative anchor against the fixed
mismatch: the anchor outranked the mismatch in **98.81%** of 1,009 batches,
above the 95% threshold. The mismatch graded 0 in **1,009 of 1,009** batches, so
its relevance rate was 0%, below the 1% ceiling.

No quality timing was collected. No corpus row was re-embedded, no bucket was
rebuilt, and no ladder, catalog, fingerprint, certification record, live store,
or production state was touched.

### Decisive none-band levels and deltas

The noise bound beside this result is **59.6% exact / 87.1% binary agreement**
under row-order permutation. Levels give context; the recipe and guidance reads
come only from adjacent-arm deltas under the shared labels.

| Result | Production rows in none band | Production relevance | ANE rows in none band | ANE relevance | ANE minus production |
|---|---:|---:|---:|---:|---:|
| Arm 1, published | 1,510 (54.9%) | 7.9% | 921 (33.5%) | 23.3% | +15.4 pp |
| Arm 1, union-judged | 1,510 (54.9%) | 8.5% | 921 (33.5%) | 21.6% | +13.1 pp |
| Arm 2, union-judged | 748 (27.2%) | 31.4% | 921 (33.5%) | 21.6% | -9.8 pp |
| Arm 3, union-judged | 764 (27.8%) | 33.5% | 867 (31.5%) | 24.1% | -9.4 pp |

| Isolated none-band lever | Production delta | ANE delta | Change in ANE-minus-production gap | Reading |
|---|---:|---:|---:|---|
| Recipe: arm 2 minus arm 1 | **+22.9 pp** | 0.0 pp | -22.9 pp | Large fixed-instrument reversal; the order control is not a confidence interval |
| Guidance: arm 3 minus arm 2 | +2.1 pp | +2.5 pp | +0.4 pp | Below the measured noise floor; unresolved |

### Arm 1 published beside union-judged

The noise bound beside the union-judged columns is **59.6% exact / 87.1%
binary agreement**. Arm 1's old values remain historical rather than being
smoothed into the new instrument. They differ because the same rows are now
graded beside rows retrieved by the other arms.

| Metric | Arm 1 published, production / ANE | Arm 1 union-judged, production / ANE | Arm 2 union-judged, production / ANE | Arm 3 union-judged, production / ANE |
|---|---:|---:|---:|---:|
| Precision@10 | 0.2455 / 0.4371 | 0.2513 / 0.4258 | 0.5262 / 0.4258 | 0.5415 / 0.4324 |
| nDCG@10 | 0.4407 / 0.7319 | 0.3850 / 0.5925 | 0.6849 / 0.5925 | 0.7038 / 0.6068 |
| Pooled recall@10 | 0.4385 / 0.8143 | 0.2869 / 0.4960 | 0.6167 / 0.4960 | 0.6378 / 0.5086 |
| MRR@10 | 0.5355 / 0.7672 | 0.5163 / 0.7501 | 0.8000 / 0.7501 | 0.8082 / 0.7850 |

Arm 1's published bootstrap intervals resampled queries while treating labels as
fixed. Labels are not fixed in practice: the earlier duplicated-row analysis
found 81.1% binary stability, and this order control found 87.1%. Those published
intervals are therefore too narrow and must not be read as covering judge
uncertainty.

### Union-judged precision by class

The noise bound beside these class levels is **59.6% exact / 87.1% binary
agreement**.

| Arm | Class | Production precision | ANE precision | ANE minus production |
|---|---|---:|---:|---:|
| 1 | Memory | 24.9% | 45.1% | +20.2 pp |
| 1 | Commit | 22.7% | 41.5% | +18.9 pp |
| 1 | Chunk | 45.0% | 46.8% | +1.8 pp |
| 2 | Memory | 53.2% | 45.1% | -8.1 pp |
| 2 | Commit | 52.0% | 41.5% | -10.5 pp |
| 2 | Chunk | 56.4% | 46.8% | -9.7 pp |
| 3 | Memory | 54.9% | 46.7% | -8.3 pp |
| 3 | Commit | 53.8% | 42.3% | -11.5 pp |
| 3 | Chunk | 55.3% | 45.0% | -10.3 pp |

### Unchanged lexical-overlap probe

`bench/spikes/ane-direct-probe/lexical_overlap_probe.py` was run unchanged over
all three arms, reading the shared union labels. The noise bound beside every
band is **59.6% exact / 87.1% binary agreement**. Counts are returned rows, so a
row retrieved by both systems appears once in each system's column.

| Arm | Overlap band | Production relevant | ANE relevant | Gap | Production rows | ANE rows |
|---|---|---:|---:|---:|---:|---:|
| 1 | none | 8.5% | 21.6% | +13.1 pp | 1,510 | 921 |
| 1 | low | 31.8% | 42.2% | +10.4 pp | 648 | 780 |
| 1 | mid | 48.0% | 52.8% | +4.8 pp | 398 | 691 |
| 1 | high | 85.1% | 77.7% | -7.4 pp | 194 | 358 |
| 2 | none | 31.4% | 21.6% | -9.8 pp | 748 | 921 |
| 2 | low | 51.0% | 42.2% | -8.8 pp | 784 | 780 |
| 2 | mid | 59.9% | 52.8% | -7.1 pp | 798 | 691 |
| 2 | high | 79.5% | 77.7% | -1.9 pp | 420 | 358 |
| 3 | none | 33.5% | 24.1% | -9.4 pp | 764 | 867 |
| 3 | low | 52.0% | 40.1% | -11.9 pp | 771 | 795 |
| 3 | mid | 62.0% | 52.2% | -9.8 pp | 784 | 717 |
| 3 | high | 80.3% | 77.4% | -2.9 pp | 431 | 371 |

Mean overlap was 0.131 / 0.212 for production / ANE in arm 1, 0.241 /
0.212 in arm 2, and 0.242 / 0.220 in arm 3. The corresponding within-system
correlations between overlap and grade were +0.584 / +0.432, +0.380 / +0.432,
and +0.374 / +0.416.

### Pool size and arm coverage per query

The noise bound for labels drawn from these pools remains **59.6% exact / 87.1%
binary agreement**. A query's arm coverage is the share of its union retrieved
by either system in that arm. Shares can sum above 100% because the same row can
be covered by more than one arm.

| Quantity across 275 queries | Minimum | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| Union pool rows | 13 | 26 | 25.84 | 37 |
| Arm 1 coverage | 48.5% | 69.2% | 69.3% | 100.0% |
| Arm 2 coverage | 45.2% | 61.5% | 63.2% | 92.3% |
| Arm 3 coverage | 43.8% | 62.5% | 63.6% | 93.3% |

`EVIDENCE.json` records all 275 per-query pool sizes, each arm's coverage count
and share, and each arm's exclusive count and share. Those records are sorted by
numeric composition and contain no query or row identity, so arm-dominated pools
remain visible without publishing private identifiers.

## Corrected-calibration re-judgment

### Verdict: the sample-size guard refused the run

The corrected literal-anchored instrument cannot support its predeclared gate on this query set. Only **38 of all 298 queries** have a qualifying real-corpus known-good row; 260 are excluded for lack of a unique literal. Among the 275 queries that can enter the vector comparison, only **36 are calibratable** and 239 are excluded. Both populations are below the required minimum of 50, so the run stopped before judging. There is no corrected calibration rate and no re-judged candidate metric for any arm.

This refusal is not sensitive to where the ordinary-English boundary is drawn. As a conservative upper-bound check, treating every parsed query token as a possible literal, including ordinary words, finds only 41 qualifying queries in the full set and 38 among the retrieval-eligible set. That still cannot reach the guard.

The two arm-3 input files matched their predeclared SHA-256 hashes. The guard then stopped the run before the rephrased queries were embedded through ANE, before arm-3 retrieval, and before any fresh label was requested. Existing arm-1 and arm-2 rankings and pools were left unchanged; no old label was transferred.

### Decisive none-band comparison

The none-band result is unavailable under the corrected instrument for all three arms. Arm 1's historical publication is shown beside the absent re-judgment rather than silently promoted to a corrected result.

| Result | Production rows with no query vocabulary | Production relevance in band | ANE rows with no query vocabulary | ANE relevance in band |
|---|---:|---:|---:|---:|
| Arm 1, published | 54.9% | 7.9% | 33.5% | 23.3% |
| Arm 1, corrected re-judgment | not run | not run | not run | not run |
| Arm 2, corrected re-judgment | not run | not run | not run | not run |
| Arm 3, corrected re-judgment | not run | not run | not run | not run |

### Corrected calibration by arm

The known-good selection is keyed by `part_id`, so every arm has the same 36 calibratable retrieval-eligible queries and the same 239 exclusions. A rate was not computed from a sample the protocol declared too small.

| Arm | Calibration rate | Calibratable queries | Excluded for no unique literal | Result |
|---|---:|---:|---:|---|
| 1 | not run | 36 | 239 | sample-size guard failed |
| 2 | not run | 36 | 239 | sample-size guard failed |
| 3 | not run | 36 | 239 | sample-size guard failed |

The literal selector parsed identifiers, paths, symbols, and hash-like tokens from each original keyword-bag query, then required an exact, case-sensitive occurrence in exactly one of the 2,204 corpus rows. When several literals qualified, it preferred hash-like IDs, then paths, structured identifiers, mixed-case identifiers, and finally non-English alphabetic tokens, with length and lexical order as deterministic tie-breakers. The selected row excerpt was fixed from the original query and keyed by `part_id`, so arm 3 could not draw a different calibration row.

A literal-anchored known-good would still be among the most lexically overlapping rows in its batch. Passing this gate would show that the judge recognises real answer-shaped evidence; it would **not** show that the judge is free of overlap bias. The overlap-band table remains the only instrument for that question, and the unchanged probe was not run because no arm cleared the guard.

### Headline and lever status

| Pure-cosine metric | Arm 1 published, production / ANE | Arm 1 re-judged | Arm 2 re-judged | Arm 3 re-judged |
|---|---:|---:|---:|---:|
| Memory precision | 0.2751 / 0.5023 | not run | not run | not run |
| Commit precision | 0.2196 / 0.4265 | not run | not run | not run |
| Chunk precision | 0.3855 / 0.4144 | not run | not run | not run |
| Overall precision@10 | 0.2455 / 0.4371 | not run | not run | not run |
| Overall nDCG@10 | 0.4407 / 0.7319 | not run | not run | not run |
| Overall pooled recall@10 | 0.4385 / 0.8143 | not run | not run | not run |
| Overall MRR@10 | 0.5355 / 0.7672 | not run | not run | not run |

The corrected hybrid headline and per-class metrics are likewise not run for any arm. Arm 1's published numbers remain historical and are not corrected-calibration results. The **recipe lever** is arm 2 minus arm 1 and is unmeasurable. The **guidance lever** is arm 3 minus arm 2—not arm 3 minus arm 1—and is also unmeasurable. Substituting the arm-3-minus-arm-1 comparison would not isolate guidance and was not done.

No quality timing was collected. No corpus row was re-embedded, no bucket was rebuilt, and no ladder, catalog, fingerprint, certification record, or production state was touched.

## Arm 2: instructed production queries

### Verdict

No arm-2 candidate quality numbers are publishable. The instructed production query-vector file passed its SHA-256, count, dimension, normalization, and query-identity gates. The pure-cosine retrieval then reused the unchanged corpus and ANE vectors, admitted the same 275 of 298 queries under the same source filters, and produced 4,408 rows in the blind pool.

Fresh judging did not pass the predeclared calibration gate. Across 666 judging batches, the known-good item was relevant in 94.89% of batches, below the required 95%; the deliberately mismatched item was relevant in 0%, and the pairwise grade separation rate was 98.05%. Arm 1 passed the same three checks at 96.14%, 0%, and 98.14%, respectively.

The gate failed before candidate metrics were computed. A 4,370-row production-style hybrid pool had already been prepared, but hybrid judging, scoring, and the lexical-overlap probe were not run: continuing would turn labels rejected by the protocol into candidate numbers. The decisive none-band comparison remains unanswered. Arm 1 measured 7.9% relevance for production and 23.3% for ANE in that band; arm 2 has no corresponding figure.

No corpus or query vector was embedded, no bucket was rebuilt, and no ladder or production state was touched in this arm.

## Second-attempt verdict

Both original input blockers are cleared.

1. **Full retrievable-corpus re-embed duration:** **554.1 s (9 min 14 s) median** for 2,204 rows, with a 523.6–588.5 s range across three warm repetitions. This is model-execution wall time summed across the four non-overlapping, model-resident bucket intervals; it excludes database extraction, tokenization, model switching, compilation, specialization, and warmup. The 8,192 bucket dominates at 444.3 s median for 176 rows.
2. **Quality loss against production:** **none in the pure-cosine arm.** At top 10, ANE precision was 43.71% versus 24.55% for production, a **19.16 percentage-point gain**, and ANE nDCG was 0.7319 versus 0.4407. The query-bootstrap 95% interval for production-minus-ANE precision was -22.18 to -16.22 points. Pure-cosine judged precision also favored ANE within memory, commit, and chunk strata. The production-style hybrid arm retained a 16.04-point overall ANE precision advantage; its chunk stratum was nearly tied, with production ahead by 0.93 point.

That published interval resampled queries while treating labels as fixed. The
later 81.1% duplicated-row binary stability result shows that assumption is
false, so the interval is too narrow and does not cover judge uncertainty.

These measurements do not change a production fingerprint, catalog entry, certification record, or live store.

## Second-attempt input gates

All four package tree hashes were recomputed before use and matched the hashes already recorded in `EVIDENCE.json`. No bucket was rebuilt.

The replacement snapshot hash and every manifest count matched: 398 active memories, 601 memory vectors across statuses, 1,261 commit vectors, 545 chunk vectors, and 436 chunk compartments. All 298 production query vectors had 4,096 dimensions, unit L2 norm, and a `query_sha256` matching the corresponding private query text.

The same canonical chunk reconstruction used in the first attempt was rerun from an immutable, read-only snapshot. It regenerated 545 windows from 436 compartments and matched all 545 stored windows on compartment, model, window index, start ordinal, end ordinal, and `chunk_hash`. Three empty conversational spans used the established title-plus-`p1` fallback and also matched. No row was dropped or substituted.

## Re-embed timing

The timed retrievable corpus contains 398 active memories, 1,261 commits, and 545 chunk windows. Two unusually long commit messages were truncated at the model's 8,192-token limit; no memory or chunk row was truncated. Every other row was routed to the smallest fixed bucket that fit its tokenized input.

Each bucket was measured three times with `CPU_AND_NE` and GPU excluded. The table reports the median and full range of bucket wall times. Ambient columns give the median of the three interval medians followed by the complete observed range; the machine was shared and conditions were not idle or uniform.

| Fixed bucket | Rows | Median wall | Repetition range | 1-minute load average | CPU idle |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1,777 | 58.44 s | 58.10–59.31 s | 6.01 (5.52–9.66) | 77.2% (32.1–86.1%) |
| 2,048 | 121 | 10.13 s | 10.13–10.37 s | 6.11 (6.00–7.22) | 71.2% (59.7–82.3%) |
| 4,096 | 130 | 41.54 s | 41.51–42.76 s | 2.69 (2.28–3.88) | 80.4% (52.1–92.6%) |
| 8,192 | 176 | 444.33 s | 412.23–477.29 s | 9.12 (5.21–29.03) | 65.4% (19.5–87.6%) |
| **Full corpus** | **2,204** | **554.07 s** | **523.56–588.52 s** | **6.10 (2.28–29.03)** | **75.2% (19.5–92.6%)** |

The machine-readable record retains the load and idle summaries beside every one of the 12 bucket intervals rather than only beside these aggregates.

The four packages were each explicitly compiled once to stable `.mlmodelc` locations. The 1,024, 2,048, and 4,096 measurements loaded and reused those locations. Three isolated 8,192 stable-path load attempts ended exactly at this harness's 40-minute, 2-hour, and 1-hour timeout budgets. During an active attempt, `ANECompilerService` was observed making progress at load near 13 while unrelated CPU-heavy work competed. The six-hour system log contained no related jetsam or memorystatus kill. An earlier successful direct load of the same verified package took 941.0 s, but its conditions were not sampled throughout, so these observations establish severe contention sensitivity without defining a controlled startup curve or showing that the bucket is impossible.

To avoid another stable-path timeout, the authorized fallback loaded the verified 8,192 package once and measured all 176 rows warm in that single process. Core ML's internal compilation and specialization took 2,518.8 s; that load and the first two predictions were excluded. The row-cost measurement is valid and directly comparable in the table; it does **not** support a worker cold-start claim for the 8,192 bucket.

## Pooled blind quality protocol

Pure cosine was evaluated first at top 10. Source filters from each captured invocation were honored: memory and commit candidates are project-scoped, while chunk candidates are restricted to the query's session. Twenty-three of 298 queries requested only note or primer sources, which have no vectors in either compared corpus, so 275 queries entered the vector comparison.

For each query, the two systems' top 10 were pooled by stable row identity. System provenance was removed, item order was deterministically randomized, and each of the 4,843 pure-cosine pooled rows was judged once. Long rows were represented by the same system-blind, query-focused excerpt policy on both arms. A local Qwen3.5-35B-A3B Q4_K_M judge ran at temperature zero with schema-constrained grades: 0 unrelated, 1 weak, 2 useful, and 3 direct. Grades 2 and 3 count as relevant.

Judging batches held at most eight pooled rows plus a known-good and deliberately mismatched calibration item. The calibration gate required at least 95% known-good relevance, at most 1% mismatch relevance, and at least 95% pairwise grade separation. The accepted pure-cosine run passed across 751 batches: 96.14% known-good relevance, 0% mismatch relevance, and 98.14% pairwise separation. A pilot whose purported known-good arm merely repeated the query was rejected because a question is not known-good answer evidence; all of that pilot's labels were discarded and no candidate metrics were computed from them.

### Pure cosine: per class first

`Judged precision` is the relevant fraction among returned top-10 rows of that class. `Pooled coverage` is the fraction of all relevant rows in the shared judged pool that the system retrieved.

| Class | Production precision | ANE precision | ANE minus production | Production pooled coverage | ANE pooled coverage |
|---|---:|---:|---:|---:|---:|
| Memory | 27.51% | 50.23% | **+22.73 pp** | 51.75% | 74.83% |
| Commit | 21.96% | 42.65% | **+20.69 pp** | 40.55% | 82.69% |
| Chunk | 38.55% | 41.44% | **+2.89 pp** | 53.04% | 60.22% |

| Overall top-10 metric | Production | ANE | Production minus ANE, query-bootstrap 95% interval |
|---|---:|---:|---:|
| Precision | 0.2455 | 0.4371 | -0.2218 to -0.1622 |
| nDCG | 0.4407 | 0.7319 | -0.3295 to -0.2532 |
| Pooled recall | 0.4385 | 0.8143 | -0.4344 to -0.3192 |
| MRR | 0.5355 | 0.7672 | -0.2820 to -0.1810 |

These published intervals resample queries with fixed labels. Because labels
were only 81.1% stable in the duplicated-row comparison, every interval in the
last column is too narrow and excludes judge uncertainty.

### Production-style hybrid arm

Because pure cosine showed a gap, a second independently pooled and calibrated run added the stated 0.7 semantic / 0.3 reciprocal-rank FTS fusion, the 0.8 single-source penalty, and source boosts of 1.30 for memory, 1.20 for commits, and 1.275 for chunks. It judged 4,762 pooled rows. Its 741 calibration batches passed at 95.28% known-good relevance, 0% mismatch relevance, and 97.84% pairwise separation.

| Class | Production precision | ANE precision | ANE minus production |
|---|---:|---:|---:|
| Memory | 22.01% | 38.84% | **+16.83 pp** |
| Commit | 24.23% | 42.50% | **+18.27 pp** |
| Chunk | 40.40% | 39.47% | **-0.93 pp** |
| **Overall** | **25.02%** | **41.05%** | **+16.04 pp** |

Overall hybrid nDCG was 0.4793 for production and 0.7401 for ANE; pooled recall was 0.4748 and 0.8171 respectively. Fusion narrowed the overall precision advantage by 3.12 points but did not absorb it.

## Does the advantage ride on lexical overlap?

The queries are keyword bags — median six words, none of them questions — and a
language-model judge grading a keyword bag against a document tends to reward
surface term overlap. If the winning system also retrieves more overlapping
rows, retrieval and judging reward the same thing and the margin is inflated.

Both halves of that mechanism are present. The ANE arm's retrieved rows share
more query terms than production's (mean query-term coverage 0.212 against
0.131 in the pure arm), and the judge's grade correlates with overlap within
each system (+0.575 for production, +0.418 for ANE).

But the advantage does not live where that mechanism operates. Splitting judged
rows by how much of the query's content vocabulary appears in the text the judge
saw:

| Query-term overlap | Production relevant | ANE relevant | Gap | Production rows | ANE rows |
|---|---:|---:|---:|---:|---:|
| none | 7.9% | 23.3% | **+15.4 pp** | 1,510 | 921 |
| low | 31.2% | 42.6% | +11.4 pp | 648 | 780 |
| mid | 49.2% | 56.4% | +7.2 pp | 398 | 691 |
| high | 80.9% | 74.0% | **-6.9 pp** | 194 | 358 |

The gap is widest where there is NO shared vocabulary and reverses where overlap
is highest. An overlap artifact would do the opposite: concentrate the advantage
in the high band, where a judge can reward surface match, and vanish in the
none band. On rows sharing no query terms at all, the ANE arm is relevant three
times as often. The hybrid arm shows the same shape (+13.8 points at none, -1.9
at high).

So the confound is real as a mechanism and does not explain the result. What the
none-band numbers describe is production retrieving a large volume of rows that
share neither vocabulary nor meaning with the query — 55% of its returned rows
fall in that band against the ANE arm's 33%.

That is consistent with the separate finding that production's query path sends
raw keyword text to an instruction-tuned embedder with no instruction, which is
off-recipe for that model family. Under those conditions the comparison measures
how the two embedders degrade on this query distribution, not their relative
quality in general. A recipe-fair re-run is the correct next measurement.

The probe is `bench/spikes/ane-direct-probe/lexical_overlap_probe.py`; it reads
private judging artifacts from scratch and emits aggregates only.

## Privacy and interpretation limits

The database was opened with SQLite `mode=ro`, `immutable=1`, and `query_only=ON`. Private rows, vectors, queries, identifiers, model artifacts, and judge packets remained in scratch. This directory contains aggregate counts and metrics only.

The reported timing is warm model execution for a complete retrievable-corpus pass, not an end-to-end indexer benchmark or a cold-start measurement. Quality labels come from one deterministic local judge and query-focused excerpts, not human editorial judgments or Magic Context's sparse usefulness telemetry. Pooled judging makes the comparison symmetric, while the bootstrap intervals quantify query sampling variation rather than judge uncertainty.

## First-attempt verdict (retained)

Neither requested number is publishable from the supplied snapshot.

1. **Full re-embed duration:** not measured. The same-row identity gate failed before stable compiled models were prepared and before any Broca row was embedded.
2. **Quality loss against the production embedder:** not measured. The same-row gate failed, and the offline inputs also do not contain production-space query vectors. Pooled judging and its calibration arms were therefore not run. No candidate quality numbers appear in this evidence.

Stopping here is intentional. Timing a changed corpus or judging vectors derived from different text would produce precise-looking answers to different questions.

## Production and privacy boundary

The experiment opened only the supplied copy with SQLite `mode=ro` and `immutable=1`. It did not open the live Magic Context store, rotate an embedding identity, edit a catalog, touch certification state, or embed a Broca row. Model packages, reconstructed private text, database-derived vectors, and the query set remained in private scratch. This directory contains aggregate counts and model diagnostics only.

No query, message, memory, commit text, session identifier, row identifier, project identifier, or absolute operator path is committed here.

## Ladder result

The four Hadamard-conditioned packages were rebuilt from `bench/spikes/ane-modernbert-full-context/` with rotation seed 0 and 256-token query/key tiles. Each package passed the unchanged 0.999 minimum-cosine gate against the streaming CPU reference before any attempt to process corpus data.

| Fixed bucket | Minimum cosine | Fixture cosines | Result |
|---:|---:|---|---|
| 1024 | 0.9999255 | 0.9999590 / 0.9999255 / 0.9999814 | pass |
| 2048 | 0.9999255 | 0.9999825 / 0.9999255 / 0.9999814 | pass |
| 4096 | 0.9999255 | 0.9999752 / 0.9999255 / 0.9999814 | pass |
| 8192 | 0.9999038 | 0.9999038 / 0.9999255 / 0.9999814 | pass |

The staged controller was externally terminated while its separately spawned 8192 reload process was active. That process completed and wrote a passing parity report with byte-deterministic repeated predictions. The termination prevented the controller from launching the rebuilt 8192 placement inspection. The accepted rotation-conditioning evidence already documents the same source graph's 8192 placement, but this report does not pass that prior placement off as a new measurement.

The package reloads also emitted fixture latencies, but they are not a re-embed timing result. The single pre-run condition was a one-minute load average of 13.62 and 0.2% CPU idle, and conditions were not sampled beside every fixture prediction. The machine-readable evidence retains those diagnostic values and labels them accordingly.

## Production-vector check

Every observed production baseline BLOB was 16,384 bytes. Decoding it as little-endian float32 yields 4,096 dimensions, agreeing with the supplied production description. This arithmetic confirms the dtype/dimension pair; it cannot uniquely identify a model. The copied store's registration uses an opaque provider identity and contains no model-name provenance, so the model name itself remains operator-supplied rather than independently derivable from BLOB size.

## Same-row identity gate

The supplied description and the copied files no longer describe the same corpus population:

| Class | Declared rows | Rows carrying production vectors in the copy |
|---|---:|---:|
| Memory | 396 | 398 |
| Commit | 1,233 | 1,252 |
| Chunk window | 543 | 561 |
| **Total** | **2,172** | **2,211** |

Count drift alone does not prove that shared rows differ, so the rig performed the stronger chunk-content check.

Canonical transcript text was reconstructed from `message_fts_rowid_map` joined to `message_history_fts`, using the exact line grammar and 8192-token production windowing in `cortexkit/magic-context` commit `70d3945bde0feb75a24e922880791f5fe7267823`. Consecutive user or assistant entries were coalesced before windowing; oversized lines used the production recursive splitter and Claude token estimator. The rig never substituted `compartments.content` for transcript text. Three empty conversational spans followed the production fallback of title plus `p1`, and all three fallback hashes matched their stored rows.

That reconstruction produced 545 current windows from 436 compartments. Only 516 of the 561 stored production windows matched on all of:

- compartment identity;
- production model identity;
- window index;
- start and end ordinals; and
- SHA-256 `chunk_hash` of the exact embeddable text.

There were 45 stored production windows without an exact current match and 29 current windows without an exact stored baseline. A production vector for stale text cannot be compared fairly with a candidate vector for current text. Re-windowing to 545 rows would likewise violate the requirement to compare the production row identities.

The gate therefore rejected the corpus before embedding. It did not silently drop the 45 stale rows, substitute summaries, or publish a timing extrapolation from fixture costs.

## Quality protocol and why it did not run

The intended quality run remains pooled and blind: retrieve pure-cosine top-k from both spaces, pool by stable row identity, remove system provenance, randomize presentation, judge every pooled row once, and score both systems from the shared labels. Results would be stratified by memory, commit, and chunk before any overall summary. The production-style 0.7 semantic / 0.3 FTS arm would run only after the pure-cosine comparison.

A known-good calibration item and a deliberately mismatched item must accompany every judging batch. Candidate metrics are publishable only if those arms separate. Because the row-identity gate failed before retrieval, neither calibration arm ran and they did not establish reliability. Consequently there are no pure-cosine, hybrid, aggregate, or per-class candidate values.

There is a second independent offline gap: the copy contains production document vectors, but neither the query JSON nor the database contains the 298 corresponding query vectors in the production 4,096-dimensional space. Rank agreement across unrelated vector spaces is invalid, and using another local model for production-side queries would not repair that. Calling the live service or sending private query text to an external provider would break the stated isolation boundary, so neither was attempted.

## What is needed to answer the questions

A replacement private evidence bundle should be captured while indexing is quiescent and should include:

1. a row manifest whose counts agree with the database snapshot;
2. canonical text for every chunk whose hash equals the stored `chunk_hash` at the same compartment/window identity, or enough immutable FTS data to reproduce it; and
3. production-space query vectors for all 298 queries, generated by an isolated exact production embedder and stored beside the private inputs.

With those gates satisfied, the rebuilt packages can be explicitly compiled once to stable `.mlmodelc` paths, specialized once, and reused for repeated full-corpus passes. Each pass can then record one-minute load average and CPU idle beside every bucket interval, yielding medians and spread without charging repeated specialization to row cost.

Compact machine-readable details, package hashes, parity values, and the exact blocker counts are in [`EVIDENCE.json`](./EVIDENCE.json).
