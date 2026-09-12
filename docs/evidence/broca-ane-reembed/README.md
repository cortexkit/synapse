# Broca ANE re-embed and retrieval-quality rig

## Second-attempt verdict

Both original input blockers are cleared.

1. **Full retrievable-corpus re-embed duration:** **554.1 s (9 min 14 s) median** for 2,204 rows, with a 523.6–588.5 s range across three warm repetitions. This is model-execution wall time summed across the four non-overlapping, model-resident bucket intervals; it excludes database extraction, tokenization, model switching, compilation, specialization, and warmup. The 8,192 bucket dominates at 444.3 s median for 176 rows.
2. **Quality loss against production:** **none in the pure-cosine arm.** At top 10, ANE precision was 43.71% versus 24.55% for production, a **19.16 percentage-point gain**, and ANE nDCG was 0.7319 versus 0.4407. The query-bootstrap 95% interval for production-minus-ANE precision was -22.18 to -16.22 points. Pure-cosine judged precision also favored ANE within memory, commit, and chunk strata. The production-style hybrid arm retained a 16.04-point overall ANE precision advantage; its chunk stratum was nearly tied, with production ahead by 0.93 point.

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

### Production-style hybrid arm

Because pure cosine showed a gap, a second independently pooled and calibrated run added the stated 0.7 semantic / 0.3 reciprocal-rank FTS fusion, the 0.8 single-source penalty, and source boosts of 1.30 for memory, 1.20 for commits, and 1.275 for chunks. It judged 4,762 pooled rows. Its 741 calibration batches passed at 95.28% known-good relevance, 0% mismatch relevance, and 97.84% pairwise separation.

| Class | Production precision | ANE precision | ANE minus production |
|---|---:|---:|---:|
| Memory | 22.01% | 38.84% | **+16.83 pp** |
| Commit | 24.23% | 42.50% | **+18.27 pp** |
| Chunk | 40.40% | 39.47% | **-0.93 pp** |
| **Overall** | **25.02%** | **41.05%** | **+16.04 pp** |

Overall hybrid nDCG was 0.4793 for production and 0.7401 for ANE; pooled recall was 0.4748 and 0.8171 respectively. Fusion narrowed the overall precision advantage by 3.12 points but did not absorb it.

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
