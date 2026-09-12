# Broca ANE re-embed and retrieval-quality rig

## Verdict

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
