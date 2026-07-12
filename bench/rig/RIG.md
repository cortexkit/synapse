# Synapse measurement rig

`synapse-rig` is the blessed benchmark entry point for the owned runtime. It is built separately from the candidate and drives the candidate as a subprocess, so candidate changes cannot alter corpus selection, canonical token accounting, timing boundaries, parity math, gates, or result serialization.

## Invocation

Embedding:

```sh
synapse-rig \
  --candidate /pinned/bin/spike-unified-rt \
  --model /models/all-MiniLM-L6-v2 \
  --tokenizer /models/all-MiniLM-L6-v2/tokenizer.json \
  --corpus /data/corpus.jsonl --reference /data/reference-vectors.jsonl \
  --out /results/run.json --vectors-out /results/vectors.jsonl \
  --device cpu --dtype f32 --shapes exact --passes 3
```

Reranking replaces `--corpus` with `--rerank-requests` and optionally uses `--scores-out`. The rig starts the candidate with the same model, tokenizer, provider, precision, execution, cache, maximum-length, attention-budget, and shape-policy settings plus `--serve-stdio`.

Campaign executors must build and stage `synapse-rig` independently, hash the staged executable, pass the expected hash with the campaign specification, and reject a result whose `rig_metadata.sha256` or `rig_metadata.git_revision` differs. The executable computes SHA-256 over `std::env::current_exe()` at runtime; `build.rs` embeds the source Git revision.

## Candidate protocol

stdin and stdout contain only length-prefixed JSON frames. Each frame starts with a four-byte little-endian unsigned payload length followed by that many UTF-8 JSON bytes. Frames larger than 256 MiB are rejected. Diagnostic text belongs on stderr. Protocol version 1 uses tagged objects whose `kind` is one of:

- Candidate to rig: `ready`, `prepared`, `embedding`, `rerank`, `shutdown`, or `error`.
- Rig to candidate: `prepare_shapes`, `embed`, `rerank`, or `shutdown`.

A candidate sends `ready` after loading its model and provider. It includes identity plus internal load timing as advisory metadata. The rig may then send `prepare_shapes` to preserve the existing warmup and eager shape-discovery behavior. An embedding request contains `texts`, `max_length`, `shape_policy`, and `{batch, seq}`. A rerank request contains `pairs` of `{query, document}` and the same policy fields. Responses contain vectors or raw scores, the candidate's reported real-token count, and its internal request wall time.

The candidate never receives reference vectors or scores, corpus IDs, the reason for corpus ordering, gate thresholds, measured rig walls, pass labels, or throughput calculations. The candidate does receive only the text or pairs needed for the current batch and an explicit shape directive. Candidate timings are recorded under `rig_metadata` but never gate a result.

## Measurement and gates

The rig loads and limits the corpus or rerank requests before starting the candidate. It sanitizes the tokenizer by disabling baked-in padding and installing the requested truncation limit. Canonical real-token counts use attention-mask positions, not encoded buffer lengths, which keeps the known 4.21% padded-token trap outside candidate control. Qwen3 terminal-token handling and ModernBERT pair encoding follow the existing owned-runtime rules.

Embedding rows remain length-sorted and use the existing exact or bucket-policy-v1 planner. Each pass wall begins immediately before the first batch is assembled and ends after the final response is reconciled; labels remain `first`, `warm`, and (for the last of at least three passes) `steady`. Rerank request walls retain pair tokenization, planning, transport, and inference. Load walls include process startup, model/provider load, and the existing warmup or eager shape preparation. The subprocess boundary is therefore part of rig-measured inference.

For every pass, the rig compares its canonical real-token total with the candidate report. Divergence greater than 1% fails loudly. Throughput always uses the rig's canonical numerator. Bucket padding waste still fails at 15%. Embedding parity uses `synapse_bench::parity` unchanged: mean cosine must be at least 0.9999 and mean top-10 overlap at least 0.995 by default. Rerank gates remain aggregate Pearson at least 0.999 and tie-aware top-1 agreement at least 0.98.

## Result and exit contract

The existing embedding and rerank result fields retain their names and meanings. `rig_metadata` is the only additive top-level block. It contains:

- rig executable SHA-256, embedded Git revision, and protocol version;
- candidate identity and advisory internal load timing;
- advisory candidate preparation and per-pass timing;
- canonical and candidate token totals plus divergence for each measured pass;
- best-effort `macmon --version` metadata on macOS, or `nvidia-smi` GPU/driver metadata for CUDA. The host probe is absent when unsupported or unavailable.

Exit status `0` means all protocol checks, reconciliation checks, and requested gates passed and the result was written. Status `1` means argument, load, protocol, reconciliation, parity, ranking, rerank, candidate-exit, or output failure. Executors must accept only status `0`, parse the result file rather than stdout, and verify both rig pin fields before publishing it.

## Semantics-drift verification

The split was checked locally on CPU with the `all-MiniLM-L6-v2` snapshot and a deterministic 400-row MiniLM semantics corpus. The legacy in-process path emitted the reference vectors; both paths then ran over the same rows, tokenizer, exact-shape policy, maximum length, attention budget, and reference.

- canonical and candidate token totals were exactly equal at 6,439, with zero reconciliation divergence;
- legacy and rig real/padded totals were exactly equal at 6,439 / 6,448;
- mean cosine was exactly equal at `0.999999999998994`;
- mean top-10 overlap was exactly equal at `1.0`;
- both parity gates passed.

The subprocess path produced a different wall-derived throughput, as expected. That timing delta is intentionally not retained as a performance claim: the local run was a semantics check, and subprocess transport is now part of the blessed boundary.

`tests/full_loop.rs` independently creates a tiny tokenizer, 12-row corpus, and reference, spawns the fixture candidate through the framed protocol, runs three passes, reconciles tokens, executes both parity gates, checks first/warm/steady labels, and validates the emitted pin metadata.
