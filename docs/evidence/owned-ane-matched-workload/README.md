# Production GTE ANE matched-workload evidence

## Scope and conclusion

This is a bounded, machine-local measurement of the installed production
`gte-modernbert-base-ane-fp16` package set through the installed production ANE worker.
It does not convert a model, change an engine, alter fleet configuration, modify the MC
shadow, or perform full certification.

Only one complete fixture case is accepted unchanged by the production ANE maximum:
`text-singleton-511`. Its first `EMBED_BATCH` IPC round trip was 30.578 ms; the next
three round trips averaged 25.494 ms and 20,043.9 real tokens per IPC second. The
three warm vectors and the first-observation vector were byte-identical. The full 513,
603, 1,203, and mixed 3,072-token cases are unsupported unchanged because at least one
row exceeds 512 tokens. Nothing was truncated, chunked, or silently dropped.

A clearly separate diagnostic measures the four distinct rows at or below 512 tokens
from the mixed fixture. This is a subset, not the original six-row request and not a
same-workload timing comparison with its Metal result.

## Timing boundaries

ANE values are **production worker IPC round-trip** measurements. The clock starts
before JSON serialization and request/raw-ID writes and stops after the response/raw
vector frames are read and converted to host `f32` values. This includes Python harness
work, Unix-socket framing, worker dispatch, Core ML execution, pooling, normalization,
response serialization, and vector transfer. It is not an engine-only timer.

The copied Metal fields keep their original metric names and scope:

- `first_use_engine_wall_s`
- `warm_engine_wall_s`
- `mean_warm_engine_wall_s`
- `aggregate_real_tokens_per_s`
- `cold_load_s`

Those values time the Metal `embed_batch` engine boundary. ANE IPC values are displayed
beside them to answer the bounded hardware question, but no speedup ratio or claim of
identical timing scope is made.

The first timed ANE observation for each workload is its single warmup and is excluded
from the three-repeat warm aggregate. For `text-singleton-511`, it is also the first
`EMBED_BATCH` request after load. The mixed subset runs second in the same worker, so its
warmup is not a cold seq512-model observation. `LOAD` itself performs dimension probes
and placement-plan inspection before either workload.

## Full fixture table

All times are milliseconds. `ANE first` and `ANE warm` are worker IPC round trips;
`Metal first` is `first_use_engine_wall_s` and `Metal warm` is
`mean_warm_engine_wall_s`, each converted to milliseconds.

| Fixture case | Row token counts | Real tokens | ANE unchanged status | ANE first | ANE warm | Metal v1 first | Metal v1 warm | Metal v2 first | Metal v2 warm |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `text-singleton-511` | 511 | 511 | measured unchanged | 30.578 | 25.494 | 253.307 | 244.362 | 171.491 | 29.860 |
| `text-singleton-513` | 513 | 513 | unsupported: 513 > 512 | — | — | 1,761.120 | 1,769.365 | 191.061 | 48.281 |
| `text-singleton-603` | 603 | 603 | unsupported: 603 > 512 | — | — | 1,756.741 | 1,739.976 | 47.092 | 47.560 |
| `text-singleton-1203` | 1,203 | 1,203 | unsupported: 1,203 > 512 | — | — | 1,747.220 | 1,752.555 | 280.720 | 114.687 |
| `text-mixed-budget-3072` | 127, 255, 383, 511, 603, 1,193 | 3,072 | unsupported: rows 603 and 1,193 exceed 512 | — | — | 3,830.399 | 3,698.269 | 1,052.846 | 945.895 |

For the sole unchanged case, original throughput labels are retained:

| Measurement | Metric | Value |
|---|---|---:|
| ANE | `aggregate_real_tokens_per_ipc_second` | 20,043.9 |
| Metal v1 | `aggregate_real_tokens_per_s` | 2,091.2 |
| Metal v2 | `aggregate_real_tokens_per_s` | 17,113.1 |

These are not used to calculate cross-interface speedups.

## Optional supported-row subset

`text-mixed-budget-3072-supported-row-subset` contains the original, unmodified
`music`, `astronomy`, `cooking`, and `software` rows in fixture order. Their lengths are
127, 255, 383, and 511 (1,276 real tokens total). `education` (603) and `transport`
(1,193) are omitted explicitly. The resulting four-row request is not named or reported
as the 3,072-token fixture case.

| Scope | First-observation/warmup IPC ms | Mean of 3 warm IPC ms | Aggregate real tokens/IPC s |
|---|---:|---:|---:|
| four-row supported subset | 100.278 | 100.468 | 12,700.5 |

All four output rows were byte-identical across the warmup and three warm repeats.
There is no corresponding exact-workload Metal timing in the copied evidence: the Metal
mixed result is for all six rows and 3,072 real tokens.

## Load, placement, and runtime maximum

The installed launcher reported `ck-synapse-worker-ane 0.1.0-alpha.1` and handed off to
its installed Swift sibling. One inference worker tree was used serially. Worker
spawn-to-`HELLO` was 6.647 ms. The all-bucket `LOAD` IPC round trip was 26,932.839 ms,
and the worker's `cold_load_ms` field was 26,932 ms. The subsequent production `PING`
reported `placement_share=0.9941245593419507`.

`LOAD` covers SHA-256 verification, extraction of all three compiled archives, Core ML
loading, dimension probes, and placement-plan inspection. It therefore differs from the
copied Metal `cold_load_s` values (17.590917416 s for v1 and 14.609868209 s for v2); no
cold-load ratio is reported.

The harness sent the exact 513-token `database` row only as a rejection probe. No
inference ran, and the production worker returned:

```text
batch requires 513 tokens but ANE buckets are [128,256,512]
```

This runtime result agrees with both the on-disk package roots and the source dispatch
rule in `crates/synapse-worker-ane/swift/ane_worker.swift`: `handleLoad` accepts compiled
Core ML package sets, while `handleEmbedBatch` selects the smallest loaded bucket large
enough for the longest row and rejects requests for which no such bucket exists. The
worker has no MiniLM-only family check; it consumes pretokenized IDs and dispatches the
loaded Core ML artifacts.

## Exact inputs and artifacts

`representative-text-fixture.json` is copied byte-for-byte from
`0de0d95a41c1606853509f87e4f04b9f86f2d411:docs/evidence/owned-metal-bucket-policy-v2/representative-text-fixture.json`.
Its SHA-256 is
`1297d1cfd09464e4b71917d317490733bf66a5e9f946652f2cff0d55e5f8b62a`.
The harness never invokes a tokenizer. It validates each exact `input_ids` array using
unsigned little-endian 32-bit IDs before sending the same positive IDs as signed
little-endian 32-bit worker payloads (identical bytes for this fixture).

Fixture lineage digests, copied rather than regenerated:

- tokenizer: `6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30`
- model config: `8ba54dc3d35d7194f5178a4194b649f146753e02dabd22bdca5c5cbac15069ed`
- model weights: `3e85899d5728cb7de79781c0c3acfb91ccef9f875f1f7e0b3c9f3dd4b6a724ba`

| Row | Tokens | `input_ids_u32le_sha256` |
|---|---:|---|
| climate | 511 | `0d3367c5fc4689301998eb69f15b9f8ad4a7a120c35435885b6c9c3d7934b8d8` |
| database | 513 | `89e903dc0ff8586bc390acfd6b682accef4899af5b380babadbcc9331acfc4a8` |
| biology | 603 | `5a125658bba9c1a987ff9decbb2c36001220c37458f4155863ebdae77d14490e` |
| history | 1,203 | `a6088c8bf7e86f96374232c1c29d274fea6a6335cf0849fb1cc0ef7fa9832a8e` |
| music | 127 | `0aa0378f8f6b28bcb83317694baad81e5d17c8217ba7922c2e6acd82d8f12a56` |
| astronomy | 255 | `44d575756ab35edb26cab6b079a24d1f837dce160c601c8dc4ff023128d2c053` |
| cooking | 383 | `39ea347470b8ebd789f7d1b276021a052e501249a110137c76181660e6d4b3e9` |
| software | 511 | `9767afd28e4bfbffe635f8ee1eb2fa395079ec6ed6b1a8cc8d87b0cbb7638268` |
| education | 603 | `c6fefe7b16745f0bdcc624f1dcc5f1e25bb3a906d79eca520d6c0c2ff380d965` |
| transport | 1,193 | `4bee0b89f6cfbc852abf60d6e13e79d428de28f2db7c5eae3d311ce4a989ce83` |

The three production CAS files existed, were regular ZIP archives, and matched their
names byte-for-byte:

| Bucket | Bytes | CAS SHA-256 / filename | Archive root |
|---:|---:|---|---|
| 128 | 275,182,932 | `8c3bf4b2a50634ec4a3eb54e986b769c84d0d11946c281ecd88d24827afd7d80` | `gte-modernbert-base-seq128.mlmodelc` |
| 256 | 275,207,675 | `a8626c487794f879b88c73bf9c8fe6f7864e3cc252a1db0d79bb5e096210b9fd` | `gte-modernbert-base-seq256.mlmodelc` |
| 512 | 275,261,188 | `9df6b44617b49e08a068602efc3fddcd7dadc0e46791929698b524c96d544f72` | `gte-modernbert-base-seq512.mlmodelc` |

Installed binary SHA-256 values recorded by the run:

- launcher: `baca4c27a4a28cff6ef86fcd6517b0f66d2f52b9625f92c34d309faffb30c7a7`
- Swift worker: `7f36edcfd737b6b831ae5017caeb560892e4097e641ad80a793570c127d9fc0a`
- inspected Swift source: `6f577dca660e84a1acdad35479dc00277bb1419e97448abc1fe79523f5d0da05`

## Vector determinism and Metal diagnostic

The worker returned 768-dimensional normalized `f32` vectors. Every invocation's exact
raw `f32le` bytes are preserved as base64 in `ane-worker-ipc.raw.json`; the table below
shows the digest shared by the warmup and all three warm repeats.

| ANE workload row | Stable vector SHA-256 | Exact observations | Metal v2 cosine | Max absolute difference |
|---|---|---:|---:|---:|
| climate | `f1d87182d985575a087db5b953b3fc28fbb1472687d0589cbf5d15d881fd4b42` | 4/4 | 0.9992788443 | 0.005612223 |
| music (subset) | `f4d1782143df576963f57ee7b34d27aa95603d10e68edc041932bd2ac39aa675` | 4/4 | 0.9993175607 | 0.009045836 |
| astronomy (subset) | `26783a4db43bafe3a9ca2918442977b28ec0297d90ffd29634a738a78e922b58` | 4/4 | 0.9997920819 | 0.004483883 |
| cooking (subset) | `54483a99165c3d113644653eba893fcfca58821fb0f849c4c325578d6a5ae204` | 4/4 | 0.9993984366 | 0.007209327 |
| software (subset) | `be6c12f90bddeacb0e2969573fc8722f1bdb0097c7321f34b979286c6718c2fb` | 4/4 | 0.9999179958 | 0.002028379 |

The cosine/max-difference values compare the first ANE warm repeat with the exact Metal
v2 reference vector for the same row IDs and token IDs. For subset rows, that Metal
vector came from the original full six-row mixed call, not a four-row timing run. ANE
and owned Metal have distinct engine/package fingerprints. These similarities are a
numerical diagnostic only, not fingerprint equivalence, mixed-space approval, or a
claim that the vectors can be interchanged in production.

## Host and reproduction

Observed 2026-09-08 UTC on an Apple M5 Max MacBook Pro (`Mac17,6`, 18 cores, 128 GB),
macOS 27.0 build `26A5425a`. The worker reported Neural Engine placement as above.

Validate all pinned fixture/reference/artifact/binary hashes without loading Core ML:

```sh
python3 docs/evidence/owned-ane-matched-workload/run_production_ane_ipc.py \
  --validate-only
```

Repeat the bounded measurement (one serial worker, one first-observation warmup, three
warm repeats per measured workload):

```sh
python3 docs/evidence/owned-ane-matched-workload/run_production_ane_ipc.py
```

The harness implements the production v1 length-prefixed Unix-socket handshake and
`LOAD`/`PING`/`EMBED_BATCH`/`UNLOAD`/`SHUTDOWN` requests used by the installed Rust
launcher and Swift worker. It always unloads before shutdown so extracted temporary
Core ML packages are removed.

Durable files:

- `run_production_ane_ipc.py`: runnable, standard-library-only IPC harness and input
  validators.
- `representative-text-fixture.json`: exact canonical fixture.
- `metal-text-comparison.json`: exact copied Metal v1/v2 comparison with original metric
  labels.
- `metal-v2-reference.raw.json`: exact copied v2 vectors used only for the numerical
  diagnostic.
- `ane-worker-ipc.raw.json`: machine, source, worker, artifact, load, placement,
  unsupported-case, timing, exact-ID, raw-vector, and determinism evidence. The
  operator's data directory appears as `$CORTEXKIT_DATA` rather than an absolute
  path; every artifact in the record is identified by its sha256, so the install
  location carried no evidentiary weight and this repository is public.
