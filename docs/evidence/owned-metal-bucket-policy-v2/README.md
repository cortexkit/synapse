# Owned Metal bucket policy v2 evidence

## Representative text-tokenized A/B

The primary evidence is the bounded serial workload in
`representative-text-fixture.json`, generated with the committed
`crates/synapse-engine-owned/examples/embed_bucket_probe.rs`. It contains diverse
text seeds, exact kept token IDs, per-row ID digests, and measured real-token counts.
No input was chunked. Baseline and candidate inputs are byte-for-byte identical.

Configuration:

- Hardware: Apple M5 Max, 18 cores, 128 GB.
- Model: GTE ModernBERT base, f16, explicit MPSGraph execution.
- Model config SHA-256: `8ba54dc3d35d7194f5178a4194b649f146753e02dabd22bdca5c5cbac15069ed`.
- Model weights SHA-256: `3e85899d5728cb7de79781c0c3acfb91ccef9f875f1f7e0b3c9f3dd4b6a724ba`.
- Tokenizer SHA-256: `6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30`.
- `max_tokens=8192`, `attention_units=67108864`, explicit execution.
- Serial order: Metal v1 process, then Metal v2 process. Each case had one timed
  first-use call followed by three measured warm calls. Separate caches were empty at
  process start. Cold load was measured separately.
- Throughput is `total real tokens / total measured wall`, not a mean of reciprocal
  timings. The five-case aggregate was 641.2 real tok/s for v1 and 4,975.2 real tok/s
  for v2 (17,706 measured tokens per arm).

| Case | Real tokens | v1 first-use ms | v2 first-use ms | v1 warm ms | v2 warm ms | v1 real tok/s | v2 real tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| text-singleton-511 | 511 | 253.307 | 171.491 | 244.362 | 29.860 | 2,091.2 | 17,113.1 |
| text-singleton-513 | 513 | 1,761.120 | 191.061 | 1,769.365 | 48.281 | 289.9 | 10,625.4 |
| text-singleton-603 | 603 | 1,756.741 | 47.092 | 1,739.976 | 47.560 | 346.6 | 12,678.8 |
| text-singleton-1203 | 1,203 | 1,747.220 | 280.720 | 1,752.555 | 114.687 | 686.4 | 10,489.4 |
| text-mixed-budget-3072 | 3,072 | 3,830.399 | 1,052.846 | 3,698.269 | 945.895 | 830.7 | 3,247.7 |

Cold load was 17,590.917 ms for v1 and 14,609.868 ms for v2. The mixed request has
six distinct rows of 127, 255, 383, 511, 603, and 1,193 tokens, totaling exactly the
module's 3,072-token quantum and remaining below eight rows.

The probe asserts the complete ordered vector array equals first use on every measured
repeat; `repeat_vector_sha256` records every row of every repeat. The comparison tool
also rejects metadata, exact input IDs, lengths, row counts, vector dimensions, profile
row counts, or repeat digests that differ. The mixed batch is byte-exact across v1/v2,
which provides non-vacuous output-order coverage because all six texts and vectors are
distinct. Singleton vectors differ slightly across shapes; the worst observed cosine was
0.99998651 and maximum absolute difference was 0.00066243. Policy v2 deliberately
rotates engine/package identity, so no same-fingerprint or mixed-space claim is made.

## ANE feasibility and exact handoff

No existing ANE harness can execute this complete workload unchanged. The installed
worker is MiniLM/mean-pooling only. The standalone GTE converter and runner support only
fixed 128/256/512 buckets; rows 513, 603, and 1,203 are over the maximum, and the runner
requires every row in a batch to have identical padded length. Therefore the mixed
request cannot be run without dropping, truncating, or splitting rows, all of which would
change the comparison.

`ane-feasibility.json` records the explicit refusals. `ane-supported-seq512.jsonl` exports
only the 511-token row to the existing fixed-512 schema, preserving all kept IDs and
adding one masked pad token (`pad_token_id=50283`). If a matching GTE seq512 Core ML
artifact is available, the bounded supported subset can be run separately:

```sh
swiftc -O -parse-as-library \
  -o target/ane-coreml bench/spikes/ane-minilm/ane_coreml.swift

target/ane-coreml run \
  --model <gte-modernbert-seq512.mlmodelc> \
  --input docs/evidence/owned-metal-bucket-policy-v2/ane-supported-seq512.jsonl \
  --output <ane-vectors.jsonl> \
  --stats-out <ane-stats.json> \
  --placement-out <ane-placement.json> \
  --batch-size 1 --pooling cls --compute-units cpuAndNeuralEngine
```

This would be a supported-subset diagnostic, not an exact three-arm workload. ANE has a
distinct engine fingerprint and must not be treated as vector-space equivalent from
cosine alone.

## Synthetic shape diagnostic

The earlier repeated-token sweep is retained only as a shape/timing diagnostic, not as
production-throughput or ordering/parity evidence. Its compact derived files are
`comparison.json`, `short-control-comparison.json`, `summary.txt`, and
`short-control-summary.txt`. Original raw synthetic JSON and profile logs were preserved
before compaction at:

`/Users/ufukaltinok/.local/share/cortexkit/alfonso/worktrees/07d868436e96de13/pool-94/.cortexkit/alfonso/evidence/owned-metal-bucket-policy-v2/synthetic-originals`

## Durable files

- `representative-text-fixture.json`: canonical diverse text/ID fixture and digests.
- `text-metal-v1.raw.json`, `text-metal-v2.raw.json`: compact raw metadata, exact IDs,
  timings, vectors, and per-repeat vector digests.
- `text-metal-v1.profile.stderr`, `text-metal-v2.profile.stderr`: engine-selected shapes
  and first-use/cache timing.
- `text-comparison.json`, `text-summary.txt`: strictly validated aggregate comparison.
- `ane-supported-seq512.jsonl`, `ane-feasibility.json`: exact supported subset and
  explicit unsupported requests.

Regenerate comparisons with
`bench/campaign/compare-owned-metal-bucket-probes.py`; run its negative tests with
`python3 -m unittest bench/campaign/tests/test_compare_owned_metal_bucket_probes.py`.
