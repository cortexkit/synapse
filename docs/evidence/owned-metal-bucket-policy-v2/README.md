# Owned Metal bucket policy v2 evidence

This directory contains durable raw output from the bounded serial probe in
`crates/synapse-engine-owned/examples/embed_bucket_probe.rs`. The baseline binary used
`ef00b483a4ee95efddab1d86d0e8b1a3bcd9e0b3`; the candidate used bucket policy v2 with
the original short buckets preserved and long shapes loaded on first use.

## Configuration and protocol

- Hardware: Apple M5 Max, 18 cores, 128 GB.
- Model: GTE ModernBERT base, f16, explicit MPSGraph execution.
- `config.json` SHA-256: `8ba54dc3d35d7194f5178a4194b649f146753e02dabd22bdca5c5cbac15069ed`.
- `max_tokens`: 8192; `attention_units`: 67,108,864; synthetic input token ID: 1.
- Main run order: baseline process, then candidate process. Each case ran once as an
  unmeasured first-use/warmup call, then twice in the listed order. Processes and cases
  were serial; caches were separate and initially empty.
- Warm milliseconds below are the arithmetic mean of the two measured engine calls.
  Token rates are the arithmetic mean of each repeat's `real_tokens / engine_wall_s`.
- Cold load, kept separate from first-use calls: baseline 17,694.413 ms; candidate
  14,923.541 ms.

| Case | Baseline warm ms | Candidate warm ms | Baseline real tok/s | Candidate real tok/s |
|---|---:|---:|---:|---:|
| singleton-511 | 221.011 | 30.314 | 2,312.1 | 16,857.5 |
| singleton-513 | 1,718.693 | 48.098 | 298.5 | 10,665.7 |
| singleton-603 | 1,697.327 | 48.458 | 355.3 | 12,443.7 |
| singleton-1203 | 1,674.083 | 117.336 | 718.6 | 10,252.9 |
| singleton-2600 | 1,687.830 | 360.841 | 1,540.5 | 7,205.5 |
| singleton-4096 | 1,742.814 | 643.978 | 2,350.4 | 6,363.6 |
| filled-131 (8 rows) | 60.088 | 71.776 | 17,441.5 | 14,601.0 |
| filled-320 (8 rows) | 122.512 | 147.939 | 20,897.3 | 17,305.1 |
| filled-448 (8 rows) | 188.463 | 215.396 | 19,019.9 | 16,641.6 |
| filled-603 (8 rows) | 14,278.730 | 392.200 | 337.8 | 12,301.8 |
| filled-1203 (8 rows) | 16,648.376 | 980.595 | 580.3 | 9,814.8 |
| filled-2600 (7 rows) | 16,151.719 | 2,435.713 | 1,130.2 | 7,472.5 |
| filled-4096 (4 rows) | 9,111.727 | 2,221.991 | 1,798.2 | 7,384.1 |
| mixed | 11,943.401 | 3,216.480 | 797.7 | 2,962.0 |

Because the baseline-first main run showed lower candidate numbers for the unchanged
131/320/448 shapes, a short bounded order-control was run candidate first and baseline
second against warm package caches. Each case had one warmup and seven measured calls:

| Case | Baseline warm ms | Candidate warm ms | Baseline real tok/s | Candidate real tok/s |
|---|---:|---:|---:|---:|
| filled-131 (8x160) | 79.777 | 61.307 | 13,144.8 | 17,096.3 |
| filled-320 (8x320) | 154.067 | 125.457 | 16,618.1 | 20,406.1 |
| filled-448 (8x448) | 212.524 | 189.268 | 16,871.3 | 18,944.7 |

The selected shapes and vectors are identical for these three short controls. The two
orders bracket host-state variance and provide no evidence of a policy-caused full-batch
regression after preserving the original short ladder.

All main-run vectors were byte-exact across policies except singleton-511, where changing
from 8x512 to 1x512 produced maximum absolute difference 0.00036131 and cosine
0.9999965398. This passes the existing 0.999 cosine gate, but policy v2 deliberately
changes engine and package-cache identity; cosine is not used to claim an unchanged
fingerprint.

## Files

- `baseline.raw.json`, `candidate.raw.json`: full metadata, per-repeat timings, and vectors.
- `baseline.profile.stderr`, `candidate.profile.stderr`: engine-emitted selected shapes and
  compile/cache timing.
- `comparison.json`: parsed shapes, padded-token ratios, throughput, and exact/cosine
  vector comparison.
- `baseline-short-control.raw.json`, `candidate-short-control.raw.json` and matching
  `*.profile.stderr`: seven-repeat reverse-order short-shape control.
- `short-control-comparison.json`: parsed short-shape control.
- `summary.txt`, `short-control-summary.txt`: compact generated tables.

Regenerate comparisons with
`bench/campaign/compare-owned-metal-bucket-probes.py`; its arguments are documented by
`--help`.
