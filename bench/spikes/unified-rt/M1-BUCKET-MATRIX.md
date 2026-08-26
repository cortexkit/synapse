# Locked-M1 bucket-policy matrix

## Verdict

The locked M1 confirms the fixed-eight-row steady-throughput regression seen on the contended local host. Against cache-matched exact mode, bucketed v1 is slower by 17.0% on MiniLM and 4.6% on gte-ModernBERT for MISS, and by 16.0% and 4.4% respectively for HIT. Qwen3 also regresses by 2.7% in both cache states. The result is consistent across first, warm, and steady passes; it is not a compilation artifact.

Policy v1 remains bounded and valid: every bucketed cell has exactly ten packages, package inventories do not change during inference, all four corpora stay below 15% padding waste, and all parity gates pass. The defined policy-v2 ladder `16/16/16/16/16/16/12/12/8/8` was subsequently tested and rejected. It improves MiniLM over v1 but remains 9.9–10.6% behind exact, barely moves gte or MC, and fails the padding gate at 17.71% MiniLM and 18.45% gte. The retained [Policy v2](#policy-v2--rejected) section contains the full matrix.

## Host, binary, and protocol

| Item | Value |
|---|---|
| Host | `<bench-host>`, Apple M1 Max, 64 GiB |
| macOS | 26.5.2, build `25F84` |
| Xcode | 26.6, build `17F113` |
| Source revision | `fb8a7276986f122b945274891843a3fe9dc8c92c` |
| Binary SHA-256 | `3b92806c6af4d9243378a9d7f25bad7aef8bf4c86fae3487816a2d4574f9e2b3` |
| Device and dtypes | Metal; MiniLM f16, gte-ModernBERT fp32, Qwen3 f16 |
| Shape settings | policy v1 or `--shapes exact`, 4,000,000 attention units, maximum length 512, three passes |
| Power tool | macmon 0.7.2, requested 100 ms interval, system-wide metrics |

The binary was built locally with the Xcode and Homebrew paths required by the M1 protocol, then copied to `$SYNAPSE_BENCH_ROOT/bench-tools/unified-rt-serving/bin/spike-unified-rt`. The requested `$SYNAPSE_BENCH_ROOT/synapse` checkout does not exist, so no M1-side source build was attempted. Every timed process acquired `$SYNAPSE_BENCH_ROOT/bench.lock`, verified that `pgrep -f Runner.Worker` was empty, and released the lock on exit. The script waits five minutes before retrying either a busy lock or an active worker; no runner process was stopped.

Each MISS deleted and recreated its mode-specific package root. The following HIT was a fresh process reusing that root. Exact compilation remains visible in pass 1 rather than `cold_load_s`. The three standard corpora have both MISS and HIT rows. MC exact mode has one fresh-root MISS, as planned, because repeating its 158-shape compile path would add little policy evidence.

The MiniLM cells were rerun as a complete four-cell submatrix after the first power capture revealed macmon's roughly two-second startup delay. The final run waits for an idle macmon sample before launching the benchmark; all earlier MiniLM measurements were discarded. The other processes are long enough that their final steady windows were fully sampled despite the initial delay.

## Inputs and gates

| Corpus | Limit | Real tokens | Reference/corpus evidence |
|---|---:|---:|---|
| MiniLM standard | 400 | 66,783 | canonical corpus SHA-256 `b7c8424f…a10`; frozen reference `7589eea5…24d` |
| gte-ModernBERT standard | 400 | 62,838 | corpus `b4ff00f6…a8`; reference `d1fb6aaf…4000` |
| Qwen3 standard | 400 | 46,716 | corpus `5a9bfdc8…630c`; reference `cacee1f…cf46` |
| gte-ModernBERT MC | 11,293 | 4,172,183 | corpus `03ff11d0…3c2`; exact reference `3d6c6f3…20fe` |

Every pass checks mean cosine `>=0.9999` and top-10 overlap `>=0.995`. All 400 rows are rank queries for standard corpora. MC uses the runner's deterministic sample of at most 100 queries against the full vector set. All reported passes passed. MiniLM's lowest overlap was 0.999000 in bucketed mode, gte-ModernBERT was 1.000000, and Qwen3 was 0.998500. Mean cosine was at least 0.99999880 in every family.

## Standard-corpus matrix

Each pass cell is `infer wall / real-token throughput`. Load is `cold_load_s`; package counts and bytes are measured after the process.

| Family | shapes | cache | load | packages | bytes | first | warm | steady |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MiniLM f16 | bucketed v1 | MISS | 0.579 s | 10 | 558,918 | 0.578 s / 115,495 tok/s | 0.571 s / 117,006 tok/s | 0.570 s / 117,139 tok/s |
| MiniLM f16 | bucketed v1 | HIT | 0.414 s | 10 | 558,918 | 0.573 s / 116,648 tok/s | 0.573 s / 116,620 tok/s | 0.566 s / 117,896 tok/s |
| MiniLM f16 | exact | MISS | 0.124 s | 6 | 335,456 | 0.628 s / 106,344 tok/s | 0.476 s / 140,188 tok/s | 0.473 s / 141,113 tok/s |
| MiniLM f16 | exact | HIT | 0.107 s | 6 | 335,456 | 0.549 s / 121,750 tok/s | 0.476 s / 140,275 tok/s | 0.476 s / 140,428 tok/s |
| gte-ModernBERT fp32 | bucketed v1 | MISS | 2.718 s | 10 | 1,296,600 | 2.847 s / 22,075 tok/s | 2.843 s / 22,100 tok/s | 2.844 s / 22,094 tok/s |
| gte-ModernBERT fp32 | bucketed v1 | HIT | 1.602 s | 10 | 1,296,600 | 2.839 s / 22,136 tok/s | 2.836 s / 22,161 tok/s | 2.839 s / 22,132 tok/s |
| gte-ModernBERT fp32 | exact | MISS | 0.419 s | 5 | 648,406 | 3.100 s / 20,271 tok/s | 2.704 s / 23,238 tok/s | 2.712 s / 23,168 tok/s |
| gte-ModernBERT fp32 | exact | HIT | 0.371 s | 5 | 648,406 | 2.891 s / 21,733 tok/s | 2.707 s / 23,216 tok/s | 2.714 s / 23,157 tok/s |
| Qwen3 f16 | bucketed v1 | MISS | 5.833 s | 10 | 2,677,344 | 6.855 s / 6,815 tok/s | 6.856 s / 6,814 tok/s | 6.855 s / 6,815 tok/s |
| Qwen3 f16 | bucketed v1 | HIT | 4.386 s | 10 | 2,677,344 | 6.851 s / 6,819 tok/s | 6.846 s / 6,824 tok/s | 6.850 s / 6,819 tok/s |
| Qwen3 f16 | exact | MISS | 1.137 s | 4 | 1,071,006 | 7.245 s / 6,448 tok/s | 6.664 s / 7,010 tok/s | 6.673 s / 7,001 tok/s |
| Qwen3 f16 | exact | HIT | 1.025 s | 4 | 1,071,006 | 6.921 s / 6,750 tok/s | 6.672 s / 7,002 tok/s | 6.665 s / 7,009 tok/s |

Exact pass 1 exposes first-use package work. This is clearest in MiniLM and ModernBERT MISS, where first throughput is below warm and steady throughput. Bucketed pass 1 has no compilation and tracks its later passes closely.

## Magic-context matrix

| shapes | cache | load | packages | bytes | first | warm | steady |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bucketed v1 | MISS | 5.380 s | 10 | 1,296,600 | 183.332 s / 22,758 tok/s | 183.323 s / 22,759 tok/s | 183.146 s / 22,781 tok/s |
| bucketed v1 | HIT | 4.873 s | 10 | 1,296,600 | 183.319 s / 22,759 tok/s | 183.319 s / 22,759 tok/s | 183.208 s / 22,773 tok/s |
| exact | MISS | 0.420 s | 158 | 20,491,457 | 188.573 s / 22,125 tok/s | 174.336 s / 23,932 tok/s | 174.360 s / 23,929 tok/s |

Bucketing reduces MC package count and bytes by 93.7%, from 158 packages and 20.49 MB to 10 packages and 1.30 MB. On this locked host it is 4.8% slower than exact at steady state, unlike the positive contended-local observation.

## Padding and package immutability

| Corpus | bucketed real tokens | bucketed padded tokens | waste | gate |
|---|---:|---:|---:|---|
| MiniLM standard | 66,783 | 77,312 | 13.62% | PASS |
| gte-ModernBERT standard | 62,838 | 72,704 | 13.57% | PASS |
| Qwen3 standard | 46,716 | 53,504 | 12.69% | PASS |
| gte-ModernBERT MC | 4,172,183 | 4,499,200 | 7.27% | PASS |

For bucketed MISS, the runner captured a recursive `path/mtime/size` inventory after all ten package roots stabilized and before inference completed. For HIT, it captured the inventory before process launch. It compared each inventory byte-for-byte after process exit. All eight MISS/HIT comparisons retained ten package roots and were unchanged. `results/m1-bucket-matrix/package-invariants.json` retains each package-root mtime plus hashes of the full before/after inventories.

## Steady-pass GPU power

macmon 0.7.2 reports system-wide GPU power and frequency-scaled effective utilization, not per-process shader occupancy. The table therefore uses effective utilization as the occupancy-like signal. Each window has the measured pass-3 duration and ends at the final sample with at least 1 W GPU power and 5% effective utilization. The requested 100 ms interval produced 3–4 samples for sub-second MiniLM windows and 17–1,150 samples for the longer cells, so the MiniLM power means are lower-confidence than the throughput timings.

| Family/corpus | shapes | cache | samples | mean GPU use | mean GPU power | max GPU power |
|---|---|---|---:|---:|---:|---:|
| MiniLM | bucketed v1 | MISS | 4 | 79.4% | 21.08 W | 25.21 W |
| MiniLM | bucketed v1 | HIT | 4 | 76.8% | 20.62 W | 23.84 W |
| MiniLM | exact | MISS | 3 | 64.5% | 24.15 W | 30.24 W |
| MiniLM | exact | HIT | 3 | 63.6% | 23.57 W | 29.20 W |
| gte-ModernBERT | bucketed v1 | MISS | 18 | 90.8% | 37.81 W | 42.54 W |
| gte-ModernBERT | bucketed v1 | HIT | 19 | 90.5% | 37.60 W | 42.16 W |
| gte-ModernBERT | exact | MISS | 17 | 92.7% | 42.46 W | 48.20 W |
| gte-ModernBERT | exact | HIT | 18 | 87.2% | 39.89 W | 46.86 W |
| Qwen3 | bucketed v1 | MISS | 43 | 96.2% | 39.72 W | 43.75 W |
| Qwen3 | bucketed v1 | HIT | 43 | 97.2% | 40.17 W | 43.06 W |
| Qwen3 | exact | MISS | 43 | 95.7% | 43.17 W | 48.14 W |
| Qwen3 | exact | HIT | 43 | 96.3% | 43.45 W | 47.56 W |
| gte-ModernBERT MC | bucketed v1 | MISS | 1,147 | 93.2% | 41.74 W | 46.22 W |
| gte-ModernBERT MC | bucketed v1 | HIT | 1,150 | 93.1% | 41.72 W | 45.94 W |
| gte-ModernBERT MC | exact | MISS | 1,095 | 92.9% | 41.61 W | 49.59 W |

The eight-row bucket policy does not leave the GPU broadly idle: effective use is high for ModernBERT, Qwen3, and MC. The MiniLM throughput loss instead accompanies lower power than exact mode despite a higher frequency-scaled use ratio, consistent with insufficient useful work per dispatch rather than package compilation during timed inference.

## Reproduction and retained artifacts

The committed `run-m1-bucket-matrix.sh` contains the lock/retry guard, model snapshots, mode and cache ordering, macmon capture, fresh-root handling, and package-inventory check. `summarize-m1-bucket-matrix.py` derives the committed power and inventory summaries from the raw M1 files. Raw process logs, macmon JSONL, and before/after inventories remain at `$SYNAPSE_BENCH_ROOT/bench-tools/unified-rt-serving/results/m1-bucket-matrix/`. The 15 harness result JSON files and two evidence summaries are committed under `results/m1-bucket-matrix/`.

## Policy v2 — rejected

### Verdict

Do **not** adopt policy v2 as specified. It raises MiniLM steady throughput by 7.7% on MISS and 7.3% on HIT relative to v1, but it still trails exact mode by 10.6% and 9.9%. It does not materially move gte-ModernBERT or MC: their MISS gains over v1 are only 0.24%, leaving gaps to exact mode of 4.4% and 4.6%. Qwen3 improves by 4.1% over v1 and finishes 1.3% above exact on MISS, but that isolated win cannot rescue the candidate.

The decisive failure is padding. The larger short-sequence rows raise MiniLM waste from 13.62% to 17.71% and gte-ModernBERT from 13.57% to 18.45%. Both violate the strict `<15%` gate. Qwen3 (13.92%) and MC (7.28%) pass. The ladder was tested unchanged; no corpus-specific tuning was applied.

Policy v2 preserves the bounded-package property: every cell has exactly ten packages, all eight recursive inventories remain unchanged during inference, and MC still uses 93.7% fewer packages and bytes than exact mode (10 and 1.30 MB versus 158 and 20.49 MB). These wins are not sufficient for adoption when two canonical corpora fail the serving gate. Policy v1 therefore remains the selectable default, and exact mode remains available for A/B measurements.

### Revision and protocol caveat

The v2 binary was built from base revision `6aea8ccd334123157c6aad91650faf9f0cf74b1a` plus the policy-v2 changes in this commit; its SHA-256 is `da4ffedce076173b4c4c51e2ee8a4c3e789d8f56968189207b7529b8d8543958`. The committed v1 comparison rows used revision `fb8a7276986f122b945274891843a3fe9dc8c92c` and binary `3b92806c…f9e2b3`, so the percentages below are a cross-revision A/B rather than a same-binary toggle. The v2 performance change is isolated to bucket selection, versioned cache identity, and auditable gate reporting, but the revision difference remains a measurement caveat. The final source subsequently corrected the version number in rerank result metadata; that branch is not reached by embedding measurements.

Current-source ModernBERT distinguishes the reranker by `classifier_pooling`, while the retained embedding snapshot also carries that metadata despite having no classification tensors. The M1 staging view removed only that metadata field so the same `e7f32e3c…` embedding weights and tokenizer continued through the established CLS-plus-L2 path. No model tensor, tokenizer, corpus, or reference changed.

All cells otherwise followed the earlier locked protocol: Metal with MiniLM f16, gte fp32, and Qwen3 f16; a 4,000,000-unit budget; maximum length 512; three in-process passes; a fresh policy-v2 package root for MISS followed by a fresh HIT process; lock acquisition with five-minute retry; an empty `Runner.Worker` check; and a trap that kills macmon before releasing the lock. macmon produced an idle sample and then waited another two seconds before each timed process. The runner retained result JSON and returned a gate failure after all three passes for MiniLM and gte, allowing the failed policy to remain auditable without weakening the exit-status gate.

### Locked matrix

Each pass cell is `infer wall / real-token throughput`. All eight processes ran three passes and every pass passed its family parity gates.

| Family/corpus | cache | load | packages | bytes | first | warm | steady |
|---|---|---:|---:|---:|---:|---:|---:|
| MiniLM f16 | MISS | 0.624 s | 10 | 558,982 | 0.529 s / 126,188 tok/s | 0.531 s / 125,655 tok/s | 0.529 s / 126,205 tok/s |
| MiniLM f16 | HIT | 0.453 s | 10 | 558,982 | 0.534 s / 125,047 tok/s | 0.531 s / 125,747 tok/s | 0.528 s / 126,550 tok/s |
| gte-ModernBERT fp32 | MISS | 2.481 s | 10 | 1,296,744 | 2.841 s / 22,119 tok/s | 2.836 s / 22,160 tok/s | 2.837 s / 22,146 tok/s |
| gte-ModernBERT fp32 | HIT | 1.934 s | 10 | 1,296,744 | 2.835 s / 22,163 tok/s | 2.836 s / 22,154 tok/s | 2.835 s / 22,168 tok/s |
| Qwen3 f16 | MISS | 6.597 s | 10 | 2,677,408 | 6.591 s / 7,087 tok/s | 6.582 s / 7,098 tok/s | 6.586 s / 7,093 tok/s |
| Qwen3 f16 | HIT | 5.513 s | 10 | 2,677,408 | 6.589 s / 7,090 tok/s | 6.593 s / 7,085 tok/s | 6.593 s / 7,086 tok/s |
| gte-ModernBERT MC fp32 | MISS | 5.743 s | 10 | 1,296,744 | 182.731 s / 22,832 tok/s | 182.759 s / 22,829 tok/s | 182.699 s / 22,836 tok/s |
| gte-ModernBERT MC fp32 | HIT | 5.132 s | 10 | 1,296,744 | 182.740 s / 22,831 tok/s | 182.726 s / 22,833 tok/s | 182.636 s / 22,844 tok/s |

### Steady comparison

The exact and v1 columns are the committed rows above; v2 is the new observation. MC exact has only the planned MISS cell.

| Family/corpus | cache | exact steady | v1 steady | v2 steady | v2 vs exact | v2 vs v1 |
|---|---|---:|---:|---:|---:|---:|
| MiniLM | MISS | 141,113 | 117,139 | 126,205 | -10.6% | +7.7% |
| MiniLM | HIT | 140,428 | 117,896 | 126,550 | -9.9% | +7.3% |
| gte-ModernBERT | MISS | 23,168 | 22,094 | 22,146 | -4.4% | +0.24% |
| gte-ModernBERT | HIT | 23,157 | 22,132 | 22,168 | -4.3% | +0.16% |
| Qwen3 | MISS | 7,001 | 6,815 | 7,093 | +1.3% | +4.1% |
| Qwen3 | HIT | 7,009 | 6,819 | 7,086 | +1.1% | +3.9% |
| gte-ModernBERT MC | MISS | 23,929 | 22,781 | 22,836 | -4.6% | +0.24% |

### Padding gate and parity

| Corpus | real tokens | padded tokens | waste | gate | lowest cosine | lowest top-10 |
|---|---:|---:|---:|---|---:|---:|
| MiniLM standard | 66,783 | 81,152 | 17.71% | **FAIL** | 0.99999932 | 0.999250 |
| gte-ModernBERT standard | 62,838 | 77,056 | 18.45% | **FAIL** | 1.00000000 | 1.000000 |
| Qwen3 standard | 46,716 | 54,272 | 13.92% | PASS | 0.99999880 | 0.998500 |
| gte-ModernBERT MC | 4,172,183 | 4,499,712 | 7.28% | PASS | 1.00000000 | 1.000000 |

The larger row buckets add 3,840 padded token slots for MiniLM and 4,352 for gte relative to v1. Qwen3 adds 768 slots and remains within the gate. MC adds only 512 because its rows remain concentrated in the conservative long-sequence buckets.

### Steady-pass GPU power

| Family/corpus | cache | samples | mean GPU use | mean GPU power | max GPU power |
|---|---|---:|---:|---:|---:|
| MiniLM | MISS | 4 | 62.3% | 20.30 W | 27.72 W |
| MiniLM | HIT | 4 | 62.7% | 20.48 W | 27.45 W |
| gte-ModernBERT | MISS | 18 | 90.7% | 40.34 W | 44.31 W |
| gte-ModernBERT | HIT | 18 | 91.4% | 41.03 W | 45.54 W |
| Qwen3 | MISS | 42 | 95.7% | 40.85 W | 45.45 W |
| Qwen3 | HIT | 41 | 97.3% | 41.94 W | 45.01 W |
| gte-ModernBERT MC | MISS | 1,143 | 92.2% | 41.63 W | 48.32 W |
| gte-ModernBERT MC | HIT | 1,289 | 92.5% | 41.72 W | 48.16 W |

The extra rows raise gte power by roughly 2.5–3.4 W versus v1 without a meaningful throughput gain. MiniLM gains throughput while its short power windows remain near 20 W, but the gain is not enough to close exact mode and comes with a gate failure.

### Retained evidence

`run-m1-policy-v2.sh` is the executed campaign script. The eight result files, macmon-derived power summary, and byte-for-byte package inventory summary are committed under `results/m1-policy-v2/`. Raw logs, macmon JSONL, gate-status files, and before/after inventories remain at `$SYNAPSE_BENCH_ROOT/bench-tools/unified-rt-serving/results/m1-policy-v2/` on the M1.
