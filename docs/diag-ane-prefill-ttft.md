VERDICT (b) — **the certification aggregation is median-of-20 and is not the source of the discrepancy.** A fresh, quiet, exact-protocol W128 f16-step run produced GPU p50 **1074.688 ms**, ANE-split p50 **181.136 ms**, and **5.933x**, so the current production paths clear the unchanged 5.0x headline gate. The supplied seven-sample blocks independently produce **6.021x**. The earlier **3.680x** failure cannot be numerically replayed because `certify.py` raises before returning or writing its raw samples; its exact formula was nevertheless `median(20 GPU worker_ttft_ms) / median(20 split worker_ttft_ms)`, not worst-of-N or p95.

The measured ratio is honest for the current worker, but it is inflated by a removable GPU-path defect: W128 GPU TTFT spends **1074.284 ms p50 (99.96% of driver TTFT)** in prefill because production calls the sequential `verify_tokens` primitive once with all 128 tokens. It does **not** use the existing 16-token batched primitive or yield between chunks. The same engine's actual eight-chunk batched prefill is **312.119 ms p50** on this M5 Max, removing about **762 ms**. This report does not implement that performance fix or alter any gate.

The split arm has a different, split-specific production tax. Its p50 is **180.963 ms inside worker prefill**: **50.504 ms** through sidecar `EXECUTE` (including **12.996 ms** Core ML prediction and **36.441 ms** K/V layout), then **123.081 ms** transferring the 14.68 MB active K/V payload over the worker/Swift Unix socket, **0.467 ms** expanding it, **6.159 ms** uploading it, and **0.133 ms** selecting/publishing logits. Relative to the locked-M1 spike's 39.437 ms, almost all of the +141.7 ms is the production sidecar transport plus a slower copy-heavy K/V materialization. It is not the GPU arm's sequential-prefill tax.

No gate constant, evidence record, production config, or routing behavior changed.

# ANE-prefill W128 TTFT attribution

## Result summary

| Measurement | Protocol | GPU p50 (ms) | Split p50 (ms) | GPU / split | Load context |
|---|---|---:|---:|---:|---|
| Supplied cross-check | 3 warmups; 7 GPU samples then 7 split samples | 1083.800 | 180.000 | 6.021x | approximately 5.3 |
| Fresh production worker | 3 warmups/engine; 20 sample-major pairs, split then GPU; `max_tokens=1` | 1074.688 | 181.136 | **5.933x** | 3.92 / 5.51 / 5.76 before; 3.43 / 5.26 / 5.66 after |
| Raw batched Metal engine | 3 discarded warmups; 20 calls; eight 16-token chunks | 312.119 | — | — | 4.78 / 5.96 / 5.86 before; 4.72 / 5.93 / 5.85 after |
| Locked-M1 spike calibration | 20-call stage p50s after warmup; not production-worker interleaving | 687.980 | 39.437 | 17.45x | 1.70 / 1.28 / 1.20 |

Machine: Apple M5 Max, macOS 26.6.1 (25G76), arm64, AC power. Measurements were taken on 2026-08-14. The fresh accepted tables began and ended with one-minute load below 6. The p95 values below use nearest-rank p95 (`ceil(0.95*N) - 1` after sorting); certification does not gate on p95.

## 1. What `certify.py` actually computes

The executable protocol is unambiguous:

1. `TTFT_SAMPLE_COUNT = 20` (`tests/ane-prefill-certification/certify.py:28`).
2. `warmup()` issues three requests per engine (`certify.py:688-691`). `ttft()` warms GPU first, then split (`certify.py:693-695`).
3. Sampling is sample-major. For each index 0 through 19, it measures `ane-split` first and `gpu` second (`certify.py:696-710`). The engines are persistent clients, keyed by `(bucket, decode_config, engine, chain_k)` in `machine_driver.py:524-543`; there is no process or model setup per `GENERATE_START`.
4. It computes `statistics.median` independently over each engine's 20 `worker_ttft_ms` values and divides GPU by split (`certify.py:711-713`). It does not aggregate per-pair ratios, p95, max, or worst-of-N.
5. The W128 check is `ratio >= 5.0` (`certify.py:716-717`), matching the contract's “pure-GPU p50 / split p50 >= 5.0” wording (`contracts/ane-prefill-split/ane-prefill-split-contract-v2.json:165-166`).
6. The machine driver uses a deterministic 128-token prompt and `max_tokens=1` (`machine_driver.py:564-570`). Its current `worker_ttft_ms` and `wire_ttft_ms` are the same Python monotonic wall around one worker request, not two independent clocks.

### Why 3.680 cannot be reconstructed from the failed run

The failed run's exact arrays were not persisted. `ttft()` constructs `samples` locally, computes the medians, then calls `require()` before returning the object containing `samples` (`certify.py:711-719`). When the 5.0x check fails, the output record is never written. Therefore the only exact statement recoverable from source and the surfaced message is:

```text
3.680 = median(gpu worker_ttft_ms[0:20])
        / median(ane-split worker_ttft_ms[0:20])
```

It was not produced by a different aggregation. If its GPU median was in the reproducible 1075-1084 ms class, 3.680 implies a split median around 292-295 ms—at least 10 of 20 split samples would have needed to be slow. That is materially different from both the supplied seven-sample block and the fresh exact-protocol population. A certification failure should preserve or print both arrays in future diagnostic work, but changing failure serialization is not needed to resolve or alter this gate.

The full certifier also runs the arm's token and routing batteries before TTFT. That history differs from the standalone seven-sample block and this focused reproduction, and is the remaining plausible source of a materially slower split population. The absent failed-run arrays prevent a stronger claim.

### Supplied seven-sample cross-check

| Engine | Raw `worker_ttft_ms` samples | Min | Median | Max |
|---|---|---:|---:|---:|
| GPU | 1073.1, 1082.1, 1089.3, 1091.8, 1083.8, 1090.9, 1078.5 | 1073.1 | 1083.8 | 1091.8 |
| ANE-split | 181.3, 180.0, 178.9, 182.0, 177.1, 177.3, 184.3 | 177.1 | 180.0 | 184.3 |

`1083.8 / 180.0 = 6.0211x`. These blocks use the same warmup count and median statistic but not certification's split/GPU sample-major interleaving.

### Fresh exact-protocol raw samples

| Sample | ANE-split (ms) | GPU (ms) |
|---:|---:|---:|
| 0 | 208.890 | 1085.836 |
| 1 | 180.203 | 1070.312 |
| 2 | 182.218 | 1073.518 |
| 3 | 178.738 | 1073.184 |
| 4 | 183.786 | 1071.970 |
| 5 | 180.231 | 1070.760 |
| 6 | 182.867 | 1078.200 |
| 7 | 180.963 | 1072.753 |
| 8 | 181.309 | 1073.671 |
| 9 | 194.799 | 1074.707 |
| 10 | 181.411 | 1074.273 |
| 11 | 178.558 | 1071.599 |
| 12 | 177.358 | 1075.850 |
| 13 | 176.891 | 1074.669 |
| 14 | 180.470 | 1075.638 |
| 15 | 188.541 | 1075.561 |
| 16 | 178.096 | 1076.644 |
| 17 | 179.695 | 1078.745 |
| 18 | 194.099 | 1080.603 |
| 19 | 183.664 | 1076.730 |

| Engine | Min (ms) | Median (ms) | p95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| ANE-split | 176.891 | **181.136** | 194.799 | 208.890 |
| GPU | 1070.312 | **1074.688** | 1080.603 | 1085.836 |

The resulting current-worker ratio is **5.9330606x**.

## 2. GPU attribution: the milliseconds are sequential prefill

Env-gated worker timings (`CK_DECODE_LOG_STAGE_TIMINGS=1`) decompose the same 20 GPU requests:

| GPU worker stage | Min (ms) | Median (ms) | p95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| Request validation/reset/chain prepare | 0.000 | 0.001 | 0.001 | 0.001 |
| Pure-GPU prefill | 1070.112 | **1074.284** | 1078.769 | 1085.648 |
| First quantum / final-frame construction | 0.001 | 0.001 | 0.002 | 0.005 |
| Worker dispatch through response construction | 1070.191 | 1074.570 | 1080.450 | 1085.709 |
| Worker response publication | 0.009 | 0.012 | 0.015 | 0.024 |
| Python driver wall | 1070.312 | **1074.688** | 1080.603 | 1085.836 |

There is no approximately one-second process/session setup, scheduler wait, continuation loop, or logits-publication stage. Prefill itself accounts for effectively all TTFT.

The source explains why:

- `DecodeEngine::prefill_greedy()` calls `decoder.prefill(prompt)` (`crates/synapse-worker-decode/src/runner.rs:1277-1295`).
- `MetalStepDecoder::prefill()` calls `verify_tokens(&mut cache, tokens)` once for the entire prompt (`crates/synapse-engine-owned/owned-decode-engine/src/qwen3_decode_metal_step.rs:510-524`).
- `verify_tokens` invokes the sequential `synapse_qwen3_metal_step_verify` path over the supplied span (`qwen3_decode_metal_step.rs:323-370`).
- The separate `verify_tokens_batch` primitive supports at most 16 positions and streams weights once per layer (`qwen3_decode_metal_step.rs:373-460`). Production already uses it for forced-token ingestion (`runner.rs:1345-1364`), but not ordinary greedy prefill.

Thus the code's comments about 16-token batched prefill and scheduler yields do not describe the worker path measured by certification. The removable fix shape is to process ordinary Qwen prefill as `prompt.chunks(MetalStepDecoder::MAX_BATCH_VERIFY_TOKENS)`, invoke `verify_tokens_batch` for each chunk, and expose a scheduler yield between chunk command buffers. That is a performance change and is deliberately not implemented here.

### Raw batched-engine comparison

The spike harness executed the same model, same 128 token IDs, same 512-position cache, and the same production `MetalStepDecoder`, using eight calls to `verify_tokens_batch`. Three initial calls were discarded, followed by these 20 measurements:

```text
311.953, 312.021, 311.790, 312.120, 311.951,
311.950, 312.109, 312.097, 312.283, 312.390,
313.487, 314.737, 313.483, 312.743, 312.314,
312.190, 312.060, 312.118, 312.290, 312.005 ms
```

| Path | Min (ms) | Median (ms) | p95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| Raw eight-chunk batched prefill | 311.790 | **312.119** | 313.487 | 314.737 |
| Production worker sequential prefill | 1070.112 | **1074.284** | 1078.769 | 1085.648 |

The measured current-code improvement is **3.44x** and about **762.2 ms** at p50. The M5 raw result is not the speculative 100-200 ms class, so this report does not claim that lower number. It does prove that the production worker is selecting the wrong existing primitive for its intended batched baseline.

### Python continuation-loop control

Three samples per cell compare the machine driver's complete wall at `max_tokens=1` and `max_tokens=64`; one-minute load fell from 5.18 to 4.85 during these cells:

| Engine | `max_tokens=1` p50 (ms) | `max_tokens=64` p50 (ms) | Added 63-token decode + 3 continuation round-trips (ms) |
|---|---:|---:|---:|
| GPU | 1076.870 | 1686.272 | 609.402 |
| ANE-split | 176.048 | 788.164 | 612.116 |

The added cost is nearly identical because both arms use the same Metal decode after prefill. More importantly, certification uses `max_tokens=1`, which returns FINAL from `GENERATE_START`; it executes no Python `GENERATE_CONTINUE` loop. Rust-side continuation driving is therefore unnecessary to exclude the driver loop as the 1074 ms cause.

## 3. Split attribution: payload IPC dominates

The split timing log retains both sidecar clocks and worker-boundary clocks. `worker EXECUTE` is wall time from sending `EXECUTE` through receiving the `EXECUTED` header; it already includes sidecar prediction, K/V layout, and logits copy. `payload IPC` begins after that header and includes the logits frame, K/V frame, and timing readback. Consequently, `contract_handoff_sum` is useful for budget accounting but must not be added to `worker EXECUTE`, because it includes sidecar K/V/logits components already inside that wall.

| Split stage | Min (ms) | Median (ms) | p95 (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| Sidecar Core ML prediction | 12.883 | **12.996** | 13.924 | 15.147 |
| Sidecar K/V layout/materialization | 35.277 | **36.441** | 48.577 | 48.710 |
| Sidecar logits copy | 0.710 | **0.738** | 0.785 | 1.367 |
| Sidecar total | 49.019 | **50.299** | 63.131 | 63.263 |
| Worker `EXECUTE` boundary | 49.228 | **50.504** | 63.495 | 63.539 |
| Raw payload IPC + timing readback | 118.863 | **123.081** | 126.428 | 127.081 |
| Worker active-K/V expansion | 0.452 | **0.467** | 0.508 | 0.539 |
| f16→q8 cache handoff | 0.000 | **0.000** | 0.000 | 0.000 |
| Metal cache upload | 5.187 | **6.159** | 6.715 | 7.329 |
| Logits validation/argmax publication | 0.125 | **0.133** | 0.143 | 0.163 |
| Complete worker split prefill | 176.736 | **180.963** | 194.549 | 208.716 |
| Worker response publication | 0.007 | **0.010** | 0.015 | 0.015 |
| Python driver wall | 176.891 | **181.136** | 194.799 | 208.890 |

The sidecar's 699.253 ms readiness measurement is the one-time model install for the persistent split client. It is reported on later timing readbacks but is not inside any warm-request worker wall above. There is no evidence of per-request ANE re-acquisition or a readiness guard wait.

### Slow-sample distribution

Three of 20 split samples were at least 190 ms:

| Sample | Driver TTFT (ms) | Worker prefill (ms) | Worker `EXECUTE` (ms) | Sidecar prediction (ms) | Sidecar K/V layout (ms) | Payload IPC (ms) | Cause |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 208.890 | 208.716 | 50.859 | 12.883 | 37.034 | 123.964 | About 28.1 ms falls between instrumented sub-stages; consistent with host scheduling/preemption, not ANE prediction or guard wait. |
| 9 | 194.799 | 194.549 | 63.539 | 15.147 | 47.219 | 124.486 | Sidecar K/V layout was about 10.8 ms above median and prediction about 2.2 ms above median. |
| 18 | 194.099 | 193.933 | 63.495 | 13.924 | 48.577 | Sidecar K/V layout was about 12.1 ms above median. |

The p95 tail is therefore sidecar host-copy/layout work, not ANE re-acquisition. The single maximum has normal ANE and IPC stages plus an unattributed scheduler gap; it does not move the median enough to explain 3.680.

### Production tax versus the M1 spike

| Component | Locked-M1 spike p50 (ms) | M5 production p50 (ms) | Delta / interpretation |
|---|---:|---:|---|
| Core ML prediction | 32.965 | 12.996 | M5 is 19.969 ms faster. |
| K/V copy/layout | 0.613 | 36.441 | Production sidecar is 35.828 ms slower. |
| Logits copy/selection | 0.466 | 0.871 | Small. |
| Core ML/host total | 34.028 | 50.299 | +16.271 ms despite faster ANE prediction. |
| Worker/sidecar payload IPC | in-process assumption | 123.081 | Dominant production-only tax. |
| Metal upload | 5.409 | 6.159 | +0.750 ms. |
| TTFT | **39.437** | **181.136** | +141.699 ms. |

The spike intentionally excluded binary serialization and modeled an in-process host buffer (`bench/spikes/ane-prefill-split/ANE-PREFILL-SPLIT.md:50-57`). Production sends explicit logits and 14,680,064 K/V bytes through a separate Swift process. The sidecar also converts each `MLMultiArray` to a Swift `Array` before appending active K/V (`workers/ane-prefill-sidecar/Sources/AnePrefillSidecarExecutable/main.swift:121-154,336-351`), introducing additional materialization.

The split fix shape is therefore a mapped/shared payload or in-process Core ML bridge, with direct stride-aware packing into the final padded cache layout. That removes the 123 ms socket transfer, the intermediate Swift arrays, and potentially the worker expansion. It is a separate future performance change, not part of this measurement commit.

## 4. Gate disposition

- **Aggregation verdict:** correct. The contract says p50, and `certify.py` uses Python medians over 20 split/GPU alternating samples. No median-of-N fix is proposed.
- **Current production verdict:** the fresh focused run is **5.933x**, above the unchanged 5.0x headline. This report does not issue or modify certification evidence.
- **Attribution verdict:** production has two removable costs, not one overhead that cancels in the ratio: sequential GPU prefill (~762 ms avoidable versus the existing batched primitive) and split-specific sidecar copy/IPC (~123 ms transport plus ~36 ms layout).
- **Contract caution:** replacing GPU prefill with the measured current batched primitive would make the raw comparison roughly `312.119 / 181.136 = 1.723x`, below 5.0x. The gate currently measures the real deployed paths, so its result is honest; after a GPU prefill fix, the owner must decide whether the headline still expresses the intended product contract. This report does not pre-decide that question.

## Instrumentation and reproduction notes

`CK_DECODE_LOG_STAGE_TIMINGS=1` now gates generate-path and protocol-publication logs. When unset, no `Instant` stage clocks are created; a cached `OnceLock` only controls the branch. `CK_ANE_PREFILL_LOG_TIMINGS=1` now emits the sidecar and worker split sub-stages rather than only readiness/prediction/handoff aggregates. The machine driver enables both only in its certification subprocess.

The ignored raw captures used for the inline tables were produced from the real W128 package, the release worker, and the release Swift sidecar. They are intentionally not an evidence record and are not committed. Rust source was not performance-modified; the batched control used the existing spike harness and existing `verify_tokens_batch` implementation.
