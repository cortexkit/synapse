# Matched GTE ModernBERT long-row comparison: ANE versus Metal

Measured on 2026-09-12 on the same `Mac17,6` M5 Max and macOS 27.0 build `26A5425a` as the full-context ANE evidence. The input model snapshot is `e7f32e3c00f91d699e8c43b53106206bcc72bb22`. ANE used the accepted Hadamard-rotated float16 packages (seed 0); Metal used the production owned-Metal float16 embed path at bucket policy v2. `EVIDENCE.json` contains all samples, package and input hashes, vector diagnostics, memory bounds, and the host condition captured around every arm.

## Decision

**Do not ship an 8192-token ANE lane.** At 8192 real tokens, production Metal completed a row in 696.84 ms median versus 1,561.83 ms on ANE. The two interleaved paired arms put Metal at 2.208x and 2.278x ANE speed (2.243x paired median). The separation is large and repeatable despite ambient host load. Shipping ANE would rotate the consumer fingerprint and force a full re-embed for a lane that is materially slower at the context length that motivates it.

At 4096 tokens the lanes were effectively tied: the paired Metal speed ratios were 0.961x and 1.054x. At 1024 Metal's aggregate median was modestly faster, but the paired ratios ranged from 0.984x to 1.275x under ambient load, so this run does not establish a clean 1024 advantage. The short control strongly favors Metal because it executes a 1x128 bucket while the smallest available rotated ANE package pads the same 128 real tokens to 1x1024.

ANE's remaining case would be energy efficiency, not latency. Power was **not measured**: another project's CPU benchmark was active throughout the available slot, so host-level power could not be attributed credibly to either lane. ANE's efficiency argument therefore remains unmeasured rather than estimated. That missing positive does not justify paying the re-embed cost against the clear 8192 latency result.

## Matched results under ambient load

Every cell is caller wall time for one pre-tokenized row. Each lane ran twice per length in `ANE, Metal, ANE, Metal` order. Each arm ran for at least 15 seconds and at least 11 repetitions; the aggregate sample count is shown in parentheses. Throughput counts real tokens, not padded tokens. P95 exposes the contention tail.

| Real tokens | ANE executed shape | ANE median / p95 | ANE real tok/s | Metal executed shape | Metal median / p95 | Metal real tok/s | Paired Metal speed ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1x1024 | 39.01 / 40.98 ms (807) | 3,281 | 1x128 | 6.35 / 7.77 ms (4,569) | 20,151 | 6.138x |
| 1024 | 1x1024 | 40.23 / 44.49 ms (764) | 25,455 | 1x1024 | 34.50 / 43.16 ms (826) | 29,683 | 1.129x |
| 4096 | 1x4096 | 325.76 / 330.10 ms (94) | 12,574 | 1x4096 | 321.97 / 393.09 ms (92) | 12,722 | 1.007x |
| 8192 | 1x8192 | 1,561.83 / 1,566.44 ms (22) | 5,245 | 1x8192 | 696.84 / 836.49 ms (43) | 11,756 | 2.243x |

The 8192 Metal p95 remains 836.49 ms, far below ANE's best observed 1,555.53 ms. The 4096 Metal tail is much wider than ANE's, which is why the conclusion there is a tie rather than a Metal win.

## Measurement boundary

The shared input for both lanes is the exact same active token-ID row. Tokenization and text transport are outside both timers.

- **ANE:** wall time immediately around `CompiledMLModel.predict`, beginning with a prebuilt padded NumPy input dictionary and ending when the normalized embedding returns. Stable-path model load, first use, a warmup, worker checks, and IPC are outside the timer.
- **Metal:** wall time immediately around the production `OwnedMetalEmbedEngine.embed_batch`, beginning with a prebuilt `TokenBatch` and ending after bucket selection, padding and mask construction, MPSGraph execution, readback, and normalized CLS pooling return. Model load, first use, a warmup, and IPC are outside the timer.

This is the closest honest common boundary available. The rotated ANE artifact is a spike package rather than a production worker lane, so there is no common worker-IPC boundary to time. ANE includes Python/Core ML bridge overhead while Metal includes Rust bucket planning. Neither number includes process startup or model load.

Batch shape is part of the result, not hidden normalization. The 1024, 4096, and 8192 rows map to equal 1-row shapes in both lanes. The short control exposes the policy difference: ANE pads to its 1024 fixed package while Metal policy v2 selects 1x128.

## Ambient-load protocol and safety

The established 15% idle gate rejected the first attempt before any model timing. This host runs concurrent projects and could not provide an idle interval, so the accepted battery used paired interleaving to distribute ambient drift across both lanes. One-minute load average ranged from 5.51 to 8.93 and sampled CPU idle ranged from 44.1% to 86.1%; values immediately before and after every arm are committed in `EVIDENCE.json`. Absolute timings are therefore ambient-load observations. The paired lane ratio is the decision result.

A benchmark lock excluded other registered measurements, and no `Runner.Worker` was active. The supervised `ck-synapse-worker-ane` was cleared by the operator before the slot. The driver checked immediately before and after every lane arm and continuously during every timed ANE block; no worker respawned. A respawn would reject the arm rather than enter the result.

The spike's memory safeguards remained active for every model child: at least 32 GiB available before the battery, abort below 16 GiB available, abort above 50% wired memory, abort above 32 GiB owned RSS, and a 30-minute child timeout. No guard fired. These observations bound owned RSS and system pressure but cannot guarantee that an ANE allocation is bounded or prevent workstation pressure.

## Vector agreement (diagnostic only)

These values are context, not a gate. The rotated ANE and production Metal outputs have distinct deployed fingerprints even though both remain close to the canonical model space.

| Real tokens | ANE-vs-Metal cosine | Max absolute difference |
|---:|---:|---:|
| 128 | 0.99997855 | 0.0007945 |
| 1024 | 0.99996017 | 0.0010482 |
| 4096 | 0.99997490 | 0.0008991 |
| 8192 | 0.99990757 | 0.0018205 |

Both lanes were byte-deterministic across their two fresh-process arms at every length. The decreasing 8192 agreement is diagnostic only and does not alter either lane's independent parity evidence.

## Reproduction

The guarded controller is `bench/spikes/ane-modernbert-full-context/compare_long_rows.py`; the Metal caller is `crates/synapse-engine-owned/examples/long_row_probe.rs`. The controller validates the model, rotation, package, compiled-path, and input identities before measuring, derives all four cases from the accepted full-context input rows, enforces sequential execution and memory guards, and parses the engine profile to prove Metal's actual bucket shape. Source packages and compiled models remain outside git.
