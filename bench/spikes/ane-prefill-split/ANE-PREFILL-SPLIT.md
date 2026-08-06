# ANE prefill + Metal decode split: locked-M1 result

## Verdict

**SURVIVES, for W128 only.** The stateless Core ML graph converts, exposes all
56 K/V tensors plus logits, places 2,114/2,116 dispatchable operations on ANE
(99.905%), and hands a token-exact cache to the production Metal step engine.
The locked-M1 W128 split reaches first-token selection in **39.437 ms** including
Core ML, explicit host copy/layout conversion, logits copy/argmax, and device
upload, versus **687.980 ms** for 16-token-batched Metal prefill (**17.45x**).
All 20 battery prompts then match pure GPU for all 64 greedy tokens.

The pre-registered transfer kill does not fire: explicit K/V copy/layout plus
upload is **6.022 ms**, 0.875% of GPU prefill wall. Request energy improves
8.981 J -> 6.140 J (1.46x); prefill-stage energy alone improves 20.91x. The
request-level energy ratio is below 5x because the same 64-token Metal decode
dominates both arms, but the decision is an OR and the split's prefill wall is
far below the GPU arm.

W32x4 is not a viable substitute. It is faster than GPU prefill but each
stateless chunk resets position and lacks prior-chunk attention. Only 9/20
prompts are exact through 64 tokens; the other 11 diverge at depths 1--12.

## Design-fork survey

Survey authority: `[bench-host]`, MacBookPro18,2 (M1 Max), macOS
26.5.2, torch 2.5.1, coremltools 8.3.0, f16, `CPU_AND_NE`.

| Export | Outputs | Logical K/V bytes/request | Package | Conversion | Core ML logits cosine vs eager | Placement |
|---|---:|---:|---:|---:|---:|---:|
| W128 | 57 | 14,680,064 | 1,505,006,243 B | 39.574 s | 0.999974 | 2,114 ANE / 2 CPU (99.905% ANE) |
| W32 | 57 | 3,670,016/call; 14,680,064 for x4 | 1,504,923,712 B | 41.184 s | 0.999821 | 2,114 ANE / 2 CPU (99.905% ANE) |

The 57 outputs are last-position logits followed by post-RoPE
`key_00,value_00,...,key_27,value_27`, each as a normal tensor, not `MLState`.
This structurally clears the fork: torch.export conversion, runtime execution,
and ANE placement all succeed. The logical K/V payload is packed into the step
engine's existing 512-token bucket, a 58,720,256-byte f16 device cache.

## Locked protocol

- `[bench-user-home]/bench.lock` acquired with `mkdir`; no `Runner.Worker`; AC power;
  one-minute load under 3.0 (recorded `{ 1.70 1.28 1.20 }`).
- Source model hash:
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- The 20 existing `decode-prompts.jsonl` prompts were repeated as text then
  tokenized/truncated to exactly 128 active tokens. There is no padding.
- GPU prefill uses `verify_tokens_batch` in eight 16-token chunks, the same
  batched machinery gated by `SYNAPSE_METAL_STEP_BATCHED_VERIFY` in production.
- Core ML stage results are 20-call p50/p95 after three warmups. GPU prefill and
  upload results are 20-call p50/p95. The continuation is exactly 64 greedy
  tokens: one selected at TTFT, then 63 Metal step calls.
- Core ML prediction, explicit K/V copy/layout packing, logits copy, host
  top-2/argmax, Metal cache upload, and Metal decode are separately timed.
- Binary artifact writes and reads are outside the handoff clocks. They stand in
  for an in-process host buffer; the exact packed bytes are imported by the
  Metal engine. Cold model load is also excluded from warm request latency.
- Power uses macmon at 100 ms on sustained stage-only cells. Energy is mean total
  package power times the corresponding measured p50 wall.
- Metal builds used
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.

The actions-runner service was paused only for measurement cells. The initial
pause was 16:04:58Z--16:11:21Z; the corrected authoritative pause was
16:27:38Z--16:33:13Z. After the corrected run, the listener was Started and the
GitHub runner API reported `[bench-host-alias]-metal` online. Between pauses, GitHub's
Actions broker returned HTTP 503 while the healthy listener retried; no CI job
was killed.

## Latency

All values are milliseconds. Component p50 values are shown independently;
`Core ML total` is the directly measured p50 of each request's summed Core ML
and host work, not the sum of component medians.

| Arm | Core ML prediction | K/V copy + layout | Logits copy | Host argmax | Core ML total | Metal upload | TTFT | Remaining 63 Metal steps | Total request |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pure GPU | -- | -- | -- | in GPU path | -- | -- | **687.980** | 1,289.865 | **1,977.845** |
| ANE W128 -> Metal | 32.965 | 0.613 | 0.184 | 0.282 | 34.028 | 5.409 | **39.437** | 1,294.346 | **1,333.783** |
| ANE W32x4 -> Metal | 107.071 | 1.976 | 0.221 | 0.282 | 109.517 | 5.439 | **114.955** | 1,297.396 | **1,412.352** |

| Arm | Core ML total p95 | Metal upload p95 | GPU prefill p95 | TTFT speedup | Total-request speedup |
|---|---:|---:|---:|---:|---:|
| W128 | 39.877 | 6.085 | 688.382 | **17.45x** | **1.483x** |
| W32x4 | 109.988 | 6.140 | 688.126 | **5.98x** | **1.402x** |

`MLModel.prediction` is an opaque runtime boundary and may include
framework-owned output publication. `K/V copy + layout` starts only after it
returns, walks the MLMultiArray strides, writes the padded Metal layout, and is
therefore the additional explicit transfer cost under our control.

## Power and energy

| Sustained stage | Calls/samples | ANE W | GPU W | CPU W | Total W | Stage p50 |
|---|---:|---:|---:|---:|---:|---:|
| Metal W128 prefill | 10 / 58 | 0.000 | **3.690** | 0.665 | 4.355 | 685.128 ms |
| Core ML W128 + host copy | 300 / 76 | **3.127** | 0.000 | 0.529 | 3.656 | 33.728 ms |
| Core ML W32x4 + host copy | 100 / 83 | **1.902** | 0.000 | 0.444 | 2.347 | 109.162 ms |
| Metal cache upload | 1,000 / 47 | 0.000 | 0.310 | 3.242 | 3.553 | 5.453 ms |
| Metal continuation (63 steps after TTFT) | 5 / 55 | 0.000 | **3.928** | 0.724 | 4.652 | 1,289.210 ms |

| Arm | Prefill compute/copy J | Upload J | Decode J | Request J | Energy win |
|---|---:|---:|---:|---:|---:|
| Pure GPU | 2.984 | -- | 5.997 | **8.981** | 1.00x |
| ANE W128 -> Metal | 0.123 | 0.019 | 5.997 | **6.140** | **1.463x** |
| ANE W32x4 -> Metal | 0.256 | 0.019 | 5.997 | **6.273** | **1.432x** |

For W128, prefill compute/copy plus upload is 0.143 J versus 2.984 J on
Metal, a **20.91x prefill-stage energy win**. Decode reduces the full-request
ratio because it is intentionally identical and GPU-resident in both arms.

## Correctness

### W128 authority

| Prompts | Exact for all 64 | Exact rate | Match-depth mean/median/range |
|---:|---:|---:|---:|
| 20 | **20** | **100%** | 64.0 / 64.0 / 64--64 |

Core ML fp16 parity is not bit-identical to eager or Metal, but no W128 token
fork surfaced. The post-RoPE K/V layout and logical rewind/import position are
therefore certified for this battery.

### W32x4 structural control

| Prompts | Exact for all 64 | Exact rate | Match-depth mean/median/range |
|---:|---:|---:|---:|
| 20 | 9 | 45% | 30.7 / 9.5 / 1--64 |

The 11 divergent prompts are listed at their first differing generated-token
index (zero-based). `GPU gap` and `ANE gap` are top-1 minus top-2 logits at that
fork.

| Prompt | Depth | GPU token | W32x4 token | GPU gap | ANE gap |
|---|---:|---:|---:|---:|---:|
| completion-02 | 1 | 11 | 18 | 9.191038 | 0.084301 |
| completion-03 | 1 | 27934 | 13 | 9.921104 | 0.665420 |
| completion-07 | 1 | 314 | 17 | 5.216038 | 2.457165 |
| completion-08 | 1 | 42578 | 11871 | 6.496294 | 0.226611 |
| completion-10 | 1 | 304 | 518 | 8.090673 | 1.256819 |
| completion-11 | 12 | 15846 | 220 | 2.165686 | 2.569712 |
| completion-14 | 4 | 18608 | 220 | 3.465769 | 0.153526 |
| completion-15 | 7 | 1416 | 419 | 2.530172 | 0.017251 |
| completion-16 | 2 | 25 | 13 | 1.841234 | 0.122505 |
| completion-17 | 6 | 374 | 304 | 6.456208 | 0.066849 |
| completion-18 | 2 | 15003 | 358 | 5.650030 | 0.669054 |

Several W32x4 ANE gaps are small, but the corresponding GPU gaps are not. This
is the expected structural loss of cross-window attention, not a defensible
near-tie band.

## Superseded stride-bug run

The 16:04:58Z run is explicitly superseded and is not used in any table above.
Its Swift runner treated the logits MLMultiArray vocabulary axis as contiguous
and read `dataPointer[index]`. Core ML returned a strided axis, so values were
associated with wrong token IDs. That corrupted logits, argmax, top-2 gaps, and
all apparent token forks; it did **not** corrupt K/V because the K/V packer
already indexed every MLMultiArray stride. The visible symptom was
`completion-01` selecting ANE token 35584 instead of GPU token 9625 and a false
0/20 W128 exact result.

The correction reads `dataPointer[index * strides[vocabularyAxis]]`. Before the
second locked run, an 8-token smoke check changed `completion-01` to identical
streams `[9625, 374, 576, 6722, 315, 9625, 374, 576]`; the authoritative battery
then passed W128 20/20 x 64. `results/superseded-stride-bug.json` preserves the
invalidated observation rather than silently deleting it.

## Decision table

| Pre-registered question | W128 result | Decision |
|---|---:|---|
| Does explicit-output torch.export conversion run? | 57 outputs; runtime succeeds | Pass |
| Is placement acceptable? | 99.905% ANE, 0 GPU dispatches | Pass |
| Does K/V copy + conversion + upload exceed GPU prefill? | 6.022 ms vs 687.980 ms | **No kill (114.25x below)** |
| Is split prefill wall below GPU prefill? | 39.437 ms vs 687.980 ms | **Yes (17.45x)** |
| Otherwise, is request energy >5x with wall <=1.5x? | 1.463x; not needed | Alternate criterion not met |
| Is the handed-off cache token-exact? | 20/20 prompts x 64 tokens | Pass |
| Final | -- | **SURVIVES: advance W128 split** |

The next step is an in-process integration that keeps the Core ML and Metal
models resident together and replaces the binary artifact with the same host
buffer. This spike certifies conversion, placement, byte layout, upload,
request latency, energy, and token behavior; it does not claim that excluded
model-load or file-serialization time belongs in steady-state TTFT.
