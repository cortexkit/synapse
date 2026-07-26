# SPIKE-A: Qwen3-0.6B ANE speculative-decode feasibility

**Host:** local Apple M5 Max, macOS 26.5.2 (`25F84`), arm64<br>
**Scope:** stateless fixed-window re-encoding only; no M1, fleet, serving, or
runtime integration<br>
**Checkpoint:** `Qwen/Qwen3-0.6B`, local snapshot; `model.safetensors`
SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`<br>
**Toolchain:** Python 3.12.12, torch 2.5.1, coremltools 8.3.0,
transformers 4.51.3, NumPy 2.3.2

## Result in one paragraph

Stateless Qwen3-0.6B re-encoding is **placement- and parity-feasible on the
M5 ANE, but only last-K batching clears the useful draft-rate rung**. A W32,
K4 `CPU_AND_NE` call takes 8.596 ms p50 and yields 465.4 effective draft
tok/s; W64, K4 takes 9.407 ms and yields 425.2 effective tok/s. K8 raises the
scheduled effective rate to 866.1 and 795.4 tok/s respectively for only a
small per-call latency increase. K1 is 112.7-121.0 tok/s at W32/W64, so the
Core ML dispatch tax makes single-token ANE drafting merely a slower-target or
batching-assisted option. The measured branch is therefore **ANE stateless
last-K draft + GPU batched verifier**, with W32/K4 as the conservative default
and W64/K4 as the longer-context option. Do not build the stateful ANE decode
path from these results.

## Decision table

Each timing cell is 200 warm calls with one batch-1 request per call. `draft tok/s` is `1000 / p50` for one output position. `effective Kx tok/s` is the
requested speculative-draft scheduling figure `K * 1000 / p50`; it does not
claim that independent last-K logits alone constitute an autoregressive
acceptance loop. ANE watts are the mean `ane_power` from an approximately
30-second sustained call loop sampled by `macmon`. `CPU_ONLY` has no ANE
placement by configuration, so its ANE share and ANE watts are zero by design.

| W | K | compute unit | p50 ms | p95 ms | draft tok/s | effective Kx tok/s | ANE share | ANE watts |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | 1 | CPU_AND_NE | 8.262 | 8.389 | 121.0 | 121.0 | 97.37% | 3.656 |
| 32 | 1 | CPU_ONLY | 12.194 | 13.172 | 82.0 | 82.0 | 0.00% | 0.000 |
| 32 | 4 | CPU_AND_NE | 8.596 | 8.797 | 116.3 | 465.4 | 97.37% | 3.581 |
| 32 | 4 | CPU_ONLY | 14.073 | 14.748 | 71.1 | 284.2 | 0.00% | 0.000 |
| 32 | 8 | CPU_AND_NE | 9.237 | 9.470 | 108.3 | 866.1 | 97.37% | 3.421 |
| 32 | 8 | CPU_ONLY | 13.984 | 17.957 | 71.5 | 572.1 | 0.00% | 0.000 |
| 64 | 1 | CPU_AND_NE | 8.873 | 8.978 | 112.7 | 112.7 | 99.86% | 5.564 |
| 64 | 1 | CPU_ONLY | 21.784 | 26.263 | 45.9 | 45.9 | 0.00% | 0.000 |
| 64 | 4 | CPU_AND_NE | 9.407 | 9.682 | 106.3 | 425.2 | 99.86% | 5.432 |
| 64 | 4 | CPU_ONLY | 24.962 | 30.210 | 40.1 | 160.2 | 0.00% | 0.000 |
| 64 | 8 | CPU_AND_NE | 10.058 | 10.264 | 99.4 | 795.4 | 99.86% | 4.895 |
| 64 | 8 | CPU_ONLY | 24.667 | 39.364 | 40.5 | 324.3 | 0.00% | 0.000 |
| 128 | 1 | CPU_AND_NE | 12.141 | 12.227 | 82.4 | 82.4 | 99.86% | 7.705 |
| 128 | 1 | CPU_ONLY | 42.705 | 50.060 | 23.4 | 23.4 | 0.00% | 0.000 |
| 128 | 4 | CPU_AND_NE | 12.489 | 12.645 | 80.1 | 320.3 | 99.86% | 7.540 |
| 128 | 4 | CPU_ONLY | 45.796 | 63.276 | 21.8 | 87.3 | 0.00% | 0.000 |
| 128 | 8 | CPU_AND_NE | 13.100 | 13.272 | 76.3 | 610.7 | 99.85% | 7.253 |
| 128 | 8 | CPU_ONLY | 203.577 | 209.586 | 4.9 | 39.3 | 0.00% | 0.000 |

The full compact matrix and placement/power aggregates are in
`results/phase-a-raw.json`. Detailed per-operation placement and raw macmon
samples remain under the ignored `results/phase-a-work/` directory from the
measurement run.

## Placement

`MLComputePlan` reported 5,171 total operations, 2,089 dispatchable operations,
and 3,082 non-dispatchable `const` nodes for every window/K package under
`CPU_AND_NE`:

| window | ANE dispatchable | CPU dispatchable | ANE share |
|---:|---:|---:|---:|
| 32 | 2,034 | 55 | **97.37%** |
| 64 | 2,086 | 3 | **99.86%** |
| 128 | 2,086 | 3 | **99.86%** |

The W32 CPU falloff is a small fixed-shape gather/reshape boundary; it does
not displace the transformer body from the ANE. W64 and W128 are effectively
all-ANE among dispatchable operations. The `CPU_ONLY` plan is intentionally
CPU-only and is a dispatch/latency control, not a placement candidate.

## Conversion and numerical sanity

The converter uses `torch.export` only, fixed batch 1, left padding, Core ML
fp16, and `CPU_AND_NE`. The body uses the Wave-2 Conv2d projection layout and
includes the causal mask, Qwen3 Q/K RMSNorm, RoPE, GQA attention, SwiGLU,
terminal RMSNorm, and tied `lm_head`. It slices the final hidden positions
before `lm_head`, so K4/K8 do not multiply the full window's logits.

All nine packages converted in 26-59 seconds and were about 1,504,9xx,xxx bytes
(the shared 0.6B fp16 weights dominate package size). The two-row conversion
smoke reports stayed between 0.9998448 and 0.9999331 mean cosine from the
float32 eager wrapper; `torch.export` was exact in the smoke rows.

The required greedy parity check ran **20 prompts x 8 steps** for each of W32,
W64, and W128, using the Core ML fp16 K1 package against Transformers CPU fp32:

| window | agreements | steps | argmax agreement |
|---:|---:|---:|---:|
| 32 | 157 | 160 | 98.125% |
| 64 | 157 | 160 | 98.125% |
| 128 | 156 | 160 | 97.500% |
| **all windows** | **470** | **480** | **97.917%** |

This clears the 95% drafter sanity threshold. It is not a correctness gate:
the GPU verifier remains responsible for exact output, and an occasional
mismatch reduces speculation acceptance efficiency only. The result is far
above the <80% warning level, so there is no evidence here of the severe
Core ML fp16 `lm_head` drift seen in the LFM2 spike.

## Threshold verdict and Phase B/C branch

The decision thresholds are: **draft rate >= 300 tok/s effective at W<=64** is
comfortably viable against a roughly 30-50 tok/s 4B-class GPU target; **100-300**
is viable only with last-K batching or for slower targets; and **<100** means
ANE drafting is dead on Core ML dispatch overhead for this model class, leaving
only a smaller parity-clean draft or ANE-prefill + GPU-decode split. This run
puts W32/K4, W32/K8, W64/K4, and W64/K8 in the first category; W32/K1 and W64/K1
are in the second; and W128/K1 is in the third. W128/K4/K8 exceed 300 only by
multiplying positions outside the W<=64 rung and cost 7.25-7.54 W on the ANE,
so they are not the preferred path. **Phase B/C should proceed with the GPU
batched verifier on the Metal step path and a Leviathan-style acceptance loop,
feeding it a W32/K4 (or W64/K4) stateless ANE draft.** The numbers do not justify
stateful ANE decode or a K1-only drafter.
