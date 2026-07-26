# PHASE-C GATE: in-graph autoregressive unroll

**Host:** local M5 Max only. No M1, fleet, unified-rt, or serving changes were
used. The model snapshot and toolchain match SPIKE-A: Qwen3-0.6B,
torch 2.5.1, coremltools 8.3.0, Transformers 4.51.3, NumPy 2.3.2.

## Experiment

`convert_qwen3_to_coreml.py --window 32 --unroll-k 4` exports four explicit
greedy passes inside one `torch.export` graph. Each pass argmaxes the final
position, shifts the fixed W32 IDs and mask left, appends the token with a
fixed-shape embedding gather on the next pass, and returns one `[1, 4]`
`token_ids` output. The generated package is
`artifacts/models/qwen3-w32-unroll-k4.mlpackage` (1,509,442,342 bytes); the
conversion completed in 78.6 s.

The phase-A harness was run with `--calls 200 --warmup 20
--power-seconds 30`, once per compute-unit setting. The `CPU_ONLY` row is the
control, not a placement candidate. Effective draft rate is `4 * 1000 / p50`.

## Decision table

The first two rows are the SPIKE-A W32/K4 reference. The final two rows are the
single Phase-C variable: autoregressive K4 unroll rather than four independent
last-K positions.

| source | W | K | mode | compute unit | p50 ms | p95 ms | effective Kx tok/s | ANE share | ANE watts |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| SPIKE-A | 32 | 4 | last-K logits | CPU_AND_NE | 8.596 | 8.797 | 465.4 | 97.37% | 3.581 |
| SPIKE-A | 32 | 4 | last-K logits | CPU_ONLY | 14.073 | 14.748 | 284.2 | 0.00% | 0.000 |
| **Phase C** | **32** | **4** | **autoregressive token-ID unroll** | **CPU_AND_NE** | **31.638** | **31.828** | **126.4** | **98.96%** | **3.813** |
| **Phase C** | **32** | **4** | **autoregressive token-ID unroll** | **CPU_ONLY** | **47.762** | **50.264** | **83.7** | **0.00%** | **0.000** |

The Core ML compute plan has 8,387 dispatchable operations: 8,300 preferred
for ANE and 87 for CPU, or **98.963% ANE share among dispatchable operations**.
The four `reduce_argmax` operations are CPU-preferred, one per unrolled pass;
this is allowed by the gate because the transformer body remains on ANE and
there is still one Core ML prediction for all four tokens.

## Parity

The harness compared the unrolled package with four sequential calls to the
resident W32/K1 package on the same 20 W-token inputs, using the same tokenizer
pad ID and CPU_AND_NE configuration. All generated IDs matched:

| windows | tokens/window | token agreements | result |
|---:|---:|---:|---|
| 20 | 4 | **80 / 80 (100%)** | pass |

The converter's eager/export smoke check also matched 8/8 token IDs. This
confirms that the unroll changes scheduling, not greedy math.

## Verdict

The pre-registered **p50 > 20 ms** threshold fires (31.638 ms), while ANE
placement remains healthy (>90%). **Phase C therefore declares stateless
in-graph unroll dead too and pivots to the ANE-prefill + GPU-decode split.**
The batched verifier branch is not pursued as a rescue for this stateless draft
experiment.

Raw measurements: `results/phase-c-raw.json`.
