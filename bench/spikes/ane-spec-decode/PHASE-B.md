# PHASE-B: composition law proven; stateless-draft economics refuted

**Host:** local M5 Max (functional numbers; the M1 is the pinned authority for decode lanes and was campaign-locked).
**What ran:** the merged phase-B loop (proposal-fed chained Metal verification, logical KV rewind, greedy acceptance, persistent ANE JSONL drafter) driven by `measure_phase_b.py` over the 20 pinned fixtures + depth-470, W32/K4 mlmodelc drafter.

## Correctness (the deliverable)

- Speculative output == target-only baseline for EVERY prompt, asserted in-binary per prompt (the loop aborts on the first mismatch). The composition law holds with the real ANE drafter at 67% acceptance — rejections and rollbacks exercised for real.
- The one reference-gate failure is the DOCUMENTED M5 Metal-compiler drift on completion-06 step 7 (present in plain single-step decode on this box since the campaign-5 integration; the M1 authority passes it). `measure_phase_b.py` recognizes exactly this case and no other.
- Depth-470: speculative path token-exact at depth, 8.7 tok/s (deep-context attention wall dominates both paths equally).

## Economics (honest negative for this pairing AND the 4B projection)

| metric | value |
|---|---:|
| baseline single-step (0.6B f16, M5) | 122.7 tok/s |
| speculative w/ ANE stateless draft | 19.9 tok/s |
| acceptance rate | 67.1% |
| verify chain (4 tokens) | 40.5 ms |
| draft compute per 4-token proposal | 109.1 ms (~27 ms/call in-loop vs 8.6 ms isolated in SPIKE-A — contention/feature-build overhead, undiagnosed) |
| draft transport (IPC) | 0.12 ms (negligible) |
| **4B break-even acceptance rate** | **1.91 (impossible)** |

The stateless re-encode drafter costs ~as much per token as a 4B-class target step. No acceptance rate rescues that: **sequential stateless ANE drafting cannot pay, even against the 4B product target.**

## What phase C must change (both, or pivot)

1. **In-graph autoregressive K-token unroll** on the drafter: k tokens per single ANE dispatch (~8.6 ms for k=4 => ~2.2 ms/token effective) instead of k dispatches.
2. **True batched verification** on the target: one prefill-style k-token forward (weights read once) instead of k chained steps.

With both, the 4B math flips positive at moderate acceptance rates; with neither, the program pivots to the ANE-prefill + GPU-decode split. The next cheap gate is converting ONE unrolled-K4 package and measuring its dispatch latency — a single-variable experiment on the phase-A harness.

Raw: `results/phase-b.json`.
