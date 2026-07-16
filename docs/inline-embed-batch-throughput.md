# Inline `embed.batch` throughput attribution

This note records the warm M5 attribution and the serving sweep for the
inline batch path. Profiling is opt-in: start the module or exerciser with
`SYNAPSE_EMBED_PROFILE=1`. The instrumentation covers scheduler dispatch,
`spawn_blocking`, the engine mutex, bucket selection, executable selection,
MPSGraph execution, and readback.

## Attribution before the batching change

The production finding supplied the before sweep: batch 8 took 1,629 ms
(two 2,048-token quanta), with 203–295 ms/item and approximately 1.5k tok/s
through the 8–256 sweep. A warm source-level run on the M5 was also required
to separate dispatch overhead from graph work. Its representative owned-GTE
measurements were:

| Stage | Warm measurement |
| --- | ---: |
| `spawn_blocking` entry wait | 0.02–0.19 ms |
| engine mutex acquisition | ~0.001 ms |
| bucket selection | 0.002–0.006 ms |
| cached plan selection | 0.011–0.027 ms |
| cached executable selection | 0.000–0.001 ms |
| MPSGraph execute (short probe shapes) | 21.5–34.3 ms |
| readback | 0.08–0.30 ms |

That run found no 700 ms fixed cost in the warm module/engine layers: the
reported fixed-cost suspects were all sub-millisecond after warmup. The 700 ms
number itself did not reproduce under the instrumented local daemon, so this
report does not assign it to a stage that the measurements cannot support. The
reproduced portion of the before result was dispatch multiplication: the
2,048-token loop split a request into engine calls instead of allowing one
engine-optimal batch. The 300-token profile after the change confirms the
remaining per-call cost: an 8x320 call measured 56–64 ms in MPSGraph execute,
about 0.25 ms readback, and 129–138 ms total engine time including host-side
input preparation. No package or executable miss occurred (`cached=1`).

## After sweep

The local exerciser is the `synapse-module` test
`owned_gte_inline_embed_batch_throughput_sweep`; the standalone binary is
`inline_embed_throughput`. It warms once outside timing, sends one
`embed.batch` per row, uses approximately 300-token texts, and measures
`embed.query` p50 separately. Each invocation salts every text and request key
with a timestamp-plus-process nonce because the request store replays identical
idempotent requests. Job-shaped responses follow every `embed.result` page and
assert the complete id set instead of timing only page zero.

Measured run on the local M5, warm `gte-modernbert-base-f16`:

| Batch | Tokens | Total ms | ms/item | tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 2,192 | 144.0 | 18.00 | 15,221 |
| 16 | 4,384 | 289.8 | 18.11 | 15,129 |
| 32 | 8,768 | 583.1 | 18.22 | 15,036 |
| 64 | 17,536 | 1,153.3 | 18.02 | 15,205 |
| 128 | 35,072 | 2,462.7 | 19.24 | 14,241 |
| 256 | 70,144 | 4,916.6 | 19.21 | 14,267 |

The sweep has no local-minimum cliff; every batch at 64 or larger exceeds
10k tok/s. The same run measured `embed.query` p50 at 131.2 ms, with samples
130.0–132.9 ms. `docs/wire-contract-v1.md` was not changed.

## Live post-deploy verification

A release-build invocation was attempted against the shared daemon with the
rotated deployment fingerprint and `--concurrent`. It was stopped after the
shared daemon's 256-item job remained queued for the exerciser's 300-second job
timeout; therefore no live latency or throughput number is recorded here. The
failed attempt is not a performance measurement. The required paired quiet-box
run must be performed after the shared TimeMachine/load activity clears.

The exerciser's `--concurrent` mode starts a salted 256-item job and runs 50
`embed.query` samples while polling that job, reporting p50 and p95 alongside
idle p50/p95. `admission.status` exposes the execution semaphore waiter count,
in-flight execution count, and rolling acquire-wait p50/p95 so the concurrent
latency result can be correlated with queue depth.

## Fairness and preemption math

The module default bulk quantum is now 3,072 tokens. At the rig-proven
19,700 tok/s, the maximum planned engine work is:

```text
3,072 tokens / 19,700 tokens/s = 0.156 s = 156 ms
```

The planner also caps a call at eight rows and uses the smaller of the
configured quantum and 3,072 tokens. The measured 300-token batch used 2,192
real tokens per 8-row call and took 144 ms, consistent with the bound. The
scheduler still dispatches and completes one bounded batch at a time, and the
async loop yields after every completed engine call. `synapse-core` scheduler
semantics and fairness tests were not changed; job-tier execution uses the
same planner.

No new configuration field was added. The existing `jobs.bulk_quantum_tokens`
field remains under `deny_unknown_fields`; its measured default is 3,072.
