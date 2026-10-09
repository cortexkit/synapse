# Module-side cost of a 64-row `embed.batch`

## Result

For an AFT-shaped call (64 rows, about 120 tokens per row, 1,024-dim vectors),
the module, transport and client codec together cost about **8 ms** at the client
on this machine, when tokenization uses the fixture tokenizer. With the real
Qwen3-Embedding tokenizer, that projects to **about 12 ms** (range 10-15 ms).
That is under 1% of a 1.7 s call, so **nothing was cut**. No request field, wire
change or scheduling change was made. The only code changes are measurement
support: new stage records and test-engine options.

The 240-370 ms "module overhead" this work set out to find does not exist. See
[The earlier 240-370 ms figure](#the-earlier-240-370-ms-figure).

## Method

- **Engine.** The `test-deterministic` engine (feature `test-support`), so engine
  time is close to zero (0.19 ms for all eight calls together). Two test-only
  options were added for this run. `SYNAPSE_TEST_DETERMINISTIC_DIMS=1024` sets
  the vector width; the default stays at 384. `SYNAPSE_TEST_DETERMINISTIC_DENSE=1`
  gives every component a non-zero value. A sparse token-bag vector prints most
  components as `0.0`. That made the reply 0.35 MB instead of 1.44 MB, and it
  understated JSON cost (client total about 7 ms). Dense mode changes each empty
  component by at most 1e-6 before normalization. The probe's reference vectors
  still match, and every value prints at full precision like a real embedding.
  `SYNAPSE_TEST_DETERMINISTIC_DELAY_MS=0` removes the engine's simulated 5 ms
  sleep per call.
- **Workload.** 64 rows of 120 whitespace words, which gives 7,680 tokens with
  the fixture word-level tokenizer. The call stays inline (64 items, within the
  8,192-token inline limit). The scheduler splits it into eight 8-row engine
  calls, as in production.
- **Real tokenizer.** The deterministic lane must pass its probe before it serves.
  The probe's reference vectors assume the fixture tokenizer, so the Qwen3
  tokenizer cannot replace it inside the module. Instead, the harness times the
  module's own `SanitizedTokenizer::tokenize_batch` in-process. It uses the real
  Qwen3-Embedding-0.6B `tokenizer.json` on 64 code-like rows (short Rust
  functions, 7,330 tokens in total, about 115 per row). The harness finds the
  tokenizer in the local Hugging Face cache, or at
  `SYNAPSE_MODULE_COST_TOKENIZER` when that is set.
- **Client.** The e2e harness `embed_batch_module_cost_profile` in
  `crates/synapse-module/tests/embed_batch_module_cost/mod.rs` starts the
  in-process test subc daemon and the release `ck-synapse` binary. It times
  request encoding, send-to-reply-frame, `serde_json` parsing of the reply, and
  extraction of the 64 × 1,024 `f32`s.
- **Module stages.** These come from the `SYNAPSE_ANE_PROFILE_DIR` stage log
  (`module-<pid>.jsonl`), extended with the stages listed under
  [Instrumentation added](#instrumentation-added). The harness reads the records
  each call appended.
- **Runs.** Each set has 1 lane-load call and 2 warm-up calls, all discarded,
  then 15 measured calls. There were 3 sets, back to back, so 45 measured calls.
  Raw per-call data is in `run-1/`, `run-2/` and `run-3/` (`module-cost.json`).
- **Host.** Apple M5 Max (18 cores, 128 GiB), macOS 27.0.1, rustc and cargo
  1.99.0, release profile. The host was shared and busy: one-minute load average
  was 13-14 throughout. No Neural Engine or GPU work ran.

Reproduce from the checkout root:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
SYNAPSE_RUN_MODULE_COST_PROFILE=1 SYNAPSE_MODULE_COST_RUNS=15 \
  SYNAPSE_MODULE_COST_OUT="$PWD/docs/evidence/embed-batch-module-cost/run-N" \
  cargo test --release --locked -p synapse-module --features test-support \
  --test skeleton_e2e embed_batch_module_cost_profile -- --nocapture
```

Without `SYNAPSE_RUN_MODULE_COST_PROFILE=1`, the test returns at once. Its
numbers only mean something in a release build.

## Stage table

All values are in milliseconds over the 45 measured calls. Per-set medians are
listed so drift between sets is visible. Stages are listed in the order they
run. Each stage is one record per call, except the four engine-call stages,
which sum 8 records per call.

| Stage | What it covers | Median | Min | Max | Set medians |
| --- | --- | ---: | ---: | ---: | --- |
| client request encode | `serde_json::to_vec` of the request | 0.013 | 0.012 | 0.026 | 0.013 / 0.013 / 0.015 |
| request decode | module parses the route body (`request_decode`) | 0.016 | 0.013 | 0.036 | 0.016 / 0.015 / 0.019 |
| resolve | params, alias table, model resolution, certification checks (`batch_resolve`) | 0.086 | 0.070 | 0.169 | 0.087 / 0.078 / 0.112 |
| tokenize, fixture tokenizer | `batch_tokenize` | 1.078 | 0.998 | 1.572 | 1.117 / 1.037 / 1.267 |
| tokenize, real Qwen3 tokenizer | in-process, same call, 64 code rows | 4.53 | 3.75 | 5.55 | 5.02 / 4.21 / 4.53 |
| prepare | composition, tokenizer policy, request digest (`batch_prepare`) | 0.067 | 0.064 | 0.180 | 0.066 / 0.065 / 0.085 |
| admission | `batch_admission` | 0.019 | 0.015 | 0.048 | 0.018 / 0.016 / 0.028 |
| engine fan-out, total | eight bounded calls with scheduling (`bulk_execution_total`) | 1.541 | 1.171 | 4.035 | 1.480 / 1.412 / 1.758 |
| · engine calls (sum of 8) | lane lock, permit, blocking-pool handoff, engine (`bulk_engine_call`) | 1.018 | 0.793 | 3.062 | 1.006 / 0.916 / 1.177 |
| · engine compute (sum of 8) | the test engine itself (`test_engine_compute`) | 0.192 | 0.162 | 0.257 | 0.191 / 0.182 / 0.218 |
| · scheduler, lane, permit waits (sum of 8 each) | `bulk_scheduler_dispatch`, `execution_lane_wait`, `execution_permit_wait` | ≤0.005 each | | | |
| reply build | row hashes, envelope, JSON value tree (`reply_build`) | 0.254 | 0.238 | 0.359 | 0.252 / 0.246 / 0.275 |
| reply encode | JSON value tree to bytes (`reply_encode`) | 1.955 | 1.866 | 2.561 | 1.937 / 1.910 / 2.030 |
| module handler total | first byte parsed to reply bytes ready (`handle_total`) | 5.377 | 4.765 | 8.246 | 5.250 / 5.030 / 6.192 |
| transport outside handler | client round trip minus handler total, for a 1.44 MB reply | 1.066 | 0.831 | 29.538 | 1.010 / 1.116 / 1.088 |
| client reply parse | `serde_json::from_slice` into a `Value` | 1.685 | 1.603 | 2.109 | 1.683 / 1.646 / 1.822 |
| client vector extract | `Value` arrays to `Vec<f32>` | 0.039 | 0.033 | 0.063 | 0.037 / 0.037 / 0.040 |
| **client total** | encode + round trip + parse + extract | **8.182** | 7.463 | 37.456 | 8.153 / 7.831 / 9.472 |

Derived figures, all from medians:

- **Engine fan-out overhead.** About 1.35 ms for the eight calls (1.541 total
  minus 0.192 compute), or about 0.17 ms per engine call.
- **Untimed time in the handler.** About 0.36 ms: the handler total minus the
  sum of its timed stages. It includes the stage log's own file writes. Each
  record opens and appends to the log file, and there are 49 records per
  call.
- **JSON vector codec, end to end.** About 4.7 ms: reply encode 1.96, the
  1.44 MB on the wire about 1.07, and client parse 1.69. It is the largest group
  that does not depend on the engine.
- **Projected total with the real tokenizer.** About 11.6 ms: the client total
  plus the 3.45 ms by which Qwen3 tokenization exceeds fixture tokenization.

The single max outliers (35 ms round trip, 29.5 ms transport, in set 3) are one
call each, which matches a host under load. They are not a pattern.

## Why nothing was cut

The two largest module-side costs are close to a tie, and both are small:

- JSON vector codec, about 4.7 ms end to end.
- Qwen3 tokenization, about 4.5 ms.

An opt-in binary vector encoding would save about 4 ms, and only after a caller
opts in. It would also add to the wire contract. Parallel tokenization would
save about 3-4 ms, but it would use more cores during admission. Either change
saves about 0.25% of a 1.7 s call, so neither was made.

## The earlier 240-370 ms figure

These figures are the task owner's, from the M4 run in
`docs/evidence/ane-lane-profile/m4-2026-10-09/`. That directory is not part of
this branch, and this run did not reproduce the figures.

The "240-370 ms of module overhead" was a misread summary column. The
`dispatch=` value that `docs/evidence/ane-lane-profile/summarize.py` prints is
the sum of the worker's per-layer `submit_wait_ms`. It leaves out the rest of
the worker's own time (per-row gather, packing, readback, tail and gaps between
rows). Subtracting that column from wall time therefore put worker time into
"module and IPC".

Corrected split for AFT's real 64-row call on the M4, as supplied by the task
owner:

| Part | M4 measurement |
| --- | --- |
| Module tokenization | 2-7 ms |
| Eight direct-ANE round trips (module `direct_ane_roundtrip`, summed) | 1,573-1,772 ms |
| Of which the worker's own `sequences` time | all but 3-50 ms |
| Wall time minus the round trips | 17-97 ms |

The 17-97 ms outside the round trips includes the client codec, transport and
admission that this run measures at about 8-12 ms. The spread fits a loaded host
and a 1.44 MB JSON reply decoded by AFT's client. Nothing in it points to a
module stage worth cutting.

## Instrumentation added

New records in the existing `SYNAPSE_ANE_PROFILE_DIR` stage log. Each is one
timestamp pair per stage, or per bounded engine call, with nothing per row.
With the variable unset, each costs one environment lookup and writes nothing.

- `request_decode` and `handle_total`. These are written for `embed.batch` only.
  The model is not resolved yet at that point, so their `model_id` field holds
  the method name.
- `batch_resolve` and `batch_prepare`, around tokenization (`batch_tokenize`
  already existed).
- `bulk_engine_call`, once per engine call, and `bulk_execution_total`, once per
  request.
- `reply_build` and `reply_encode`, on the inline embed reply.
- `test_engine_compute`. This is written only by the test-only deterministic
  engine.

## Not measured

- **Neural Engine and worker time.** Worker-side per-row cost and the module's
  direct-ANE exchange were not measured. The exchange covers token framing, the
  `f32` frame decode and residency leases. None of this can run here, because
  production holds the Neural Engine lock. The M4 split above is the task
  owner's.
- **Catalog-lane request path.** The deterministic lane is a preloaded,
  non-catalog lane. The catalog path's resolver lane lock, profile check status
  read, and per-call catalog lane lock were not exercised. On the M4 they fall
  inside the 17-97 ms above.
- **Real tokenizer inside the module.** The Qwen3 tokenizer was timed in-process
  with the module's own tokenizer wrapper, not inside the running module. The
  code rows are synthetic, not AFT's real chunks.
- **Production daemon hop.** The test daemon runs inside the test process. A
  production client also crosses a separate subc daemon process, which was not
  measured.
- **AFT's own client decode.** The client parse figure is `serde_json` into a
  `Value` in the harness, not AFT's decoder.
- **Concurrency.** Only serial calls were measured: no concurrent calls and no
  contention on the execution permit.
- **Quiet-host numbers.** The host was shared and under load (load average
  13-14), so absolute values carry that noise.
