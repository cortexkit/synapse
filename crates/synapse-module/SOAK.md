# Synapse module soak harness

The ignored integration test in `tests/soak.rs` drives the Lane 1 hardening cases without adding a new runtime surface. It is intentionally capped for a developer laptop: 4 sustained consumers, a short 64-request burst, and one 5,000-item job-tier batch.

## Run

Build the llama worker, then run the ignored soak explicitly:

```sh
cargo build -p synapse-worker-llama --bin synapse-worker-llama
SYNAPSE_LLAMA_WORKER_BIN=target/debug/synapse-worker-llama \
  cargo nextest run -p synapse-module --test soak --run-ignored ignored-only --no-capture
```

If model snapshots are not in the default Hugging Face cache locations, set:

- `SYNAPSE_MINILM_ONNX_SNAPSHOT` — directory containing `model.onnx` and `tokenizer.json`.
- `SYNAPSE_MINILM_GGUF_SNAPSHOT` — directory containing `all-MiniLM-L6-v2-ggml-model-f16.gguf`.
- `SYNAPSE_LLAMA_WORKER_BIN` — built `synapse-worker-llama` binary.

`cargo test -p synapse-module --test soak -- --ignored --nocapture` is equivalent when nextest is unavailable.

## What it drives

- Mixed load: 4 consumer connections race the same `request_key` for one 5,000-item `embed.batch` job, keep 64 total `embed.query` calls moving on the ORT lane, and run `probe.start` mid-soak.
- Burst: 64 simultaneous `embed.query` calls with `max_queue_ms = 0`; every response must be a vector or typed `queue_full` / `deadline_exceeded`, with no hangs and no response past 2x the request deadline.
- Crash budget: a wrapper launches the llama worker with hidden `--test-abort`, which allows LOAD and aborts on inference/probe requests. Repeated llama probe attempts produce typed `engine_crashed`, then `probe_required` with a quarantined message after the crash budget is exhausted. ORT `embed.query` traffic is checked before and after each llama crash.
- Drain invariants: `module_generation` remains consistent and `admission.status.inline_in_flight_bytes` returns to zero after the job and burst drain.

## Numbers asserted by the harness

- 5,000 job-tier items returned exactly once across result pages.
- 4 concurrent submissions of one `request_key` produce exactly 1 `job_id`.
- 64 sustained query calls complete with vectors or typed rejections.
- 64 burst query calls complete with vectors or typed fast-fail rejections.
- 2 worker crashes exhaust the default llama crash budget; the next llama probe reports quarantine.
- ORT lane queries continue to return 384-dimensional vectors throughout the llama crash sequence.

Observed on this worktree with local MiniLM assets: mixed soak 49s, crash soak 43s under nextest.

## What broke and was hardened

The mixed soak exposed a job-drain hang: the job scheduler used disclosed real token counts, while dispatched token chunks use padded token-id buffers. Large batches could exhaust the scheduler's token budget before every padded chunk had been dispatched, leaving the job permanently `running`. Job quanta now budget from the actual token-id chunks they dispatch.

The crash path exposed that `WorkerHost` cleared all loaded model records when a worker died. The module still held the old `LoadedModel`, so a restarted worker could not reload the model and later requests could turn into stale `model_ref` failures instead of budgeted crash recovery. The host now returns stable host-side model IDs, keeps the artifact/runtime config after a crash, clears only worker-local refs, and lazily reloads the model into the restarted worker before the next request. Quarantine is surfaced as permanent `probe_required` with a quarantined detail.

Worker health is cached from supervisor state: recent crashes mark a llama lane degraded until the crash window expires, and quarantined models keep the lane degraded until a future reprobe path clears them.
