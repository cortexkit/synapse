# Development: pooled operations and live executable budget

This is development evidence, not a release qualification.

## Serving policy

Each full-shape admission and inference runs inside an Objective-C autorelease pool. Eviction releases the selected shape inside a pool and responds only after that pool drains. Model unload, replacement, and connection-close cleanup also release cached programs inside a pool. The isolated compile control in `reclaim-paths-development.md` identifies retained compilation temporaries: without compilation pools, pooled inference/eviction still cannot load replacement layer14; pooled compilation permits complete replacements and preserves finite, byte-identical outputs from a different resident sequence length (256) that was never evicted.

The supervisor continues using the measured **100 live-executable budget** (22 per GTE shape,28 per Qwen shape), alongside shape caps4 per model/8 overall. Pending admissions reserve the complete layer count; evicting slots retain that charge through the eviction acknowledgment. Completed eviction removes its slot, refunds that reservation, and wakes FIFO waiters. If all candidates are leased, admission waits until its deadline and returns the existing transient refusal. The budget counts currently reserved programs, not every compile performed since process start.

A first hardware resource refusal still chooses one unleased LRU victim and retries once. Persistent exhaustion after that completed eviction invokes worker recovery, returns a transient refusal, and does not recursively evict/compile. Recovery prevents new leases on that owner and waits for outstanding leases and already dispatched admissions. The connector now provides an owned-process exit-confirmation future: closing a stream alone is insufficient. Real hardware connectors wait on their own child, or terminate and wait for that same child after ten seconds. Only after confirmation is a replacement connected and its cached successful LOAD requests replayed with checked model references. Reservations are removed only after successful recovery. An unconfirmed exit or failed recovery stays closed with reservations retained.

The prior test claiming that a second exhaustion must not restart is intentionally replaced by `persistent_exhaustion_after_eviction_restarts_and_refuses_transiently`: persistent exhaustion despite a completed eviction can indicate resources still held by the worker, so confirmed process exit provides a recovery backstop. Its one-eviction/two-attempt/transient-refusal claims remain; it now also requires pinned metadata restoration and readiness for a later request.

## Unit verification and mutation proof

Rust/cargo1.99.0 on macOS. Eighteen supervisor tests pass, including refund-after-ACK ordering, retained leases, FIFO deadlines, pending budget, persistent exhaustion, process-exit ordering, and model restoration. The worker's twelve ordinary tests pass; hardware probes remain ignored unless explicitly invoked.

Safe refund mutation: staged the live implementation and confirmed empty `git diff --stat`; disabled only completed-eviction slot removal (marked `NON-VACUITY BREAK`) to prove the test detects a missing budget refund; observed `1 file changed,1 insertion(+),1 deletion(-)` for `crates/synapse-module/src/worker_host/mod.rs`. A three-test run failed **only** `worker_host::ane_residency::tests::eviction_refunds_executables_only_after_completed_ack` (budget wait deadline exceeded). `replacement_never_starts_before_owner_exit_confirmation` and `restart_reloads_pinned_models_before_returning_ready` stayed green. Restored from the staged index with checkout and touch; `git diff --stat` was empty again. No mutant is committed.

## Hardware regression

The full real-worker stress test, not a unit assertion about syntactic pool placement, is the reclamation regression guard. Prebuild the worker and module driver before waiting for unitless macOS load averages below16 over one minute and below20 over five minutes. Record the fresh JSON separately. At this commit the post-fix stress run is pending.

## Low-load post-fix stress result: FAILED on queue deadlines

`stress-pooled-live100-development.json` records commit `6f6abcf0a7e70aa637f4dca80e2253a7d66dc1d7`, Mac17,6, OS27.0.1/build26A434, debug. Start 1-/5-/15-minute macOS load averages **13.39/19.37/25.62** passed both gates; end **17.66/26.82/28.80**. Production's Core ML GTE lane was reported resident by the operator and was not modified. The launcher waited approximately81minutes for a quiet start; the actual hardware test ran1094.94seconds. All processes started for the run have exited; no background measurement remains.

JSON summary: **37 requests,8 completed,29 failed**, all29 failures exactly `admission budget wait deadline exceeded` with250ms retry. **No hardware `no ANE resources` errors**, no nonresident inference errors, no leased eviction. Eight shape admissions, five completed evictions, thirteen residency samples, peak two shapes per model/three overall. The reranker finite/repeat checks did not complete successfully; Metal ranking remains explicitly skipped because its package/lane is unavailable.

This is not a passing hardware regression. The original pre-fix flood failed33 of37 requests while loading layer0 with `Program load failed — no ANE resources`, underlying0x5. This run instead failed only on the admission queue's wait deadline: pooled reclamation has avoided all observed hardware resource refusals, but the37-way cold/debug flood exceeds the unchanged600-second admission wait deadline. No timeout or success assertion was relaxed to turn this run green. A separate decision is needed on cold-start stress scheduling/deadline policy before another run; the evidence does not establish37-request acceptance or a release-quality throughput claim.
