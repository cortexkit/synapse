# Worker supervision

This document describes the behavior at repository commit `e5d51cb626de`. The main implementation is in:

- `crates/synapse-module/src/worker_host/mod.rs` (`WorkerHost`, `WorkerEngine`, and the owned-decode adapter);
- `crates/synapse-core/src/worker_protocol.rs`, `worker_framing.rs`, `worker_framing_sync.rs`, and `worker_transport/` (the common wire protocol);
- `crates/synapse-engine-owned/owned-decode-worker/src/` (the owned-decode supervisor and persistent crash budget); and
- the `crates/synapse-worker-*` binaries (the worker side).

## The two supervision paths

There are two related but different paths.

1. **The generic worker host** runs the llama, MLX, ANE, CUDA, and decode executables. A `WorkerEngine` owns a private Tokio runtime and a mutex-protected `WorkerHost`. The host owns one child connection at a time and implements the common v1 `LOAD`, inference, `PING`, `UNLOAD`, and `SHUTDOWN` protocol. Its default crash authority is an in-memory rolling window in that host.
2. **The owned-decode supervisor** drives resident generation through `Supervisor<FileBudgetStore>`. It uses `OwnedDecodeWorkerFactory` and `OwnedDecodeWorkerSession` as an adapter over the same generic host, but it changes the crash authority to `OwnedDecodeSupervisor`. This disables the host's rolling crash book for that process. The owned supervisor is then the only component allowed to charge its persistent `(machine profile, decode fingerprint, runtime digest)` budget.

The process host and the owned state machine therefore have separate responsibilities. `WorkerHost` owns process creation, the launch nonce, IPC framing, request I/O, request timeouts, forced termination, and lazy model reload. `Supervisor` owns generation validation, quantum sequencing, boundary cancellation/deadline decisions, failure classification, one possible crash redispatch, and persistent quarantine. Do not add owned-decode charging to `WorkerHost`: `OwnedDecodeWorkerFactory::new` deliberately sets `CrashAuthority::OwnedDecodeSupervisor`, and `record_crash_and_maybe_restart` returns immediately in that mode.

A “worker” is an executable child process, not a thread. The common protocol is deliberately serial: the worker request loops process one request at a time, and `WorkerEngine` holds the host mutex across each `Runtime::block_on`. Concurrency comes from multiple engine/host instances outside this protocol, not concurrent messages on one connection.

## Spawn and handshake

### Before the child exists

`WorkerHost::start_worker` first creates the module-owned endpoint:

- Unix: `<runtime_dir>/wk-<first-8-bytes-of-SHA256(worker_id)>.sock`. The parent directory is created, and an existing filesystem entry at that path is removed before bind.
- Windows: `\\.\pipe\synapse-<same-digest>`, opened as the first named-pipe instance.

Failure to create/remove/bind the endpoint returns `WorkerHostError::Io` before a process is spawned. It is not recorded as a crash.

The host generates a 16-character hexadecimal nonce from time, PID, and a process-local counter. It passes `--socket <path> --nonce <nonce>` on Unix or `--pipe <name> --nonce <nonce>` on Windows, sets `SYNAPSE_WORKER_ID`, pipes stdout and stderr, and enables Tokio's `kill_on_drop`. Configured extra arguments follow those mandatory arguments. If a catalog engine identity is configured, the host also sets `SYNAPSE_WORKER_EXPECTED_ENGINE`; that environment variable exists for the timeout test worker and does not weaken the host's identity check.

A command-spawn failure is wrapped as `WorkerHostError::Protocol("spawn worker …")`, not `EngineCrashed`, and consumes no crash budget.

### What each side believes during HELLO

The host is always the listener and the worker is always the connector. This is a single-accept endpoint: the listener is consumed by the handshake function. The launch nonce prevents a process from an older spawn from being accepted as the new child.

The state progression is:

| Point | Host state and belief | Worker state and belief |
| --- | --- | --- |
| Listener bound, child spawned | `connection` is still `None`; the host has an OS child handle and expects this spawn's nonce. | It has parsed its command line and is trying to connect. Unix workers try once. Windows workers retry opening the pipe every 50 ms for at most 30 seconds. |
| Connected, before HELLO | The host has accepted a stream but has not authenticated it. | It believes it reached the endpoint named by its parent. |
| HELLO sent | The host reads a length-prefixed JSON value with its configured maximum. | It has advertised common protocol `v = 1`, the exact nonce it was given, its engine identity, PID, and maximum frame size. The owned decode worker additionally advertises `protocol_version = 2`. |
| HELLO validated | The host has verified common version and nonce, optionally the engine **name**, and, for owned decode only, exact extension version 2. It does not compare the advertised PID to the child handle, and it does not validate engine version or build flags. | It is still waiting; a rejected handshake does not receive a negative acknowledgment from this host. |
| HELLO_ACK received | Only after writing the ACK and returning from the handshake does `start_worker` install `WorkerConnection`. | It accepts only common version 1 and `accept = true`, then enters its serial request loop. |

The common HELLO and ACK types are `WorkerHello` and `WorkerHelloAck` in `worker_protocol.rs`. The host emits only a positive ACK. `accept: false` exists in the type and worker-side checks, but no host branch constructs it; rejection is close plus kill.

The accepted frame size in the ACK is `min(host max_frame, worker max_frame)`, normally 64 MiB. Frames are a little-endian `u32` byte length followed by payload bytes. A length over the supplied maximum, a payload too large for `u32`, short reads/writes, or invalid JSON all fail the framing operation. Control frames are JSON. Token IDs and float vectors use separate raw little-endian frames.

Owned decode has two simultaneous protocol versions:

- the common process protocol remains v1 (`WorkerHello.v`); and
- `synapse-worker-decode` must advertise `protocol_version: 2` for `owned-decode-envelope-v2`.

The host requires the latter only when crash authority is `OwnedDecodeSupervisor` and includes it in the ACK. The decode worker validates it if the ACK field is present, but accepts an ACK that omits the field. This host always includes it on the owned path.

### Handshake failure branches

The accept operation has one full handshake timeout, and reading HELLO after accept has another full handshake timeout. With the 5-second default, a connector can therefore consume nearly 5 seconds and then HELLO can consume nearly another 5 seconds; this is not one end-to-end 5-second deadline. ACK encoding/writing has no additional timeout.

The host rejects and then calls `child.kill()` and `child.wait()` (ignoring both results) for all errors returned by the handshake:

- no connection before the accept timeout (`"worker handshake timed out"`);
- accept/connect I/O error;
- no complete HELLO before the second timeout (`"worker HELLO timed out"`);
- malformed HELLO JSON or missing/wrongly typed required fields;
- common version other than 1 or nonce mismatch;
- configured engine-name mismatch; or
- for owned decode, absent, non-byte, or non-2 extension version.

ACK serialization or write failure follows the same kill-and-wait path. These failures occur inside `start_worker`, outside `send_request`'s request timeout, and are not charged by the generic host. On the owned path, a factory failure while constructing/loading a worker is later classified as `startup_failure` and is charged by the owned supervisor.

The nonce-to-`u64` conversion used as owned `worker_generation` can also return a protocol error. The generator always produces valid 16-digit hex, so that failure is unreachable from the nonce generator I found. Were it reached after a successful handshake, construction of `WorkerConnection` would abort; dropping the moved child still has `kill_on_drop` protection.

## Generic host lifecycle

### Stable host state versus process-local state

`WorkerHost` retains these across child restarts:

- stable host model IDs (`host-model-<counter>`), artifact and runtime configuration, dimensions/load timing/buckets, and each model's crash key;
- rolling crash timestamps and permanent host-local quarantine keys;
- request/model counters, cached placement share, log-job context, and the stdout/stderr forwarding limiter.

`WorkerConnection` is process-local: transport stream, child handle, that process's 8 KiB log ring, and worker generation. Worker-side `model_ref` values are process-local. `kill_current` therefore sets every tracked model's `worker_model_ref` to `None` while retaining the stable host record for lazy reload.

The field names are not liveness guarantees. `worker_connected` in a health snapshot means only `connection.is_some()`; the host does not poll `Child::try_wait`. `loaded_worker_models` counts non-`None` host-side worker refs, not models confirmed live by a ping.

### Load

`WorkerHost::load_model`:

1. Requires `runtime_config["artifact_path"]` or, as a fallback, `runtime_config["model_path"]`. Absence is a protocol error before spawn.
2. Builds a crash key from the whole runtime-config map after canonicalizing `artifact_path`. Owned CUDA also inserts the stable worker ID so equal artifact/config values in different model hosts are isolated.
3. Refuses immediately if that key is in the host-local quarantine set.
4. Calls `load_worker_model_ref`, which repeats the quarantine check, lazily starts a worker if needed, and sends `LOAD` under the load timeout.
5. Accepts only `LOADED` with the expected request ID, then allocates a stable host model ID and records the process-local model ref and metadata.

`ERR` becomes `WorkerErr` and is not a crash; the host does not validate the optional request ID in an `ERR`. A different response type or mismatched successful response ID is `Protocol` and is not a crash. I/O/EOF during the timed request or request expiry becomes `EngineCrashed`; the child is killed, all existing worker refs are forgotten, the load key is charged, and an immediate replacement process is attempted if the key is still below its threshold. A failed initial load is never inserted in `loaded_models`, even if the host started a replacement process.

The worker binaries perform defense-in-depth digest and format/config validation:

| Worker | Worker load state and clean load errors |
| --- | --- |
| llama | A `HashMap` of models; accepts GGUF. Backend mismatch, digest, runtime, and llama load failures are returned as `unsupported`, `artifact_invalid`, or `config_invalid`. |
| MLX | A `HashMap`; accepts safetensors/MLX safetensors and resolves MiniLM or Qwen. Load failures become `artifact_invalid` or `config_invalid`. |
| ANE | The Rust launcher `exec`s the Swift binary, so the host's child handle follows the Swift process. The Swift worker keeps a model map, accepts Core ML packages, verifies all bucket digests, and returns `artifact_invalid` or `config_invalid`. Partial temporary artifacts are removed on load failure. |
| CUDA | A single `Option<LoadedModel>` rather than a map. A build without the CUDA feature returns `backend_missing`; invalid inputs/engine load return `artifact_invalid`. A successful later load replaces the previous option. |
| owned decode | Exactly one loaded model and no load during a resident generation. It accepts `safetensors`, `owned-safetensors`, or `q8_0`, validates digest, family, context bucket, production N, chain K, tokenizer, and required identities. All ordinary load failures are sent as `load_failed`. In the owned factory those clean worker errors are collapsed into a charged `startup_failure`, unlike generic catalog loads. |

Production catalog construction overrides `WorkerHostConfig`'s 180-second load default with `worker.load_timeout_ms`, whose module default is 900,000 ms (15 minutes). A separate control-path waiter also defaults to 900,000 ms; its expiry returns `model_loading` to that waiter but does not cancel the background catalog load.

### Dispatch and lazy reload

`ensure_worker_model` first requires the stable host model ID to exist and not be quarantined. If both the connection and process-local model ref are present, it uses them without checking child liveness. Otherwise it starts a worker as needed and repeats `LOAD` from the retained artifact/config, then updates the stable record. Thus a crash after a successful load leaves the logical model available for lazy reload on the next request.

The ordinary operations then behave as follows:

- `EMBED_BATCH` flattens module-side token batches into one i32 raw frame. A token ID over `i32::MAX` is a local protocol error. Success requires `VECTORS`, the expected request ID, a raw response, and exactly `n * dims` floats; multiplication overflow, malformed raw floats, and shape mismatch are protocol errors. A missing raw frame remains inside the timed transport read and therefore becomes an I/O failure or timeout/crash.
- `RERANK` concatenates query and candidates into an i32 raw frame. Success requires `SCORES`, matching request ID, and a decodable f32 frame. A malformed delivered raw frame is a protocol error; a missing frame is a transport failure/timeout.
- `GENERATE` sends prompt IDs as an i32 raw frame. Success requires `TEXT` and a matching request ID. The host accepts the worker's text, counts, finish string, and generated IDs without additional cross-checking.

For each operation, a returned `ERR` is a clean `WorkerErr`, and its optional request ID is ignored. A request transport failure or timeout is `EngineCrashed`, charges that model key, and invokes restart logic. Local preprocessing errors and post-response protocol validation errors do not kill or charge the worker. For embed/rerank, however, `send_request` tries to read a raw response after **any** non-`ERR` JSON response before the public method checks its variant. An unexpected header without a following raw frame therefore times out and counts as a crash; the unexpected-response `Protocol` branch is reached only if a raw frame also arrives.

The workers' unsupported-operation branches matter for stream alignment:

- llama implements embed, rerank, and generate;
- MLX and ANE consume the accompanying raw frame before returning `unknown_type` for rerank/generate;
- CUDA returns `backend_missing` for rerank/generate; see the anomaly below; and
- owned decode returns `unknown_type` for common embed/rerank/generate and expects generation through the owned commands instead.

Except for ANE's string-based default arm, an unknown request type or malformed request that cannot deserialize as `WorkerRequest` exits the blocking worker loop with an error rather than returning `ERR`. The host then observes EOF while awaiting the response and treats it as a crash. Extra fields accepted by the common serde shape are a different case: they are ignored unless a more specific owned shape uses `deny_unknown_fields`.

### Ping and health

`ping` is an active request and can spawn an otherwise empty worker. `PONG` must have the matching request ID; it updates the host's cached `last_placement_share` and returns RSS/model count/share. `ERR`, wrong request ID, and wrong response type do not charge the crash budget.

If the ping transport crashes, the host copies the crash key of **every tracked stable model** and calls crash/restart accounting once for each entry before returning the error; with no tracked models it charges nothing and starts no replacement. This behavior was added by commit `39d7fd03b826` so ANE placement probes leave crash and lazy-restart state consistent with inference. There is no proactive death detector: a worker that dies between requests remains “connected” until the next write/read (often this ping) discovers it.

The regular health endpoint calls `health_snapshot`; it does not ping. The optional performance sampler tries to lock each worker engine and pings only unlocked engines, discarding ping failures. ANE certification also performs an explicit ping off the module's async runtime to read placement share.

### Unload

`WorkerHost::unload` removes the stable host record **before** attempting worker I/O.

- Unknown stable model ID: return success without sending anything.
- Known model whose worker ref is already `None` (for example after a crash): return success after removing the stable record.
- Live ref: send `UNLOAD`; accept only matching `UNLOADED`; map `ERR` or unexpected/mismatched replies as above.
- Transport crash/timeout: charge the removed model's key and maybe restart, then return the crash. The model record remains removed regardless of the result.

The trait-level `EmbedEngine::unload` has no result, so `WorkerEngine` discards the host unload result. Llama, MLX, and ANE remove the named model and acknowledge even when it did not exist. CUDA returns `model_not_loaded` for a foreign ref. Owned decode refuses unload while a resident generation exists and exits with an error if a nonresident unload names an unknown ref; the production owned factory does not issue ordinary unloads.

### Shutdown and destruction

Although `WorkerRequest::Shutdown` and every worker loop have a `SHUTDOWN` arm, I found no production host caller that constructs this request. `request_stage` has a `shutdown` arm, but it is correspondingly unreachable from current `send_request` callers. Normal teardown is forcible:

- `kill_current` takes the connection, calls child kill and wait while ignoring both results, forgets all process-local model refs, and returns the process log tail. With no connection it still forgets refs and returns an empty tail.
- `WorkerEngine::drop` takes its private runtime. If drop occurs while the thread is driving any Tokio runtime, it moves kill plus runtime destruction to a newly spawned OS thread; otherwise it runs them inline. The teardown thread is not joined. The special thread exists because the old implementation panicked by calling/dropping a runtime inside another runtime (commit `ca50adc332cf`). A poisoned host mutex skips explicit `kill_current`, but eventual connection/child drop retains `kill_on_drop` protection.
- Owned session `kill` marks the session non-reusable, invokes `kill_current`, and clears queued normalized frames. Dropping a non-reusable session drops its engine, which runs the same teardown. Dropping a reusable session puts the engine/model/generation into the factory's single idle slot.

A worker receiving `SHUTDOWN` exits its loop (MLX and owned decode also explicitly shut down the socket; owned decode clears resident, closed, and loaded state). Since the host never sends it, those are test/direct-client paths, not module shutdown behavior.

## Generic crash window and quarantine

The default generic policy is `max_crashes = 2` in a 60-second rolling window. `record_crash` prunes timestamps whose age is **greater than** the window (an event exactly at 60 seconds remains), appends the new event, and quarantines when the retained count is greater than or equal to the threshold. There is no validation that `max_crashes` is positive, so zero also quarantines on the first recorded event.

What counts is narrower than the word “crash” suggests:

| Event | Generic host action |
| --- | --- |
| I/O/EOF while writing or reading a timed request | Kill/wait, clear all worker refs, return `EngineCrashed`, charge at the public operation. |
| Request timeout | Same, with stage `timeout`. |
| Malformed JSON read through common framing | The framing helper wraps JSON decode as `io::ErrorKind::InvalidData`; therefore this is treated as I/O and charged as a crash. |
| Worker `ERR` | `WorkerErr`; no kill, no charge. |
| Unexpected but decodable response, mismatched request ID, bad raw shape/token conversion | `Protocol`; no kill, no charge. |
| Listener/spawn/HELLO/ACK failure | No generic charge. A spawned child is killed on handshake failure. |
| Successful request or successful restart | No reset. |

Below the threshold, the host immediately calls `start_worker`; failure of that restart is ignored and leaves `connection = None`, so the next operation tries again. At the threshold, it does not restart and inserts the key into `quarantined`.

Crash timestamps age out when `health_snapshot` or the next crash prunes them. Successful work does not clear them. More importantly, the host-local `quarantined` set is never pruned and has no lift API: time-window expiry does **not** remove quarantine. It is lifted only by destroying/recreating that `WorkerHost` (for example module/model lifecycle reconstruction) or by changing the crash key. Quarantine blocks `load_model` and `ensure_worker_model`, hence all model inference and lazy reload. It does not block health snapshots, active ping, unloading an already tracked record, or teardown.

`WorkerHostError::to_engine_error` marks only `EngineCrashed` safe to retry as the same request. Quarantine is reported at the worker-crash stage with no retry delay. `WorkerErr` also has no retry delay; all other host errors advertise 250 ms but are not marked safe for same-request retry.

## Owned-decode process pooling and request setup

Production builds one `SupervisedDecodeDispatch` from `build_supervised_decode_dispatch_for_chain_k` in `synapse-module/src/lib.rs`. The module refuses non-macOS or absent worker binaries before dispatch. It constructs:

- a host configured for `OwnedDecodeSupervisor`, with load timeout from module config and request timeout set to the request deadline (at least 1 ms);
- a persistent budget file `<owned worker runtime dir>/<model_id>-crash-budget.json`;
- a `QuarantineKey(machine_profile_hash, decode_fingerprint, runtime_config_digest)`;
- a start frame/context with immutable model/runtime identities; and
- a wall-clock dispatch clock (milliseconds since the Unix epoch), because the same reading sets deadlines and is persisted as the crash budget's `quarantined_until`, which the routing precheck and later processes read back.

Configured-shape dispatches are cached by model ID behind `Arc<Mutex<_>>`; the mutex serializes logical generations for that model. `set_request` replaces prompt and constraint, sets an absolute boundary deadline to `clock.now() + deadline_ms`, and resets the sidecar hint source. Dispatch then replaces generation ID, decode fingerprint, and max tokens from the routed command. The only production constructor sets `TerminalControl.cancel_at` to `None`, and `set_request` has no cancellation setter, so caller-recorded cancellation does not currently reach this production dispatch.

Despite the `WorkerFactory::spawn` documentation saying “fresh worker,” `OwnedDecodeWorkerFactory::spawn` is a checkout operation. It first takes the one reusable idle engine/model/generation shared by factory clones. Only an empty/poisoned idle slot creates a new `WorkerEngine`, performs `LOAD`, and reads its nonce-derived generation. There is at most one idle process, not a multi-worker pool. A non-reusable session causes the next attempt to create/load a process; clean completions normally return the same process to the idle slot for later logical requests.

Reusability is independent of the supervisor's clean/chargeable classification. Start marks protocol/version/JSON and worker-`ERR` failures reusable, but other host failures and generation mismatch non-reusable. Install/continue mark only host protocol/version errors reusable and other host errors non-reusable; a structure error found after a successful continue response leaves the previous reusable flag unchanged. Cancel marks host protocol/version errors reusable and other failures non-reusable. Supervisor-detected bad generation/sequence/count and an empty pending queue do not call session `kill`, so they also leave the flag unchanged. Consequently, an abstract `Crash` does not by itself prove the next factory call creates a new OS process.

The decode worker itself enforces one loaded model and one resident generation. A second `GENERATE_START` while resident produces failure frames. The host adapter also has exactly one `owned_decode_stream` state per host, containing logical/session ID, cumulative generated IDs, stream and quantum sequence, max tokens, and optional constraint identity.

## Owned-decode generation state machine

### Before the first attempt

`Supervisor::run_generation_with_hint_bank` takes these branches in order:

1. If `BudgetRecord::is_quarantined(clock.now())`, return `owned_decode_quarantined`; spawn nothing and charge nothing.
2. If the latest budget save failed, retry saving it. Continued failure returns `owned_decode_unavailable`; spawn nothing and charge nothing. If the key is already considered quarantined, step 1 returns before this persistence retry.
3. Run module-side `validate_start`: generation ID and prompt must be nonempty, max tokens nonzero, sampling exactly greedy top-1, loaded model/decode/runtime identity equal, and constraint presence and every identity field equal. The first failure returns its dedicated typed error with no spawn/charge.
4. Start an attempt.

`Supervisor::new` panics unless production N is exactly 8, 16, or 32. The shipped decode worker's load path is stricter and currently requires N = 16.

### Attempt start

An attempt asks the factory for a session. Any factory error, including process construction or initial model load failure, becomes charged `startup_failure` with worker generation/sequence zero. It does not qualify for redispatch.

After checkout, the supervisor records the session generation and calls `DecodeWorker::start`:

- A typed start refusal is clean and returns without kill or charge.
- A `WorkerStartFailure::Fault` is classified, the worker is killed, and the result is chargeable.
- Success authorizes the first `min(production_n, max_tokens)` quantum.

The production session locally repeats `validate_start`, then calls the host adapter. `owned_decode_start` ensures/reloads the model, substitutes the real worker model ref, captures the connected generation, initializes adapter stream state, and sends `GENERATE_START`. The real decode worker again validates against its actually loaded identities before committing a token.

Host protocol/version/JSON errors at start become clean `owned_decode_protocol_mismatch`; worker `ERR` IDs become their recognized typed error or protocol mismatch. Other host errors are faults: a host timeout is `Timeout`; other crash/I/O/quarantine errors are `Crash`. A returned generation different from the session's immutable generation is a crash and makes the session non-reusable. Each normalized frame is structure-checked and queued; a structure-check failure at this point is typed and leaves the session's initial reusable flag true.

### Frames, progress, and continuation

`step` pops one already queued normalized frame. An empty queue is `WorkerFault::Crash`. In the production adapter all socket I/O happened in start/continue, so its `step` cannot itself produce Timeout, StartupFailure, FailedCancellation, or Protocol; those arms exist for other `DecodeWorker` implementations and fixtures.

For a stepped fault:

- `Crash`, `Timeout`, `FailedCancellation`, and `StartupFailure` are chargeable with their same classification;
- `Protocol` is a clean `owned_decode_protocol_mismatch`, with no strike.

For a frame:

- A worker-generation mismatch is `protocol_fatal`.
- `WorkerFrame::Error` parses a known stable ID and returns it clean; unknown IDs become clean `owned_decode_protocol_mismatch`. `DecodeError::from_id` does not recognize `deadline_exceeded` or `cancelled`, so those IDs in an error frame also become protocol mismatch.
- Progress with a wrong logical generation, repeated/skipped quantum sequence, or non-increasing cumulative committed count is `protocol_fatal`. Sequence accounting is updated before the committed-count check, so provenance records the bad progress sequence.
- `ProgressBoundary::Continuing` immediately reads the next queued frame. It does not evaluate cancellation/deadline, poll/install a hint bank, or send a continue.
- At `ProgressBoundary::Yield`, terminal-control is evaluated. If accepted, zero remaining tokens is `protocol_fatal` because the worker owed a final. Otherwise the next budget is `min(production_n, max_tokens - committed)`; one nonblocking hint-bank poll may install a not-already-installed digest, then `GENERATE_CONTINUE` authorizes the exact next sequence and budget.

A protocol fault while installing a bank or sending continue is clean protocol mismatch. Any other fault from either call is labeled `protocol_fatal` by these match arms, even if the adapter originally returned `Timeout` or `Crash`. This is behavior, not the `WorkerFault` variant's name. A ready hint bank is retained by its source across crash redispatch, but each attempt tracks installation independently and installs a digest at most once. An empty/disconnected source simply does nothing and never delays continue.

For a final frame, a wrong logical generation is `protocol_fatal`. Otherwise boundary evaluation runs. A natural `stop_token`, `max_tokens`, or `grammar_complete` completion wins over already-recorded cancellation and deadline and is returned clean. A worker-reported `cancelled` finish takes the cancellation path and suppresses output. The real v2 adapter never constructs that final reason: it derives only max-tokens, grammar-complete, or stop-token; an aborted v2 error becomes an error-frame ID `cancelled`, which `DecodeError::from_id` does not recognize and therefore surfaces as protocol mismatch. The supervisor copies final IDs/accounting/identity into `SuccessOutput`; the generic v2 adapter has already performed stricter stream accounting before constructing this normalized final. Neither layer compares the terminal's decode fingerprint, runtime digest, or terminal `worker_generation` with the expected values here; the real worker constructs them from loaded state. Provenance uses the session generation, while `SuccessOutput.worker_generation` uses the terminal field.

The progress-side `BoundaryDecision::AcceptCompletion` arm cannot be reached because progress passes `completion: None`. The final-side `AcceptProgress` arm cannot be reached with the current exhaustive finish reasons: three are terminal completions and `Cancelled` maps to cancellation. The final deadline branch is likewise unreachable with those reasons because natural completion wins and `Cancelled` maps to cancellation before deadline.

### v2 adaptation on the real transport

`read_owned_decode_response` reads values until it has a yield or terminal result:

- Every v2 envelope must use schema `owned-decode-envelope-v2`, version 2, the current command's request ID, the generation-derived session ID, and the exact next stream sequence.
- Progress IDs are appended to host adapter state. Their cumulative count must equal that vector's length. A continuing progress is accumulated and reading continues; a yield returns all accumulated normalized frames.
- A final must repeat request/session IDs, match cumulative count, have `tokens_emitted == committed_token_count`, and have terminal state `Completed`. Its finish reason is **derived by the adapter**, not copied: count at least max tokens means `MaxTokens`; otherwise any configured constraint means `GrammarComplete`; otherwise `StopToken`. Adapter state is then cleared.
- An error terminal validates request/session and counts. `Aborted` maps to `cancelled`; disabled/revoked maps to `artifact_poisoned`; `Failed` and the otherwise-invalid `Completed` error map to protocol mismatch. Adapter state is cleared.
- On cancel, a non-envelope `CancelledTransportResponse` is accepted and clears state.
- On install, a strict `HINT_BANK_INSTALLED` shape is accepted without clearing state.
- A standard `ERR` becomes `WorkerErr`. Anything else is a malformed-response `Protocol` error.

`owned_decode_continue` increments its adapter quantum sequence before sending. Overflow is a local protocol error. Start/continue can only return normalized frames from this parser; their “other response” match arms have no producer I could find. Similarly, install's cancelled response and cancel's installed response cannot be produced because parsing those shapes is gated by the request variant.

The decode worker validates continuation generation, exact next sequence, positive budget, budget no greater than 16, and budget no greater than remaining tokens. Hint installation is accepted only while paused awaiting continue, for the active generation, on a constrained request, with matching schema identity and bounded bank. Cancellation requires an active matching generation; mismatch restores the resident and produces protocol failure frames.

### Cancellation and deadline boundaries

Terminal control is checked only when the supervisor observes a yield or final, not asynchronously during a quantum. Precedence is:

1. natural terminal completion;
2. cancellation recorded at or before observation;
3. deadline at or before observation; then
4. accept progress.

At a cancel/deadline boundary the supervisor sends `GENERATE_CANCEL` on the same session. Acknowledgment destroys resident worker state and returns the boundary's clean `cancelled` or `deadline_exceeded`, with no strike. In the current production wiring only the deadline can create this boundary; cancellation timestamps are exercised by direct supervisor callers and fixtures. A protocol failure while cancelling returns clean protocol mismatch. Any other failure invokes `worker.kill()` and charges exactly one `failed_cancellation` strike.

There is no separate numeric cancel timeout in the production adapter. `GENERATE_CANCEL` uses the host's configured owned request timeout; the “cancel timeout” named by the trait is that transport timeout in this implementation.

### Charging, redispatch, and quarantine

The owned default policy is two strikes and a 60,000 ms quarantine. A record contains monotonically increasing strikes, ordered classifications, and an optional `quarantined_until`. It is stored in a JSON map keyed by `machine|decode|runtime`; the full map is rewritten to a temporary file and renamed on each save. `BudgetPolicy` does not reject zero strikes: remaining starts at zero, and the first charge exhausts/quarantines.

Exactly these classifications are chargeable: `crash`, `protocol_fatal`, `startup_failure`, `timeout`, and `failed_cancellation`. A single failure receives one classification even if several descriptions could apply. Clean typed errors, validation refusals, acknowledged cancellation/deadline cleanup, and transport-level `Protocol` consume none. Success does not reset strikes.

After the first charge, only classification `crash` can redispatch, and only when all are true at the same `now`:

- at least one budget unit remains;
- the key is not quarantined;
- cancellation is not yet effective; and
- the original deadline remains strictly later than `now`.

Redispatch is exactly once from the original prompt and constraint state. It preserves logical generation ID/deadline and resets supervisor attempt accounting to token count zero and expected quantum sequence one. It calls the factory again; after a real host transport crash the session has been marked non-reusable, so this creates a replacement process with a new nonce-derived generation. The supervisor itself neither enforces a different generation nor forces every `Crash` variant to make the session non-reusable, so a synthetic/empty-queue crash can check out the same idle process. A clean second result ends the operation. Any second charge is terminal and returns quarantined if that charge exhausted the budget, otherwise unavailable. There is never a third attempt.

If first-crash redispatch is barred, exhausted/quarantined returns `owned_decode_quarantined`; an expired deadline returns `deadline_exceeded`; effective cancellation returns `cancelled`. The final fallback in the crash branch also returns quarantined; with internally consistent default budget state I found no way to reach that fallback without one of the preceding conditions. A non-crash charge returns quarantined if exhausted and unavailable otherwise.

A save failure retains the updated record in the in-memory file store, marks the key unpersisted, and returns unavailable. A later non-quarantined dispatch retries that exact record and spawns nothing while the retry fails. If the failed save was an exhausting charge, the pre-dispatch quarantine check runs first and returns quarantined without retrying persistence until the local clock considers that quarantine lapsed.

At the budget abstraction, quarantine is active only while `now < quarantined_until`; equality lifts it. Strikes and classifications are not reset when time passes. Therefore an expired key may run clean work, but it has zero remaining redispatch budget, and its next charge immediately establishes another quarantine period. Key rotation or deleting/resetting the persistent record is the only way to restore strike capacity. The production routing precheck has an additional issue described under Anomalies.

## Protocol faults versus crashes

There are three easily confused categories.

1. **Common-host `Protocol` errors** are complete/understandable host-side contract failures such as unexpected response type or request-ID mismatch. `send_request` returns them without killing the worker. Common framing's malformed JSON is an exception in representation: it is wrapped as I/O and therefore becomes a crash.
2. **Owned transport `WorkerFault::Protocol`** means a complete transport interaction violated the protocol without process failure. At start, step, install, continue, or cancel, the owned supervisor returns typed `owned_decode_protocol_mismatch` cleanly and does not charge a strike (except the currently unused `WorkerStartFailure::Fault(Protocol)` route, which `classify_worker_fault` would label protocol-fatal).
3. **Owned `protocol_fatal`** is charged when a syntactically delivered worker frame violates session/generation/sequence/accounting invariants, when a yield at max tokens lacks a final, when a stale worker generation speaks, or when a non-Protocol fault occurs during install/continue. It is terminal after the charge and never redispatched.

This distinction was introduced in commit `5a061d82e7c9` while the host began consuming v2 envelopes. The code added `WorkerFault::Protocol` with the definition “a complete transport frame violated the worker protocol without a process failure” and added the test `protocol_transport_error_returns_typed_failure_without_a_strike`. That change also made start-time host protocol/version/JSON and worker typed errors reusable, with narrower reusability rules for later commands as listed above. The commit message itself gives no longer rationale; the source definition and regression test are the recorded reason: receiving an incompatible complete frame is a typed compatibility failure, not evidence that the child process crashed, so it must not spend the crash budget. Semantic corruption of an otherwise accepted session is separately and deliberately budgeted as `protocol_fatal`.

Do not collapse these categories based on their English names. In particular, changing a decoder/parsing error from `Protocol` to I/O/Crash changes quarantine behavior, and changing an invariant branch from `ProtocolFatal` to `Protocol` removes a strike.

## Death timing and residual state

| When the process dies | Who notices | Result and state left behind |
| --- | --- | --- |
| Before connect / during HELLO | Host accept/read timeout or I/O; it does not otherwise poll the child. | Spawned child is kill/waited; no `WorkerConnection`; generic budget unchanged. Owned factory reports charged startup failure. |
| During initial generic `LOAD` | Timed write/read sees EOF/I/O or expires. | `EngineCrashed`; connection removed, all older refs cleared, load key charged; failed new model is not tracked. Below threshold a bare replacement process may be started. |
| During owned factory `LOAD` | Same generic host machinery, then error is hidden by `WorkerFactory::spawn`. | Session is never returned; supervisor charges `startup_failure`, not `crash`, so no redispatch. |
| During owned `GENERATE_START` after model load | Host request sees I/O/timeout and kills. | Session marks itself non-reusable. Supervisor charges `crash` (or `timeout`) and only a crash can get the one fresh redispatch. |
| During ordinary inference, ping, unload, or an owned transport command | Next write/read/timeout. | Host kills/waits and clears every process-local model ref. Ordinary public operation charges its key; ping charges every tracked entry; owned classification depends on call site (continue/install non-Protocol faults become protocol-fatal, cancel becomes failed-cancellation). |
| Between requests | Nobody immediately. | `connection` and refs remain present; health can report connected/loaded. The next request discovers it and follows the corresponding mid-request path. |
| After a progress frame but before continue | The host has already returned that yield. | The next continue write/read discovers death. The logical generation has committed only its reported prefix, but any permitted crash redispatch restarts from token zero and never exposes the failed attempt's prefix. |
| During forced shutdown/drop | `kill_current` does not distinguish already-dead from live; kill/wait errors are ignored. | Connection is taken and refs cleared. No crash strike is recorded for teardown. Drop may complete on an unjoined helper thread. |
| Worker exits cleanly because its peer closed | Worker loop treats `UnexpectedEof` as normal exit. | The host normally caused closure through teardown; there is no worker-to-host exit notification beyond EOF on a later read. |

The process log ring is per process and stores both stdout and stderr despite the `STDERR_RING_BYTES` name. It is byte-capped at 8 KiB and is attached only to `EngineCrashed`. Forwarding is line-oriented and shares one per-host budget across both streams (default 50 lines per one-second window); dropped-line count is attached to the next admitted line. Pipe read errors and a final non-newline-terminated partial line stop/disappear from forwarding, but bytes already read remain in the ring.

## Timeout and time-window inventory

| Mechanism | Value in production/default | Expiry behavior |
| --- | --- | --- |
| Unix/Windows host accept | `WorkerHostConfig.handshake_timeout`, default 5 s | Protocol handshake error; spawned child kill/wait; no generic strike. |
| Host HELLO read | Another full `handshake_timeout`, default 5 s | Same. Not shared with accept. |
| Windows worker pipe-open retry | 30 s, sleeping 50 ms between failures | Last open error is returned after the deadline. The host normally has already bound the pipe and independently gives the handshake only 5 s. ACK read has no worker-side deadline. |
| Common host request | Default 30 s for embed/rerank/ping/unload and any non-load request | Covers JSON/raw write and response/raw read as one Tokio timeout. Expiry kills and returns `EngineCrashed(stage="timeout")`. |
| Common host generate override | 180 s when catalog `spec.task == "generate"` | Same. |
| `WorkerHostConfig` load default | 180 s | Same, but selected only for `LOAD`. |
| Module production worker load | Configurable `worker.load_timeout_ms`, default 900,000 ms | Overrides the preceding 180 s for catalog and owned worker loads. Request expiry kills; generic load charges, owned factory load becomes startup failure. |
| Control-path model-load wait | Request override or 900,000 ms | Returns `model_loading`; background load continues. |
| Owned transport request | Deadline used when the dispatch/factory was constructed, at least 1 ms; ordinary default caller deadline is 30,000 ms | Each start/continue/install/cancel gets a fresh timeout of this duration. Expiry kills in the generic host, then is classified by the owned call site. It is not a single generation-wide I/O timer. |
| Owned boundary deadline | `set_request` stores `dispatch_clock.now() + deadline_ms`; ordinary default is 30,000 ms | Evaluated only at yield/final boundaries. Equality is expired. Natural completion wins; otherwise cancellation/cleanup is attempted. |
| Owned certification dispatch budget | 900,000 ms (`OWNED_DECODE_PROBE_TIMEOUT_MS`) | Passed as both the cached dispatch's per-command transport timeout and each probe request's boundary deadline; it is not a separate outer timer. |
| Generic crash window | Default 60 s | Timestamps older than 60 s are pruned; quarantine is not lifted. |
| Owned quarantine | Default 60,000 ms from an exhausting charge | Budget abstraction lifts at `now >= until`, without resetting strikes. See production precheck anomaly. |
| Worker log-forward window | 1 s | Resets forwarded count; accumulated dropped count is reported on the next forwarded line. |
| Nested ANE prefill readiness/prediction/handoff | Positive millisecond values required from runtime keys `ane_prefill_readiness_budget_ms`, `ane_prefill_prediction_budget_ms`, and `ane_prefill_handoff_budget_ms`; no fallback numeric default exists in the runner | Described below; failure falls back to GPU within the decode worker rather than charging the module's process crash budget. |

Request timers begin after `ensure_worker`, so a lazy spawn/handshake is governed by handshake timers and a lazy model load by its own load timer before the inference timer starts. There is no idle-worker timeout, and forced `kill`/`wait`, ACK writes, worker-side ACK reads, and graceful shutdown have no additional deadline in this layer.

## Nested ANE prefill sidecar

When owned decode runtime config includes `ane_prefill_artifact_path`, `synapse-worker-decode` supervises a second out-of-process Swift sidecar inside the already supervised decode process. This is a separate local protocol and budget domain, not another `WorkerHost` or `Supervisor` attempt.

`AnePrefillClient::connect` creates a unique temporary Unix socket and shared-memory file, spawns the sidecar, polls nonblocking accept every 5 ms until the configured readiness budget, and notices early child exit with `try_wait`. Bind/config/spawn/accept failures return a load fault; timeout kills the child. It then uses the same readiness budget to read a strict five-field HELLO (protocol, nonce, engine name/version, positive max frame), sends a positive ACK, and performs artifact `INSTALL`. Readiness covers launch, handshake, and install.

Execution sets one I/O deadline to `prediction_budget + handoff_budget` for command/response, then a handoff-budget deadline for publication validation and timing readback. It also rejects an observed prediction wall time over the prediction budget even if the combined I/O deadline succeeded. A failed request sets the cached client to `None`; client drop kill/waits the child and removes the socket, specifically preventing reuse of a socket with a late response. The decode worker records the sidecar failure and performs GPU prefill fallback. Since the outer worker remains alive and continues generation, this does not itself charge the owned crash budget. Absence of the sidecar config takes the explicit no-sidecar branch.

## Sharing and isolation boundaries

The boundaries enforced by code are:

- **Per `WorkerEngine`:** private Tokio runtime, one mutexed host, one connection, one request at a time, stable models and generic crash book. There is no repository-global generic worker registry or global crash threshold.
- **Per generic child generation:** child/stream/log ring/nonce-derived generation and all worker `model_ref`s. Death invalidates all refs in that host, not only the model whose request detected it.
- **Usually per catalog model:** module catalog loading constructs a separate worker engine and worker ID for each stored model. The protocol/host can track several models, and llama/MLX/ANE workers implement maps, but production does not depend on sharing them. CUDA and owned decode enforce a single loaded model more tightly.
- **Per owned model dispatch:** one mutex, one monotonic clock, one factory, one idle process slot, one quarantine key, and one current request's mutable start/control/hint source. Configured-shape dispatches are cached by model ID.
- **Per owned worker process:** exactly one loaded key and at most one resident generation. `worker_generation` comes from the launch nonce and is checked at session and frame boundaries.
- **Per owned logical generation:** logical generation ID, prompt/constraint, deadline, attempt provenance. A crash retry preserves these but resets process/session generation, sequence, and token history.
- **Persistent budget sharing:** one JSON file per model ID can contain multiple storage IDs. The storage ID includes machine profile, decode fingerprint, and runtime digest, so one key's strike does not intentionally charge another.
- **Logs:** log context and rate limiter are per host and survive child replacement; the 8 KiB ring is per child. The nested ANE sidecar inherits the decode worker's stderr, so its stderr is ultimately captured in the outer worker's ring.

Endpoint uniqueness is based only on `(runtime_dir, worker_id)`. On Unix a second host with the same pair removes the first host's socket path before binding; on Windows first-instance creation fails. Module-generated worker IDs include engine/model identity to avoid ordinary collisions.

## Arms with no current production caller

These branches are intentionally called out rather than silently omitted:

- The host never sends `WorkerRequest::Shutdown`; every worker's graceful shutdown arm and `request_stage(Shutdown)` are reachable only from tests or another direct client.
- The host never emits `WorkerHelloAck { accept: false }`; worker rejection checks exist but this host rejects by closing/killing.
- `WorkerHostError::Json` has no construction site in `worker_host/mod.rs` beyond its `From` implementation that I could find. Common JSON framing errors become `TransportError::Io`. The request-body `Protocol`/`ProtocolVersion` alternatives in ordinary `send_request` and the `ProtocolVersion` alternative in `send_owned_request` have no current producer; version negotiation occurs earlier in `ensure_worker`, while owned response parsing can produce `Protocol`.
- `kill_current`'s no-connection branch is reachable from teardown before any spawn or repeated kill and deliberately still clears refs.
- Owned start/continue “unexpected command response” arms have no parser producer for their request variants; install cannot produce cancel acknowledgment and cancel cannot produce install acknowledgment.
- `OwnedDecodeWorkerSession::drop`'s `engine is None` branch is defensive; the only `take` I found is that same one-shot drop implementation.
- The supervisor's progress/`AcceptCompletion`, final/`AcceptProgress`, and current final/deadline arms are unreachable for the reasons given in the frame section. Its cancellation-decision branch is reachable through the public state machine and fixtures, but not current production wiring: the sole production `TerminalControl` has `cancel_at: None`, and the real adapter emits no cancelled final.
- `WorkerStartFailure::Fault(WorkerFault::Protocol)` would be charged as protocol-fatal, but the production adapter maps start protocol errors to `WorkerStartFailure::Typed`. I found no production constructor for that pair.
- `worker.step()` fault arms other than empty-queue `Crash` are reachable through scripted/custom `DecodeWorker` implementations, not the production `OwnedDecodeWorkerSession`, because production I/O occurs in start/continue.
- The owned decode worker's common-protocol embed/rerank/generate error arm, ping arm, active-generation unload refusal, and graceful shutdown are available to direct clients/tests. The production factory uses load plus owned generation commands and tears down by kill; it does not call those arms.

## Safe modification checklist

A change in this path must preserve, or deliberately revise together, all of the following:

1. Bind before spawn; retain single-accept ownership; validate launch nonce, common version, configured engine name, and owned extension version before storing a connection.
2. Keep the one-in-flight-message invariant or redesign both host locking and every blocking worker loop.
3. Invalidate every process-local model ref whenever a connection is killed, while retaining stable artifact/config state for lazy reload.
4. Decide explicitly whether a new error is `WorkerErr`, clean `ProtocolMismatch`, chargeable `ProtocolFatal`, `Crash`, `Timeout`, `StartupFailure`, or `FailedCancellation`; this controls kill, retry, provenance, and quarantine.
5. Leave only the owned supervisor charging owned decode. A second generic charge book would double count one process event under a different key/lifetime.
6. Preserve token-zero-only crash redispatch: same logical generation/deadline, reset attempt sequence and committed count, at most once, and never for timeout/protocol-fatal/startup/cancel failure. A real dead process must produce a non-reusable session and new generation; do not assume the abstract `Crash` variant alone enforces that today.
7. Keep boundary precedence and check timing exact: completion, then cancellation, then deadline, and only at yield/final boundaries.
8. Keep persistence fail-closed. A save error must not silently permit dispatch with a budget record that can disappear on restart.
9. Treat process reuse as the current behavior of `WorkerFactory::spawn`; changing reusability on an error changes whether the next logical request sees the same resident process.
10. Remember teardown can run inside another Tokio runtime; do not move runtime `block_on` or runtime destruction back onto that thread.

## Anomalies

These were encountered while following the path; no source was changed:

- **Negotiated frame size is not used by the host after HELLO.** `handshake_on_stream_*` ACKs `min(host, worker)`, and workers use that value, but `WorkerHost::send_request` and `send_owned_request` continue passing `config.max_frame`. Shipped workers advertise the same 64 MiB, but a smaller worker maximum would make the two sides enforce different limits.
- **A cached owned dispatch keeps its first transport timeout.** `set_request` updates the boundary deadline but not `WorkerHostConfig.request_timeout`, which was fixed when the cached factory was constructed. Later requests with a different deadline use the new boundary deadline but the first request's per-command transport timeout.
- **CUDA unsupported rerank/generate does not consume the raw input frame.** The host always sends that frame, while CUDA immediately returns `backend_missing`; a subsequent request on the same connection would read the leftover raw frame as JSON. Current catalog routing uses this worker for embedding, so I found no production caller for those unsupported operations.
- **One failed ping can account/restart more than once.** `ping` loops over tracked model entries without deduplicating crash keys, and each below-threshold iteration directly calls `start_worker` rather than `ensure_worker`. Production normally gives each catalog model its own host, but a multi-model host can charge one physical death repeatedly and replace a just-started connection during that loop.
- **The production factory does not enforce the supervisor's documented fresh-generation retry contract.** A real host I/O crash marks its session non-reusable, but supervisor-detected `Crash` paths such as an empty pending queue do not. The session can return the still-running process to the idle slot, and the one redispatch can then reuse the same `worker_generation`; the supervisor does not compare first and second generations.
- **Owned cancellation is not wired from production requests into terminal control.** The only production `TerminalControl` sets `cancel_at: None`, and `set_request` updates only the deadline. Cancellation behavior exists in the reusable state machine and fixtures, but the module path can currently invoke `GENERATE_CANCEL` only as deadline cleanup.
