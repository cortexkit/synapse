# Synapse engine-worker protocol v3

Module-internal protocol; deploy the module and every packaged worker together.
See [worker progress and liveness](design-worker-progress-liveness.md) for the
completed-work watchdog and engine-specific boundaries.

## Purpose

Abort-capable engines (llama.cpp class) run in supervised worker processes so
engine faults never kill the module (Oracle F7; GGML_ASSERT abort observed
live). This protocol is the seam between synapse-module and its workers. It is
module-internal: not subc, not federation, no foreign consumers — which is why
it stays deliberately minimal. Lane 2's owned runtime implements the same
in-process traits instead; a worker binary is just one packaging of an engine.

## Transport

Unix domain socket (macOS/Linux), named pipe (Windows). Socket path:
<runtime_dir>/synapse/workers/<worker_id>.sock, created by the MODULE (listener)
before spawn; the worker connects to it (module-as-listener avoids a
stale-socket-file race on worker crash loops).

Framing: length-prefixed — u32 LE payload length, then payload. Control frames
are JSON (small, rare, debuggable); tensor payloads are raw little-endian
buffers described by the preceding JSON header (no base64, no serde of float
arrays). Max frame size negotiated at handshake (default 64 MiB; requests
exceeding it are module bugs — admission byte budgets cap upstream).

## Handshake (SUBC r2 requirements)

1. Module spawns worker with args: --socket <path> --nonce <hex16> and env
   SYNAPSE_WORKER_ID.
2. Worker connects and sends HELLO:
   {v: 3, nonce: "<hex16>", engine: {engine, version, build_flags},
    pid, max_frame: u32, manifest_digest?, kernel_revision?, packed?}
   - v is the protocol version BYTE (integer); module accepts only an exact match.
   - Owned embed/rerank workers report their manifest digest and kernel revision.
     `packed` optionally advertises multi-row execution support. Every v3 worker
     follows the completed-work protocol/watchdog, even without that capability;
     short or opaque calls can finish with just a terminal reply (see Requests).
   - nonce must equal the spawn nonce: a stale worker from a previous module
     generation can never answer a new module's socket (launch-nonce pattern).
3. Module replies HELLO_ACK {v: 3, accept: true, max_frame} or closes.
4. Anything else before HELLO_ACK = protocol violation, close + worker kill.

## Requests

One in-flight request per worker connection (concurrency = worker pool size,
owned by the module's scheduler — workers are intentionally single-lane so a
hung request maps to exactly one process to kill). Request/response pairs:

- LOAD {req_id, artifact_path, artifact_digest, format, runtime_config{...}}
  → LOADED {req_id, model_ref, dims, cold_load_ms}
  | ERR {req_id, code, msg}  (codes: artifact_invalid, oom, config_invalid)
  Worker re-verifies digest before mmap (defense in depth; module already
  validated at cache-ingest).
- EMBED_BATCH {req_id, model_ref, pooling, normalize: bool,
    items: [{id, n_tokens}]} + one raw frame of concatenated i32 token ids
  → VECTORS {req_id, dims, n: usize} + one raw frame of concatenated f32
  | ERR {req_id, code, msg}
- RERANK {req_id, model_ref, query_n_tokens, candidates: [{n_tokens}]} + raw
  ids frame → SCORES {req_id} + raw f32 frame | ERR. The llama worker builds
  BOS/SEP/EOS query-document pairs internally and keeps Qwen3 reranker GGUFs
  disabled until their templates are faithful.
- GENERATE {req_id, model_ref, max_tokens, grammar?} + raw ids frame
  → TEXT {req_id, text, n_prompt, n_gen, finish_reason} | ERR. Grammar is a
  protocol field; the current llama-cpp-2 build does not expose GBNF because it
  is compiled without the optional common feature, so non-empty grammar requests
  are rejected instead of silently ignored.
- UNLOAD {req_id, model_ref} → UNLOADED {req_id} | ERR
- PING {req_id} → PONG {req_id, rss_mb, models_loaded, placement_share?} —
  module-initiated, NEVER on the health dispatch path (health serves cached
  state; PING feeds the background refresher that stamps it). ANE workers set
  placement_share to the last MLComputePlan Neural Engine dispatchable-operation
  share so certification can gate quality and placement together.
- SHUTDOWN {} → worker exits 0 after current request.

Owned rerank uses `RERANK_SEQUENCES` with already-composed token sequences;
direct ANE also has explicit shape/packed admission and eviction exchanges.
The complete schemas are in `crates/synapse-core/src/worker_protocol.rs`.

While an exchange runs, a worker can send
`PROGRESS {req_id, completed: u64, phase: setup | inference | control}`.
Counters strictly increase for completed work, never for timer ticks or kernel
submission alone. Each frame belongs to the current request and generation.
Progress precedes the terminal reply; a VECTORS/SCORES header is immediately
followed by its raw payload, with no progress frame inserted between them.
Short or opaque operations can send only their terminal reply within the idle
window. The host validates IDs, phase transitions and counters before renewal.

Tokenization stays module-side (sanitized tokenizers, padding rules, token
counts) — workers receive ids only. Pooling is done by the worker's engine
(llama.cpp builtin pooling where faithful; the padding incident is upstream
of this boundary and stays fixed there).

## Timeouts, crash, and recovery

- There is no worker-specific total request cap. Caller deadlines stop the
  caller's wait with `deadline_exceeded`, without killing or striking a live
  worker. Its reply-reading task keeps running after the caller stops waiting,
  retaining the connection and lifecycle guards until the full reply drains so
  a subsequent request cannot consume the abandoned request's result.
- The idle watchdog starts only after owning the stream and, for inference,
  its execution permit. Compilation holds no execution permit.
  All inference has a fixed 120-second silence window; LOAD, shape/packed
  compilation, eviction and CUDA cold setup use 600 seconds. Completed work
  renews that window. A no-progress expiry faults/kills the owned worker and
  counts as a strike, not a caller deadline.
- Worker exit/socket EOF mid-request → engine_crashed, request fails typed
  (never hangs — the module holds the consumer contract, Oracle F8).
- Two consecutive genuine crashes, protocol faults or hangs quarantine a lane.
  Only drained successful serving work clears strikes; LOAD/reload, PING and
  setup do not. Catalog quarantine is transient `engine_crashed`, with remaining
  ten-minute cool-down and `{lane_id, quarantined: true}`; after expiry the next
  request starts a fresh worker with zero strikes. Unload/reload or module
  restart clears it immediately. Legacy lanes retain `probe_required`.
  The independent owned-decode supervisor keeps its own policy.
- Worker stdout/stderr → module log ring buffer, tagged, size-capped; last 8 KiB
  attached to engine_crashed detail for diagnosis.

## Versioning

Version 3 adds mandatory completed-work progress and is a clean cutover from
v2. Reject v2 at HELLO, not after starting inference. A v3 worker which sends no
progress during long work times out under its operation's silence window; there
is no legacy total-cap fallback or heartbeat exemption. Numerical lane
fingerprints and the separate owned-decode envelope version do not change.

## Non-goals

Streaming generation (micro-LLM is ≤64 tokens; post-v1 agentic serving will
revisit), worker-side batching decisions (scheduler owns batching), worker
pools per model (scheduler owns placement), TLS/auth beyond the nonce (same
UID, private runtime dir, module-owned lifecycle).
