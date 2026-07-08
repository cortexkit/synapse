# Synapse engine-worker protocol v1

Status: DRAFT for implementation (design-synapse-module.md names this spec a
prerequisite; SUBC r2 handshake requirements folded).

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
   {v: 1, nonce: "<hex16>", engine: {name, version, build_flags},
    pid, max_frame: u32}
   - v is the protocol version BYTE (integer); module rejects unknown majors.
   - nonce must equal the spawn nonce: a stale worker from a previous module
     generation can never answer a new module's socket (launch-nonce pattern).
3. Module replies HELLO_ACK {v: 1, accept: true, max_frame} or closes.
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
  ids frame → SCORES {req_id} + raw f32 frame | ERR
- GENERATE {req_id, model_ref, max_tokens, grammar?} + raw ids frame
  → TEXT {req_id, text, n_prompt, n_gen, finish_reason} | ERR
- UNLOAD {req_id, model_ref} → UNLOADED {req_id} | ERR
- PING {req_id} → PONG {req_id, rss_mb, models_loaded} — module-initiated,
  NEVER on the health dispatch path (health serves cached state; PING feeds
  the background refresher that stamps it).
- SHUTDOWN {} → worker exits 0 after current request.

Tokenization stays module-side (sanitized tokenizers, padding rules, token
counts) — workers receive ids only. Pooling is done by the worker's engine
(llama.cpp builtin pooling where faithful; the padding incident is upstream
of this boundary and stays fixed there).

## Timeouts, crash, and recovery

- Per-request deadline set by the module from the scheduler's quantum budget;
  on expiry: SIGKILL the worker (single-lane means no collateral), classify
  engine_crashed{stage: timeout}, restart per crash budget.
- Worker exit/socket EOF mid-request → engine_crashed, request fails typed
  (never hangs — the module holds the consumer contract, Oracle F8).
- Crash budget: N crashes per (model, config) within window → that
  (model, config) is quarantined (permanent error until re-probe); M crashes
  across all work → lane degraded, health reflects it (cached, stamped by the
  supervisor path).
- Worker stdout/stderr → module log ring buffer, tagged, size-capped; last 8 KiB
  attached to engine_crashed detail for diagnosis.

## Versioning

v1 is intentionally complete for Lane 1's needs. Additions are new message
types (old workers ERR unknown_type, module treats as capability absence);
breaking changes bump v and the module refuses mismatched workers (it ships
its own worker binaries, so version skew only happens on packaging bugs —
fail loud, not compatible).

## Non-goals

Streaming generation (micro-LLM is ≤64 tokens; post-v1 agentic serving will
revisit), worker-side batching decisions (scheduler owns batching), worker
pools per model (scheduler owns placement), TLS/auth beyond the nonce (same
UID, private runtime dir, module-owned lifecycle).
