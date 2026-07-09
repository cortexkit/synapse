# Synapse module v1 design (Lane 1)

Status: DRAFT r2 — Oracle adversarial review folded in (16 findings, [task-id]).
Constraints inherited from: SUBC review rounds (pm_aa948240, pm_5f7e36ff), MC
contract (pm_56c42b11, pm_82691068), AFT riders (pm_4373925e, pm_74b2aff8),
spike evidence (bench/lanes/llama-inproc, bench/spikes/unified-rt,
bench/lanes/candle-embed), Decision #1 doc.

## Shape

One Rust workspace, module template per ai-provider-quota (two-crate split):

- `synapse-core`: pure logic — engine seam, scheduler/admission, fingerprints +
  alias table, model cache, batching, tokenization. No subc, no I/O beyond
  injected traits.
- `synapse-module`: wire binary — subc-client-rs serve(), ModuleHandler, config,
  storage descriptor consumption, vault (post-v1, remote endpoints).

Pins: subc-protocol 0.7.0, subc-transport 0.3.1 (D-008). Singleton module,
enforced by a machine-wide lock (not just convention); dev instances take a
separate cache namespace or run no-GC (Oracle F14). ManagementSurface role.

## Engine seam and crash domains

```
trait EmbedEngine {
    fn identity(&self) -> EngineIdentity;      // engine, version, build flags — hashed into numeric_profile_id
    fn load(&mut self, artifact: &ValidatedArtifact, cfg: &RuntimeConfig) -> Result<LoadedModel, EngineError>;
    fn embed_batch(&self, model, batch: TokenBatch) -> Result<Vectors, EngineError>;
    fn embed_one(&self, model, ids: TokenIds) -> Result<Vector, EngineError>;
    fn unload(&mut self, model);
}
// RerankEngine, GenerateEngine: same lifecycle, different call shapes.
```

**Engines are risk-classified (Oracle F7):**

- **abort-safe** (mature Rust API, no observed process-fatal paths): ort. May
  run in-process. The owned runtime (Lane 2) targets this class — it's Rust,
  error-returning by construction.
- **abort-capable** (C/C++ asserts abort the process — GGML_ASSERT observed
  live in the spike): llama.cpp. v1 runs these in SUPERVISED WORKER PROCESSES
  owned by Synapse: our lane binary speaking a thin length-prefixed protocol
  over a unix socket (pre-tokenized ids in, vectors out) — not llama-server
  HTTP, not a foreign app. Worker crash = classified error + worker restart
  with crash budget; the module process never dies from engine faults. The
  measured cost of process separation is small at batch granularity.
  (History: an earlier draft cited the child's "latency win" on the Qwen hot
  path as a second justification; the latency spike root-caused that as a
  benchmark artifact — the child was replaying a cached prefix. Same-workload
  comparison has in-process/worker FASTER than the HTTP child, so the worker
  decision now rests on crash-domain grounds alone, which were sufficient
  anyway. Honest fresh-query cost, M1-class: Qwen3-0.6B 14-17ms, MiniLM
  ~2.5ms.) Promotion of llama.cpp to in-process remains possible LATER,
  per-engine-version, once a crash-free record exists under the probe's shape
  envelope.

Artifact digest + format validation happens before any engine (worker or
in-process) touches a file; engine-load failure = classified permanent error,
never retried, never process-fatal. The probe certifies (model-family, config,
shape-envelope) combos; anything outside the envelope is rejected at admission,
not discovered in ggml.

## Tokenization ownership

Synapse owns tokenization (HF tokenizers) for all owned engines and workers.
Published tokenizer.json artifacts are SANITIZED at cache-ingest (padding
config stripped — the Qdrant Fixed(128) incident — truncation ours); the
sanitized-tokenizer digest is part of the numeric profile. Reported token
counts are real-token counts.

## Fingerprints, numeric profiles, and the alias table

Strict fingerprint per the signed contract, with the canonical form now fully
defined (Oracle F9):

```
numeric_profile_id = hash(model digest, quant, engine identity (name+version+build flags),
                          sanitized-tokenizer digest, pooling, normalization, dtype,
                          flash-attention setting, certified shape envelope
                          (ctx/batch/ubatch/n_seq bounds), prompt/prefix templates,
                          thread policy class)
fingerprint = hash(model digest, quant, numeric_profile_id)
```

Certification must prove SHAPE-INVARIANCE across the declared envelope (our own
bench: batch shape perturbs bf16 numerics — if an engine is not shape-invariant,
the shape class splits the profile). Certification rows are keyed by
`machine_profile_hash` = hash(OS build, driver/backend versions, GPU ids, CPU
features, RAM class, engine build) (Oracle F13); on mismatch the lane demotes
to uncertified until re-probe (staleness flag in status, never auto-burn).

**Query/batch profile unification (Oracle F10 — the subtle one):** the hot
single-query path and the bulk path MUST share a numeric profile, or be
explicitly alias-certified against each other by the probe, or embed.query
returns vectors under a DIFFERENT fingerprint than the corpus — silently
splitting consumers' vector spaces. The probe certifies query-config ≡
batch-config as a first-class check; if it fails, the lane's hot path is
disabled rather than allowed to fork the space.

**Alias table**: Synapse-owned, versioned; every row carries a validity
interval; table_epoch bumps on any change. Responses carry (fingerprint,
table_epoch, provenance, dims, equivalent_to inline). Revocation semantics
formalized (Oracle F11/F12):

- Vectors written under a certified fingerprint remain READABLE forever.
- An index whose written-provenance set spans a retracted pair becomes
  `migration_required` — response names the retracted pair and the safe
  rebuild target. This is index-level, not vector-level; the distinction is
  explicit in the contract.
- `aliases.check_index {index_fingerprint, provenance_set}` — atomic verdict
  (valid | migration_required {retracted_pair, rebuild_target} ) for
  write-commit-time checks; response-epoch caching alone is insufficient
  against mid-flight retraction.
- Re-probe triggers explicit: engine version bump, runtime-config change,
  model file hash change, machine_profile_hash change.

Day-1 declared pair: llama-f16 ≡ ort-fp32 (1.00000 full corpus). MLX bf16 and
ANE fp16/Core ML profiles are distinct fingerprints always; ANE certification
also gates the MLComputePlan Neural Engine placement share.

## Surface (ops)

Poll-first; anything slow is job-shaped (SUBC 32-credit route window). Requests
carry acceptance constraints; admission is atomic accept-or-reject (Oracle F2).

Common request fields: `deadline_ms` / `max_queue_ms` / `fail_if_not_ready`,
`required_fingerprint`, `allow_equivalent` (default false — MC's hard
substitution rejection preserved byte-for-byte), `required_epoch` (Oracle F15).

- `embed.query` — single text, interactive lane. Accepts `target_fingerprint`;
  routes only to a matching or currently-equivalent certified profile, else
  `substitution_rejected` / `not_certified` (Oracle F10).
- `embed.batch` — small batches inline; anything over the inline budget
  (token/item/byte caps) is job-shaped: accept → {job_id}, paged result
  retrieval, `queue_full` when the lane's memory budget is exhausted (Oracle
  F3). Response envelope everywhere: {fingerprint, table_epoch, dims,
  provenance, vectors, real_token_counts, module_generation} —
  module_generation on EVERY response (not just job errors) so consumers
  detect restarts mid-conversation cheaply (SUBC r2 note). Every item also carries an explicit
  truncation disclosure: {submitted_tokens, effective_tokens, truncated: bool}
  (AFT r2 note 1 — silent truncation is a retrieval-quality bug invisible
  downstream; the era of guessing what max_length did is over).
- `rerank.score` — query + candidates → per-candidate RAW scores +
  fingerprint; candidate-count/token budgets; large requests job-shaped.
- `microllm.oneshot` — prompt → text, max_tokens ≤ 64, greedy default.
  The protocol reserves a `grammar` field, but this llama-cpp-2 worker build
  does not expose GBNF support; non-empty grammar requests fail as invalid
  rather than running unconstrained.
- `model.load` / `model.status` — control-class job (below).
- `models.list` — cached catalog, states, fingerprints, alias rows.
- `probe.start` / `probe.status` / `probe.report` — explicit ops; start/status
  are job-shaped. Probe writes certification + perf rows stamped with
  (machine_profile_hash, os_build, module_generation) and persists the
  per-workload knob assignments. `probe.report` returns the full measured
  capability table (quality, perf, stale flags, current knob, assignments) for
  the onboarding screen. v1 reads the knob at startup; changing it requires a
  module restart.
- `aliases.check_index` — see above.
- `cache.pin` / `cache.gc` — cache management.
- `admission.status` — advisory only; the contract lives in per-request
  budgets, not in this snapshot (Oracle F2). It does expose, cheaply from
  cached scheduler state: per-lane `meeting_deadlines: bool` + rolling p50
  start-delay, plus certification/perf staleness for loaded lanes (AFT r2 note
  3 — feeds consumer-side degradation notices; advisory for UX, never a
  substitute for per-request budgets).

**Error contract (Oracle F16)**: stable codes, not just classes —
`queue_full`, `deadline_exceeded`, `model_loading`, `not_certified`,
`substitution_rejected`, `artifact_invalid`, `engine_crashed`,
`probe_required`, `migration_required`, `module_restarted` — each carrying
`class` (transient|permanent), `retry_after_ms` (transient), and
`safe_to_retry_same_request`.

## Scheduler and admission (machine-wide)

Three queue classes per engine lane (Oracle F1/F4/F5):

- **interactive** (embed.query, small rerank): deadline-priority; a request
  either STARTS within its max_queue_ms or is rejected immediately.
- **bulk** (batch jobs): executes in bounded MICRO-BATCH QUANTA (token-budget
  sized, ≈100-300ms of GPU work), never one monolithic batch — the interactive
  lane preempts BETWEEN quanta, so fast-fail semantics are real. Weighted fair
  share with aging guarantees bulk a minimum quantum share so steady query
  traffic cannot starve indexing.
- **control** (load/unload/probe): own state machine; artifact validation and
  download happen OUTSIDE engine locks; loads admit only at quantum
  boundaries; interactive calls never wait on an implicit load (that's
  `model_loading`, typed, with retry_after).

Execution: engine calls run on bounded dedicated worker pools
(spawn_blocking / worker-process channels) behind scheduler-owned semaphores —
per-request tokio spawns never run engine work directly, so the wire loop and
health/status stay responsive under full route credits (Oracle F6). Memory:
each lane has a byte budget covering queued inputs + in-flight tensors +
  pending results; admission rejects beyond it (F3).

**Idempotent job resubmission (AFT r2 note 2)**: job-shaped requests carry a
consumer-supplied `request_key`; resubmitting after `module_restarted` with the
same key returns the existing terminal job or admits a fresh one, never two —
and budget accounting is keyed on request_key so a retry is never double-billed
against lane budgets.

## Jobs, restart, and durability (Oracle F8)

Accepted jobs are persisted in the module's daemon-delivered storage
(HELLO_ACK.storage descriptor — the jobs DB lives in the same store.db as the
alias table and certification rows, no hand-rolled paths) with
`module_generation`.
On startup, any prior-generation job in queued/running becomes terminal
`failed_transient: module_restarted`; results pages of completed jobs survive
restart until TTL. Direct (non-job) calls fail via transport closure on crash —
nothing hangs. Consumers see exactly one of: result, typed error, or
transport-level closure.

## Model cache

~/.local/share/cortexkit/models/ per SUBC convention: content-addressed sha256,
tmp+fsync+atomic-rename, cortexkit-lease (module=models-cache, scope=digest).
GC hardening (Oracle F14): loads take SHARED read leases; GC takes an EXCLUSIVE
lease per digest and two-phase tombstones (mark, grace period, delete) so a
concurrent validate/mmap never loses its file. Refcount/pin metadata beside
blobs; Synapse-owns-GC; never touches other modules' pins. Artifact record:
{digest, source, format, sanitized-tokenizer digest, validation state, pins}.

**Lease primitive gap (SUBC r2 review, source-verified)**: cortexkit-lease
currently ships try_lock_exclusive ONLY — no shared mode. Resolution: option
(a), extend cortexkit-lease with a shared-lock mode (fs4 exposes
try_lock_shared; small commons addition). RESOLVED 2026-07-08: commons PR #1
merged (16aed47, CI green 3 OSes) — acquire_shared exists with the exact
semantics F14 needs (shared blocks exclusive until last holder drops, epoch-
neutral shared handles). Path-dep to commons master until SUBC's next release
pass publishes the semver-minor bump. Cache GC is UNBLOCKED for the cache
implementation wave.

## Health

Cached in-memory state ONLY (SUBC invariant): model/lane states, queue depths,
worker liveness, and probe-row staleness — stamped by the serving paths; ages
computed at probe time. Cold load in progress = degraded + "loading <model>".
No engine/GPU/worker probes on the dispatch path.

## Config

~/.config/cortexkit/synapse.jsonc: knob (performance|balanced|quiet,
default=balanced), model pins, cache budget, remote endpoints (post-v1).
Everything else measured (probe) or contracted (subc). v1 samples config at
startup; knob changes take effect after restart.

## v1 cut line

IN: embed (query + batch incl. job-shaped), rerank.score, microllm.oneshot,
model lifecycle + cache, probe + certification + machine profiles, scheduler
(3-class, quanta, budgets), fingerprint/alias surface incl. aliases.check_index,
durable jobs + restart semantics, health, engines: ort in-process + llama.cpp
supervised workers, knob mapping on Apple + x86.
OUT (designed-for, post-v1): remote endpoint lane (vault), MLX/ANE graduation,
llama.cpp in-process promotion, image/STT/TTS, agentic LLM serving, DirectML
activation, alias events beyond the day-1 pair.

## Open items

- ~~Qwen hot-query latency mechanism~~ RESOLVED: benchmark artifact
  (cached-prefix replay in llama-server slots; see
  bench/lanes/llama-inproc/SPIKE.md). Worker protocol spec now exists:
  docs/design-worker-protocol.md. Harness rule adopted: latency loops must
  vary query text (llama-server slot reuse cannot be disabled by flags).
- ANE spike verdict → quiet-tier reality on M1-gen hardware.
- Probe workload contents (built-in corpus + reference vectors: size/licensing
  pass pending).
- Worker protocol spec (length-prefixed frames over unix socket; Windows named
  pipes) — write before implementation. Handshake carries a protocol-version
  byte + a module-issued generation nonce so a stale worker from a previous
  module generation can never answer a new module's socket (SUBC r2 note,
  subc launch-nonce pattern).
- ~~cortexkit-lease shared-mode commons PR~~ MERGED (commons #1, 16aed47).
