# Synapse wire contract v1 (consumer snapshot)

Status: SNAPSHOT of the additive v1.1 surface as of 2026-08-29.
Authoritative examples: crates/synapse-module/tests/skeleton_e2e.rs and
tests/soak.rs. This doc restates the contract for consumer integration (AFT,
MC); on any disagreement, the e2e tests win and this doc gets fixed.

## Envelope (every response, every op)

```json
{
  "fingerprint": "…",            // strict identity string — persist VERBATIM, compare only, never parse
  "table_epoch": 0,               // alias-table version; persist (fingerprint, table_epoch) per index
  "dims": 384,
  "provenance": { "engine": { "engine": "ort", "version": "…", "build_flags": {} } },
  "module_generation": 3,         // bumps every module boot — cheap restart detection mid-conversation
  "equivalent_to": [],            // live alias reads; inline, no second call
  …op payload flattened here…
}
```

- fingerprint = hash(model digest, quant, numeric_profile_id); the numeric
  profile folds engine identity, sanitized-tokenizer digest, pooling,
  normalization, dtype, shape envelope, thread class. Anything that moves
  vector space mints a new string. Alias flips do NOT change fingerprints.
- Per-item fields on embed responses: real_token_counts,
  truncation_disclosures {submitted_tokens, effective_tokens, truncated},
  and content_sha256 (see Divergence detector below).

## Errors (stable codes, typed recovery)

Every error: {code, class: transient|permanent, retry_after_ms?,
safe_to_retry_same_request}. Codes: queue_full, deadline_exceeded,
model_loading, not_certified, substitution_rejected, artifact_invalid,
engine_crashed, probe_required, migration_required, module_restarted, grammar_disabled.
Transient carries retry_after_ms. Never poll-hammer a permanent code.

## Common request fields (acceptance constraints)

deadline_ms (absolute budget), max_queue_ms (fast-fail bound — admission is
atomic accept-or-reject against it), required_fingerprint,
allow_equivalent (default false: hard substitution rejection),
required_epoch, target_fingerprint (embed.query: route only to a matching or
certified-equivalent profile, else substitution_rejected/not_certified).

## Ops

The management registry in this snapshot is `embed.query`, `embed.batch`,
`embed.result`, `job.resume`, `rerank.score`, `microllm.oneshot`,
`owned_decode.admit_session`, `owned_decode.decode`, `owned_decode.snapshot`,
`owned_decode.continue`, `owned_decode.abort`, `owned_decode.close`,
`owned_decode.session_status`, `owned_decode.disable`, `owned_decode.revoke`,
`model.load`, `model.status`, `model.unload`, `models.list`, `probe.start`,
`probe.status`, `probe.report`, `aliases.check_index`, `alias.retract`,
`alias.declare`, `cache.pin`, `cache.gc`, `admission.status`,
`approvals.migrate_owned_decode`, `approvals.enable`, `approvals.disable`, and
`approvals.emergency_rollback`.

- **embed.query** {model, text, …constraints} — interactive class, the
  latency path. Full envelope incl. fingerprint/equivalent_to (one-comparison
  hot path).
- **embed.batch** {model, items[{id, text}], …} — inline under the byte/item
  budget; over budget → job: {job_id} back immediately.
  - **embed.result** {job_id, cursor?, max_page_bytes?} — committed pages are
    readable while the job is still running; `pages_available` on job status
    is the visible committed-page count. The default page is 512 KiB; page
    order is job-internal (length-sorted execution) — items carry ids, KEY
    WRITES BY ID, never by position. This behavior is identical for local and
    remote jobs. Pages survive restart until the result-retention TTL.
    `request_key` makes resubmission idempotent: the same key and digest attach
    to or resume the same work; a different digest is `idempotency_conflict`.
    After a terminal restart failure, a fresh attempt keeps prior committed
    pages readable and resumes by item id.
- **rerank.score** {model, query, candidates[], …} — RAW per-candidate
  scores (no server-side ordering opinions); interactive ≤20 candidates,
  bulk beyond. Qwen3-Reranker architectures are rejected at load (measured
  broken template path in llama.cpp b9580).
- **microllm.oneshot** {model, prompt, max_tokens, grammar?, …} — greedy.
  `max_tokens` is capped by module config `microllm_max_tokens` (default 512);
  requests above the ceiling are rejected with both numbers in the error.
  `grammar` (GBNF): unset or empty = free-text. When set, constrained
  decoding is permitted only if module config `grammar_enabled` is true (default
  false); otherwise grammar requests fail closed with typed `grammar_disabled`.
  When the constrained runtime is unavailable or not certified, grammar requests
  fail closed with the same reason; the retired `grammar_unavailable_in_build` code
  is not emitted.
- **models.list** — catalog + per-model state, fingerprints, alias rows,
  recommended_batch (ADVISORY batch sizing per model; admission remains the
  enforcement — consumers should not carry their own batch knobs). When present,
  `recommended_batch` is `{ "rows": positive_integer, "token_budget": positive_integer }`;
  `rows` is the advised request-row cap and `token_budget` is the corresponding
  aggregate token cap. The field is omitted when an engine has no measured policy.
- **model.load / model.status, probe.start / probe.status / probe.report** —
  load and probe start/status are job-shaped (poll-first); `probe.report` is a
  query. Probe is EXPLICIT, never auto-triggered; certification + perf rows are
  keyed by machine profile and stamped with os_build/module_generation; the
  report returns the full measured capability table plus per-knob assignments.
  Embed/rerank ops and `microllm.oneshot` routed to an owned-engine lane refuse
  (`not_certified`/`probe_required`) until the (machine, fingerprint) pair is
  certified. Lane-1 worker-backed `microllm.oneshot` routes retain their existing
  dispatch path and do not acquire the owned-lane certification gate.

### Fixed-bucket ANE package sets

An ANE `model.load` manifest may put the primary compiled Core ML package in
`files.model` and additional fixed-sequence packages in ordered `files.extra`
entries. Each entry is a detailed `{url, sha256}` file specification. The
module caches every file separately, then derives one composite artifact digest
from the ordered role set before minting the model fingerprint. The ANE worker
loads all packages and selects the smallest bucket that can hold the longest
item in an `embed.batch`; it never silently truncates an item to a smaller
bucket. `files.tokenizer` remains required because Synapse owns tokenization.

The production worker accepts a zipped `.mlmodelc` bundle for each package.
The archive is only a transport/cache representation; the worker verifies its
per-file digest, materializes it into a temporary compiled-model directory,
and loads it with `CPU_AND_NE`. A package set's placement gate is the minimum
reported Neural Engine share across its buckets.
- **aliases.check_index** {index_fingerprint, provenance_set[]} → valid |
  migration_required {retracted_pair, rebuild_target} + table_epoch. Call at
  write-commit when your index holds mixed provenance. Revocation is never
  retroactive: vectors written under a certified fingerprint stay readable;
  demotion affects the future only.
- **admission.status** — advisory snapshot: queue depths, per-lane
  meeting_deadlines + rolling p50 start-delay, current knob, and
  certification/perf staleness for loaded lanes. The contract lives in
  per-request budgets, not this snapshot.
- **cache.pin / cache.gc** — model cache management (content-addressed,
  shared-lease readers, two-phase GC; GC never deletes under a live reader
  or a foreign pin).

### Decode lane certification (additive)

The shipped decode fixture is selected by model family and dtype. For
`Qwen/Qwen3-0.6B` f16 it contains 20 raw-completion prompts, each with 64 token
IDs generated by the pinned greedy reference implementation. Decode
certification is token-exact across every prompt; there is no numerical
similarity tolerance. An uncertified row includes `blocking_reason` and the
first differing token position, expected/actual token IDs, and the diverging
prompt in its evidence.

`probe.report.result.lanes[]` adds:

- `certification_required`: whether requests on this lane are gated;
- `certification_status`: `certified`, `uncertified`, or `not_required`;
- `certification.status`: `certified` or `uncertified` for a persisted probe
  outcome;
- `blocking_reason`: `token_mismatch`, `fixture_unavailable`, a typed probe
  failure reason, or the existing `probe_required`/platform/quarantine reason.

A failed re-probe replaces the current machine-profile outcome with
`uncertified`, so later owned-lane `microllm.oneshot` calls fail closed with
`not_certified`. This demotion is never retroactive: results produced while the
pair was certified remain valid. Successful probes also write the decode
performance row used by knob assignment: cold-load milliseconds, median
single-stream generated tokens/second over the fixture, and median one-item
latency.

Sampling and all other non-greedy certification semantics are deliberately out
of scope until sampling ships.

### Owned-decode session operations (envelope v2)

Every successful operation below has the management response envelope
`{"result": <payload>}`. Malformed parameters are channel errors with
`invalid_request`; the owned-decode domain refusals below instead use
`{"result":{"module_generation": <u64>, "error": {"code", "class":
"permanent", "safe_to_retry_same_request": false, "message"}}}`. The domain
refusal list is exhaustive for the checks in these handlers; a worker dispatch
can additionally return the normal owned-route errors described by its route.

#### `owned_decode.admit_session`

- Params: `{catalog_fingerprint, caller_id, context_ceiling_tokens,
  generation: {mode: "greedy_top1"}, kv_configuration:
  {block_size_tokens, recurrent_state_grain_tokens}}`. All fields are required
  and unknown fields are rejected. The only accepted generation mode is
  `greedy_top1`; KV block size is one of 256, 512, or 1024 tokens and the
  recurrent-state grain must be nonzero.
- Result payload: `{session_id, catalog_fingerprint, approval_generation,
  reservation: {reserved_embed_rerank_bytes, reserved_artifact_weight_bytes,
  reserved_session_kv_bytes, context_ceiling_tokens}}`.
- Errors: `invalid_request`, `sampling_unsupported`, `artifact_unapproved`,
  `artifact_disabled`, `artifact_revoked`, `artifact_mismatch`,
  `unsupported_machine`, `invalid_context_ceiling`, `insufficient_memory`,
  `incompatible_resident_artifact`, `invalid_kv_configuration`,
  `owned_decode_unavailable`, and `store_failure`.
- Semantics: admission reserves the certified artifact and session KV budget,
  records the serving-session approval generation, and writes no session on a
  refusal. A session can only be admitted against the requested catalog
  fingerprint's current serving approval and matching certification.

#### `owned_decode.decode`

- Params: `{session_id, req_id, prompt, max_tokens, grammar?, deadline_ms?,
  max_queue_ms?}`. All fields before `?` are required; `req_id` is nonempty and
  `max_tokens` is positive. `grammar` is free text when absent or empty.
- Result payload: `{session_id, req_id, frames: [FrameEnvelope...]}`. Each
  frame is envelope-v2: `{protocol: "owned-decode-envelope-v2",
  protocol_version: 2, req_id, session_id, stream_seq, kind, ...}`. `stream_seq`
  starts at 1 and is monotonic. A progress frame has
  `{kind: "progress", progress: {committed_token_ids: [u32...],
  committed_token_count}}`; the IDs are newly committed, while the count is
  cumulative. The terminal is `{kind: "final"|"error", terminal: {req_id,
  session_id, committed_token_count, tokens_emitted, decode_fingerprint,
  processing_fingerprint, runtime_config_digest, worker_generation,
  derived_digest?, terminal_state, decode_mode, speculative_telemetry?}}`.
  The session handler emits serial terminals; `speculative_telemetry` is absent.
- Errors: `invalid_request`, `unknown_session`, `session_still_in_flight`,
  `decode_scheduler_busy`, `request_cancelled`, `worker_protocol_error`,
  `scheduler_boundary_error`, `store_failure`, `owned_decode_unavailable`, and
  normal owned-route errors such as `grammar_disabled`, `queue_full`,
  `deadline_exceeded`, `model_loading`, `artifact_invalid`, and
  `engine_crashed`.
- Semantics: the response carries every frame produced by this invocation. A
  token becomes visible only after its scheduler and durable serving-session
  boundary both commit. The terminal accounts for exactly that committed prefix;
  an emergency revoke returns an `artifact_revoked` error terminal after the
  last committed progress frame rather than retracting tokens.

#### `owned_decode.snapshot`

- Params: `{session_id, position_tokens}`; both fields are required and unknown
  fields are rejected.
- Result payload: `{session_id, retained_kv_session_id, retained_position,
  reused_blocks}`.
- Errors: `invalid_request`, `unknown_session`, `session_still_in_flight`,
  `invalid_kv_configuration`, `invalid_kv_alignment`, `artifact_unapproved`,
  `artifact_disabled`, `artifact_revoked`, `artifact_mismatch`,
  `incompatible_resident_artifact`, `store_failure`, and
  `owned_decode_unavailable`.
- Semantics: snapshots are accepted only while the session is idle, at the LCM
  boundary of its KV block size and recurrent-state grain. The retained state
  is durable and belongs to that session and catalog fingerprint.

#### `owned_decode.continue`

- Params: `{session_id, retained_kv_session_id, req_id?}`; unknown fields are
  rejected.
- Result payload: `{session_id, retained_kv_session_id, retained_position,
  reused_blocks}`.
- Errors: `invalid_request`, `unknown_session`, `session_still_in_flight`,
  `retained_kv_unavailable`, `artifact_unapproved`, `artifact_disabled`,
  `artifact_revoked`, `artifact_mismatch`, `incompatible_resident_artifact`,
  `store_failure`, and `owned_decode_unavailable`.
- Semantics: continuation is only for the matching session's retained state at
  an idle boundary. When `req_id` is supplied, its retained stream prefix must
  match the snapshot position, so continuation neither replays nor skips an
  already committed token.

#### `owned_decode.abort`

- Params: `{session_id, req_id, retain_kv}`; all fields are required and
  unknown fields are rejected.
- Result payload: `{session_id, req_id, abort: "requested", disposition:
  "removed_queued"|"deferred_to_boundary"}`.
- Errors: `invalid_request`, `unknown_session`, `session_still_in_flight`, and
  `owned_decode_unavailable`.
- Semantics: abort is an acknowledgement, not a terminal. The active decode
  observes it at its next committed boundary. If retained, its decode result
  carries `cancelled: {req_id, generation_id, committed_token_count}` beside
  the final error frame; status recovery uses that same committed count.

#### `owned_decode.close`

- Params: `{session_id}`; unknown fields are rejected.
- Result payload: `{session_id, closed: true, unload_artifact}`.
- Errors: `invalid_request`, `unknown_session`, `session_still_in_flight`,
  `routing_error`, `store_failure`, and `owned_decode_unavailable`.
- Semantics: close is idempotent after a successful close, but cannot close an
  active decode. It completes durable serving state and may release the
  artifact only when no active sessions still require it.

#### `owned_decode.session_status`

- Params: `{session_id, req_id}`; both fields are required and unknown fields
  are rejected.
- Result payload: `{session_id, req_id, committed_token_count, state,
  retained_kv_session_id?}`. `state` is either `"in_flight"` or
  `{state: "terminal", terminal_state: "completed"|"aborted"|
  "artifact_disabled"|"artifact_revoked"|"failed"}`.
- Errors: `invalid_request`, `unknown_session`, `unknown_request`, and
  `owned_decode_unavailable`.
- Semantics: use this result after a lost or gapped envelope-v2 frame. It is
  authoritative for the monotonic committed-token count and never revises
  history already committed by a progress frame.

#### `owned_decode.disable`

- Params: `{catalog_fingerprint, reason}`; both fields are required and
  unknown fields are rejected.
- Result payload: `{approval: {schema_revision, catalog_fingerprint,
  certification_record_id, artifact_id, state: "disabled"|"revoked",
  reason, approved_by, approved_at_ms, updated_at_ms, generation,
  semantic_digest}, invalidated_retained_states, active_sessions,
  termination_requested_sessions, unload_artifact}`.
- Errors: `invalid_request` and `store_failure`.
- Semantics: disable fences new session admissions and invalidates retained KV
  states for that catalog fingerprint, but lets an already active decode finish
  normally. A previously revoked serving approval remains revoked.

#### `owned_decode.revoke`

- Params: `{catalog_fingerprint, reason}`; both fields are required and
  unknown fields are rejected.
- Result payload: the same `ServingControlOutcome` shape as
  `owned_decode.disable`, with `approval.state: "revoked"`.
- Errors: `invalid_request` and `store_failure`.
- Semantics: revoke fences new admissions, invalidates retained state, and
  marks active sessions for `artifact_revoked` terminal accounting at their
  next committed boundary. It also requests the in-memory scheduler revoke and
  can unload the artifact when no active session remains.

### Model and alias mutations

#### `model.unload`

- Params: `{model_id}`. `model_id` is required; this handler does not reject
  additional fields.
- Result payload: `{module_generation, model_id, fingerprint, state:
  "unloaded", engine, task}`.
- Errors: `invalid_request`, `model_loading`, and `engine_crashed`.
- Semantics: unloading a ready model removes its loaded engine and owned-decode
  dispatch cache, but keeps the catalog model registered and its cached artifact
  available for later lazy reload.

#### `alias.declare`

- Params: `{left, right, evidence?}`; `fingerprint_a` and `fingerprint_b` are
  accepted aliases for `left` and `right`. Both fingerprints must be nonempty;
  omitted `evidence` becomes `{}`. This handler does not reject additional
  fields.
- Result payload: `{module_generation, changed, table_epoch}`.
- Errors: `substitution_rejected`, `invalid_request`, and `store_failure`.
- Semantics: alias administration must be enabled in module configuration.
  Declaring an already-live pair is idempotent (`changed: false`) and preserves
  its epoch; a new live pair records the evidence and advances `table_epoch`.

#### `alias.retract`

- Params: the same shape as `alias.declare`.
- Result payload: `{module_generation, changed, table_epoch}`.
- Errors: `substitution_rejected`, `invalid_request`, and `store_failure`.
- Semantics: retracting a live pair records its end time and evidence, then
  advances `table_epoch`; retracting a pair that is not live is idempotent and
  leaves the epoch unchanged. Retraction affects future alias reads only, not
  vectors or job pages already produced under the pair.

### Owned-decode approval administration

Approval rows are identified by the pair `(model_id, decode_fingerprint)`, not
by model ID alone. The approval mutation path is a fenced store transaction:
each changed identity receives a monotonic `generation`, and migration compares
its single pinned seed marker before writing (the store epoch/CAS boundary).
These wire operations do not accept a caller-supplied expected epoch. The
server recomputes the row's semantic digest in the same transaction.

#### `approvals.migrate_owned_decode`

- Params: `{seed_revision, schema_revision}`; both fields are required and
  unknown fields are rejected.
- Result payload: `{outcome, seed_revision, rows, marker, rendering}`. Outcomes
  include `applied`, `already_applied`, `invalid_seed`, `unmappable_identity`,
  `duplicate_identity`, and `transaction_failed`; non-applied outcomes leave
  approval rows and the migration marker unchanged.
- Errors: `invalid_request`, `approval_migration_state_corrupt`, and
  `store_failure`.
- Semantics: the one-shot migration validates the pinned seed, resolves every
  seed identity to exactly one eligible catalog model, compares its marker under
  the fence, then writes all approval rows and the marker atomically.

#### `approvals.enable`

- Params: `{model_id, decode_fingerprint, grammar_enabled}`; all fields are
  required and unknown fields are rejected.
- Result payload: `{model_id, decode_fingerprint, enabled, grammar_enabled,
  approved_by, approved_at_ms, semantic_digest, generation}`.
- Errors: `invalid_request`, `operator_identity_unavailable`, and
  `store_failure`.
- Semantics: an authenticated operator identity supplies `approved_by`. Enable
  creates or re-enables exactly the requested identity; a new row starts at
  generation 0 and a re-enable increments its generation. The model ID must
  resolve to exactly one owned-metal-decode generate catalog entry.

#### `approvals.disable`

- Params: `{model_id, decode_fingerprint, reason}`; all fields are required and
  unknown fields are rejected.
- Result payload: `{model_id, decode_fingerprint, enabled: false,
  disabled_reason}`.
- Errors: `invalid_request` and `store_failure`.
- Semantics: disable changes exactly one approval identity. The reason must be
  nonempty; disabling a missing identity or a storage validation failure is
  reported as `store_failure` by this handler.

#### `approvals.emergency_rollback`

- Params: `{reason}`; `reason` is required and unknown fields are rejected.
- Result payload: `{disabled, reason}`.
- Errors: `invalid_request` and `store_failure`.
- Semantics: the reason must be nonempty. One fenced transaction disables every
  owned-decode approval identity across the approval lanes, writes that reason,
  recomputes each changed row digest, and increments each changed row generation.
  This approvals-table rollback does not itself invoke `owned_decode.revoke` or
  terminate already active serving sessions.

## Divergence detector (content_sha256)

Each embed item response echoes content_sha256 = hash of the EXACT text
embedded (post-truncation). It is a DIVERGENCE DETECTOR, not a substitute
key: if the echo differs from your own hash of what you stored, the vector
does not represent your content (truncation or normalization drift) — the
correct consumer behavior is reject + investigate loudly, never adopt the
provider hash.

## Consumer patterns (as agreed)

- AFT: persist envelope fingerprint verbatim + (fingerprint, table_epoch) in
  index headers; check_index at index commit; job-tier cold builds keyed by
  item id with request_key resumption.
- MC: no client batch knob (recommended_batch advisory + admission);
  fingerprint mismatch = hard reject (allow_equivalent stays false until MC
  opts in); content_sha256 mismatch = reject-vector + log-loud; free-text
  oneshot with fail-closed parsing for the dreamer A/B, grammar as a later
  third arm.

## Operational notes

- Crash domain: llama.cpp runs in supervised single-lane workers; engine
  crashes are typed (engine_crashed, transient with budget → quarantine as
  permanent probe_required). The ort lane is isolated from llama crashes
  (soak-proven). Nothing hangs: every request ends in result, typed error,
  or transport closure.
- Health: cached state only; worker liveness/crash-window and probe-row
  staleness ride models.list / probe.report / health detail. ANE worker PING
  refreshes the last Neural Engine placement share used by explicit
  certification probes.
- module_generation on every envelope: if it changes mid-conversation,
  re-poll jobs (prior-generation jobs are terminal module_restarted).


## Additive wire contract v1.1: remote gateway and durable pages

All v1.1 fields are additive. Local responses retain their existing shapes unless
this section explicitly aligns durable-page behavior across both lanes.

### Provenance

Remote responses use the real `EngineIdentity` object and add `remote` as an
optional sibling of `engine`:

```json
"provenance": {
  "engine": {
    "engine": "remote_openai_compatible",
    "version": "<adapter semver>",
    "build_flags": {}
  },
  "remote": {
    "provider": "openai",
    "deployment": "api.openai.com",
    "assurance": "declared"
  }
}
```

`build_flags` is a string-keyed object: it is an empty object or is omitted by
serde when empty, never an array. `remote` is absent for local lanes. No existing
provenance field changes shape. Owned-decode responses may add the optional
`chain_k` provenance field; legacy readers must ignore additive provenance fields
and no request wire field is introduced.

For `assurance: "declared"`, `content_sha256` hashes the exact post-truncation
text Synapse submitted to the provider. In this lane, “embedded” means
“submitted to the provider”; provider-side preprocessing is part of the declared
risk accepted by the caller.

### Request opt-in

`embed.query`, `embed.batch`, `rerank.score`, and `microllm.oneshot` accept
`accept_declared: bool`, defaulting to `false`. Requesting a model whose identity
assurance is declared without `accept_declared: true` fails permanently with
`declared_identity_not_accepted`.

### Durable job status and resume

A credential pause has this status payload:

```json
{
  "state": "paused_needs_reauth",
  "logical_handle": "provider-handle",
  "paused_at_ms": 1780000000000,
  "resume_deadline_ms": 1780086400000,
  "action": "reauth"
}
```

The execution TTL is suspended while paused; `resume_deadline_ms` is the only
active clock. Consumers resume with `job.resume {job_id}` after repairing
credentials, or by resubmitting the same request key and digest. Deadline expiry
transitions durably to terminal `needs_reauth_expired`; result retention starts
at that terminal transition.

Every job status carries `pages_available`. `embed.result` serves any committed
page whose index is below that count while the job is queued, running, paused,
done, or failed. A page commit atomically makes its item results, page metadata,
and incremented visible count observable. Previously committed pages remain
readable after a later failure, including continuity quarantine. This
page-while-running rule applies equally to local and remote jobs.

The request digest is bound at admission to the operation, `synapse_model_id`,
constraints, ordered item ids, and per-item content hashes. Remote-bound digests
also bind `remote_profile_hash` and `logical_handle`; local digests omit those
two fields. Reusing a request key with different bound content is the permanent
`idempotency_conflict` error. Same-key/same-digest resubmission resumes from
committed item ids.

### Provider request ids

When the provider returns no request id, `provider_request_id` is omitted
entirely; it is never `null` or an empty string. Success envelopes and error
payloads carry the final attempt's `provider_request_id`, limited to 128 bytes.
Job pages carry `provider_request_ids: [string]`, one per sub-batch, limited to
16 entries. Missing entries are omitted, and `provider_request_ids` itself is
omitted when the resulting array is empty.

### Stable error-code union and credential mapping

Every error retains `{code, class, retry_after_ms?,
safe_to_retry_same_request}`. The complete remote additions are:

- `declared_identity_not_accepted` — permanent.
- `remote_identity_drift` — permanent; the model is quarantined.
- `provider_unavailable` — transient.
- `provider_protocol_violation` — permanent; the model is quarantined.
- `idempotency_conflict` — permanent.
- `needs_reauth` — permanent-shaped inline rejection naming the logical handle.
- `needs_reauth_expired` — permanent terminal job failure.
- `remote_deployment_changed` — permanent boot or serve refusal.
- `credential_config_invalid` — permanent vault-handle `not_found` or malformed
  configuration; it is never retried.

Remote paths reuse these baseline codes unchanged: `module_restarted` (the
existing post-restart job state), `invalid_request`, and `probe_required`.
Credential mapping is exact: vault `not_found` or malformed data maps to
`credential_config_invalid`; an unreachable or restarting credentials module
maps to transient `provider_unavailable`; `vault_locked` or `needs_reauth`
enters the pause path for jobs and returns inline `needs_reauth` for inline
operations.

## Changelog

### 2026-08-29: complete management-operation coverage

Added the owned-decode session controls, owned-decode approval administration,
`model.unload`, and alias declaration/retraction operations. The snapshot now
records their parameter and result shapes, refusal codes, durable identity and
fencing semantics, and envelope-v2 decode frames.

### Grammar refusal retirement

`grammar_unavailable_in_build` is retired in favor of `grammar_disabled`. For a
constrained `microllm.oneshot` request on a platform or machine without a
certified owned-decode lane, the stable `grammar_disabled` refusal is returned;
when applicable, the original owned-decode refusal remains available as additive
`underlying_owned_decode_refusal_id` provenance. Constrained requests remain
owned-only: they never fall back to llama GBNF or retry unconstrained.

**Consumer notice:** consumers matching the legacy
`grammar_unavailable_in_build` string (ALF sidekicks and the persona plane) must
switch to `grammar_disabled` before deploy; the legacy code is no longer emitted.
