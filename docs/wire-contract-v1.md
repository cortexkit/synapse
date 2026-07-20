# Synapse wire contract v1 (consumer snapshot)

Status: SNAPSHOT of the additive v1.1 surface as of 2026-07-10.
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
engine_crashed, probe_required, migration_required, module_restarted.
Transient carries retry_after_ms. Never poll-hammer a permanent code.

## Common request fields (acceptance constraints)

deadline_ms (absolute budget), max_queue_ms (fast-fail bound — admission is
atomic accept-or-reject against it), required_fingerprint,
allow_equivalent (default false: hard substitution rejection),
required_epoch, target_fingerprint (embed.query: route only to a matching or
certified-equivalent profile, else substitution_rejected/not_certified).

## Ops

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
  `grammar` (GBNF): unset or empty = free-text. When set, constrained decoding
  runs only if module config `grammar_enabled` is true (default false); otherwise
  typed `invalid_request` naming the gate. If the llama worker build lacks GBNF
  support, grammar requests fail with reason `grammar_unavailable_in_build`.
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
provenance field changes shape.

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
