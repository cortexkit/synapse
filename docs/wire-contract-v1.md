# Synapse wire contract v1 (consumer snapshot)

Status: SNAPSHOT of the implemented surface as of 2026-07-08 (post wave 7).
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
  - **embed.result** {job_id, cursor?, max_page_bytes?} — pages become
    readable AS COMPLETED (streaming-bounded cold builds); default page
    512 KiB; page order is job-internal (length-sorted execution) — items
    carry ids, KEY WRITES BY ID, never by position. Pages survive restart
    until TTL. request_key makes resubmission idempotent (same key → same
    job; after module_restarted terminal → fresh job, old pages readable
    until TTL).
- **rerank.score** {model, query, candidates[], …} — RAW per-candidate
  scores (no server-side ordering opinions); interactive ≤20 candidates,
  bulk beyond. Qwen3-Reranker architectures are rejected at load (measured
  broken template path in llama.cpp b9580).
- **microllm.oneshot** {model, prompt, max_tokens, grammar?, …} — greedy.
  `grammar` is RESERVED: unset = free-text; set → typed rejection until the
  GBNF feature lands (then: constrained decoding, no wire change). The
  current 64-token cap is a v1-dev cap, not a design ceiling — the intended
  ceiling covers multi-hundred-token one-shot classes (dreamer manifests);
  it moves via config, not contract change.
- **models.list** — catalog + per-model state, fingerprints, alias rows,
  recommended_batch (ADVISORY batch sizing per model; admission remains the
  enforcement — consumers should not carry their own batch knobs).
- **model.load / model.status, probe.start / probe.status** — job-shaped
  (poll-first). Probe is EXPLICIT, never auto-triggered; certification rows
  are keyed by machine profile; ops refuse (not_certified/probe_required)
  until the (machine, fingerprint) pair is certified against shipped
  reference vectors with tail-sensitive gates.
- **aliases.check_index** {index_fingerprint, provenance_set[]} → valid |
  migration_required {retracted_pair, rebuild_target} + table_epoch. Call at
  write-commit when your index holds mixed provenance. Revocation is never
  retroactive: vectors written under a certified fingerprint stay readable;
  demotion affects the future only.
- **admission.status** — advisory snapshot: queue depths, per-lane
  meeting_deadlines + rolling p50 start-delay, certification staleness. The
  contract lives in per-request budgets, not this snapshot.
- **cache.pin / cache.gc** — model cache management (content-addressed,
  shared-lease readers, two-phase GC; GC never deletes under a live reader
  or a foreign pin).

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
- Health: cached state only; worker liveness/crash-window and certification
  staleness ride models.list and health detail.
- module_generation on every envelope: if it changes mid-conversation,
  re-poll jobs (prior-generation jobs are terminal module_restarted).
