# Synapse Remote Endpoint Gateway — design r2

Status: r2 — post-adversarial-review rewrite. Owner: Synapse. 2026-07-10.
r1 review: Oracle, 20 findings (F1-F20), 8 design-breaking. All folded in
below; each section notes the findings it resolves. Items needing consumer
sign-off are marked [CONSUMER] and collected at the end — none block r2
review, all block Wave A ship.

## Why (unchanged)

Synapse's founding charter is local inference PLUS a gateway for remote AI
endpoints, one surface via subc. The gateway half is what lets consumers
delete their provider code: credential handling, error classification,
retry/breaker stacks, rate-limit lore. MC still carries all of that; AFT
wraps LMStudio/Ollama by hand.

## Scope

- One adapter kind v1: `openai_compatible` — plain OpenAI-shaped REST only.
  Azure-OpenAI is OUT of v1 claims (path/query-version/auth-header rules
  need their own tagged variant; F16). Anthropic-native etc. are v-next
  variants behind the same seam.
- Ops routed remote: `embed.query`, `embed.batch`, `rerank.score`,
  `microllm.oneshot` (stateless one-shots; llm-runner scope line holds).
- Wave A ships embed only, and only for remote models with a
  **Synapse-owned tokenizer + input profile** (F5): we pre-truncate below
  the provider limit with our own tokenizer, submit exact text, and hash
  exactly what we submitted — `content_sha256` and truncation disclosures
  stay truthful. Remote models whose input handling we cannot own are out
  until the wire can express `input_assurance: unknown` (v1.1, [CONSUMER]).

Non-goals v1: streaming, provider failover/aggregation, cost accounting
(NO reserved `cost` field — removed per F20; an accounting contract comes
first), Azure.

## Identity: declared assurance, opt-in, drift-quarantined (F1, F2, F3)

Remote identity cannot be measured. r2 makes that a first-class, opt-in,
enforced property instead of a weaker fingerprint:

- **Canonical remote numeric profile** (hashed into the fingerprint):
  `{adapter_kind, adapter_semver_major, provider_deployment_id,
  provider_model_id, identity_revision, task, dims, input_profile_id}`.
  - `provider_deployment_id`: operator-declared stable deployment identity
    (for plain OpenAI: the provider account host, e.g. "api.openai.com";
    for self-hosted: the operator's name for that deployment). base_url is
    TRANSPORT, not identity — but changing base_url to a different
    deployment REQUIRES an identity_revision bump; Synapse detects base_url
    changes under an unchanged profile at boot and refuses to serve until
    the operator bumps `identity_revision` or attests continuity in config
    (`same_deployment: true` one-shot acknowledgment).
  - `identity_revision`: operator-managed integer; bumping it is the
    operator's statement "the vector space may have changed" → new
    fingerprint, consumers re-embed.
  - Display `name` and transport URL are NOT in the hash (renames don't
    churn fingerprints; deployment moves do).
- **Assurance class**: every remote model carries
  `identity_assurance: "declared"` (locals are `"measured"`). Surfaced in
  models.list, probe.report, and every envelope (placement per wire section
  below). **Consumers must opt in per request** (`accept_declared: true`)
  to receive embeddings from a declared-identity model; absent opt-in →
  typed permanent rejection `declared_identity_not_accepted`. Default
  behavior is thus exactly the v1 strict contract. [CONSUMER: field name +
  their default posture]
- **Sentinel drift check**: at probe time Synapse embeds the shipped
  fixture corpus subset and PERSISTS those vectors keyed by remote profile
  hash. A scheduled/manual re-probe re-embeds sentinels; cosine drift
  beyond the certification gate → **quarantine the remote profile**
  (typed `remote_identity_drift`, permanent until operator bumps
  identity_revision). Drift detection is a tripwire, not a guarantee — the
  wire contract says so explicitly. A silent same-dim swap between probes
  is undetectable until the next sentinel check; consumers accepted that
  when they opted in.
- **Routing** (F3): every remote model gets an operator-chosen **Synapse
  alias** (`synapse_model_id`, globally unique across local+remote,
  validated at boot). `provider_model_id` never keys the catalog. Remote
  models are NEVER eligible for knob/default assignment or measured-
  equivalence aliases (F6); explicit selection only.

## Certification rows (F6)

New column `assurance_class: measured | declared` on certification rows.
Declared rows key on `(remote_profile_hash, identity_revision)` — NOT
machine profile (OS updates don't invalidate reachability; deployment
changes do). Declared rows can never satisfy a measured gate, never seed
alias-table entries, never participate in `equivalent_to`. Serving a
declared model requires a live declared cert row (reachability + auth +
dims + sentinel baseline + latency sample recorded).

## Wire surface (F7, F8)

No invented enum values, no shape changes to frozen fields:

- `provenance` keeps its object shape. Remote adds a valid EngineIdentity:
  `{engine: {kind: "remote_openai_compatible", version: <adapter semver>}}`
  plus new OPTIONAL sibling field `remote: {provider: <name>,
  deployment: <provider_deployment_id>, assurance: "declared"}`. Optional
  fields are additive per the snapshot's serde-skip rule; consumers that
  don't read `remote` see a valid v1 envelope. [CONSUMER: confirm their
  deserializers tolerate the optional field — serde/TS both should]
- **Error matrix is published in the wire doc as exact JSON** before Wave A.
  New stable error codes (`declared_identity_not_accepted`,
  `remote_identity_drift`, `provider_unavailable`, `idempotency_conflict`,
  `needs_reauth`) ship as a **wire v1.1 additive evolution with explicit
  consumer ack** [CONSUMER], not silent enum growth. Classes stay
  transient/permanent only — no new class. Reauth is NOT an error class:
  see job states.
- `provider_request_id`: returned on the FINAL attempt only, both on
  success envelopes and error payloads (errors are where it earns its
  keep), bounded 128 bytes (F20).

## Durable jobs: pause, resume, idempotency (F9, F10, F11)

- New job state `paused_needs_reauth` (nonterminal), carrying
  `{logical_handle, paused_at_ms, resume_deadline_ms, action: "reauth"}`
  in job_status — a state with an action descriptor, not an error.
  - Bounded: `resume_deadline_ms` (config, default 24h). Deadline passes →
    durable terminal transition to `failed` with permanent
    `needs_reauth_expired`. No zombie occupancy (F11).
  - Module restart: prior-generation paused jobs become `module_restarted`
    (existing contract preserved); same request_key resubmission resumes
    from checkpoints (below). No new restart semantics invented.
- **TTL separation** (F11): `execution_ttl` (admission → terminal, existing
  semantics) vs `result_retention_ttl` (starts at terminal transition).
  Purge only consults retention TTL for terminal rows. A job finishing
  near its execution deadline gets full retention for its pages.
- **request_key binds a canonical request digest** (F10): sha256 over
  (op, synapse_model_id, remote profile hash, constraints, ordered item
  ids, per-item content hashes). Same key + different digest → typed
  permanent `idempotency_conflict`. Same key + same digest → attach to
  existing job (current behavior, now safe).
- **Checkpointed pages** (F10): remote batch jobs persist each completed
  sub-batch as a readable page BEFORE dispatching the next (this also
  closes the gap vs the snapshot's "pages readable as completed" promise —
  current local implementation commits pages at completion; fix ships with
  this wave for both lanes). Resume (post-reauth or post-restart
  resubmission) re-dispatches only ids without persisted results.
  At-most-once upstream billing is impossible without provider-side
  idempotency; the wire doc states double-submit-on-ambiguous-timeout as
  a property of remote serving. [CONSUMER: FYI line, no decision needed]

## Credentials (F12, F13, F14)

- **One supervised lazy vault client**, shared by all remote paths, built
  at first remote admission (never during module init — a credentials
  outage can never affect local serving). Reconnect with backoff; route
  reopen on route_gone. Classification: `not_found`/malformed handle →
  permanent config error; credentials module unreachable/restarting →
  transient `provider_unavailable`-family (bounded retry then breaker);
  `vault_locked`/`needs_reauth` → pause path (jobs) or typed rejection
  naming the handle (inline).
- **Per-attempt acquisition** (F13): tokens are fetched AFTER queue
  admission, immediately before each outbound sub-batch, with
  `min_ttl_ms = attempt_timeout + 60s margin` (not a fixed 10min); never
  one token for a whole job. Vault lock mid-job: already-issued tokens are
  used until expiry (vault owns revocation semantics; we don't cache past
  expiry), next acquisition pauses the job.
- **Trust boundary** (F14 — security): `remote_providers`, base_url, and
  vault_handle bindings are legal ONLY in the user-tier config
  (`~/.config/cortexkit/synapse.jsonc`) or the SYNAPSE_CONFIG_PATH test
  override. If the project-tier file contains `remote_providers` → boot
  fails loud with a typed config error naming the field and the rule
  (project config can reference user-declared synapse_model_ids in future
  fields, never define endpoints/credentials). This is the founding
  project-tier-URL-trust rule applied here.

## Endpoint security (F15)

- `auth: "none"` (replaces vault_handle:null): allowed only for loopback
  IP LITERALS (`127.0.0.0/8`, `::1`) — hostname "localhost" is rejected
  (DNS/hosts tricks), redirects disabled, final connection target
  re-validated post-connect. Anything else with auth:none → boot error.
- Vault-backed providers require `https` base_url. URL userinfo and query
  strings carrying secrets are rejected at parse. Response bodies capped
  (config, default 32 MiB). Redirects disabled globally for API calls.

## Provider runtime: one state machine, both tiers (F17, F18, F19)

A single shared `ProviderRuntime` per configured provider, used by inline
and job execution alike:

- **Resource pools, not a queue class**: remote work gets its own pool —
  `remote_max_jobs`, `remote_max_queued_bytes` — RESERVED alongside (not
  competing with) local budgets, plus a global emergency cap. Local job
  admission is unaffected by remote saturation and vice versa. Within a
  provider: weighted interactive-vs-bulk aging (reusing the scheduler's
  aging math), permits released between sub-batches so a big batch cannot
  monopolize a provider while queries wait (F17).
- **Breaker**: provider-level transport breaker + per-model quarantine
  (protocol violations quarantine the model, not the provider). Single
  half-open LEASE: one probe attempt total across both tiers, no retry
  inside the half-open attempt, epoch-checked transitions so a stale
  worker can't flap state (F19). Retry budget is shared per (provider,
  request): a request's retries stop when the breaker opens.
- **Classifier presets**: transient/permanent classification lives in a
  per-adapter preset table; `lmstudio` preset marks 400-under-load
  transient with bounded retry (bench-measured behavior), generic preset
  keeps all 400s permanent. 429 handling is provider-wide pacing (AIMD
  concurrency + honored Retry-After), never per-request retry storms.
- **Deadline admission** (F18): deadline semantics frozen and tested as
  ABSOLUTE-BY-MODULE (module converts request deadline_ms to absolute at
  admission; core compares absolute — one interpretation, e2e-pinned).
  Remote predicted-finish uses per-(provider, op, size-bucket) conservative
  quantile (p90) INCLUDING censored timeouts, current queue depth,
  concurrency state, open-breaker cooldown remaining, and Retry-After
  holds. Remaining budget re-checked before every retry; attempts canceled
  at deadline. A breaker-open provider fails admission immediately
  (typed transient) instead of queueing doomed work.

## Health (F20 — moved to Wave A)

Cached-state rule holds: ProviderRuntime stamps breaker state, last
sentinel-check outcome, credential-pause counts, and rolling latency
summary into the health snapshot from its own bookkeeping. Breaker-open →
degraded naming provider + cooldown remaining. No live provider calls on
the dispatch path, ever.

## Config surface (F16)

Tagged, nested-strict, evolution-ready:

```jsonc
{
  "remote_providers": [            // USER-TIER CONFIG ONLY (see trust)
    {
      "kind": "openai_compatible", // serde tag; azure/anthropic are future kinds
      "name": "openai",            // display + logs; NOT identity
      "deployment_id": "api.openai.com",  // identity component
      "identity_revision": 1,      // operator bumps = vector space may differ
      "base_url": "https://api.openai.com/v1",
      "auth": { "kind": "vault", "handle": "apikey:openai" },
      // or       { "kind": "none" }   — loopback literals only
      "max_concurrency": 4,
      "timeouts": { "connect_ms": 5000, "embed_read_ms": 60000,
                    "generate_read_ms": 120000 },
      "classifier_preset": "generic",   // or "lmstudio"
      "models": [
        {
          "task": "embed",              // tagged capability variant
          "synapse_model_id": "openai-te3-small",   // global unique alias
          "provider_model_id": "text-embedding-3-small",
          "dims": 1536,
          "input_profile": "cl100k_base@8191",  // Synapse-owned tokenizer ref
          "max_batch": 128
        }
      ]
    }
  ]
}
```

Every nested struct is `deny_unknown_fields` (typo test per level). Dims
requested explicitly where the API supports it and **enforced on every
response**: exact item count, exact index permutation, uniform dims,
finite values, body cap — violation quarantines the model and returns
typed `provider_protocol_violation`, never a vector envelope (F4).

## Open consumer items (collected; none block r2, all block Wave A ship)

1. `accept_declared` opt-in field name + default posture (MC, AFT).
2. Wire v1.1 additive evolution ack: new stable error codes + optional
   `remote` provenance sibling (MC, AFT).
3. FYI: double-submit-on-ambiguous-timeout property; page-checkpoint fix
   also improves local job paging (MC, AFT).
4. Day-1 provider presets worth shipping (MC); remote-rerank appetite (AFT).
5. LMStudio/Ollama as auth-none loopback presets (Ufuk — carve-out is now
   loopback-literal-only, so presets are safe by construction).

## Waves (re-scoped per review)

- **Wave A** (embed only): adapter + ProviderRuntime (pools, breaker,
  classifier presets, deadline estimator), vault client, declared
  identity/cert rows/sentinels, config + trust boundary + URL policy,
  page checkpointing (both lanes), health stamping, request-digest
  idempotency, paused_needs_reauth + TTL split, mock-provider e2e
  (protocol-violation, drift-quarantine, reauth-pause/resume/expiry,
  half-open lease, idempotency-conflict, project-config-rejection,
  starvation, latency-spike admission) + LMStudio skip-guarded live test.
- **Wave B**: rerank + microllm one-shots, remote-side soak (fault
  injection: breaker trip/recovery, vault outage mid-job, sentinel drift).

Estimate honestly revised: Wave A is 3+ mason-days equivalent (durable-job
migration + wire evolution + security boundary are each real); consumer
acks gate ship, not start — mock-provider work can proceed once r2 passes
re-review.
