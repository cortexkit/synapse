# Synapse Remote Endpoint Gateway — design r4

Status: r4 — folds the third-pass closure review (F2/N1 and F6/N4 closed
there; this rev closes the six residuals it listed). Owner: Synapse.
2026-07-10. History: r1 (20 findings, 8 DB) → r2 (8 resolved, 10 partial,
2 papered-over per second pass) → r3 (start-blocker closures) → r4. Sections cite the findings
they close. [CONSUMER] items are collected at the end; they gate Wave A
SHIP, not start.

## Why (unchanged)

Founding charter: local inference PLUS a gateway for remote AI endpoints,
one surface. The gateway lets consumers delete their provider stacks:
credentials, error classification, retry/breaker, rate-limit lore.

## Scope

- v1 adapter: `openai_compatible` (plain OpenAI-shaped REST). Azure and
  Anthropic-native are future tagged variants. No streaming, no failover,
  no cost accounting.
- Ops: embed (Wave A), rerank + microllm one-shot (Wave B). Stateless only.
- Trust statement (N1): writers of the user-tier config are in the trusted
  computing base — they already control vault-handles.json and the module
  binary path. Identity rules below defend against ACCIDENTS (config
  drift, copy-paste, provider-side swaps), not against a hostile user-tier
  writer. Project-tier config is UNTRUSTED (see Trust boundary).

## Identity (F1, F2, F3, N1)

- **Canonical remote profile, frozen bytes**: the fingerprint hashes
  `remote-profile-v1` + canonical JSON (sorted keys, no whitespace) of
  `{adapter_kind, adapter_semver_major, provider_deployment_id,
  provider_model_id, identity_revision, task, dims, input_profile_id}`.
  `remote_profile_hash` is defined ONCE as this hash; identity_revision is
  inside it (never a separate key component; N4-redundancy fix).
- **URL transitions (N1)**: the `same_deployment` attestation carve-out is
  REMOVED. Boot compares each provider's normalized base_url against the
  persisted `(remote_profile_hash → last_base_url)` row; any change under
  an unchanged profile → typed boot error naming the provider; the only
  path forward is bumping `identity_revision` (new fingerprint, consumers
  re-embed). Renames stay free (name is not identity); URL moves always
  cost a revision. No one-shot state machine to wedge. URL-binding rows
  are immutable and live independently of certification rows (a cert
  invalidation or re-probe never touches the binding; the row is removed
  only when its profile hash no longer appears in config after a
  config-removal grace sweep).
- **Assurance + opt-in (F1)**: `identity_assurance: declared` on every
  remote model; requests must carry `accept_declared: true` to receive
  declared vectors, else typed permanent `declared_identity_not_accepted`.
  Strict v1 semantics are the default posture. [CONSUMER #1]
- **Routing (F3)**: operator-chosen `synapse_model_id`, globally unique
  across local+remote, boot-validated. Remote models never enter knob or
  default assignment, never participate in measured equivalence/aliases.

## Input assurance (F5)

For declared models, `content_sha256` is the hash of the exact
**post-truncation text Synapse submitted**. The wire contract states this
verbatim for `assurance: declared` responses: "embedded" means "submitted
to the provider"; provider-side preprocessing is part of the declared risk
a consumer opts into. [CONSUMER #2] Eligibility hardening at cert time:
the probe submits a deliberately over-limit input and requires a 4xx
rejection — a provider that silently truncates instead is INELIGIBLE for
declared certification (`input_handling: silent_truncate` → refusal).
Synapse pre-truncates with its own tokenizer (`input_profile_id`) below
the provider limit, so the rejection path never fires in normal serving.

## Declared certification vs runtime health (F6, N4)

Two concepts, two stores:

- **Identity certification** (persisted, deployment-scoped, keyed by
  `remote_profile_hash` alone): dims verified via enforced response
  validation, protocol capability (batch shape, over-limit rejection),
  sentinel baseline vectors, calibration variance (below). Machine
  profile is NOT in the key; OS updates never invalidate it.
- **Runtime health** (machine-local, in-memory ProviderRuntime state,
  never persisted as certification): reachability, auth state, breaker
  state, rolling latency. Serving requires BOTH a live identity cert and
  a healthy runtime (breaker closed, credentials available).

Declared cert rows carry `assurance_class: declared` and can never satisfy
a measured gate, seed aliases, or appear in `equivalent_to` (F6, held).

## Sentinel calibration and drift (N3)

- At certification: the sentinel subset is embedded **5 times**; the
  calibration floor is the minimum pairwise self-cosine across runs. If
  the floor is below `1 - (1 - drift_gate)/2` (i.e. self-noise eats more
  than half the drift budget), declared certification is REFUSED —
  provider too nondeterministic to monitor honestly.
- Drift checks (scheduled or `probe.start`): re-embed sentinels, compare
  against the persisted baseline. One failing run → `suspect` (serving
  continues; health surfaces the flag) + an immediate independent
  confirmation run with fresh calls. Confirmed → quarantine with typed
  permanent `remote_identity_drift` (operator exits via identity_revision
  bump). Single noisy calls can no longer force consumer re-embeds.
- Drift gate derives from calibration: `drift_gate = calibration_floor -
  (1 - calibration_floor)` (i.e. twice the observed self-noise distance),
  capped at 0.9999 for numerical headroom when a provider is perfectly
  deterministic — NOT the local measured-lane gates (those were designed
  for deterministic comparisons). Edge rules:
  - Certification is REFUSED when `calibration_floor <
    (1 + drift_gate_min) / 2` (the derived gate would fall below the
    config minimum `drift_gate_min` — self-noise too high to monitor).
  - Any zero-norm, NaN, or non-finite sentinel vector during calibration
    → refusal (`provider_protocol_violation` reason).
  - `drift_gate_min` config changes after certification apply at the
    NEXT drift check: gates are recomputed from the stored
    calibration_floor and current config; existing certs are not
    invalidated by a config edit.

## Wire contract v1.1 (F7, F8, F20)

Exact shapes, published in docs/wire-contract-v1.md as an additive v1.1
section before any implementation hardens; consumer ack gates ship
[CONSUMER #3]:

- Provenance (real `EngineIdentity` shape — `{engine, version,
  build_flags}`):
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
  (`build_flags` is a string-keyed object in the real EngineIdentity —
  empty object or serde-omitted when empty, never an array.)
  `remote` is an optional sibling; absent for local lanes. No existing
  field changes shape.
- New request fields: `accept_declared: bool` (embed/rerank/microllm ops).
- New job state: `paused_needs_reauth` with status payload
  `{state: "paused_needs_reauth", logical_handle, paused_at_ms,
  resume_deadline_ms, action: "reauth"}`.
- Complete new stable error codes (exhaustive; F8):
  `declared_identity_not_accepted` (permanent),
  `remote_identity_drift` (permanent),
  `provider_unavailable` (transient),
  `provider_protocol_violation` (permanent, model quarantined),
  `idempotency_conflict` (permanent),
  `needs_reauth` (permanent-shaped inline rejection naming the handle),
  `needs_reauth_expired` (permanent, terminal job),
  `remote_deployment_changed` (permanent; boot/serve refusal surface),
  `credential_config_invalid` (permanent; vault handle not_found or
  malformed — config errors, never retried).
  Baseline codes that remote paths REUSE unchanged (not new, listed for
  the union): `module_restarted` (job state after a restart, existing v1
  contract), `invalid_request`, `probe_required`. Vault error mapping is
  exact: not_found/malformed → `credential_config_invalid`; credentials
  module unreachable/restarting → `provider_unavailable` (transient);
  vault_locked/needs_reauth → pause path (jobs) or `needs_reauth`
  (inline).
- `provider_request_id` when the provider returns none: the field is
  OMITTED entirely (never null, never empty string); page-level
  `provider_request_ids` omits missing entries and is itself omitted when
  empty.
- Page availability: pages become readable WHILE THE JOB RUNS
  (`pages_available` on status; `embed.result` serves committed pages for
  running jobs). This fixes the local lane to match the snapshot's
  "readable as completed" promise too — one behavior, both lanes.
- `provider_request_id` placement: success envelope and error payload
  carry the final attempt's id (≤128 bytes); job PAGES carry
  `provider_request_ids: [string]` (one per sub-batch in the page,
  bounded 16). [CONSUMER #3 includes this]

## Durable jobs: checkpoints, pause, resume (F9, F10, F11, N2)

- **Store split — attempts vs results**: new table
  `remote_checkpoints(request_digest, item_id, result BLOB, page_no,
  provider_request_id, committed_at)` — immutable, keyed by request
  digest, NOT by job id. Job rows are attempts that REFERENCE a digest.
  Fresh admission with the same request_key/digest no longer deletes
  pages: prior pages remain readable (retention TTL applies), the new
  attempt resumes from committed item_ids.
- **Atomic page commit (N2)**: one SQLite transaction writes item
  results + page metadata + increments the job's visible page count.
  Crash → committed ids are skipped on resume, an uncommitted partial
  page is re-dispatched (double upstream submission on ambiguous timeout
  is disclosed as a property of remote serving [CONSUMER #4]); local
  results can never be missing or duplicated.
- **Request digest binds auth identity (N2)**: digest = sha256 over
  (op, synapse_model_id, remote_profile_hash, logical_handle,
  constraints, ordered item ids, per-item content hashes). Same key +
  different digest → `idempotency_conflict`.
- **Continuity is checkpoint-driven, not state-driven**: ANY attempt that
  would append to a NONEMPTY checkpoint set for its digest runs the
  sentinel continuity check first — regardless of how the prior attempt
  ended (reauth pause, module_restarted, crash). Restart cannot erase the
  need for the check because the trigger is the checkpoints' existence,
  not remembered pause state. Drift at continuity → quarantine, prior
  pages stay readable, no mixing.
- **Single-owner resume (CAS)**: the job row carries
  `active_attempt_id`; `job.resume` and same-digest resubmission both
  claim ownership via compare-and-swap on it (SQLite conditional update).
  Exactly one claimant dispatches; the loser attaches to the winner
  (status polling) instead of double-dispatching.
- **Pause semantics (F9, F11)**: entering `paused_needs_reauth` RELEASES
  execution slots and queued-byte budget; the job retains only its row +
  digest + checkpoints. Wakeup paths: explicit `job.resume {job_id}`
  (consumer calls it after fixing auth) or same-key/same-digest
  resubmission. Execution TTL is suspended while paused;
  `resume_deadline_ms` (default 24h) is the only clock, enforced lazily
  on any status/resume touch AND by the existing periodic purge sweep —
  expiry is a durable terminal transition to `failed
  needs_reauth_expired`, then retention TTL owns the row. Restart:
  prior-generation paused jobs → `module_restarted` (unchanged contract);
  resubmission resumes from checkpoints.

## Credentials (F12, F13 — held from r2)

Lazy supervised vault client (first remote admission, never module init);
per-attempt token acquisition post-admission with
`min_ttl_ms = effective_attempt_timeout + 60s` where effective attempt
timeout is capped by the remaining request deadline; reacquire after any
pacing/backoff wait. Classification: not_found/malformed → permanent;
credentials module unreachable → transient; vault_locked/needs_reauth →
pause (jobs) / typed rejection (inline).

## Trust boundary and config layering (F14)

The current first-file-wins loader cannot express "project may tune knobs
but never touch providers", so the loader changes:

- `remote_providers` is a PRIVILEGED field: read EXCLUSIVELY from the
  user-tier file (`~/.config/cortexkit/synapse.jsonc`) or
  SYNAPSE_CONFIG_PATH. If the project-tier file contains it → typed boot
  error naming the field and the rule.
- When the project-tier file wins general selection, the user-tier file
  is STILL loaded for privileged fields only. (Two reads, one merge
  point, boot-tested both ways.)

## Endpoint security (F15, N6)

- `auth: {kind: "none"}`: allowed only when the URL host parses (typed
  parser result, not string matching) to canonical IPv4 loopback
  (127.0.0.0/8) or exact IPv6 `::1` — zone-qualified (`::1%en0`),
  unspecified (`0.0.0.0`, `::`), and IPv4-mapped (`::ffff:127.0.0.1`)
  forms are all rejected; boot-tested each.
- Environment proxies bypassed for gateway calls (no_proxy semantics
  enforced in the client build); redirects disabled globally. Peer
  validation (connect → getpeername → verify → send) applies to
  `auth: none` endpoints ONLY — it enforces the loopback rule. Vault-
  backed endpoints are protected by mandatory https/TLS, not peer
  checks.
- Vault-backed providers: https only, no URL userinfo, no query secrets,
  response body cap (default 32 MiB).

## Provider runtime (F17, F18, F19, N5)

- Pools: `remote_max_jobs` + `remote_max_queued_bytes` reserved alongside
  local budgets; invariant `local_reserved + remote_reserved <= global
  cap` validated at config load.
- **Interactive reservation (F17)**: providers with `max_concurrency >= 2`
  permanently reserve 1 permit for the interactive class. Bulk sub-batches
  are SIZED to a latency target (`target_subbatch_ms`, default 10s, from
  the rolling estimate) so bulk permits recycle on that cadence — worst-
  case interactive wait is one sub-batch, not one read timeout. At
  `max_concurrency: 1` there is NO hard interactive isolation: sizing
  bounds expected (not worst-case) wait, and the only guarantee is strict
  interactive-first priority at every permit turnover. The config docs
  state this plainly and default shipped presets to `max_concurrency: 2`
  minimum; operators choosing 1 accept bulk-behind-interactive latency.
- Breaker (held from r2): provider transport breaker + per-model
  quarantine, single half-open lease across tiers, epoch-checked
  transitions, shared retry budget. Explicit rule (F19): 429 feeds AIMD
  pacing only — it never increments transport-breaker failure counts.
- **Deadline estimator (F18, N5)**: absolute-by-module semantics frozen
  and e2e-pinned. Fully specified:
  - Size buckets by total input tokens: ≤1k, ≤8k, ≤64k, >64k.
  - Window: per (provider, op, bucket) ring buffer of 256 samples;
    samples expire after 30 minutes.
  - p90 = nearest-rank on the sorted live window.
  - Hierarchy: (1) bucket p90 when ≥8 live samples; (2) merged-op p90
    (all buckets pooled) when ≥8; (3) configured cold estimate
    (`cold_estimate_ms` per op — defaults: embed 15s, rerank 15s,
    generate 30s).
  - Censored timeouts never enter the sample set as latencies; they set
    a conservative FLOOR: estimate = max(chosen estimate, MAX unexpired
    censored duration in the bucket's window) — maximum, not
    most-recent, so a later short censor can never shrink the floor.
  Sustained failure is the breaker's job, not the estimator's. Admission
  against an open breaker fails immediately (transient), never queues.

## Health (F20 — held)

ProviderRuntime stamps breaker state, suspect/drift state, credential
pauses, and latency summary into health from its own bookkeeping. Breaker
open → degraded naming provider + cooldown. No live calls on dispatch.

## Config (F16 — held from r2, plus r3 fields)

Tagged provider variants, nested `deny_unknown_fields` with per-level typo
tests. r3 additions: `cold_estimate_ms` (per op), `target_subbatch_ms`,
`drift_gate_min`, `resume_deadline_ms`. `same_deployment` does not exist.
Schema example unchanged from r2 otherwise (auth tagged {kind: vault,
handle} | {kind: none}; models tagged by task; synapse_model_id required).

## Consumer items (ship gates, not start gates)

1. `accept_declared` opt-in: field name + default posture (MC, AFT).
2. Declared `content_sha256` = submitted-text semantics (MC pin owner).
3. Wire v1.1 additive ack: provenance.remote sibling, paused state, page-
   while-running availability, full error-code list, provider_request_id
   placement (MC, AFT).
4. Double-submit-on-ambiguous-timeout disclosure (FYI).
5. Day-1 presets (MC); remote-rerank appetite (AFT); LMStudio/Ollama
   loopback presets (Ufuk).

## Waves

- **Wave A start-safe now** (per review): mock HTTP provider harness,
  no-redirect/no-proxy client with peer validation, response validator
  (count/permutation/dims/finite/body-cap), classifier preset table +
  tests, adapter request/response parsing.
- **Wave A remainder** (after this r3 passes re-review): store migration
  (checkpoints, TTL split, paused state), wire v1.1 section, declared
  cert + calibration + drift, ProviderRuntime, config layering + trust
  boundary, vault client, e2e battery (drift-quarantine, reauth-pause/
  resume/expiry incl. restart+rotated-credential+half-written-page, half-
  open lease, idempotency-conflict, project-config rejection, starvation,
  cold-start admission, loopback parser table).
- **Wave B**: rerank + microllm, remote soak with fault injection.

Estimate: Wave A remainder 3+ mason-days after re-review; page-while-
running fix lands with the store migration (benefits local lane too).
