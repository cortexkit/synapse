# Synapse Remote Endpoint Gateway — design r1 (draft)

Status: DRAFT r1 — pre-review. Owner: Synapse. 2026-07-10.
Reviewers wanted: SUBC (credentials + surface), AFT/MC (scope questions only,
async — both are busy; nothing here blocks on them until implementation).

## Why

Synapse's founding charter is local inference PLUS a gateway for remote AI
endpoints, one surface via subc. The local half shipped in v1. The gateway
half is what lets consumers delete their own provider code: MC still carries
an OpenAI-compatible client + retry/circuit-breaker stack of its own, AFT
wraps LMStudio/Ollama. Every consumer-owned provider stack duplicates
credential handling, error classification, and rate-limit lore that Synapse
already owns for local engines.

## Scope (v1 of the gateway)

- One adapter kind: **openai-compatible** (covers OpenAI, Azure-OpenAI-shaped,
  LMStudio, Ollama, vLLM, together.ai, fireworks, etc.). Anthropic-native and
  others are v-next adapters behind the same seam.
- Ops routed remote: `embed.query`, `embed.batch`, `rerank.score` (where the
  provider offers it), `microllm.oneshot` (stateless one-shots only — the
  llm-runner scope line holds: conversation/session/tool semantics stay in
  llm-runner; Synapse serves stateless capability inference, local or remote).
- Job tier + paging work identically for remote batches (durable jobs,
  request_key idempotency, pages readable as completed).

Non-goals v1: streaming token output (subc consumer call() is unary; poll-first
jobs are the shape), provider failover/aggregation (a request targets one
declared provider-model), cost accounting (bank the envelope field, see below).

## Identity and fingerprints for remote models

The fingerprint contract's job — "changes iff the vector space changes" —
cannot be *verified* for a remote endpoint the way we certify local engines
(no artifact digest, no runtime config we control, provider can silently swap
weights). The honest contract:

- Remote model identity: `remote:<provider-name>:<provider-model-id>` plus
  dims. Fingerprint string embeds provider name, model id, dims, and the
  adapter kind — **declared identity, not measured identity**.
- Envelope `provenance: "remote"` on every response (consumers already read
  provenance; local lanes say e.g. "ort-cpu"). MC/AFT store fingerprints
  verbatim, so a provider-side silent swap is INVISIBLE to us — this is an
  inherent property of remote serving, stated loudly in the wire contract.
  Consumers choosing remote accept re-embed risk that local lanes eliminate.
- Probe for remote models: reachability + auth + dims + self-consistency
  (same input twice → cosine ≈ 1.0) + latency sample. Produces a cert row of
  a distinct class `declared` (vs `measured` for local). `probe.report` shows
  the class so the CK app can render remote entries honestly.
- Hard substitution guarantee unchanged: requested remote model unavailable →
  typed rejection, never silent fallback to another model, local or remote.

## Credentials (vault integration)

FromVault pattern, copied from llm-runner (studied day one):

- Handles file: `~/.config/cortexkit/synapse/vault-handles.json`
  `{"handles": {"apikey:openai": "ckh_..."}}`; env override
  `SYNAPSE_VAULT_HANDLES_PATH` (test hook, same rule as SYNAPSE_CONFIG_PATH).
- Second outbound subc consumer connection to `cortexkit-credentials`:
  poll catalog.list until live → route.open management_surface →
  `credential.get {handle, min_ttl_ms: 600_000, force_refresh?}` →
  VaultToken. Tokens live in memory only, never logged, never in config.
- Error mapping: `needs_reauth` → typed RESUMABLE pause error to the consumer
  naming the handle (consumer surfaces "reconnect provider X" UX; job parks).
  All other vault errors (not_found/vault_locked/corrupt/...) fail admission
  as permanent config errors.
- `report_auth_failure {handle, provider_status}` ONLY on actual provider
  401/403, never on 5xx/429, reported post-loop. Vault owns OAuth refresh.

## Resilience contract (the part consumers get to delete)

Per provider endpoint:

- Timeout tiers: connect (5s) / read (60s embed, 120s generate) — config
  overridable per provider.
- Bounded retry with jittered backoff on transient classes (connect errors,
  5xx, 429-with-retry-after honored); NO retry on 4xx-permanent (400 shape
  errors, 401/403 auth, 404 model). The lesson MC paid hours for: transient
  vs permanent is explicit in OUR response envelope; consumers never infer.
- Circuit breaker per (provider, model): consecutive-failure trip → open for
  a cooldown → half-open probe. While open: typed `provider_unavailable`
  (transient class) immediately, no queue pileup.
- Rate-limit backpressure: 429s reduce per-provider concurrency (AIMD-style),
  recovering slowly. Provider concurrency cap in config, default conservative.

## Admission and scheduling

Remote ops do not burn local GPU/CPU, so they bypass the local compute
classes: a fourth scheduler class REMOTE_IO with its own concurrency budget
(per-provider caps beneath it). Machine-wide admission still applies to
queue depth and job counts (durable job table is shared), so a remote batch
storm cannot starve the job store, but it never contends with local engine
quanta. Predicted-finish math uses measured per-provider latency (probe +
rolling window) instead of token-throughput models.

## Config surface (synapse.jsonc)

```jsonc
{
  "remote_providers": [
    {
      "name": "openai",                     // unique, lowercase
      "kind": "openai_compatible",
      "base_url": "https://api.openai.com/v1",
      "vault_handle": "apikey:openai",       // key into vault-handles.json
      "max_concurrency": 4,
      "connect_timeout_ms": 5000,
      "read_timeout_ms": 60000,
      "models": [
        {
          "model_id": "text-embedding-3-small",
          "task": "embed",
          "dims": 1536,
          "max_batch": 128,                  // becomes recommended_batch
          "max_input_tokens": 8191
        }
      ]
    }
  ]
}
```

No secrets in config. deny_unknown_fields applies. Models appear in
models.list with provenance "remote", the declared cert class, and
recommended_batch from provider limits.

## Envelope additions

- `provenance: "remote"` (existing field, new value).
- `provider_request_id` (optional string): provider's own request id when
  returned — consumers debugging provider issues need it; costs nothing.
- RESERVED, unset in v1: `cost` (object) — provider-reported usage/cost for
  future accounting. Typed-rejected if a consumer sends expectations on it.

## Health

Remote provider state follows the cached-state rule (SUBC fleet invariant):
the circuit-breaker state machine and last-probe results are stamped into
health from the request path's own bookkeeping — never a live provider call
on the dispatch path. Breaker-open providers surface as degraded with the
provider name and cooldown remaining.

## Open questions (async, non-blocking for design review)

1. **MC**: which remote providers do you actually run today besides
   LMStudio/Ollama wraps? Day-1 provider config presets worth shipping?
2. **AFT**: any remote-rerank appetite, or is rerank local-only for you?
3. **SUBC**: second consumer connection to cortexkit-credentials — any
   liveness/reconnect lore beyond llm-runner's implementation worth stealing?
   And: is a REMOTE_IO scheduler class visible anywhere in manifest metadata,
   or purely module-internal (my assumption: internal)?
4. **Ufuk**: LMStudio/Ollama — v1 ships them as openai_compatible presets
   (localhost base_url, no vault handle needed → `vault_handle: null` allowed
   for localhost-only providers)? That makes the wrap-embed bench lane's
   lessons (pre-truncation, 400-under-load retry) part of the adapter.

## Implementation shape (post-review)

Wave A: adapter trait + openai_compatible impl (embed only) + vault client +
resilience stack + REMOTE_IO class + config/catalog/models.list + probe
`declared` class + e2e against a local mock provider (mockito-style) and a
real LMStudio if present (skip-guarded, hollow-green-proof).
Wave B: rerank + microllm one-shots + provider_request_id + health/breaker
surfacing + soak (breaker trip/recovery under fault injection).
Two waves, each ~1 day at current pace.
