# Synapse decisions log

Running log of design decisions. Entries made while Ufuk is AFK are marked with their
approval state: **decided** (reversible, AFT-approved or within pre-approved scope),
**banked** (recommendation recorded, needs Ufuk's call), **locked** (Ufuk-approved).

## D-001: Module skeleton built on subc-client-rs SDK, not a hand-rolled frame loop

- State: decided (SUBC-blessed, pm_fe84493f: SDK is the path for all new modules;
  quota's hand-rolled loop is historical. SDK verified cancellation-safe at source:
  dedicated reader, per-request spawns.)
- Date: 2026-07-04
- Context: ai-provider-quota (the canonical template) hand-rolls its frame loop, but
  subconscious now ships subc-client-rs (serve() + ModuleHandler trait) implementing the
  same conventions: per-request tokio::spawn, channel-0 control handling, cancellation
  tokens, bind hooks.
- Decision: synapse-module implements ModuleHandler on the SDK. Two-crate split kept:
  synapse-core (pure logic) + synapse-module (wire binary).
- Consequence if reversed: rewrite of the wire binary only; core is wire-agnostic.

## D-002: Storage via HELLO_ACK descriptor + cortexkit-store (alfonso-core pattern)

- State: decided (convention-mandated)
- Date: 2026-07-04
- Context: quota ignores HELLO_ACK.storage (it has no persistent state). Synapse owns a
  vector store, so it needs the real path: deserialize HELLO_ACK.storage into
  StorageDescriptor, open via cortexkit-store, run migrations. Reference:
  alfonso-core/crates/alfonso-core-module/src/main.rs (resolve_storage_descriptor) +
  alfonso-core-store/src/lib.rs (open + migrate).

## D-003: Long model loads exposed as job ops, never blocking requests

- State: decided (SUBC-blessed, pm_fe84493f: poll-first model.load/model.status is the
  blessed shape; no deferred-response primitive planned ever; RequestCtx::emit progress
  is additive-only since the Rust consumer is unary today. Status bodies kept cheap.)
- Date: 2026-07-04
- Context: subc-client-rs has no deferred-response primitive; requests must terminate
  with Response/Error/StreamEnd. Daemon route.bind timeout is 12s; consumer default call
  timeout is 30s; cold model loads run 60s+.
- Decision: explicit job ops (model.load start / status) returning fast
  building/queued/ready states; heavy work inside handle() (spawned per request, wire
  loop stays healthy); nothing heavy in on_bind. Model-load serialization enforced
  inside the module (ManagementSurface has no manifest concurrency field; subc gives a
  32-credit ModuleManaged window).

## D-004: Test gate — nextest for integration from day one

- State: decided (AFT directive)
- Date: 2026-07-04
- Decision: unit = libtest, integration = cargo-nextest, retries = 0, one gate script
  shared by CI and release. Reference: cortexkit/aft scripts/rust-test-gate.sh +
  .config/nextest.toml.

## D-005 (open): Decision #1 — in-house hybrid vs wrapping existing servers

- State: evaluation in progress (empirical bench per AFT's steer)
- Ufuk decides after the tradeoff doc; AFT may green-light direction if evidence is
  one-sided (formal lock still Ufuk's).

## D-006: Shared content-addressed model blob cache

- State: decided (SUBC-blessed as a new family convention, pm_fe84493f)
- Date: 2026-07-04
- Decision: model artifacts live in ~/.local/share/cortexkit/models/ (cross-module,
  content-addressed hash-keyed paths, checksummed TOFU downloads, atomic tmp+rename
  writes, never delete other modules' pins, reproducible-from-network so GC/backup
  treats it as cache). Synapse-private state (metadata, vector store via HELLO_ACK
  sqlite descriptor) stays in ~/.local/share/cortexkit/synapse/. Locking/pinning
  contract formalized when a second consumer module lands.

## D-007: Consumer contract requirements banked from MC (consumer #2)

- State: decided (consumer requirements, folded into API design round)
- Date: 2026-07-04
- Machine-wide admission/queuing for embed calls is a v1 requirement, not a consumer
  convention (AFT live incident: 6 concurrent processes' per-process backoff did not
  compose; buried LMStudio). The module is the single admission point.
- Model fingerprint on every embed response; changes iff vector space changes (runtime
  lanes with different float outputs = different fingerprints); dims explicit; hard
  reject-on-substitution (never silently serve a different identity). input_type
  (query|document) is a request param, not identity. Embed-only consumption is
  first-class (vector store optional, AFT-only). Rerank + structured-output micro-LLM
  shapes reserved in v1 surface. Transient-vs-permanent error typing is a top-level
  field in every error response.
