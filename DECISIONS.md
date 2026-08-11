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

- State: evaluation in progress (empirical bench per AFT's steer); DIRECTION set by
  Ufuk 2026-07-04: "borrow the kernels but own the stack fully so we can have
  optimal control for our use cases" — i.e. native engine layers (MLX, llama.cpp,
  ort) under a fully-owned Rust serving stack; no Python packaging layers adopted.
- The measured matrix still runs: it picks WHICH engine lanes carry which
  workloads (and provides the evidence record), not whether we own the stack.
- Formal lock lands with the tradeoff doc review.

## D-009: Model matrix + speed-vs-energy knob (Ufuk, 2026-07-05)

- Synapse recommends models per user hardware; a user-facing speed-vs-energy
  knob ships later (fan-noise matters; bench shows 14.7-62W spread for the same
  workload across engine/batch configs).
- Model matrix: class (22M / 150M ModernBERT / 600M Qwen3) x format x quant.
- Cross-model quality: public benchmark data (MTEB/CoIR) — no in-house
  retrieval eval for now. Intra-model quant quality: our parity + rank-overlap
  tooling (4-bit DWQ already disqualified as default by rank instability).

## D-008: Pin subc-protocol 0.7.0 + subc-transport 0.3.1; health() carries model state

- State: decided (SUBC directive, 2026-07-04)
- Pin 0.7.0/0.3.1 from day one (lockstep; released after the founding study which
  observed 0.6.0/0.3.0). Adopt the HEALTH system (docs/specs/subc-health.md):
  override ModuleHandler::health() with model-serving state — ok|degraded|failing
  plus detail/metrics (cold load in progress = degraded "loading <model>", GPU
  queue depth as metric). Daemon probes on cadence; restart policy default
  report-only. Pairs with the poll-first job-ops design (D-003).

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

## D-009: Lane 2 (owned runtime) is the primary engine program; Lane 1 frozen as reference

- State: decided (Ufuk, 2026-07-12)
- Date: 2026-07-12
- Context: the two-lane program (D-005) assumed the owned runtime was a months-scale
  bet gated behind the adopt lane. Three days of Lane 2 work produced: one ModelFamily
  seam over four backends (CPU/Accelerate, Metal MPSGraph for three families, CUDA
  MiniLM), parity-exact serving vs frozen ORT references everywhere, MLX-class Metal
  throughput on locked hardware (MiniLM f16 1.03x MLX, Qwen3 f16 0.88x, gte fp32 with
  no incumbent baseline), ~2.6x llama.cpp-CUDA-class directional on a 130W RTX 3060,
  compile-at-load + per-shape package caching, a bounded bucket policy, and three
  optimization branches pruned with clock-matched evidence. Pace, not just numbers,
  drove the decision: construction compresses under agentic parallelism, and there is
  no consumer deadline forcing the adopt seam (AFT/MC cutover is not urgent).
- Decision:
  - Lane 2 is the primary engine program. All new engine build effort lands there.
  - Lane 1 is FROZEN as the reference implementation and fallback: no new investment,
    kept CI-green, kept shippable. It is not deleted — it is the incumbent the
    graduation probe measures against and the instant fallback if a Lane 2 path
    regresses in the field.
  - Cutover is per-workload-per-hardware through the existing certification/alias
    machinery (fingerprint + probe + graduation), never a big-bang engine swap.
    Consumers see fingerprints, not lanes.
  - Sequence: (1) embeddings + reranking nailed on all target hardware classes at
    at-least-LMStudio speed (the user-visible product bar; the internal engineering
    bar stays MLX/llama-direct, which Mac embeddings already meet), (2) decode
    (micro-LLM) as the next big campaign — designed instrumentable from day one
    because the research endgame (mid-thought injection, PAQ) requires owning the
    decode loop regardless, (3) quants + remaining hardware breadth (Vulkan, x86
    CPU floor, DirectML) gradually behind measurement.
  - Footprint is a named driver: the owned Metal/Accelerate path ships zero extra
    engine bytes (OS frameworks) vs Lane 1's llama.cpp worker binaries and the ort
    dylib; Synapse curates a small model set for CortexKit's own use cases rather
    than serving arbitrary GGUFs, which is what makes per-family graph work viable.
- Known gaps accepted at decision time: no decode path, no int4/int8 quants, no
  Vulkan/DirectML, CPU floor is Apple-only (Accelerate), per-family graph work is a
  recurring cost for new models (~mason-day per family at current pace).
- First action: same-harness graduation probe on the locked M1 (owned runtime vs
  llama-server-Metal vs MLX python, identical corpus/token accounting) to convert
  directional ratios into certifiable ones.

## D-010: Owned-decode approvals are stable decisions; evidence is current local state

- State: decided (runtime-bound decode cutover contracts v1)
- Date: 2026-08-11
- Decision: store owned-decode approval rows in the module SQLite store, keyed
  exactly by `(model_id, decode_fingerprint)`. An approval records a human
  decision about one artifact; it never contains a machine profile, a numeric
  profile ID, or a concrete certification row. The row's unkeyed semantic
  digest detects corruption and accidental edits inside the existing
  local-operator trust boundary; it is not a claim of cryptographic resistance
  to a principal that can rewrite the store and recompute the digest.
- Admission: approval structure and digest are load-time checks. Evidence is a
  serve-time check: an enabled approval with no current measured evidence is
  structurally loadable and refuses owned serving. Current evidence must match
  the runtime's `MachineProfile::revisioned_hash()` and positive activation
  epoch, plus the current processing, runtime, worker-path, constraint, and
  gate requirements. This is local profile-and-epoch binding, not physical-host
  identity; a complete copied store on an identical collected profile is out of
  scope. Epochs prevent reuse across observed A-to-B-to-A transitions but do
  not claim to detect an intermediate profile that no module observed.
- Operations: deployment is one replacement binary, one fenced migration, and
  probe re-certification. The interval between deployment and the first
  successful probe is deliberately fail-closed: eligible unconstrained traffic
  uses llama fallback, identity-pinned traffic receives
  `owned_decode_not_certified`, and constrained traffic keeps its existing
  refusal behavior. A supported profile rotation is healed by probe alone;
  approval bytes and the deployed binary must not change. A new
  `decode_fingerprint` remains a distinct artifact and needs explicit approval.
- Rollback: disable one exact approval identity with a reason, or atomically
  disable every owned-decode approval for an emergency rollback. Re-certifying
  evidence never overrides disabled state. The retired profile-keyed
  `disable_profile(machine_profile_hash)` procedure is not an operational
  path.
- Release rule: the fleet transition report is the release record. It must
  contain the unavailable-window measurement, trigger and ledger evidence,
  binary and approval byte identities, clean source/build state, and an
  instrumented zero approval-write result. Release remains blocked until every
  execution-manifest job and every preceding slice pass.
