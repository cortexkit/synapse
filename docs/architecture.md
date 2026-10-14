# Architecture

Synapse is a local inference module on the subc bus. Consumers use the `synapse`
route for embedding, reranking and generation. The module owns tokenization,
admission, identity, durable jobs and result pages. Inference runs in an owned
engine or a supervised worker; configured remote profiles use the gateway.

## Module boundary

`crates/synapse-module/src/lib.rs` registers and dispatches management operations.
Shared types, tokenization, fingerprints, scheduling and IPC live in
`crates/synapse-core`. `synapse-opctl` is the operator client. The root
[code map](../STRUCTURE.md) lists implementation locations; the
[wire contract](wire-contract-v1.md) is the consumer-facing API.

Engines receive composed token IDs rather than re-tokenizing text. Numeric
profiles bind model and tokenizer digests, engine identity, readout, normalization,
precision and shape limits. Changing vector-space semantics changes the
fingerprint; callers must not treat different fingerprints as interchangeable.
An alias changes routing, not the underlying fingerprint.

## Catalog and legacy lanes

Catalog installation is explicit through `models.download`; serving does not
fetch missing files. Catalog lanes verify cached members, load lazily and perform
sealed self-checks. A failed check stays failed for its identity key. Catalog lane
IDs use `<catalog-id>-<backend>` and are separate from legacy preload IDs.
Moving an index between them requires deliberate identity handling.

Legacy probe-gated lanes retain their certification requirements. Owned-generation
serving additionally needs an enabled approval row. Backend source code existing
in the workspace does not mean that backend is runnable on a host or published
in a release; consult `models.catalog` and the release matrix.

There is no automatic CPU embedding fallback. A host without a runnable backend
gets a typed refusal rather than an undisclosed execution change.

## Requests and durable state

The module resolves the requested model, applies token policy and admits bounded
work. Requests exceeding inline limits become durable jobs. The SQLite store
holds job identity, committed item results and page metadata. Request keys bind
content and operation identity: resubmission can resume work, but changed content
cannot reuse the same key. Read result pages by item ID rather than assuming
page order is input order.

The store uses WAL and refuses a schema newer than the running binary understands.
A rollback is not permission to open a migrated store with an older schema.

## Process and hardware boundaries

Workers connect over a Unix socket or Windows named pipe and must pass a nonce,
protocol-version and engine-identity handshake. The host owns restart and fault
accounting; a worker fault must not terminate the module. Completed-work progress,
caller deadlines and detached drain have separate roles, described in the
[worker protocol](worker-protocol.md).

The direct Neural Engine lane has shared admission and residency ownership.
Leases prevent eviction during an exchange; resources are refunded only after
acknowledged release or confirmed process exit. See [ANE serving](ane-serving.md).
Core ML materialization uses digest-keyed stable paths so compiled specialization
can be reused rather than extracted to a new path on every load.

## Remote boundary

The gateway uses declared provider identity, input assurance, bounded retries and
circuit-breaker state. Declared identity requires caller opt-in; credentials and
HTTP transport failures have distinct recovery behavior. Redirects are refused.
Remote requests use the same durable item/page discipline as local requests;
provider preprocessing is not represented as local numerical certification.
The additive remote fields and recovery codes are in the wire contract.

## Documentation and evidence

Public documentation is flat and topic-named under `docs/`. Canonical certification
JSON remains tracked beneath `docs/evidence/certification/` because validation and
release tooling consume it. Historical records are not current certification:
schema, source, model, row and artifact validation still apply. Internal research,
audits, investigations and operational records are not public release inputs.
