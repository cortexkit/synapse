# Architecture

## What synapse is

Synapse is a local inference service. It runs as the `synapse` module on the subc
bus (binary `ck-synapse`) and serves embedding, reranking and small-model generation
to other fleet modules. Some engines run inside the module process; the rest run in
supervised worker processes that the module spawns and talks to over local IPC. A
remote gateway can forward work to external providers instead. The same repository
holds a benchmark and research tree (`bench/`) where candidate backends are measured
before they become production engines or workers.

## Major pieces

**`synapse-module`** (`crates/synapse-module`). The subc module and the only process
consumers talk to. It registers the management operations, loads layered config
(`SYNAPSE_CONFIG_PATH`, else `.cortexkit/synapse.jsonc` in the working directory,
else `cortexkit/synapse.jsonc` in the user config dir; when both exist, remote
providers come from the user file),
tokenizes requests, admits jobs through the scheduler, owns the SQLite store (model
cache, durable jobs and result pages, certification and approval rows, aliases), runs
certification probes, and dispatches work to an in-process engine, a worker, or the
remote gateway. Owned-generation logic lives in sibling directories of `src/`:
`owned-decode-routing` (lane selection, ANE split-prefill routing),
`owned-decode-grammar-scheduler` (JSON-schema grammar compilation, decode quantum
scheduling), `owned-decode-certification`, `owned-decode-sidecar` and
`owned-decode-manifests`.

**`synapse-core`** (`crates/synapse-core`). Shared vocabulary for the module and the
workers: engine traits (`EmbedEngine`, `RerankEngine`, `GenerateEngine`), the
sanitized tokenizer, numeric profiles and fingerprints, the machine profile, the
scheduler, the stable error-code contract, response envelopes, and the worker
protocol, framing and transports (UNIX socket and Windows named pipe).

**Owned Metal engine** (`crates/synapse-engine-owned`, macOS). In-process embedding
and reranking on Apple GPUs through MPSGraph (ModernBERT, Qwen3, MiniLM) with a
bounded sequence-bucket ladder. It also holds the direct Metal decode kernels for
Qwen3 and LFM2 (`owned-decode-engine`) and the owned-decode supervisor, protocol and
crash budget (`owned-decode-worker`) used by the decode worker and the module.

**CUDA engine** (`crates/synapse-engine-cuda`). Embedding on NVIDIA GPUs (MiniLM,
ModernBERT, Qwen3) from ported PTX kernels, with a hardware-floor check. The module
does not link it; it runs only inside `ck-synapse-worker-cuda`.

**ORT engine** (`crates/synapse-engine-ort`). In-process ONNX Runtime embedding on
CPU. It is the portable floor on any machine and the parity reference for other lanes.

**Workers.** Each is a separate binary the module spawns, handshakes with, and
supervises:
- `ck-synapse-worker-llama` (`crates/synapse-worker-llama`): llama.cpp for GGUF models.
- `ck-synapse-worker-ane` (`crates/synapse-worker-ane`): a small Rust launcher that
  execs a Swift Core ML worker built by `build.rs`; embeds on the Neural Engine using
  fixed-bucket compiled models.
- `ck-synapse-worker-cuda` (`crates/synapse-worker-cuda`): wraps the CUDA engine;
  `--probe-floor` reports hardware readings before the module commits to it.
- `ck-synapse-worker-decode` (`crates/synapse-worker-decode`): owned Metal token
  generation for Qwen3 and LFM2, driven quantum by quantum by the owned-decode
  supervisor.
- `ane-prefill-sidecar` (`workers/ane-prefill-sidecar`): a Swift package that runs
  fixed-window Qwen3 prefill on the Neural Engine and hands the KV cache to the decode
  worker.

**Remote gateway** (`crates/synapse-module/src/remote`). Sends work to external
OpenAI-compatible providers through `ProviderRuntime` pools with circuit breakers and
latency estimators. Credentials come from the vault over the subc `claustrum` route;
the HTTP client refuses redirects and checks loopback rules before sending.

**`synapse-opctl`** (`crates/synapse-opctl`, binary `ck-synapse-opctl`). Operator CLI
that calls the module's management operations over the subc daemon: model status,
probes, admission stats, approvals and rollback, batch submission, paged results.
`crates/synapse-module/src/bin/subc_call.rs` is a lower-level tool that sends one raw
method call to any module.

**Bench tree** (`bench/`). `bench/harness` (`synapse-bench`) builds corpora, wraps runs
with power telemetry behind an idle gate, and computes parity against the CPU ORT
reference. `bench/rig` (`synapse-rig`) drives a candidate as a subprocess over framed
stdio so timing and token accounting are measured outside the candidate. `bench/lanes`
holds one runner per backend under test (Rust crates, Python and JS scripts);
`bench/spikes` holds research prototypes, `unified-rt` being the largest;
`bench/campaign` holds controller scripts that build and score a candidate tree in a
sandbox against pinned fixtures; `bench/eval-coir` scores retrieval and rerank
quality. `bench/run-matrix.sh` and `bench/run-night.sh` run the lanes in sequence.

## Code map

| Area | Path | What is there |
| --- | --- | --- |
| Module entry and ops | `crates/synapse-module/src/lib.rs` | `dispatch_request`, `management_operations`, handlers, config |
| Store | `crates/synapse-module/src/store.rs` | SQLite schema, migrations, cache, jobs, certification, approvals |
| Approvals rollback | `crates/synapse-module/src/rollback.rs` | disable and emergency rollback |
| Core ML materialization | `crates/synapse-module/src/ane_artifact.rs` | digest-keyed extraction of Core ML bundles |
| Worker host | `crates/synapse-module/src/worker_host/mod.rs` | spawn, handshake, requests, restarts |
| Remote gateway | `crates/synapse-module/src/remote/` | provider pools, HTTP client, vault |
| Decode routing | `crates/synapse-module/owned-decode-routing/` | lane choice, ANE prefill router |
| Grammar and decode scheduler | `crates/synapse-module/owned-decode-grammar-scheduler/` | schema → token constraint, quantum scheduling |
| Engine traits | `crates/synapse-core/src/engine.rs` | `EmbedEngine`, `RerankEngine`, `GenerateEngine`, `TokenBatch` |
| Tokenizer | `crates/synapse-core/src/tokenizer.rs` | sanitized tokenization, digests, truncation disclosure |
| Fingerprints | `crates/synapse-core/src/fingerprint.rs` | `NumericProfile`, `Fingerprint`, aliases |
| Machine profile | `crates/synapse-core/src/machine_profile.rs` | profile fields and hash |
| Scheduler | `crates/synapse-core/src/scheduler.rs` | `QueueClass`, aging arbitration |
| Error codes | `crates/synapse-core/src/error_contract.rs` | `StableErrorCode` |
| Worker protocol | `crates/synapse-core/src/worker_protocol.rs`, `worker_transport/` | messages, HELLO, socket and pipe transports |
| Worker names | `crates/synapse-core/src/worker_engine_names.rs` | HELLO identities, binary names |
| Owned Metal embed | `crates/synapse-engine-owned/src/` | MPSGraph models, bucket policy in `runtime.rs` |
| Owned decode supervisor | `crates/synapse-engine-owned/owned-decode-worker/src/` | supervisor, crash budget, streaming |
| CUDA kernels | `crates/synapse-engine-cuda/src/port/` | PTX kernel sources |
| Bench harness | `bench/harness/src/` | metrics, parity, results schema |
| Bench rig | `bench/rig/src/main.rs` | candidate supervisor |

## Request flow

1. A consumer calls an operation on the `synapse` route. `dispatch_request` in
   `crates/synapse-module/src/lib.rs` matches the method name.
2. The module resolves the model or alias from the catalog, tokenizes the input with
   the sanitized tokenizer, and applies per-row token ceilings.
3. Admission: the request enters the scheduler under a `QueueClass` (interactive,
   bulk, decode, control). Batches over the inline budget become a durable job in the
   store and the caller gets a `job_id` back; a `request_key` makes resubmission
   idempotent.
4. The lane is checked against the store: the model is fetched or verified in the
   content-addressed cache, and lanes that need certification refuse with
   `not_certified` unless evidence exists for the current machine profile (owned
   generation lanes also need an enabled approval).
5. The work runs on the chosen lane: an in-process engine (owned Metal, ORT), a worker
   over the worker protocol, or the remote gateway.
6. Results are committed in pages as the job runs. Callers read them with
   `embed.result`; pages are readable while the job is still running and survive a
   restart until the retention TTL. Items carry ids and page order is not item order.

## Rules that bite

- **Serving needs certification and approval.** An owned-generation lane serves only
  with an enabled approval row and current certification evidence; certified-but-not-
  approved lanes refuse. Certification is keyed by the machine profile hash, which
  covers OS build, arch, chip, RAM class, ANE subtype and engine identities, so an OS
  update or engine change rotates the hash and requires an explicit re-probe
  (`probe.start`). Probes never run automatically.
- **Changing the model changes the fingerprint.** `NumericProfile::fingerprint` hashes
  the model digest, quant and numeric profile id; the profile includes engine,
  tokenizer digest, pooling, normalization, dtype and the certified shape
  (`max_context_tokens` and batch limits). Change any of them and consumers must
  re-embed stored vectors.
- **The module owns tokenization.** Engines and workers receive token ids
  (`TokenBatch`), never text. Do not add a tokenizer inside an engine or worker.
- **Workers are processes, not threads.** They connect back over a UNIX socket or a
  Windows named pipe and must pass a HELLO with the launch nonce and the expected
  engine identity from `worker_engine_names.rs`. A worker crash, hang or bad handshake
  becomes an error and a restart or quarantine; it must never take the module down.
- **Core ML bundles are materialized once, at a digest-keyed stable path.** Use
  `materialize_core_ml_artifact`; never extract to a per-load temp directory. Core ML
  caches its compiled specialization by path, so a new path pays the full compile
  again.
- **Metal crates need full Xcode.** `synapse-engine-owned` compiles Metal shaders with
  `xcrun metal`. If `xcrun` points at the Command Line Tools, build with
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- **The store is SQLite in WAL mode and refuses a newer schema.** It is opened through
  `cortexkit-store` (WAL). If the recorded schema version is newer than the
  migrations this binary carries, `open` fails with `SchemaAheadOfBinary` instead of
  serving; migrations are not additive-only, so a rollback binary cannot run on a
  migrated store.
- **The subc wire crates move together, at exact versions.** The subc and commons
  crates come from crates.io. `subc-daemon` is pinned with `=`, because a caret
  requirement can resolve a newer daemon built on a different protocol minor and leave
  two copies of `subc-protocol` in the lock. After any bump, check that the lock holds
  one version of each `subc-*` crate.

## Longer pages

- Wire contract and error codes: [`docs/wire-contract-v1.md`](docs/wire-contract-v1.md)
- Module design: [`docs/design-synapse-module.md`](docs/design-synapse-module.md)
- Worker protocol: [`docs/design-worker-protocol.md`](docs/design-worker-protocol.md)
- Worker supervision as built: [`docs/explain/worker-supervision.md`](docs/explain/worker-supervision.md)
- Remote gateway: [`docs/design-remote-gateway.md`](docs/design-remote-gateway.md)
- Placement profiles: [`docs/design-placement-profiles.md`](docs/design-placement-profiles.md)
- Runtime decision: [`docs/decision-1-runtime.md`](docs/decision-1-runtime.md)
- Serving gates audit: [`docs/audits/meaning-serving-gates.md`](docs/audits/meaning-serving-gates.md)
- Measured evidence: [`docs/evidence/`](docs/evidence/)
- Decisions log and founding constraints: [`DECISIONS.md`](DECISIONS.md), [`FOUNDING.md`](FOUNDING.md)
- Bench rig: [`bench/rig/RIG.md`](bench/rig/RIG.md); campaigns: [`bench/campaign/README.md`](bench/campaign/README.md)
- ANE worker packaging: [`crates/synapse-worker-ane/README.md`](crates/synapse-worker-ane/README.md)
