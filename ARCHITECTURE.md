# Architecture

## What synapse is

Synapse is a local inference service. It runs as the `synapse` module on the subc
bus (binary `ck-synapse`) and serves embedding, reranking and small-model generation
to other fleet modules. Embedding and reranking run only on synapse's own engines:
Metal and the Neural Engine on Apple hardware, CUDA and Vulkan on Linux and Windows.
There is no CPU embedding lane and no remote gateway: a machine with no runnable
backend gets `backend_unavailable`, and modules that want hosted inference call their
provider directly. Some engines run inside the module process; the rest run in
supervised worker processes that the module spawns and talks to over local IPC. The
repository also holds a benchmark and research tree (`bench/`).

Models come from two places:
- **The catalog**, compiled into the binary (`crates/synapse-module/src/catalog/models.json`):
  ModernBERT (gte-modernbert-base, gte-reranker-modernbert-base) and Qwen3 0.6B
  (Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B), each pinned per backend with its
  fingerprint. Installs are explicit (`models.download`); a catalog lane checks
  itself on the device at first load and needs no probe or approval.
- **Legacy lanes**, loaded from config preloads or `model.load`. They keep their
  existing ids, fingerprints and probe/approval gates, because stored vectors depend
  on them.

## Major pieces

**`synapse-module`** (`crates/synapse-module`). The subc module and the only process
consumers talk to. It registers the wire operations, loads config
(`SYNAPSE_CONFIG_PATH`, else `.cortexkit/synapse.jsonc` in the working directory,
else `cortexkit/synapse.jsonc` in the user config dir), tokenizes requests, admits
jobs through the scheduler, owns the SQLite store (catalog installs and self-checks,
durable jobs and result pages, certification and approval rows, aliases), runs
legacy certification probes, and dispatches work to an in-process engine or a
worker. It also answers `diagnostics.findings` for unattended health checks, runs
the timed Neural Engine pause (`lanes.pause`), and writes its backup descriptor for
Engram at startup. Owned-generation logic lives in sibling directories of `src/`:
`owned-decode-routing`, `owned-decode-grammar-scheduler`,
`owned-decode-certification`, `owned-decode-sidecar` and `owned-decode-manifests`.

**`synapse-core`** (`crates/synapse-core`). Shared vocabulary for the module and the
workers: engine traits (`EmbedEngine`, `RerankEngine`, `GenerateEngine`), the
sanitized tokenizer, numeric profiles and fingerprints, the machine profile, the
scheduler, the stable error-code contract, response envelopes, the worker protocol,
framing and transports (UNIX socket and Windows named pipe), the child-process spawn
helper and the `ckdev-` test-binary helper.

**`synapse-native-probe`** (`crates/synapse-native-probe`). The few unsafe native
calls the machine profile needs on Windows, kept out of `synapse-core`.

**Owned Metal engine** (`crates/synapse-engine-owned`, macOS). In-process embedding
and reranking on Apple GPUs through MPSGraph (ModernBERT, Qwen3, MiniLM) with a
bounded sequence-bucket ladder. It also holds the direct Metal decode kernels for
Qwen3 and LFM2 (`owned-decode-engine`) and the owned-decode supervisor, protocol and
crash budget (`owned-decode-worker`) used by the decode worker and the module.

**CUDA engine** (`crates/synapse-engine-cuda`). Embedding and reranking on NVIDIA
GPUs (MiniLM, ModernBERT, Qwen3) from ported PTX kernels, with a hardware-floor
check. The module does not link it; it runs only inside `ck-synapse-worker-cuda`.

**Workers.** Each is a separate binary the module spawns, handshakes with, and
supervises. Which ones a release ships is decided by the release workflow:
- `ck-synapse-worker-ane-direct` (`crates/synapse-worker-ane-direct`): ModernBERT and
  Qwen3 on the Neural Engine through Apple's private framework, compiling one
  program per transformer layer per padded length. Serves the `-ane` catalog lanes.
- `ck-synapse-worker-ane` (`crates/synapse-worker-ane`): a small Rust launcher that
  execs a Swift Core ML worker built by `build.rs`; embeds on the Neural Engine with
  fixed-bucket compiled models. Serves legacy Core ML lanes.
- `ck-synapse-worker-vulkan` (`crates/synapse-worker-vulkan`): owned Vulkan compute
  shaders for embedding and reranking; it loads the Vulkan loader at runtime and
  refuses software rasterizers.
- `ck-synapse-worker-cuda` (`crates/synapse-worker-cuda`): wraps the CUDA engine;
  `--probe-floor` reports hardware readings before the module commits to it.
- `ck-synapse-worker-decode` (`crates/synapse-worker-decode`): owned Metal token
  generation for Qwen3 and LFM2, driven quantum by quantum by the owned-decode
  supervisor.
- `ck-synapse-worker-llama` (`crates/synapse-worker-llama`): llama.cpp generation for
  GGUF models. The serving allow-list never routes embedding or reranking to it.
- `ane-prefill-sidecar` (`workers/ane-prefill-sidecar`): a Swift package that runs
  fixed-window Qwen3 prefill on the Neural Engine and hands the KV cache to the decode
  worker.

**Certification.** `crates/synapse-certify` holds the release record format and its
validator; `ck-synapse` exposes only `certify validate` and `certify source`. The
live runner is `ckdev-synapse-certify` (`crates/synapse-certify-runner`), a
development binary that is never shipped, because it links the subc daemon.
`crates/synapse-release-checks` holds the Python gates the tag workflow runs over
prebuilt release assets.

**`synapse-opctl`** (`crates/synapse-opctl`, binary `ck-synapse-opctl`). Operator CLI
over the module's operations: lanes (pause, resume, status), models, probes,
admission stats, approvals and rollback, batch submission and paged results.
`crates/synapse-module/src/bin/subc_call.rs` sends one raw method call to any module.

**Bench tree** (`bench/`). `bench/parity` (`synapse-parity`, its own workspace) holds
the model manifest, the fp32 reference fixtures, their generator and the parity
evaluator that certification and catalog self-checks grade against, plus the
`parity-manifest` tool that converts checkpoints into engine packages.
`bench/harness` (`synapse-bench`) builds corpora, wraps runs with power telemetry and
computes parity. `bench/rig` (`synapse-rig`) drives a candidate as a subprocess over
framed stdio so timing and token accounting are measured outside it. `bench/lanes`
holds one runner per backend under test, `bench/spikes` research prototypes,
`bench/campaign` sandboxed campaign controllers, and `bench/eval-coir` retrieval and
rerank quality scoring.

## Code map

| Area | Path | What is there |
| --- | --- | --- |
| Module entry and ops | `crates/synapse-module/src/lib.rs` | `dispatch_request`, `management_operations`, handlers, config |
| Model catalog | `crates/synapse-module/src/catalog/` | `models.json` pins, backend declarations, manifest digests |
| Store | `crates/synapse-module/src/store.rs` | SQLite schema, migrations, engine allow-list, jobs, certification, approvals |
| Worker host | `crates/synapse-module/src/worker_host/mod.rs` | spawn, handshake, requests, restarts, quarantine |
| Worker liveness | `crates/synapse-module/src/worker_host/liveness.rs` | no-progress limits |
| Direct ANE residency | `crates/synapse-module/src/worker_host/ane_residency/` | compile scheduler, executable budget, eviction |
| Neural Engine pause | `crates/synapse-module/src/lane_pause.rs` | timed pause of the ANE lanes |
| Diagnostics | `crates/synapse-module/src/diagnostics_findings.rs` | `diagnostics.findings` |
| Engram descriptor | `crates/synapse-module/src/engram_catalog.rs` | backup catalog written at startup |
| Approvals rollback | `crates/synapse-module/src/rollback.rs` | disable and emergency rollback |
| Core ML materialization | `crates/synapse-module/src/ane_artifact.rs` | digest-keyed extraction of Core ML bundles |
| Decode routing | `crates/synapse-module/owned-decode-routing/` | lane choice, ANE prefill router |
| Engine traits | `crates/synapse-core/src/engine.rs` | `EmbedEngine`, `RerankEngine`, `GenerateEngine`, `TokenBatch` |
| Tokenizer | `crates/synapse-core/src/tokenizer.rs` | sanitized tokenization, digests, truncation disclosure |
| Fingerprints | `crates/synapse-core/src/fingerprint.rs` | `NumericProfile`, `Fingerprint`, aliases |
| Machine profile | `crates/synapse-core/src/machine_profile.rs` | profile fields and hash |
| Scheduler | `crates/synapse-core/src/scheduler.rs` | `QueueClass`, aging arbitration |
| Error codes | `crates/synapse-core/src/error_contract.rs` | `StableErrorCode` |
| Worker protocol | `crates/synapse-core/src/worker_protocol.rs`, `worker_transport/` | messages, HELLO, socket and pipe transports |
| Worker names | `crates/synapse-core/src/worker_engine_names.rs` | HELLO identities, binary names |
| Owned Metal embed | `crates/synapse-engine-owned/src/` | MPSGraph models, bucket policy in `runtime.rs` |
| Direct ANE worker | `crates/synapse-worker-ane-direct/src/` | ModernBERT and Qwen3 graphs, shape compile, packing |
| CUDA kernels | `crates/synapse-engine-cuda/src/port/` | PTX kernel sources |
| Parity and references | `bench/parity/` | model manifest, fp32 fixtures, evaluator |

## Request flow

1. A consumer calls an operation on the `synapse` route. `dispatch_request` in
   `crates/synapse-module/src/lib.rs` matches the method name.
2. The module resolves the model or alias (an omitted embed or rerank model uses the
   task's catalog default), tokenizes the input with the sanitized tokenizer, and
   refuses rows over a catalog lane's 8,192-token limit with `sequence_too_long`
   before admission.
3. Admission: the request enters the scheduler under a `QueueClass` (interactive,
   decode, bulk, control). Batches over the inline budget become a durable job in the
   store and the caller gets a `job_id` back; a `request_key` makes resubmission
   idempotent.
4. The lane is made ready. Serving never downloads: a catalog model that isn't
   installed is refused with `model_not_installed`. The module verifies the installed
   files, loads the lane and runs its self-check on first use. Legacy lanes need
   current certification; owned generation also needs an enabled approval.
5. The work runs on the chosen lane: an in-process engine, or a worker over the
   worker protocol.
6. Results are committed in pages as the job runs. Callers read them with
   `embed.result`; pages are readable while the job is still running and survive a
   restart until the retention TTL. Items carry ids and page order is not item order.

## Rules that bite

- **Production fingerprints never change.** `NumericProfile::fingerprint` hashes the
  model digest, quant and numeric profile; the profile includes engine identity,
  tokenizer digest, pooling, normalization, dtype and the certified shape. Consumers
  such as AFT store vectors under that fingerprint, so changing any input forces a
  full re-embed. A faster kernel or graph ships as a new catalog lane, never as a
  change to a live one. Empty optional profile fields are left out of the hashed bytes.
- **Catalog self-checks are keyed to the machine.** The key is the catalog id,
  manifest digest, backend, fingerprint, engine identity, OS build and fixture
  revision. A failed check persists and the lane refuses. After an OS update, startup
  re-runs the checks of lanes that passed before. Catalog lanes never answer
  `probe_required` or `not_certified`.
- **Legacy lanes need certification and approval.** Certification is keyed by the
  machine profile hash (OS build, arch, chip, RAM class, ANE subtype and engine
  identities), so an OS update rotates it and the lane refuses until an operator
  re-probes (`ck-synapse-opctl probe run`). Health reports degraded meanwhile.
  Probes never run on their own. Owned generation also needs an enabled approval row.
- **The module owns tokenization.** Engines and workers receive token ids
  (`TokenBatch`), never text. Do not add a tokenizer inside an engine or worker.
- **Workers are processes, not threads.** They connect back over a UNIX socket or a
  Windows named pipe and must pass a HELLO with the launch nonce, the expected engine
  identity and the same protocol version as the module, so a protocol bump deploys the
  module and every worker together. A request is alive while the worker sends
  progress frames: 120 s without progress during inference, or 600 s during load and
  compile, is a hang. Two consecutive faults quarantine a lane, and only a completed
  serving request clears the count. A worker fault must never take the module down.
- **The direct Neural Engine is one machine-wide resource.** Its lanes share a
  cross-process lock and one budget of 100 compiled layer programs. Apple's compiler
  handles one layer at a time for the whole machine, so the scheduler runs one cold
  compile at a time. Compiled programs are reused across restarts only because the
  worker gives each program a stable identifier. Never let test or profiling daemons
  bypass the lock, and never redirect `TMPDIR` for a process that compiles.
- **Core ML bundles are materialized once, at a digest-keyed stable path.** Use
  `materialize_core_ml_artifact`; never extract to a per-load temp directory. Core ML
  caches its compiled specialization by path, so a new path pays the full compile
  again.
- **Flow-scoped routes are read-only diagnostics.** A route stamped with a `flow_id`
  may call only `diagnostics.findings`; every other method answers
  `flow_scope_refused`.
- **Metal crates need full Xcode.** `synapse-engine-owned` compiles Metal shaders with
  `xcrun metal`. If `xcrun` points at the Command Line Tools, build with
  `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- **The store is SQLite in WAL mode and refuses a newer schema.** It is opened through
  `cortexkit-store`. If the recorded schema version is newer than the migrations this
  binary carries, `open` fails with `SchemaAheadOfBinary`; migrations are not
  additive-only, so a rollback binary cannot run on a migrated store.
- **The subc wire crates move together, at exact versions.** The subc and commons
  crates come from crates.io. `subc-daemon` is pinned with `=`, because a caret
  requirement can resolve a newer daemon on a different protocol minor and leave two
  copies of `subc-protocol` in the lock. After any bump, check that the lock holds one
  version of each `subc-*` crate. `ck-synapse` must not link `subc-daemon`.

## Longer pages

- Wire contract and error codes: [`docs/wire-contract-v1.md`](docs/wire-contract-v1.md)
- Module architecture: [`docs/architecture.md`](docs/architecture.md)
- Worker protocol and supervision: [`docs/worker-protocol.md`](docs/worker-protocol.md)
- Direct Neural Engine serving: [`docs/ane-serving.md`](docs/ane-serving.md),
  [`docs/design-ane-compile-scheduling.md`](docs/design-ane-compile-scheduling.md),
  [`docs/ane-graph-layout.md`](docs/ane-graph-layout.md)
- Configuration: [`docs/configuration.md`](docs/configuration.md)
- Development and verification: [`docs/development.md`](docs/development.md)
- Reference generation: [`docs/reference-generation.md`](docs/reference-generation.md)
- Catalog token identity: [`docs/catalog-token-identity.md`](docs/catalog-token-identity.md)
- Certification and release evidence: [`docs/backend-certification.md`](docs/backend-certification.md),
  records in [`docs/evidence/certification/`](docs/evidence/certification/)
- Decisions log and founding constraints: [`DECISIONS.md`](DECISIONS.md), [`FOUNDING.md`](FOUNDING.md)
- Bench rig: [`bench/rig/RIG.md`](bench/rig/RIG.md); campaigns: [`bench/campaign/README.md`](bench/campaign/README.md)
- Core ML worker packaging: [`crates/synapse-worker-ane/README.md`](crates/synapse-worker-ane/README.md)
