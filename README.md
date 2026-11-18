# Synapse

Synapse is a local inference service on the subc bus. It provides embedding,
reranking and small-model generation through the `synapse` module, with explicit
model installation, stable numerical identity, bounded scheduling and durable
result pages. Owned engines and supervised workers isolate inference from the
management surface. Modules needing hosted inference call providers directly.
A missing local backend produces a typed refusal, not an automatic CPU fallback.

## Workspace crates

Production and supporting workspace members live under `crates/`. A worker's
presence in source does not imply that a release ships it; the catalog and release
matrix determine supported backends and published assets.

| Member under `crates/` | Responsibility |
|---|---|
| `synapse-build-stamp` | Shared clean/dirty Git build provenance. |
| `synapse-core` | Protocol, errors, fingerprints, tokenizer, cache and scheduler types. |
| `synapse-speech` | Portable WAV validation, resampling, Parakeet log-mel frontend, TDT greedy decoding and word timing. |
| `synapse-native-probe` | Safe native platform-profile probes. |
| `synapse-certify` | Schema-3 certification records, self-checks and offline validation. |
| `synapse-certify-runner` | Non-shipped hardware evidence producer with isolated run state. |
| `synapse-engine-owned` | Owned Metal embedding/reranking and decode kernels. |
| `synapse-engine-owned/owned-decode-worker` | Owned-decode supervisor, protocol and crash budget. |
| `synapse-engine-cuda` | Owned CUDA engine and hardware-floor checks. |
| `synapse-module` | Management operations, model cache and durable-job orchestration; `ck-synapse`. |
| `synapse-owned-decode` | Owned-decode contracts, sidecar normalization, routing, grammar scheduling and pure certification. |
| `synapse-store` | SQLite persistence, migrations, durable jobs, certification and event-outbox rows. |
| `synapse-worker-host` | Supervised worker processes, transcription exchange and ANE residency. |
| `synapse-opctl` | Operator client; `ck-synapse-opctl`. |
| `synapse-worker-llama` | Supervised GGUF worker. |
| `synapse-worker-ane-direct` | Direct Neural Engine compilation, admission and inference worker. |
| `synapse-worker-decode` | Parked owned Metal generation worker, built in CI but not shipped. |
| `synapse-worker-cuda` | Owned CUDA worker. |
| `synapse-worker-vulkan` | Owned Vulkan embedding/reranking worker. |

## Manage local models

With Synapse running, use `ck-synapse-opctl` on Linux, macOS, or Windows:

```sh
ck-synapse-opctl models list
ck-synapse-opctl models download CATALOG_ID
ck-synapse-opctl models remove CATALOG_ID
```

`models list` shows the registered model lanes and their states. Use a catalog
ID, not a lane ID, with `download` and `remove`. Synapse downloads models only
on request. `download` shows progress and reports the installed lanes on
success, including when the model was already installed. Ctrl-C stops watching
without cancelling the download. Rerun the same command against the same daemon
to resume watching. Use `--json` to get only the final reply on standard output.

`remove` uninstalls the catalog model and reports the bytes freed. Synapse
refuses removal while the model is in use.

## Build and test

Use the committed `Cargo.lock`. The workspace requires Rust 1.89 or newer and
native tools appropriate to the selected package. Apple Metal builds need full
Xcode, not only Command Line Tools; verify `xcrun --find metal` on macOS.

From the repository root:

```sh
cargo build --locked -p synapse-module -p synapse-opctl
cargo build --locked --no-default-features -p synapse-worker-cuda -p synapse-worker-vulkan
cargo check --locked -p synapse-module --features test-support
cargo test --locked -p synapse-module --features test-support,e2e --lib --test it --test skeleton_e2e --test wire_contract
cargo test --locked -p synapse-certify --lib --test records
cargo fmt --all --check
bash scripts/public-banlist-check.sh
bash scripts/check-cortexkit-layout.sh
```

The feature-off GPU workers above are test fixtures, not GPU certification.
Hardware and checkpoint-dependent tests are separate from portable tests; a skip
is not a hardware pass. See [development verification](docs/development.md) and
[backend certification](docs/backend-certification.md) for those boundaries.

## Documentation

- [Architecture](docs/architecture.md) and [configuration](docs/configuration.md).
- [Wire contract](docs/wire-contract-v1.md) and [worker protocol](docs/worker-protocol.md).
- [Direct ANE serving](docs/ane-serving.md) and [graph layout](docs/ane-graph-layout.md).
- [Reference generation](docs/reference-generation.md) and [token identity](docs/catalog-token-identity.md).
- [Contributing](CONTRIBUTING.md), [security reporting](SECURITY.md), and [third-party notices](THIRD-PARTY-NOTICES.md).

`ARCHITECTURE.md` and `STRUCTURE.md` remain root code/session maps. Public
certification artifacts are tracked because release tooling reads them; internal
research and operational records are not part of the public documentation set.
