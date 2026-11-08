# Codebase Structure

## Directory map

```
bench/                      benchmark and research tree
  campaign/                 sandboxed campaign controllers, their fixtures and tests
  data/                     prompt sets and corpus inputs for bench runs
  eval-coir/                CoIR retrieval and rerank quality scoring (Python)
  fixtures/                 shared bench fixtures
  harness/                  synapse-bench: corpus, idle gate, telemetry, parity, results schema
  lanes/                    one runner per backend under test (Rust, Python, JS)
  parity/                   synapse-parity (own workspace): model manifest, fp32 reference
                            fixtures and generator, parity evaluator, parity-manifest tool
  results/                  ignored local bench outputs
  rig/                      synapse-rig: candidate supervisor over framed stdio
  spikes/                   research prototypes (unified-rt, ane-*, laya-owned, stt-bias)
contracts/                  interface contracts and their validators (ane-prefill-split)
crates/                     production Rust crates
  synapse-build-stamp/      standard-library-only Git provenance helper for build scripts
  synapse-certify/          release certification records and their validator
  synapse-certify-runner/   ckdev-synapse-certify, the live certification runner (never shipped)
  synapse-core/             shared types, engine traits, tokenizer, worker protocol, scheduler
  synapse-engine-cuda/      CUDA embedding and reranking engine (PTX kernel ports)
  synapse-engine-owned/     Metal embedding engine, Metal decode kernels, owned-decode supervisor (macOS)
  synapse-module/           the subc module (ck-synapse): ops, catalog, store, worker host
  synapse-native-probe/     safe wrappers for the Windows calls the machine profile reads
  synapse-opctl/            operator CLI (ck-synapse-opctl)
  synapse-release-checks/   Python gates the tag workflow runs over prebuilt release assets
  synapse-worker-ane/       Core ML Neural Engine embedding worker (Rust launcher + Swift)
  synapse-worker-ane-direct/ direct Neural Engine worker for the -ane catalog lanes
  synapse-worker-cuda/      CUDA embedding and reranking worker
  synapse-worker-decode/    owned Metal generation worker (macOS)
  synapse-worker-llama/     llama.cpp generation worker for GGUF models
  synapse-worker-vulkan/    Vulkan embedding and reranking worker
docs/                       public architecture, contracts and operator guides
  evidence/certification/   public machine-readable certification records
manifests/                  component manifests (ane-prefill-split, semantic-sidecar-v1)
results/                    token-identity capture instrumentation patch
scripts/                    gate, train landing, CI watch, path and layout checks, packaging
tests/                      standalone certification and attribution scripts (outside cargo test)
tools/                      gather-distill (Bun/TS dataset and SFT harness), stt-voice-test
workers/                    non-Rust worker processes: ane-prefill-sidecar (Swift package)
.cortexkit/                 tracked: WORKER.md (worker guide) and skills/; the rest is ignored,
                            including internal-docs/ (designs, investigations, audits, research)
.github/workflows/          tests.yml (push CI, calling portable.yml), release-candidate.yml,
                            release.yml, design-gate.yml
```

Inside `crates/synapse-module`, the owned-generation code sits next to `src/` in
`owned-decode-routing/`, `owned-decode-grammar-scheduler/`,
`owned-decode-certification/`, `owned-decode-sidecar/` and `owned-decode-manifests/`.
Inside `crates/synapse-engine-owned`, `owned-decode-engine/` holds the Metal decode
kernels and `owned-decode-worker/` the supervisor and protocol.

The model catalog is `crates/synapse-module/src/catalog/models.json`, validated by
`catalog/mod.rs`. Worker supervision is in `crates/synapse-module/src/worker_host/`:
`mod.rs` (spawn, handshake, requests, quarantine), `liveness.rs` (no-progress limits)
and `ane_residency/` (the direct Neural Engine compile scheduler and executable budget).

Root files: `Cargo.toml` (workspace members, default members and the pinned subc and
commons crates; `bench/parity` is excluded and has its own lock), `DECISIONS.md`,
`FOUNDING.md`, `CONTRIBUTING.md`, `SECURITY.md`, `THIRD-PARTY-NOTICES.md`.

## Where to add new code

**New worker backend.** Create `crates/synapse-worker-<name>` with a `ck-synapse-worker-<name>`
binary that speaks `crates/synapse-core/src/worker_protocol.rs` over the transports in
`worker_transport/` and reports progress while it works. Add its HELLO identity and
binary name to `crates/synapse-core/src/worker_engine_names.rs`, wire spawning in
`crates/synapse-module/src/worker_host/mod.rs`, add the engine to the allow-list in
`crates/synapse-module/src/store.rs` if it serves embedding or reranking, and add the
crate to the root `Cargo.toml` members. The worker receives token ids, not text.

**New catalog model or backend.** Add the model to `bench/parity/models.json` and its
reference fixtures (see `docs/reference-generation.md`), then add the entry or backend
to `crates/synapse-module/src/catalog/models.json` with its pinned fingerprint. A new
backend for an existing model is a new lane; it must not change any live fingerprint.

**New wire operation.** Add the handler in `crates/synapse-module/src/lib.rs`, add a
match arm in `dispatch_request`, and register the name and kind (query or mutate) in
`management_operations`. Document it in `docs/wire-contract-v1.md`. If operators need
it, add a subcommand in `crates/synapse-opctl/src/main.rs`.

**New benchmark lane.** Add `bench/lanes/<lane>/`. A Rust lane is a workspace crate
(add it to `Cargo.toml` members, not default members) that depends on `bench/harness`
or runs under `bench/rig`; Python or JS lanes carry their own manifest. Emit the
`LaneResult` schema from `bench/harness/src/results.rs` and add the run to
`bench/run-matrix.sh` (and `bench/run-night.sh` if it belongs in the nightly set).

**New campaign harness.** Add a controller script `bench/campaign/<name>-harness.sh`
with its pinned fixtures beside it, following the existing controllers and
`bench/campaign/README.md`. Tests for controllers go in `bench/campaign/tests/`.

**Tests.** Unit tests go in `#[cfg(test)]` modules or a `*_tests.rs` file beside the
source. Integration tests go in the crate's `tests/` directory (for the module,
`crates/synapse-module/tests/`; the whole-process e2e suites there run on schedule and
on demand, not on every push). Tests that run a built binary run it as a `ckdev-` copy
through `synapse_core::dev_binary::ckdev_binary`. Hardware certification and
attribution scripts that cannot run under `cargo test` go in the top-level `tests/`.

## Naming conventions

- Shipped binaries carry the `ck-` prefix (`ck-synapse`, `ck-synapse-worker-cuda`).
  Development, test and benchmark binaries never do: they run as `ckdev-*`, and a
  source scan refuses direct `ck-` invocations in tests. Crate names and the subc
  module id (`synapse`) carry no prefix.
- Catalog lane ids are `<catalog_id>-<backend>` (`qwen3-embedding-0.6b-ane`).
- Directories and crate names are kebab-case (`synapse-worker-decode`, `ort-embed`).
- Rust source files are snake_case (`worker_engine_names.rs`, `ane_artifact.rs`).
