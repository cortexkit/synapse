# Codebase Structure

## Directory map

```
bench/                      benchmark and research tree
  campaign/                 sandboxed campaign controllers and their fixtures
  data/                     prompt sets and corpus inputs for bench runs
  eval-coir/                CoIR retrieval and rerank quality scoring (Python)
  fixtures/                 shared bench fixtures
  harness/                  synapse-bench: corpus, idle gate, telemetry, parity, results schema
  lanes/                    one runner per backend under test (Rust, Python, JS)
  results/                  saved bench outputs
  rig/                      synapse-rig: candidate supervisor over framed stdio
  spikes/                   research prototypes (unified-rt, ane-*, laya-owned, stt-bias)
contracts/                  interface contracts and their validators
crates/                     production Rust crates
  synapse-core/             shared types, engine traits, tokenizer, worker protocol, scheduler
  synapse-engine-cuda/      CUDA embedding engine (PTX kernel ports)
  synapse-engine-ort/       ONNX Runtime CPU embedding engine
  synapse-engine-owned/     Metal embedding engine, Metal decode kernels, owned-decode supervisor (macOS)
  synapse-module/           the subc module (ck-synapse): ops, store, worker host, remote gateway
  synapse-opctl/            operator CLI (ck-synapse-opctl)
  synapse-worker-ane/       Core ML / Neural Engine embedding worker (Rust launcher + Swift)
  synapse-worker-cuda/      CUDA embedding worker
  synapse-worker-decode/    owned Metal generation worker (macOS)
  synapse-worker-llama/     llama.cpp worker for GGUF models
docs/                       design notes, wire contract, audits, measured evidence
manifests/                  component manifests (ane-prefill-split, semantic-sidecar-v1)
results/                    saved measurement outputs from tests/ scripts
scripts/                    packaging, CI watch, sibling lock refresh, git hooks
tests/                      standalone certification and attribution scripts (outside cargo test)
tools/                      gather-distill (Bun/TS dataset and SFT harness), stt-voice-test
workers/                    non-Rust worker processes: ane-prefill-sidecar (Swift package)
```

Inside `crates/synapse-module`, the owned-generation code sits next to `src/` in
`owned-decode-routing/`, `owned-decode-grammar-scheduler/`,
`owned-decode-certification/`, `owned-decode-sidecar/` and `owned-decode-manifests/`.
Inside `crates/synapse-engine-owned`, `owned-decode-engine/` holds the Metal decode
kernels and `owned-decode-worker/` the supervisor and protocol.

Root files: `Cargo.toml` (workspace members), `siblings.lock` (pinned commits of the
`subconscious` and `commons` path dependencies), `DECISIONS.md`, `FOUNDING.md`,
`CONTRIBUTING.md`.

## Where to add new code

**New worker backend.** Create `crates/synapse-worker-<name>` with a `ck-synapse-worker-<name>`
binary that speaks `crates/synapse-core/src/worker_protocol.rs` over the transports in
`worker_transport/`. Add its HELLO identity and binary name to
`crates/synapse-core/src/worker_engine_names.rs`, wire spawning in
`crates/synapse-module/src/worker_host/mod.rs`, and add the crate to the root
`Cargo.toml` members. The worker receives token ids, not text.

**New wire operation.** Add the handler in `crates/synapse-module/src/lib.rs`, add a
match arm in `dispatch_request`, and register the name and kind (query or mutate) in
`management_operations`. Document it in `docs/wire-contract-v1.md`. If operators need
it, add a subcommand in `crates/synapse-opctl/src/main.rs`.

**New benchmark lane.** Add `bench/lanes/<lane>/`. A Rust lane is a workspace crate
(add it to `Cargo.toml` members) that depends on `bench/harness` or runs under
`bench/rig`; Python or JS lanes carry their own manifest. Emit the `LaneResult` schema
from `bench/harness/src/results.rs` and add the run to `bench/run-matrix.sh` (and
`bench/run-night.sh` if it belongs in the nightly set).

**New campaign harness.** Add a controller script `bench/campaign/<name>-harness.sh`
with its pinned fixtures and registration beside it, following the existing
controllers and `bench/campaign/README.md`. Tests for controllers go in
`bench/campaign/tests/`.

**Tests.** Unit tests go in `#[cfg(test)]` modules in the source file. Integration
tests go in the crate's `tests/` directory (for the module,
`crates/synapse-module/tests/`). Hardware certification and attribution scripts that
cannot run under `cargo test` go in the top-level `tests/`.

## Naming conventions

- Shipped binaries carry the `ck-` prefix (`ck-synapse`, `ck-synapse-worker-cuda`);
  crate names and the subc module id (`synapse`) do not.
- Directories and crate names are kebab-case (`synapse-worker-decode`, `ort-embed`).
- Rust source files are snake_case (`worker_engine_names.rs`, `ane_artifact.rs`).
