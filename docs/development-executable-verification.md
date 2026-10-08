# Development executable aliases

Development children use `ckdev-*` file names so process listings distinguish
build/test executions from installed `ck-*` images. Cargo test harnesses retain
their normal underscore-and-hash names. Production sibling resolution, worker
HELLO identities, version text, release archive names and certification record
roles/file names are unchanged.

## Site table

`alias` below means `synapse_core::dev_binary::ckdev_binary`: a hard link in the
caller's scratch tree, falling back to a copy only across volumes. `.exe` is
preserved. Each alias gets a separate directory to avoid replacing a live image
on Windows. `strict alias` means the same implementation with copying forbidden:
certification refuses a cross-volume layout rather than executing a different
inode from the attested build file.

| File | Before | After |
| --- | --- | --- |
| `crates/synapse-core/src/dev_binary.rs` | No common alias facility | Shared alias helper and strict certification variant; byte/name/inode tests |
| `crates/synapse-core/src/lib.rs` | No development alias module | Exposes the shared helper |
| `crates/synapse-core/Cargo.toml`, `Cargo.lock` | No AST parser in core test dependencies | Existing `syn 2.0.118` is a dev dependency with `full`/`visit`; no version updates |
| `crates/synapse-core/tests/dev_binary_guard.rs` | No executable-name source fence | AST scan of integration and inline test code, including nested members and bench sources; positive/negative planted controls |
| `crates/synapse-core/tests/candidate_workers.rs` | Candidate module and worker files executed directly | Aliases for the module and each worker; explicit aliased Swift override for ANE |
| `crates/synapse-module/tests/skeleton_e2e.rs` | Direct `CARGO_BIN_EXE_ck-synapse`, direct mock/optional owned worker paths | Module, mock and optional owned worker aliases; binary-byte inspection still reads the original build |
| `crates/synapse-module/tests/common/mod.rs` | Module implicitly found `ck-*` siblings beside its build output | Canonical worker env overrides point to aliases; Swift and legacy owned-generation overrides are isolated too |
| `crates/synapse-module/tests/soak.rs` | Direct module launch; abort wrapper execed the original llama worker | Module alias; `ckdev-*` abort wrapper execs a worker alias; source asset discovery stays unchanged |
| `crates/synapse-module/tests/it/launch_refusals.rs` | Direct module launch | Module alias in the isolated home |
| `crates/synapse-module/tests/it/ane_workers.rs` | WorkerHost launched the supplied ANE launcher, which could exec a `ck-*` Swift companion | Aliased launcher plus private shell wrapper pinning its existing Swift override to an alias |
| `crates/synapse-module/tests/it/main.rs` | Common support included separately by child modules | Includes common support once; avoids Clippy's duplicate-module error and duplicate sweep-test registration |
| `crates/synapse-module/tests/it/worker_host_timeout.rs` | Direct timeout mock path | Shared alias |
| `crates/synapse-module/tests/worker_logging.rs` | Direct logging mock path | Shared alias |
| `crates/synapse-module/src/lib.rs` (CUDA probe test only) | Ignored test executed a supplied/release CUDA worker directly | Shared alias before probing; production code unchanged |
| `crates/synapse-module/src/worker_host/mod.rs` (ANE residency tests only) | `ANE_TEST_WORKER` executed directly | Shared alias; production host unchanged |
| `crates/synapse-worker-cuda/tests/hosted.rs` | Direct built worker for protocol and probe tests | Shared aliases |
| `crates/synapse-worker-cuda/tests/hosted_windows.rs` | Direct `.exe` | Shared `.exe` alias |
| `crates/synapse-worker-cuda/tests/gpu_parity.rs` | Direct built worker in opt-in GPU test | Shared alias |
| `crates/synapse-worker-llama/tests/protocol_v2.rs` | Direct built worker | Shared alias in protocol scratch tree |
| `crates/synapse-worker-llama/tests/worker_host.rs` | Direct worker in WorkerHostConfig | Shared alias |
| `crates/synapse-worker-vulkan/tests/protocol_v2.rs` | Direct built worker for hosted/probe/version tests | Shared aliases |
| `crates/synapse-worker-decode/tests/it/worker_transport.rs` | Direct built worker in commands and WorkerHostConfig | Shared aliases |
| `crates/synapse-worker-decode/tests/it/semantic_sidecar_phase1.rs` | Direct measurement worker | Shared alias; evidence still identifies the original built file |
| `crates/synapse-certify/src/live.rs` | Floor worker, module and configured worker executed from `--assets` | Strict hard links in `.live` scratch trees; record hashing still reads the original assets |
| `crates/synapse-certify/tests/live_hardware.rs` | Candidate producer executed from `.live/metal-candidate/ck-synapse` | Strict producer alias |
| `crates/synapse-opctl/tests/connection_discovery.rs` | Direct CLI binary | Same helper source included in the integration test, with an alias under its TempTree |
| `scripts/check-release-candidate.py` | Vulkan version and CUDA missing-runtime probes executed build files directly | Scoped development aliases with cross-volume copy fallback; original files still inspected/hashed |
| `scripts/test-owned-cuda-package.ps1` | Extracted and isolated package executables retained `ck-*` names | `ckdev-*.exe` hard link beside packaged DLLs and a renamed isolated copy |
| `scripts/sample-dev-images.py` | No scoped live measurement | Samples image names and process births; excludes recycled PIDs and reports leftovers without killing ambient processes |
| `docs/direct-ane-serving.md` | Certification CLI executed `target/release/ck-synapse` | CLI starts through a `ckdev-*` hard link, assets remain original files |
| `docs/audits/pr15-cuda-verification.md` | Historical raw build-output commands could be mistaken for current instructions | Adds current alias-based reproduction instructions; historical measurements and diagnostics remain intact |

Additional sites found beyond the initial inventory: candidate protocol matrix,
llama protocol-v2 tests, opctl connection discovery, timeout/logging mocks,
ANE residency tests driven by `ANE_TEST_WORKER`, and release-candidate Python
smokes used by CI. Vulkan GPU parity runs its engine in-process and does not
spawn a binary. ANE's production launcher already supports an explicit Swift
path; tests use that hook without adding a production naming scheme.

## Source fence proof

Toolchain: Cargo 1.99.0 (`5f94df478`), rustc/Clippy 1.99.0 (`b940084d7e`).
The guard parses Rust expressions, follows local path bindings, and inspects
individual Command/WorkerHostConfig constructor arguments. A wrapped launch
cannot authorize a direct launch in the next statement. Reading the original
binary as evidence is not a launch.

Command:

```sh
cargo test --locked -p synapse-core --test dev_binary_guard -- --nocapture
```

Passing final scan: **266 Rust sources**, **3 passed, 0 failed**. Both planted
control tests passed:

- `planted_direct_spawn_after_wrapped_spawn_is_rejected`
- `planted_paths_aliases_and_multiline_wrappers_are_judged_separately`

Two real-tree mutations were staged/restored independently without executing
the planted subprocesses:

1. Add a direct `Command::new(env!("CARGO_BIN_EXE_ck-synapse")).spawn()` in
   `skeleton_e2e.rs` immediately after its wrapped module command.
2. Add a direct `Command::new("target/debug/ck-synapse").spawn()` at the same site.

Each red run named **only `test_launches_use_development_images`** as failed;
both planted control tests stayed green (**2 passed, 1 failed**). Each mutation
had `skeleton_e2e.rs | 2 ++` as its nonempty working diff, and an empty working
diff after checkout from the staged live implementation and touch. Restored
runs passed. The first run scanned 118 sources; the final guard also walks
nested crate/bench test trees, and its target-path mutation scanned 266.

The sampler's process-birth fence was also neutralized: accepting retained PIDs
without comparing birth times failed only
`test_recycled_pid_is_not_a_descendant`; the orphan control remained green.
Restoring the staged script returned both controls to green.

## Live measurement

The first concurrent hardware test run failed the existing GTE throughput
cliff and Qwen cold-load time bounds under machine contention. No assertion was
weakened. Serial runs passed both tests and the rest of the suite.

An initial sampler retained bare compiler PIDs during a long build and later
misattributed recycled PIDs belonging to another worktree's `ck-engram` images.
That measurement was rejected, not counted as a Synapse violation. The sampler
now binds ownership to PID plus `lstart`; its recycled-PID and orphan controls
both pass. It never signalled those unrelated processes.

Final command and measurements are recorded below after the final companion
isolation checks:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  python3 scripts/sample-dev-images.py \
  cargo test --locked -p synapse-module --test skeleton_e2e -- --test-threads=1
```

Python 3.9.6; sampling `ps -axo pid=,comm=` every **0.5 seconds**, with a
separate parent/birth snapshot for ownership:

| Measurement | Result |
| --- | ---: |
| Samples | 3540 |
| Unique `ck-*` process images from this run | **0** |
| Unique `ckdev-*` process images from this run | **63** |
| `ck-*` observation rows | 0 |
| `ckdev-*` observation rows | 2823 |
| Remaining owned `ck-*` / `ckdev-*` processes after the run | **0** |
| Skeleton e2e result | **66 passed, 0 failed, 4 existing ignores** |

The test command exited 0. Four additional half-second samples after exit
confirmed the empty leftover set.

## Gates

- `cargo fetch --locked`: completed after adding the existing AST parser to
  core's dev dependencies; only the core dependency entry changed in Cargo.lock.
- `cargo fmt --all -- --check`: passed (rustfmt 1.10.0-stable).
- macOS Clippy `--locked --all-targets -- -D warnings`: passed for 13 packages:
  core, module, certify, workers CUDA/Vulkan/llama/decode/ANE/ANE-direct, opctl,
  engines owned/CUDA, and owned-decode-worker. This includes the macOS preflight
  crates from `scripts/train-push.local.sh`.
- Linux and Windows-GNU Clippy preflights: passed using the same zig CC/AR shims
  as `scripts/train-push.local.sh`, with the seven CI packages after its llama
  exclusion, plus certify and Vulkan (9 packages each). Zig 0.16.0. The llama
  cross-build exclusion is the existing preflight's CMake limitation, not a new
  skip. Windows-GNU checks cfg(windows), not execution on an MSVC runner.
- Initial combined core/module/certify/worker build: core's 80 library tests
  passed; module's 618 library tests passed with 9 existing ignores. Integration
  suites before skeleton passed; the two hardware timing failures above were
  subsequently resolved by serial execution without changing test contracts.
- Final module `it`/`soak`/`worker_logging`: 19 + 1 + 1 passed, 2 soak tests
  ignored. Moving common support to the `it` root removes one duplicate
  sweep-test registration; it does not remove an independent assertion.
- Certify, opctl, CUDA, decode, llama and Vulkan test command: **104 passed,
  7 existing ignores**, including certification alias byte/name/inode identity,
  opctl real-binary discovery, hosted CUDA, worker transports and protocols.
- Release-candidate Python self-tests: **7 passed**, including alias identity
  and `.exe` preservation (Python 3.9.6).
- Sampler self-tests: **2 passed** (Python 3.9.6).
- Train-precondition self-test: clean control passed, **6 planted breaks refused**.
- External-path dependency fence: passed, **2 workspaces / 31 manifests** scanned.
- AFT inspection was partial (Rust indexing/callgraph not ready). Authoritative
  Rust compilation/tests and the three platform Clippy checks passed instead.

Existing opt-in checkpoint, hardware-certification, soak and CUDA/Vulkan GPU
feature tests were not forced on. The Windows PowerShell package smoke requires
Windows and packaged runtime DLLs; its executable paths were updated but it was
not executed on macOS. No production process was stopped or modified. The
production-name resolver in synapse-core and the ANE launcher are unchanged.
