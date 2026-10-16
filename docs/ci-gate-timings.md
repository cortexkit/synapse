# CI and landing gate timing

## Measurement boundary

Measured on 2026-10-16 before editing the gates, at base `9d8d0d4f`.
The landing workflow is `.github/workflows/tests.yml` (named **CI**); there is
no `ci.yml`. No `train/` branch was pushed for this change. These branches are
the CI-gated staging area before master advances. Consequently the CI **after**
numbers below are critical-path accounting or estimates, not a new hosted run.
Do not treat cache savings or shard speedup as measured until a hosted run exists.

The sample is the latest ten completed runs returned by
`gh run list --workflow tests.yml`, including two cancelled runs. Job and step
durations come from `gh run view RUN --json jobs` timestamps. Skipped jobs have
invalid/inverted timestamps and are excluded. The unusually long elapsed time
of run `38888816956` includes runner queueing; it is not test execution time.
Raw job/step JSON and representative Linux/Windows logs are retained in the
delivery's internal evidence, not in build output directories.

| Run | Branch | Result | Run elapsed (s) | Linux job / nextest / replay (s) | Windows job / nextest / replay (s) |
| --- | --- | --- | ---: | ---: | ---: |
| 38953352962 | train/lane-pause | failed | 4382 | 4295 / 1226 / 2956 failed | 2784 / 1049 / 1499 |
| 38941072766 | train/lane-pause | cancelled | 5879 | 5875 / 1877 / 3871 cancelled | 1692 / 1386 / 144 failed |
| 38929092280 | master | passed | 1796 | 1644 / 1402 / 99 | 1791 / 1392 / 190 |
| 38920846479 | train/deadline-ceil | passed | 3614 | 3609 / 1771 / 1707 | 3404 / 1354 / 1824 |
| 38916909476 | master | passed | 2142 | 2127 / 1894 / 98 | 1785 / 1392 / 187 |
| 38912758157 | train/train-relift | passed | 1792 | 1435 / 1243 / 71 | 1788 / 1396 / 187 |
| 38912407636 | master | cancelled | 2025 | 2020 / 1784 / 101 | 1819 / 1420 / 187 |
| 38891096494 | master | passed | 2184 | 2180 / 1918 / 107 | 1770 / 1354 / 185 |
| 38888816956 | train/liveness-log-level | passed | 12001 | 1561 / 1331 / 85 | 1748 / 1374 / 177 |
| 38814373765 | master | passed | 2172 | 2168 / 1898 / 112 | 1699 / 1327 / 169 |

### Hosted stages: before and after

Medians below use successful steps only; the failed/cancelled replay durations
above are lower bounds on the troublesome large changes, not successful timings.
Unchanged steps have no measured after-run; their existing timings are the budget.

| Stage | Linux median (s) | Windows median (s) | After on push |
| --- | ---: | ---: | --- |
| Checkout | 2 | 7.5 | unchanged |
| Rust toolchain setup | 9 | 22 | unchanged |
| Main Rust cache restore | 23.5 | 34.5 | workspace outputs now retained; new source key, savings unmeasured |
| MiniLM fixture cache | 2 | 2 | 0; scheduled/manual e2e job only |
| Nextest install | 0.5 | 2 | unchanged |
| Windows build environment | n/a | 1 | unchanged |
| fmt | 2.5 | 40.5 | unchanged |
| clippy | 30.5 | 51 | unchanged, now explicitly locked |
| nextest (whole step) | 1777.5 | 1380 | ordinary targets still block merges; `skeleton_e2e` moved to scheduled/manual CI |
| Catalogue check + touched mutation replay | 100 (8 successes) | 187 (9 successes) | 0 replay; catalogue check still blocks and can compile targets |
| Python setup / checker dependency / runner install | 0 / 2 / 1 | 0 / 5 / 3 | unchanged |
| Train preconditions | 1 | n/a | still blocking, adds fast planner/audit tests |
| Release-candidate / tag / benchmark helper tests | 0 / 1 / 0 | n/a | unchanged |
| Parity cache / tests | 1 / 38 | n/a | unchanged; only 38 s, not worth delaying release feedback |
| Banlist / layout / external path dependencies | 1 / 0 / 4 | n/a | unchanged |
| Module runtime dependency / ANE harness checks | 0.5 / 0 | n/a | unchanged |
| Llama CPU build | 11 | 23 | unchanged |
| Mutation evidence upload | 1 | 1 | no push replay artifact; master/nightly retain their reports |
| Llama evidence upload | 1 | 1 | unchanged |
| E2e skip guard | 0 | 0 | unchanged guard, now alongside scheduled/manual e2e |

Run `38929092280`, with successful cache restores, illustrates where the time in
the two module-inclusive nextest invocations went:

| Substage | Linux (s) | Windows (s) | After |
| --- | ---: | ---: | --- |
| First nextest compile | 86 | 716 | still required before merging; workspace outputs may reduce recompilation |
| Ordinary tests | 380.773 (1154 passed) | 51.058 (1058 passed) | still blocking, same selection |
| Second nextest compile | 74 | 538 | moved with skeleton e2e |
| Skeleton execution | 859.872 (57 passed) | 83.122 (55 passed) | scheduled/manual; **0 s on push** |

The root workspace and parity Rust caches, and the MiniLM model-fixture cache,
were hits in that run. Blindly adding another
dependency cache would not solve it. Retaining workspace outputs with a source
hash permits exact-source reuse and dependency fallback without reusing a cache
forever under one immutable lockfile key. Cargo still validates fingerprints.
No prebuilt fixture binary is reused across mutant source states.

In that run the catalogue-success log timestamp is about **100 s Linux / 187 s
Windows** after the replay step starts: nearly the entire normal tail is
catalogue validation, not mutation execution. Removing replay does **not** save
that normal-case tail; its payoff is on changes selecting many rows, such as the
3871 s cancelled Linux replay. The Windows post-check remainder is about 3 s;
step timestamps have only one-second precision.

Subtracting only the measured second compile, skeleton execution and fixture
cache gives about **708 s Linux / 1168 s Windows**, versus **1644 / 1791 s**
before. This is a *model*, not a hosted benchmark: cache transfers, catalogue
checking and queueing can change it. No reduction in catalogue-check time is
assumed. Its Windows CRT profile settings are retained because check mode still
compiles test-discovery targets.

Mutation rows must build their own mutant: sharing one clean executable would
test the wrong source. The pinned `ckdev-mutate` runner already shares clean-tree
baseline results for rows with the same runner, package, targets and features.
Nightly broad replay now has four independent checkouts per OS via `--shard i/4`.
Do not parallelize mutation processes in one checkout. No row-weight
file is introduced: the pinned 0.10 CLI exposes `--shard` but not
`--shard-weights`. This uses sorted-ID sharding rather than invented weights; skew
and repeated per-shard baselines may limit speedup. Master full replay remains
asynchronous, unsharded, and starts after landing.

An attempted four-process nextest e2e run on Linux was **rejected**, not shipped:
`catalog_e2e::catalog_endpoint_with_path_refuses_module_start_by_name` timed out
after 31 s; only 10 of 57 tests passed before fail-fast. Several normal 12 s
cases took 37–46 s under contention. `--no-capture` and serial e2e execution are
preserved. Moving `skeleton_e2e` off the push path avoids trading speed for a
flaky blocking run or relaxing its timeouts.

## Train preflight: measured locally

The scripts copied from AFT (`train-push.sh`, `train-push.test.sh`, and
`watch-ci.sh`) remain byte-identical to base commit `9d8d0d4f`. Only the local
hook changes. Commands were timed
individually on the same Mac with Cargo/Rust 1.99.0, four build jobs and warm
native artifacts. Zig caches were placed in the worktree because the sandbox
refuses the default home cache. Cross-target times include their initial target
compilation; they are not a claim about a fully warm cross-target run.

| Stage | Before (s) | After (s) | Disposition |
| --- | ---: | ---: | --- |
| Workflow preconditions | 0.57 | 1.97 | kept |
| External path dependencies | 4.62 | 3.22 | kept |
| Runtime dependencies | 0.86 | 0.89 | kept |
| macOS fmt | absent | 2.21 | added; native fmt before master, like Linux/Windows CI |
| macOS clippy | 1.81 | see native-hook timing below | kept and extended to the module, which no longer lints in the default landing script |
| macOS owned-engine lib | 2.91 | 1.09 | kept (61 passed) |
| Linux zig cross-clippy | 157.29 | 0 | native Linux CI instead |
| Windows GNU zig cross-clippy | 93.56 | 0 | native MSVC CI instead |
| Schedule-health query | 0.90 | 2.10 | kept as notice |
| Sum of individually measured original/retained stages | 262.52 | 11.99 before adding native module clippy | cross-target removal saves 250.85 s for this cache state; final hook timing below |

No cross-platform check is deferred beyond master: native CI gates the train
SHA before fast-forwarding. The loss is earlier local feedback, not coverage;
the old Windows-GNU cross-check did not exercise MSVC in any case.

## Local landing policy and timing

Use `bash scripts/land-gates.sh` for fast feedback before submitting a branch.
It requires cargo-nextest, as does CI. It checks the merge-base
with `origin/master`, including committed changes, staged/unstaged edits and
untracked files. `--base REF` overrides that base; `--plan` prints the exact
commands without claiming they passed. A root manifest/lockfile change requires
`--full`. Unit targets of changed workspace packages and directly changed
integration targets are added; a shared integration helper selects all of that
package's integration tests. Nested crates are attributed to the innermost
package. Review the plan and run additional property-specific tests where needed:
this is targeted feedback, not proof of all transitive dependents.

The default always retains fmt, banlist, layout, path/runtime dependency checks,
workflow/planner self-tests, wire/error documentation tests, launch-nonce and
child-environment scans, and store/catalog migration tests. A changed module
still builds its required stub and GPU-refusal fixtures before its narrow tests.
`--full` adds catalogue validation, stub/GPU fixtures, module check, module lib,
`it`, `skeleton_e2e` and module clippy. It is the full *portable module landing*
set, not all workspace/hardware/certification tests. The pinned mutation checker
refuses dirty mutation targets even in check mode, so commit those edits before
using `--full`; the default delegates that check to clean train CI.

Before timing used the serial portable module gate list on Linux (four CPUs).
It passed 737 lib tests (14 ignored), 26 integration tests, 57 skeleton tests
(1 ignored) and four wire tests. This is portable behavior, not hardware proof.

| Old local stage | Before Linux (s) | Fast default after, infrastructure/docs-only change |
| --- | ---: | --- |
| fmt / banlist / layout | 2.35 / 1.00 / 0.01 | kept |
| Path dependencies | remote dependency cache lacked locked `bench/parity` crates | kept; measured natively below |
| Mutation catalogue validation | 304.35 (includes exact-test discovery/builds) | 0; full/CI |
| Stub / GPU refusal builds | 1.13 / 22.35 | 0 unless selected module tests need them |
| Module check | 28.59 | 0; full/CI |
| Entire module lib | 114.01 | only mandatory store/migration filters |
| Entire module `it` | 263.20 | only mandatory nonce source scans |
| Skeleton e2e | 301.51 | 0; full/nightly/manual |
| Wire contract | 0.46 | kept |
| Module clippy | 55.78 | 0; full/native CI |

The first fast-default implementation still used four separate module Cargo
invocations. On the Mac these took **105.01 s of stage time** with initial test
builds; a warm rerun still took **83.53 s wall time**. Combining the four module
selections into one nextest invocation avoids repeatedly invalidating native
outputs through different Cargo feature resolutions.

The final default passed **10 stages in 23.78 s wall time** (23.24 s summed stage
time), with warm artifacts on the same Mac:

| Final default stage | Measured Mac time (s) |
| --- | ---: |
| fmt | 2.25 |
| Banlist / layout | 0.75 / 0.09 |
| Path dependencies | 3.11 |
| Workflow / runtime dependencies | 0.42 / 0.76 |
| E2e shell tests / planner and catalogue guard tests | 1.82 / 0.40 |
| Core child-environment and development-image scans | 8.65 |
| Combined module wire, nonce, store and migration tests | 4.89 |

It passed six core tests, 75 selected module tests and 15 Python tests. The
opt-in store drill stays ignored; unselected module tests are deliberately
filtered, not claimed as passes. The native train hook initially passed in
5.07 s with the original owned/ANE lint selection; after extending it to retain
native module clippy, the final hook passed in **72.47 s**, including the first
module lint build. The ~18-minute Linux serial-list baseline and
the Mac default are **not** a paired speed ratio: platforms and cache states
differ. The measured per-stage removals and the same-Mac 83.53 → 23.78 s batching
experiment establish where time was saved. A cold Rust build can still dominate
the default. Neither mode implies hardware or pinned-checkpoint certification.

## Removed or moved checks: explicit cost and risk

| Removed from | Check | Runs now | Why / measured critical-path saving | Caught-late risk |
| --- | --- | --- | --- | --- |
| Train/main CI serial tail | Touched-row executable replay | Master full catalogue and four-shard/OS nightly broad audit | Normal 100/187 s tail is mostly catalogue check and is **not** saved; large Linux replay ran 3871 s before cancellation (~3771 s beyond a representative 100 s check) | A surviving mutation or platform-specific replay failure is learned after landing; ordinary tests and catalogue validation still block |
| Push CI | Whole-process skeleton fixtures, model staging and skip guard | Daily 06:00 CI and manual dispatch, plus local `--full` | Representative second compile + execution: ~934 s Linux / ~621 s Windows; cache restore ~2 s | Wiring, model fixture, restart or end-to-end-only regressions may wait until the next daily run; no public deployment is assumed |
| Local train preflight | Zig Linux/Windows cross-clippy | Native blocking Linux/MSVC CI | 157.29 + 93.56 s in measured cache state | Platform lint errors are found in CI instead of the author's terminal, still before master |
| Default local landing | `ckdev-mutate check` | Blocking clean-tree CI and local `--full` | 304.35 s on first exact-test discovery; not actually a cheap cold static check | Bad catalogue anchors are caught on the train, not immediately locally |
| Default local landing | Unconditional stub/GPU fixtures and module check | Local `--full`, narrow module selections and native CI builds | 1.13 + 22.35 + 28.59 s | Build-only issues outside selected packages reach CI first |
| Default local landing | Entire module lib / `it` | Local `--full`, affected target selection, blocking native CI | 114.01 + 263.20 s baseline, minus retained mandatory subsets | Non-selected/transitive behavior reaches CI first; migrations and nonce boundaries still run locally |
| Default local landing | Skeleton e2e | Local `--full`, scheduled/manual CI | 301.51 s measured local portable baseline | Same daily end-to-end feedback delay as above |
| Default local landing | Module clippy | Local `--full`, blocking native Linux/Windows CI; native macOS module and owned/ANE clippy in preflight | 55.78 s Linux module baseline; Mac work is moved, not removed | Portable lint feedback moves to CI; Mac module lint still runs before pushing |

Kept before master: wire-contract/error-name documentation tests, native fmt and
clippy across Linux/Windows/macOS (Mac in preflight), launch-nonce/child-environment
scans, layout, public banlist, external path dependencies, and store migrations.
Nightly failures need attention: the preflight's schedule-health notice is still
present, but it is an alert, not a promise that every scheduled run was reviewed.
