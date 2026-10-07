# Mutation controls

`mutations.toml` records deliberate breaks of guards whose absence could silently
lose data, leak authority, or change a consumer's wire/vector-space contract.
Each automated row names the exact test that must fail, and `expect_message`
ties that failure to the intended property rather than an unrelated panic.
Compilation errors and timeouts are not catches.

## Install and replay

Use the reviewed runner, not a floating release:

```sh
cargo install --locked --git https://github.com/cortexkit/commons --rev 46cc166b0df2edcfd14b3eb54ed6eeac588fed69 cortexkit-mutate
ckdev-mutate --version # 0.8.0
cargo nextest --version
mkdir -p target/mutations
ckdev-mutate check
ckdev-mutate run --all --report target/mutations/all.json
ckdev-mutate run --diff origin/master --report target/mutations/diff.json
ckdev-mutate run --only nonce-sync-child --report target/mutations/one.json
ckdev-mutate run --all --broad --report target/mutations/broad.json
```

`--diff` compares **committed** changes to `HEAD`; uncommitted source, helper, and
fixture edits are not a substitute for a full replay. `check` validates every
anchor exactly once and lists guarding test names for this host. Nextest rows
need cargo-nextest installed. Python command rows need Python 3; the CI workflow
checker also needs PyYAML. The production allow-list row compiles a normal library
because Cargo test builds enable `test-support` through a self dev dependency.

Run from a clean checkout, one mutation process per checkout. Never check out a
source file while the runner is testing its mutant: that would remove the break
and manufacture a survivor. Reports belong under ignored `target/mutations/`.
The runner restores source bytes and verifies `Cargo.lock`, including on errors.

CI replays touched rows unconditionally in the same Linux/Windows step list on
trains and master. A separate master workflow replays the whole catalogue; a
schedule-only workflow audits all package test targets with `--broad`. Both retain
JSON evidence. A broad catch must be narrowed or marked `hub` with the shared
property and the observed stable collateral target names, never executable hashes.
Deadlines are hang bounds, not performance assertions. Tune budgets only with
clean CI evidence, never timings from a busy developer machine.

## Add a control

Read the production guard and its callers first. Copy the anchor from the current
source, then use `prove` rather than handwriting an unexecuted claim:

```sh
ckdev-mutate prove --id example-refusal --guards 'explain the costly failure prevented' \
  --file path/to/source.rs --old 'exact live anchor' --new 'deliberate break' \
  --test-file path/to/test.rs --package package-name --target=--lib \
  --expect-red module::tests::exact_full_name --expect-message 'property-specific failure' \
  --report target/mutations/example.json
ckdev-mutate check
```

`prove` appends only a caught row. If the break survives, add coverage through the
real production path and prove it again; testing the underlying predicate alone
cannot prove that its caller uses it correctly. Exactly-once recovery needs a
production-path row. Source scanners need planted violations in a scanned file,
not only synthetic snippets handed to their matcher. If a mutant hangs, bound the
wait in the test so it fails **by name**, assert order/outcome, and retain the row.

Use `features`, `no_default_features`, and `ignored = "include"|"only"` for the
selection the test actually needs. Declare fixture binary builds in a root
`prebuild` list with `--locked`; the runner refreshes them after every mutant build
so a stale executable cannot falsely defend a guard. An `equivalent` disposition
requires an `equivalent_guard` explaining why behavior really cannot differ; a
green test suite alone is not evidence of equivalence. See the [pinned runner
README](https://github.com/cortexkit/commons/blob/46cc166b0df2edcfd14b3eb54ed6eeac588fed69/crates/cortexkit-mutate/README.md)
for multi-edit anchors, command rows, platform selection, and HUB review.

## Mac replay and desktop-only rows

Unix mock-stream residency tests are platform-gated to Linux and macOS, not
Windows; they do not require ANE hardware. Tests compiled only on macOS carry
`platforms = ["macos"]`. A `desk_only` reason is an explicit unautomated
obligation, **not CAUGHT credit**, and replay skips it even on a Mac. Run that row
on a real Mac by copying it to a temporary catalogue under `target/mutations/`,
removing only `desk_only`, and replaying with `--only` and a retained JSON report.
Keep `platforms`, features, ignored selection, anchors and expectations unchanged.
Install the required Apple developer tools/artifacts before a hardware proof.

For the portable baseline on a Mac, isolate HOME from optional model snapshots
while retaining the real Cargo and rustup installations:

```sh
real_home="$HOME"
portable_home="$PWD/target/mutations/portable-home"
mkdir -p "$portable_home"
HOME="$portable_home" CARGO_HOME="${CARGO_HOME:-$real_home/.cargo}" \
  RUSTUP_HOME="${RUSTUP_HOME:-$real_home/.rustup}" \
  ckdev-mutate run --all --report target/mutations/portable.json
```

`prove` prepares the **whole package** baseline even for a narrow target, because
it can diagnose survivors package-wide. With a developer's cached weights that
baseline reached
`catalog_e2e::catalog_real_metal_redirect_download_self_checks_and_serves_without_probe`
and did not terminate before the runner's test deadline. An isolated HOME keeps
optional hardware fixtures out of portable guard proofs; their absence/skip is
not a hardware pass. Apply the same environment to `prove` and a portable `--broad`
audit. Do not use this environment for a desk-only hardware proof: supply the real
fixtures and report that proof separately.
