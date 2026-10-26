# Development and verification

## Build and test a change

Implement the change, then run only the filtered tests for the code you touched,
plus formatting and Clippy for the touched crates. Run them on the Linux build
server (`ck motor run --vcpus 4 --cwd "$PWD" -- COMMAND...`, or
`runon: "linux,4c"` from the agent shell), from the repository root, with
`--locked` so Cargo refuses dependency resolution that would rewrite the
committed `Cargo.lock`:

```sh
cargo test --locked -p <crate> <test-filter>
cargo fmt --all --check
cargo clippy --locked -p <crate> --all-targets -- -D warnings
```

For a bug fix, add one ordinary test that fails without the fix, and show that
once: revert only the fix, run that test, see it fail, then restore the fix.

Do not write mutation rows, break proofs or deliberate-break controls (edits
that break production code on purpose to prove some test notices), and do not
run full suites or repeat runs to prove stability. The head runs the full suite
once per integrated batch, and CI runs it before master. For reference, the
module's full portable set is:

```sh
cargo build --locked --no-default-features -p synapse-worker-cuda -p synapse-worker-vulkan
cargo check --locked -p synapse-module --features test-support
cargo test --locked -p synapse-module --features test-support --lib --test it --test skeleton_e2e --test wire_contract
cargo fmt --all --check
bash scripts/public-banlist-check.sh
bash scripts/check-cortexkit-layout.sh
```

The feature-off GPU worker builds supply the library's refusal fixtures, not
hardware execution. Apple Metal builds need the full Xcode toolchain; verify `xcrun --find metal`
before building those crates. Linux tests prove portable behavior, not Apple
hardware execution. Checkpoint-dependent or ignored tests are separate gates.
To reverse a source change, restore the compatible source and rerun the same
narrow tests; a passing unrelated package is not a substitute.

Before submitting a change for merge, run `bash scripts/land-gates.sh` for fast
static checks and targeted test feedback.
Use `--plan` to inspect its changed-target selection or `--base REF` to choose
the comparison base. `--full` retains the serial portable module landing set;
it is not hardware certification. Train CI no longer runs the whole-process
`skeleton_e2e` suite: it runs on the daily schedule, on manual dispatch and in
`--full`, so an end-to-end-only regression can surface after landing.

## Execute development binaries without confusing installed ones

Tests and development drivers use `ckdev-*` executable names. Cargo's own test
harness filenames are unaffected. `synapse_core::dev_binary::ckdev_binary` makes
an atomic content-addressed copy under `$TMPDIR/synapse-exec/<sha256-prefix>/`
only when the `TMPDIR` value as written is absolute and both that path and its
filesystem-resolved canonical path are outside macOS's `/var/folders` and
`/private/var/folders` trees; checking both catches `/var` links into
`/private/var`. Otherwise it uses
`/tmp/synapse-exec/<sha256-prefix>/`. These directories are private to the
current user, and callers with identical bytes reuse the same copy. Windows
keeps its existing artifact-adjacent location. Certification uses
`ckdev_binary_hard_link` instead: it remains beside the candidate scratch area
and refuses cross-filesystem layouts rather than executing a copy of the
attested assets.
Worker overrides in module tests point to these development images; installed
worker discovery is unchanged.

Verify that tests launch development-named images rather than installed binaries:

```sh
cargo test --locked -p synapse-core --test dev_binary_guard
```

`test_launches_use_development_images` scans real test sources;
`planted_direct_spawn_after_wrapped_spawn_is_rejected` and
`planted_paths_aliases_and_multiline_wrappers_are_judged_separately` prove the
scanner does not accept a nearby safe spawn as protection for an unsafe one.
Remove only copies or links created by your own completed run, never a live
installed executable.
