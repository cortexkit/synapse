# Development and verification

## Build and test a narrow change

From the repository root, keep the committed `Cargo.lock` unchanged and use
`--locked` so Cargo refuses dependency resolution that would rewrite it:

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

## Validate and replay mutation controls

Use the mutation-runner version pinned by `.github/workflows/tests.yml` and its
required cargo-nextest/Python dependencies. The executable is `ckdev-mutate`.
`mutations.d/` is discovered beside the default catalogue path; do not create a
second root catalogue. Each TOML control in `mutations.d/` names the exact test
that must fail when its production guard is deliberately broken, plus a
property-specific expected failure message. Compilation failures and timeouts
are not successful catches.

```sh
ckdev-mutate --version
cargo nextest --version
ckdev-mutate check
mkdir -p target/mutations
ckdev-mutate run --diff origin/master --report target/mutations/diff.json
```

`--diff` selects committed changes. Run one mutation process per checkout. The
runner restores source bytes; never check out a target while a mutant is running,
or you may erase the deliberate break and manufacture a passing proof. Keep any
evidence needed for review outside regenerable build directories after replay.

## Add a proved control

Read the guard and its production callers. Use the exact live anchor and exact
full test name, then prove the row rather than hand-writing a catch claim:

```sh
ckdev-mutate prove --id example-refusal --guards 'state the prevented failure' \
  --file path/to/source.rs --old 'exact live anchor' --new 'deliberate break' \
  --test-file path/to/test.rs --package package-name --target=--lib \
  --expect-red module::tests::exact_full_name --expect-message 'property-specific failure' \
  --select expected --report target/mutations/example.json
ckdev-mutate check
```

Verify that the named test goes red for the intended assertion and that unrelated
tests do not fail. A filtered proof is not a breadth claim. A `hub` row names the
shared property and additional affected targets and must not use filtered
selection. `desk_only` and platform skips are obligations, not hardware proof.
After a run, inspect the diff and lockfile before committing; restore only owned
mutations, not somebody else's uncommitted implementation.
