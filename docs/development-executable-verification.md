# Development executable names

On a machine that also runs the installed CortexKit fleet, a process whose
executable file name starts with `ck-` must be an installed binary. Process
listings (Activity Monitor, `ps`) then show only production under `ck-`, and a
check that flags stray `ck-` processes can trust that. Anything a test, script
or certification run executes from a build directory runs under a `ckdev-`
name instead. Cargo's own test harness binaries keep their usual
`<crate>_<hash>` names and are not affected.

## How the code does it

- `synapse_core::dev_binary::ckdev_binary(binary, scratch)` copies a built
  executable beside its artifact into `ckdev-exec/<sha256-prefix>/ckdev-<name>`
  (the `ck-` prefix is dropped, and `.exe` is kept). The scratch argument is
  retained for callers but is no longer used. The first 16 hexadecimal
  characters of the full SHA-256 select a content-addressed directory, so every
  caller for the same bytes reuses one copy. A caller copies to a unique temporary
  file in that directory and publishes it with an atomic rename; concurrent callers therefore only see a
  complete copy. Unix copies have the executable bit set.
- The copy is deliberate: under machine load, fresh hard links to Cargo build
  outputs were observed to have spawned processes exit on macOS signal 9, while
  a content-addressed copy beside the artifact was stable in the same
  reproduction. macOS did not provide a reason for those signals, so the copy
  is the tested remedy, not a proven explanation of the failure mechanism.
- `ckdev_binary_hard_link` is the same helper with copying forbidden.
  `ckdev-synapse-certify run` uses it (including its offline candidate source
  probe) because the certification record attests
  the SHA-256 of the built binaries: running a hard link executes those exact
  bytes, and a layout that would need a copy is refused. This helper remains a
  hard link in the caller's scratch directory and still rejects cross-volume
  layouts. The record keeps the original role and file names.
- The module finds an unconfigured worker by its `ck-synapse-worker-<engine>`
  name beside its own executable. Tests therefore set each engine's worker
  variable (and `SYNAPSE_ANE_SWIFT_WORKER` for the Core ML launcher) to a
  `ckdev-` copy, in `crates/synapse-module/tests/common/mod.rs`. Production
  resolution is unchanged.
- `scripts/check-release-candidate.py` and
  `scripts/test-owned-cuda-package.ps1` smoke-test extracted release assets
  through a temporary `ckdev-` link. Placing and staging a production binary
  (under `~/.local/share/cortexkit/bin/` or `staging/`) keeps the `ck-` name.

## The guard

`crates/synapse-core/tests/dev_binary_guard.rs` parses every Rust test source
in the workspace with `syn`, including inline `#[cfg(test)]` modules, and fails
if a process spawn's program refers to a `ck-` executable (a
`CARGO_BIN_EXE_ck-*` path, a string literal whose file name or any path
component starts with `ck-`, or a variable bound to one) without going through
the helper. It judges each statement on its own, so a direct spawn on the line
right after a wrapped one is still caught; two planted-control tests prove
that, and the reverse case.

## Checking it by hand

`scripts/sample-dev-images.py` samples `ps` every 0.5 s while a command runs
and counts the `ck-` and `ckdev-` executables that belong to that command's
process tree. It matches processes by PID plus start time, so a PID the system
reuses for an unrelated process is not counted. A run of the module's e2e
suite should show zero `ck-` executables, and none left afterwards.
