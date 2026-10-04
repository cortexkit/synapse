# Vendored Apple Neural Engine binding

Source: https://github.com/mutable-state-inc/siliconswarm-at-ensue-plugin

Pinned revision: `ec54af9501d4bfd0cf3a4b162e59022dee2118cb`

Source directory: `ane_kernel/crates/ane`

The upstream workspace declares `license = "MIT"` at
`ane_kernel/Cargo.toml:11`. No license text exists in that upstream revision;
this copy does not invent one. The library's explicit license field remains
`MIT`. Only the library sources, build script and manifest are copied, not
examples or benchmarks.

Local compatibility changes:

- Workspace-inherited metadata and dependency versions are explicit. The
  unavailable workspace-relative readme and example declarations are removed.
- The library, framework linkage and dependencies are macOS-only so the
  containing workspace also builds on Linux and Windows.
- A read-only `Graph::source_payload` accessor exposes the exact submitted MIL
  and weight bytes for resource-limit diagnostics; it does not change compilation.
- `metal` uses the containing workspace's 0.29 API (the binding builds against
  it); `hf-hub` enables only the synchronous `ureq` transport, and `tokenizers`
  disables default features, retaining `onig`. This avoids introducing unused
  asynchronous/download/trainer infrastructure into the serving binary.

`.cargo-checksum.json` records SHA-256 for every copied source, build script,
manifest and this provenance document (excluding the checksum manifest itself).
The `vendored_binding_checksums_match` test recomputes the hashes and checks the
file inventory. Any intentional edit must update the checksum manifest and this
provenance document if it changes the local compatibility patch.
