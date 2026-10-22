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
- `Executable::run_cached_profiled` times cached-request preparation and the
  synchronous evaluate call without changing the request or execution semantics.
- The crate re-exports the standard `objc2::rc::autoreleasepool` utility so worker
  operations drain temporary Objective-C ownership before acknowledging release.
- `diagnostics::reclaim` is an explicitly invoked development probe of model
  purge, client purge, fresh-client unload, and shared-client reference release.
  Selector names and signatures were enumerated from the Objective-C runtime on
  OS build 26A434; serving code does not call it or assume these calls reclaim capacity.
- `ANEInMemoryModel::unload` logs the BOOL and NSError description when
  `ANE_UNLOAD_DIAGNOSTICS` is set. This observes `Executable::Drop` unload calls
  without changing their return type or retry policy; success is not a
  guarantee that hardware capacity is immediately reclaimed.
- A read-only `Graph::source_payload` accessor exposes the exact submitted MIL
  and weight bytes for resource-limit diagnostics; it does not change compilation.
- `metal` uses the containing workspace's 0.29 API (the binding builds against
  it); `hf-hub` enables only the synchronous `ureq` transport, and `tokenizers`
  disables default features, retaining `onig`. This avoids introducing unused
  asynchronous/download/trainer infrastructure into the serving binary.
- Local modification, not upstream behaviour: `client::compile_network` removes
  each layer program's compiler temp directory (`<temp>/<hex identifier>/`,
  holding `model.mil`, `weights/weight.bin` and the compiler's `net.plist` and
  `data`) once the program has loaded, and also when writing the files, the
  compile or the load fails. Upstream never removes it, so every cold compile
  left about twice the layer's weight size in the per-user temp folder. Only
  that exact path is removed, never by pattern, because other applications
  share the private compiler and temp folder. Compiles of the same program
  share the directory under a shared `flock`; the last one to finish removes
  it. The `client::tests` unit tests cover this with a stand-in for the
  private compile and load.

`.cargo-checksum.json` records SHA-256 for every copied source, build script,
manifest and this provenance document (excluding the checksum manifest itself).
The `vendored_binding_checksums_match` test recomputes the hashes and checks the
file inventory. Any intentional edit must update the checksum manifest and this
provenance document if it changes the local compatibility patch.
