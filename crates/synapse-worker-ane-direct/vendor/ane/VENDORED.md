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
  compile or the load fails. Before this change the client never removed it:
  a directory stayed at least while its program was loaded, and processes
  that ended without unloading left them behind for good. Only
  that exact path is removed, never by pattern, because other applications
  share the private compiler and temp folder. On macOS 27.0 (26A428) every
  compile received a fresh identifier; in case two compiles ever share a
  directory, each holds a shared `flock` on it and the last one to finish
  removes it. On an M4 under that build, deleting the directories after load
  left every inference output byte-identical, across eviction, recompilation
  and unload. The `client::tests` unit tests cover the removal with a
  stand-in for the private compile and load.
- Local modification, not upstream behaviour: `Graph::to_ops_and_shapes`
  emits constants in tensor-id order instead of `HashMap` iteration order, so
  identical graphs emit byte-identical MIL text and weights, within a process
  and across processes. The private compiler identifies a program by hashes
  of those bytes, so before this change every build of the same layer got a
  new identifier. On an M6 with macOS 27.0 (26A425), three random-order
  compiles and the sorted compile of one gte-modernbert layer gave
  byte-identical outputs. The `graph::compile::tests` unit tests cover it.
- Local modification, not upstream behaviour: `client::compile_network`
  reuses aned's compiled-program cache. When `_ANEInMemoryModel
  compiledModelExists` reports a cached program for the identifier (the
  selector is feature-detected; a missing selector counts as not cached), it
  calls `loadWithQoS:` alone, writing no sources and calling no compile. On a
  miss, or if that load fails because the cached copy is unusable, it
  compiles as before (write sources, `compileWithQoS:`, `loadWithQoS:`,
  remove the sources) from a fresh model object. A cached load refused for
  lack of Neural Engine resources (`Error::is_resource_refusal`) is returned
  unchanged without compiling. The binding keeps only the `NSError`
  description, so that test matches the refusal's phrase on load errors
  only. A load without a cached program fails aned's bundle verification,
  and `compileWithQoS:` always recompiles, so neither shortcut is used.
  `Graph::compile_gated` and `client::compile_network_gated` run a caller
  check just before a compile (never before a cached load), and
  `Executable::origin` reports `ProgramOrigin::CacheHit` or `Compiled`. The
  development accessor `Graph::program_identifier` returns the identifier
  without compiling. The `client::tests` unit tests cover the hit, miss,
  failed-hit, resource-refused hit and refused-compile sequences with a
  stand-in program.
- Local modification, not upstream behaviour: `client::compiled_program_exists`
  and `Graph::compiled_program_exists` build a graph's program sources and
  ask `compiledModelExists` the same way `client::compile_network` does
  before it loads, without compiling, loading or writing anything. The
  worker uses it while loading a model to report which ladder rungs aned
  already holds compiled, so the module can predict a fast load for them.

`.cargo-checksum.json` records SHA-256 for every copied source, build script,
manifest and this provenance document (excluding the checksum manifest itself).
The `vendored_binding_checksums_match` test recomputes the hashes and checks the
file inventory. Any intentional edit must update the checksum manifest and this
provenance document if it changes the local compatibility patch.
