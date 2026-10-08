# Synapse cross-platform runtime census — October 2026

## Scope, ruling and evidence

Source baseline: `6b172f4db9932ea8b8a6fa0e27653420ba231c6d`. This is a report, not a portability implementation. Findings concern the installed module, its workers, certification and the checked-in delivery/tooling paths. Experimental bench paths are identified separately from installed runtime defects. Line numbers refer to that baseline, not to a future implementation.

The product contract is **macOS, Linux and Windows support**. Embedding and reranking must use owned accelerators: Metal/Neural Engine on Apple, CUDA/Vulkan on Linux and Windows. There must be **no CPU lanes**. A host without a supported accelerator correctly receives `backend_unavailable` and can use an explicitly configured remote gateway; that refusal is not a defect. Decode currently means llama.cpp through `ck-synapse-worker-llama`. The owned Metal decode stack is parked; its non-Apple refusal is not an active decode portability defect.

Evidence labels below:

- **Read**: inspected code/configuration in this checkout; a source-derived outcome, not a GPU execution claim.
- **Linux run**: commands executed on the Linux build server with `runon: "linux"`, recorded below. No accelerator discovery or GPU inference was run.
- **Unresolved**: an explicit finding requiring a dependency, installed environment, platform run or hardware result that this census could not establish. These are not silently counted as working.

Severity: **Critical** blocks ordinary module startup or the primary local-inference installation path; **High** blocks a supported lane or undermines machine-bound certification; **Medium** is a conditional installation/security/reliability problem; **Low** is ancillary tooling or a safe-but-slower path. A label such as *refuses* includes a returned error or startup error, not necessarily a crash. *Crashes (loader)* means the OS loader exits before Rust can return its structured refusal. None of the evidence establishes an inference-time segmentation fault.

### Safe Linux observations made for this census

1. Python 3.14.4 on `Linux-7.0.0-34-generic-x86_64-with-glibc2.43`: `uname -sr` returned `Linux 7.0.0-34-generic`; `uname -m` returned `x86_64`. Both completed under a three-second subprocess deadline. This runner had `HOME`, not `XDG_DATA_HOME`. This establishes the prerequisites of the Linux profile branch **on this host**, not native Windows identity or GPU identity.
2. `cargo test --locked -p synapse-core --lib machine_profile::tests -- --nocapture`: **10 passed**, 0 failed, 69 filtered out. Rust `1.99.0 (b940084d7 2026-09-28)`, Cargo `1.99.0 (5f94df478 2026-08-27)`. These test hashing, injected collector answers/refusals, missing/failing utilities and a child deadline. Most collector tests use a prober seam; this is not a real installed module boot.
3. `cargo test --locked -p synapse-core --lib worker_transport -- --nocapture`: **6 passed**, 0 failed, 73 filtered out, same Cargo version. These cover endpoint-name helpers and Unix handshake validation. The handshake helper uses `tokio::io::duplex`, **not a filesystem socket or a spawned GPU worker** (`crates/synapse-core/src/worker_transport/unix.rs:297-323`). No Windows run was available.

No ambient process was suspended or killed. The profile deadline test starts and kills its own `sleep` child. No source, manifest, lockfile, configuration or script was changed.

## Ranked findings

### F1 — Native Windows home resolution can stop the module before it serves

**Critical on Windows; Linux works with its usual HOME, degraded with custom XDG layouts.**

- **Locations:** `crates/synapse-module/src/lib.rs:254-258`, `:377-414` (singleton lease); `:16446-16470` (fallback store); `crates/synapse-core/src/cache.rs:125-140` (model cache); initialization order `crates/synapse-module/src/lib.rs:2180-2192`.
- **Linux:** lease and cache work with `HOME`; the store honors `XDG_DATA_HOME` then `HOME/.local/share`. Lease/cache ignore `XDG_DATA_HOME`, so the three roots can diverge. An environment with only XDG data home still **refuses** lease/cache resolution unless their explicit overrides are supplied.
- **Windows:** a normal native account/service may have `USERPROFILE`, `APPDATA` and `LOCALAPPDATA` but no `HOME`. Synapse's lease resolver **refuses** before `serve`; a supplied lease root only exposes the independent store/cache HOME dependencies next. Daemon-supplied storage bypasses the store fallback, **not** the cache or lease resolver. Supplying `HOME`/explicit overrides can work around these particular errors; it does not fix F2.
- **How known:** Read; the Linux prerequisite run confirms HOME only for that runner. Do not infer native Windows success from CI's shell environment.
- **Fix shape:** one platform-aware root policy with native Windows known folders and coherent XDG data/cache/state placement on Linux; preserve explicit overrides. Test a native Windows environment with HOME absent, plus service and custom-XDG environments. Do not merely fix the user-config resolver: that is a different path (U1).

### F2 — The runtime profile requires `uname` on native Windows and fails the whole boot

**Critical on Windows; conditional refusal on minimal Linux.**

- **Locations:** `crates/synapse-core/src/machine_profile.rs:86-98`, `:157-203`, `:287-350`; propagation `crates/synapse-module/src/lib.rs:2199-2205`, overrides `:3590-3598`.
- **Linux:** **works** where `uname` is installed and answers; the two probes succeeded on the build server. Without it, or with an empty/failing/hanging answer, profile collection **refuses**. This external utility dependency is unnecessary for a native collector.
- **Windows:** the non-macOS branch also executes `uname -m` and `uname -sr`. Native Windows does not supply that command. Without Git/MSYS/Cygwin or a deliberately installed utility on PATH it **refuses startup**, not just local inference. Initialization collects the profile before constructing the remote gateway, so remote-only use cannot escape this error. The OS-build override is applied **after** collection and cannot rescue a missing probe.
- **How known:** Read plus Linux run/unit tests. A Windows utility supplied by a development shell is not a native implementation, and its output's meaning is unresolved.
- **Fix shape:** native Linux identity (`uname(2)`/OS release data as appropriate) and native Windows OS/build, architecture, processor and memory APIs. Keep bounded, explicit failures for genuinely unavailable identity; do not substitute a rotating placeholder or require a Unix shell for an installed Windows service.

### F3 — Catalog GPU discovery is empty on both non-Apple platforms

**Critical for the primary catalog installation/serving path on Linux and Windows.**

- **Locations:** runnable snapshot `crates/synapse-module/src/lib.rs:2274-2281`; detector `:22634-22665`; download refusal `:23076-23118`, `:23149-23160`.
- **Linux:** **refuses** catalog download/selection with `backend_unavailable` even with a supported CUDA/Vulkan GPU and installed workers: production detection returns an empty set.
- **Windows:** **refuses** for exactly the same reason. The detector has no CUDA or Vulkan probe on either OS. Metal/ANE receive `not_supported_on_platform`; other absent backends are called `worker_missing`, which does not establish that the worker is actually missing.
- **macOS comparison:** only `metal::Device::system_default()` populates the production set. ANE is not detected there either; this is not a universal accelerator-discovery implementation.
- **How known:** Read. `SYNAPSE_TEST_RUNNABLE_BACKENDS` can inject a set only in `test-support` builds; it is not a release-install workaround. No GPU was probed here.
- **Fix shape:** populate supported lanes from platform device/floor and installed-worker discovery, with accurate distinction among missing worker, loader, driver and unsupported hardware. Decide refresh/invalidation of the startup snapshot. Preserve the intentional refusal for a genuinely unsupported GPU. Explicit profile preloads have separate routing; this finding does **not** prove all manually configured GPU workers are unreachable.

### F4 — Repairing discovery alone cannot make the shipped catalog cross-platform

**Critical for Linux/Windows catalog functionality; independent of F3.**

- **Locations:** `crates/synapse-module/src/catalog/mod.rs:35-45`, `:871-936`; `crates/synapse-module/src/catalog/models.json`; construction `crates/synapse-module/src/lib.rs:23766-23838`; load gate `:6418-6424`.
- **Linux:** **refuses**: the frozen release catalog has Metal-only backends for three models and no backend for Qwen3 reranking. Allowed catalog engines contain only `owned-metal`. Even a fixture declaring a CUDA/Vulkan backend is constructed as an `owned-metal` model by `catalog_lane_spec`, not dispatched by its backend/engine.
- **Windows:** **refuses** identically. Selecting a backend in a test catalog is not proof the corresponding worker gets launched.
- **How known:** Read of allowed/frozen values and the unconditional `"owned-metal"` argument. The generic/profile preload path separately recognizes `owned-cuda` and `owned-vulkan` (`crates/synapse-module/src/lib.rs:3063-3120`, `:6401-6464`); it does not supply missing release-catalog wiring.
- **Fix shape:** ship evidence-backed CUDA/Vulkan entries, allow their actual engines, construct lane specs/identities/load payloads from those entries, and resolve the corresponding supervised worker. Validate each backend's package/profile/fingerprint rather than renaming a Metal lane. Test the real release catalog without runnable-set or deterministic-engine overrides. The Qwen3 reranker omission is shared with macOS, not a Windows-only defect.

### F5 — llama.cpp defaults to a CPU lane off macOS, contrary to the ruling

**High for Linux/Windows decode installations; an actual unsupported default, not a request to add CPU fallback.**

- **Locations:** `crates/synapse-worker-llama/Cargo.toml:10-16`, `:32-38`; `crates/synapse-worker-llama/src/runner.rs:436-449`, `:465-492`, `:1101-1125`; module fallback `crates/synapse-module/src/lib.rs:6857-6870`, `:6941-6947`.
- **Linux:** ordinary/default llama worker builds choose `cpu`, use zero GPU layers, and the module's undeclared/legacy backend default also says `cpu`. A load can therefore **work on CPU**, but is **degraded and prohibited** by the product contract instead of selecting a supported accelerator or refusing.
- **Windows:** same **degraded/prohibited CPU execution** in ordinary builds. Named-pipe support exists; there is no blanket missing Windows runner. macOS's default `cpu` feature paradoxically selects Metal because its dependencies enable Metal.
- **How known:** Read, not inference. Backend equality validation prevents a differently compiled worker from silently satisfying the declared backend; it does not prohibit declaring CPU. The legacy preload/model path also accepts llama embedding/reranking (`crates/synapse-module/src/lib.rs:16765-16795`), so the owned-accelerator-only policy needs enforcement there, not just in the curated catalog. GPU layer count is configurable and backend identity alone is not proof all layers execute on GPU.
- **Fix shape:** remove production CPU choices, define CUDA/Vulkan decode selection and executable/artifact identity, require actual accelerator admission and offload policy, and return `backend_unavailable` for unsupported hosts. Keep deterministic/mock tests separate from shipping decode policy. Do not revive parked owned Metal decode as the portability fix.

### F6 — Candidate builds select both llama GPU features while module defaults still declare CPU

**High for both platforms, especially Vulkan-only/AMD decode hosts.**

- **Locations:** `.github/workflows/release-candidate.yml:60-73`, `:180-191`; llama feature mapping `crates/synapse-worker-llama/Cargo.toml:13-16`; precedence `crates/synapse-worker-llama/src/runner.rs:436-449`, load equality `:1109-1125`; module fallback `crates/synapse-module/src/lib.rs:6941-6947`.
- **Linux:** the multi-package build passes unqualified `--features cuda --features vulkan`, enabling those same-named features on the selected llama package as well as the owned workers. The llama worker advertises **CUDA** because that branch wins. A preload/legacy row without an explicit backend requests **CPU** and **refuses** load. There is no selection of Vulkan for a Vulkan-only host in this binary's backend declaration.
- **Windows:** same **refusal**/CUDA precedence; the manual Windows Vulkan-only build is a different build command, not the candidate artifact. An explicitly declared CUDA model can avoid the identity mismatch, but does not prove this artifact serves Vulkan-only hardware.
- **How known:** Read; no candidate native compilation/inference was attempted here. Dependency-loader behavior of the combined llama artifact remains U4.
- **Fix shape:** qualify owned-worker features by package; build and package intentional, unambiguous llama backend variants or implement proven runtime dispatch. Bind module model/backend selection to the shipped worker identity rather than the OS's old CPU default. Verify decode on both CUDA and Vulkan hardware with the actual candidate bytes.

### F7 — Linux CUDA archives omit required userspace runtime libraries

**High: a real Linux install may crash in the loader before even probing its GPU.**

- **Locations:** links `crates/synapse-engine-cuda/build.rs:76-99`; ELF contract `scripts/check-release-candidate.py:104-124`; packaging `.github/workflows/release-candidate.yml:235-258`; compensating smoke setup `.github/workflows/release.yml:75-105`; Windows recipe `scripts/package-owned-cuda.ps1:10-43`.
- **Linux:** the worker is required to depend on `libcublas.so.13`, `libcublasLt.so.13`, `libcudart.so.13`. Its archive stages just the executable. A machine with an adequate GPU driver but without these matching CUDA userspace libraries **crashes (loader)**, rather than reporting `backend_unavailable`. The release smoke downloads pinned CUDA libraries and injects `LD_LIBRARY_PATH` first, so its success would not prove the archive is self-contained.
- **Windows:** the owned-CUDA archive **works at the packaging-contract level** by placing runtime DLLs, hashes and licenses beside the executable. Runtime loading is explicit/delay-loaded and produces `cuda_runtime_missing:<dll>` if absent (`crates/synapse-engine-cuda/src/cuda.rs:128-164`). Hardware success was not run; the driver is intentionally system-installed on both platforms.
- **How known:** Read. The checked-in ELF test deliberately checks the Linux loader failure. No loader failure was induced against an actual candidate in this census.
- **Fix shape:** bundle legally redistributable Linux userspace libraries with a private loader path/RPATH, or supply a real dependency-install contract with exact versions. Test extraction/launch in a clean install image, not only with the build job's toolkit or injected runtime paths. Do not bundle the NVIDIA driver.

### F8 — Linux/Windows certification profiles do not identify their accelerators

**High: evidence can survive an accelerator/driver/memory change that should trigger re-certification.**

- **Locations:** `crates/synapse-core/src/machine_profile.rs:57-65`, `:86-98`, `:101-154`, `:178-203`; runtime use `crates/synapse-module/src/lib.rs:2206-2211`, certification lookup `:16284-16304`.
- **Linux:** **degraded** identity: `chip_model` is just `uname -m`, RAM is always `unknown`, and no GPU vendor/device/UUID, VRAM or driver is collected. Engine identities are software declarations. Machines sharing the same kernel/architecture and declared engines can therefore have the same profile despite different GPUs/RAM. A GPU/driver change alone need not change the serving profile hash.
- **Windows:** **refuses** without uname (F2); with a supplied uname it has the same **degraded** hardware identity. A Unix compatibility utility is also an unresolved source of native OS-build semantics.
- **macOS comparison:** chip brand/model and bucketed memory are measured with sysctl; these at least distinguish Apple chip/memory classes. Neither implementation establishes a unique physical-machine identifier. This is a certification-input defect, not a claim of a unique-host hash on macOS.
- **How known:** Read plus Linux uname outputs. Release certification's separate machine record *does* collect GPU/driver details (`crates/synapse-certify/src/live.rs:780-842`), but that record is not the runtime `MachineProfile` schema.
- **Fix shape:** version the runtime profile schema to include the selected accelerator class/device and relevant driver/runtime/memory identity; bind evidence to the actual chosen adapter and rotate/re-certify intentionally on changes. Keep ephemeral identifiers out unless their stability is defined. Test GPU replacement and driver updates with injected readings, then native probes.

### F9 — Windows file verification loses both the persistent fast path and change detection

**Medium: Windows rehashes files but loses concurrent-modification detection and the cold-load verification cache.**

- **Locations:** `crates/synapse-module/src/lib.rs:24470-24525`, validation caller `:24567-24598`.
- **Linux:** **works** with Unix device/inode/size/nanosecond mtime/ctime stamps. A matching persisted stamp avoids hashing; differing before/after stamps refuse the verification result.
- **Windows:** stamp collection is an explicit **no-op** returning `None`. Consequently verification **degrades** to hashing on each unloaded-lane verification and never records/reuses a verified stamp. More importantly, `before != after` compares `None` with `None`, so it cannot detect a file replacement/write during hashing. Digest and size checks still execute; this is not a blanket digest bypass or proof of an exploit.
- **How known:** Read. The dormant Windows catalog problem in F3/F4 currently limits reachability through the production catalog; this will become active when catalog routing is fixed.
- **Fix shape:** use Windows stable file IDs plus appropriate change metadata/open-handle protection, or a platform-safe immutable/leased snapshot protocol. Preserve the rule that restoring mtime alone cannot revive stale verification. Do not replace ctime with a coarse mtime or call an always-None comparison a race guard.

### F10 — Worker endpoint isolation differs: shared Unix temp paths versus global Windows pipe names

**Medium conditional start refusal and local isolation risk.**

- **Locations:** default runtime dir/worker id `crates/synapse-module/src/lib.rs:6667-6678`; digest/names `crates/synapse-core/src/worker_transport/mod.rs:29-45`; Unix bind `crates/synapse-core/src/worker_transport/unix.rs:45-52`; Windows bind `crates/synapse-core/src/worker_transport/windows.rs:36-49`.
- **Linux:** normally **works**, but the default is a shared `temp_dir()/synapse-workers`, not a private per-user/process runtime directory. An existing endpoint entry is removed without a file-type/owner check. A long configured runtime root can still **refuse** AF_UNIX binding: only the basename is shortened, not the complete path. Same shared-temp/length concern exists on macOS.
- **Windows:** runtime_dir is ignored; `\\.\pipe\synapse-<digest>` depends only on the stable worker id, not the user, module generation or process. `first_pipe_instance(true)` **refuses** if a pipe with that name is still held by an old/other instance. Distinct users have per-home singleton roots but can address the same machine-global pipe names. Unlike Unix, there is no stale filesystem entry to unlink: a live server object is a real conflict.
- **How known:** Read. Explicit pipe DACL/remote-client restrictions are unresolved in U2; do not assume `first_pipe_instance` is an authorization control. Collision frequency was not measured.
- **Fix shape:** private per-user/generation runtime endpoints, ownership/type checks before Unix cleanup, native pipe ACL/local-only policy, and collision tests across users and stale children. Retain nonce binding; do not weaken it to resolve a collision. Preflight Unix complete-path limits.

### F11 — Standard Windows file URLs are parsed as native path text, not URLs

**Medium for explicit local artifact sources on Windows.**

- **Locations:** URL construction `crates/synapse-module/src/lib.rs:3586-3588`; base joining `:7755-7762`; downloads `:7940-7982`; cache input `crates/synapse-core/src/cache.rs:549-555`.
- **Linux:** **works** for the internally generated `file:///absolute/path` convention. Encoded URLs still **refuse** for paths requiring percent decoding, and URL authorities are not handled as authorities; this is a shared limitation.
- **Windows:** internal `file://` plus native `C:\...` can work because the reader strips the prefix and opens the remainder. A standard `file:///C:/...` becomes `/C:/...`, and `file://server/share/...` becomes a relative `server/share/...`, rather than a drive path/UNC share. These inputs **refuse** or address the wrong relative path, depending on layout. This is a code-derived path interpretation, not a Windows filesystem run.
- **How known:** Read of literal prefix stripping. Native PathBuf joins and the sibling executable resolver are separate and Windows-aware (`crates/synapse-module/src/lib.rs:6543-6556` appends `.exe`).
- **Fix shape:** use a tested `Url::from_file_path`/`to_file_path` conversion, distinguish native paths from URI sources, and cover drive letters, UNC, spaces/percent encoding and non-Unicode path policy. Do not normalize a native Windows path by just deleting its leading slash.

### F12 — Windows certification aliases leave package DLLs behind

**Medium-to-High for obtaining release certification on a clean Windows rig.**

- **Locations:** `crates/synapse-core/src/dev_binary.rs:19-53`; floor aliases `crates/synapse-certify/src/live.rs:80-101`; serving-worker aliases `:265-290`; DLL search `crates/synapse-engine-cuda/src/cuda.rs:140-154`; bundled sidecars `scripts/package-owned-cuda.ps1:10-25`.
- **Linux:** **works conditionally** when its required CUDA libraries are installed/on `LD_LIBRARY_PATH`; an executable alias does not affect that search contract. Cross-volume certification aliasing **refuses intentionally**, since attestation requires the same file, not a copy.
- **Windows:** the certify helper hard-links only the worker executable into a new `dev-bin-*` subdirectory beneath its run tree. It does not stage the CUDA DLLs there or add the asset directory to the child environment. In a sidecar-only installation with no toolkit on PATH, CUDA's executable-directory search cannot find them and **refuses** `cuda_runtime_missing`. System/PATH-installed libraries may mask this, so the exact rig outcome remains unrun, not a universal failure claim.
- **How known:** Read. `fs::hard_link` is not itself a Unix-only API, and `.exe` names are preserved; the defect is the alias's runtime closure, not lack of Windows hard links.
- **Fix shape:** keep certification executable attestation while preserving its bundled runtime dependency layout, or set an explicit scoped trusted DLL directory for the child. Hash/attest the runtime files as well. Test with only the extracted archive available and no toolkit paths. Never replace the hard-link requirement with an unverified copy solely to make the test pass.

### F13 — Ordinary source builds produce accelerator workers whose backends are disabled

**Medium for source installers: CUDA/Vulkan engines are disabled by default, although release-candidate builds explicitly enable their features.**

- **Locations:** `crates/synapse-worker-cuda/Cargo.toml:14-16`, `crates/synapse-worker-cuda/src/main.rs:351-365`; `crates/synapse-worker-vulkan/Cargo.toml:13-16`, `crates/synapse-worker-vulkan/src/lib.rs:26-34`; root defaults `Cargo.toml:6-20`.
- **Linux:** a default root build includes executable names for CUDA/Vulkan but their production engines **refuse** model loading/discovery (`backend_missing`/`vulkan_no_device`) even on supported hardware. Presence of the binary alone is not readiness.
- **Windows:** same **refusal** for default source builds. Proper feature/toolkit builds contain real Windows code; these are not OS-wide stubs. The corresponding ordinary macOS build enables its Metal dependencies without requiring the CUDA/Vulkan toolkit and feature opt-ins.
- **How known:** Read. The opt-in feature contract is deliberate, so this is an installation/default-build integration problem, not a claim that all shipped candidate workers are disabled. `.github/workflows/release-candidate.yml:60-73` enables owned CUDA/Vulkan.
- **Fix shape:** source-install/build entrypoints must select the actual supported accelerated workers and validate enabled features/identity, with package-qualified flags. Diagnose a disabled binary as disabled, not absent hardware. Prefer explicit installation recipes over blindly enabling CUDA on machines without a toolkit; never fill the gap with CPU serving.

### F14 — Live non-Apple certification adds utility dependencies and unbounded identity subprocesses

**Medium for real certification rigs; not an ordinary serving startup path.**

- **Locations:** `crates/synapse-certify/src/live.rs:752-842`, floor deadline `:80-101`.
- **Linux:** **works conditionally** with DMI `product_name`, uname, and `nvidia-smi` or `vulkaninfo`. It **refuses** on a system without readable DMI (e.g. some containers/non-PC hosts), without the external utility, or with multiple visible NVIDIA GPUs: it requires exactly one, although GPU presence itself is not the issue. A hanging identity utility can **degrade to a hang** because the helper uses unbounded `Command::output()`.
- **Windows:** uses native-oriented PowerShell/CIM commands, unlike the runtime collector's uname dependency (F2). It **works conditionally** with `powershell`, CIM, and the vendor/Vulkan utilities; missing commands **refuse**, and the same multiple-GPU rule and unbounded wait apply. `vulkaninfo` is not necessarily present in a driver-only install.
- **How known:** Read, no live certification/GPU probe run. macOS also has identity subprocesses; this helper's missing deadline is shared, while non-Apple DMI/CIM/GPU-utility requirements differ.
- **Fix shape:** collect identity for the adapter actually selected, not every visible NVIDIA GPU; use native/platform queries or provision explicit operator prerequisites; bound each child and capture a typed probe error. Do not turn unsupported identity evidence into a passing certification. Certification prerequisites need not be bundled into every end-user install.

### F15 — Windows cache publication silently skips parent-directory durability

**Medium crash/recovery difference.**

- **Locations:** `crates/synapse-core/src/cache.rs:217-222`, `:596-624`, `:702-707`; Windows directory-handle precedent `crates/synapse-module/tests/common/mod.rs:388-409`.
- **Linux:** file writes are synced, then the parent directory is opened and synced; this **works** on supporting local filesystems. Failures are ignored, so unsupported filesystems can also **degrade silently**.
- **Windows:** `sync_parent` uses plain `File::open(parent)`, without `FILE_FLAG_BACKUP_SEMANTICS`. The repository's Windows directory-handle helper explains why that fails with access denied. The `if let Ok` then makes directory sync a silent **no-op**. Files themselves still receive `sync_all`; this is missing namespace durability, not a claim that file content is never flushed.
- **How known:** Read; no power-loss or native Windows test. A successful rename/error-free API call alone is not evidence of equivalent crash durability.
- **Fix shape:** a platform-appropriate durable publication protocol and explicit treatment of unsupported directory flushing, with recovery tests. Validate native Windows rename/delete semantics and filesystem support rather than assuming Unix directory fsync has a direct drop-in counterpart.

### F16 — POSIX-only process measurement cannot run natively on Windows

**Low for end-user installs; Medium for operator measurement used as acceptance evidence.**

- **Locations:** `scripts/sample-dev-images.py:12-23`, `:53-60`, `:79-97`.
- **Linux:** **works conditionally** with procps-compatible `ps -axo pid=,comm=` and `pid=,ppid=,lstart=`. No process measurement was run in this census.
- **Windows:** native Windows lacks that executable/format; after spawning its target the script **crashes** on `subprocess.check_output` if ps is absent. MSYS/Git ps is not established to enumerate native descendants with equivalent birth identities. It does not contain a native fallback.
- **How known:** Read. This is auxiliary instrumentation, not the worker host's supervision implementation, and the script intentionally never signals ambient processes.
- **Fix shape:** Windows process enumeration including parent and creation-time identity, with a target-child cleanup/exit policy when measurement fails. Preserve protection against PID reuse. Do not infer Windows worker leaks from failure of this Unix measurement tool.

### F17 — Rig ACL/handover and BSD inventory tools are Mac-specific, not portable install gates

**Low for installed runtime; conditional refusal of those operator experiments on Linux/Windows.**

- **Locations:** `scripts/rig-acl-acceptance.sh:24-28`, `:51-57`, `:95-108`; `scripts/rig-hardlink-acceptance.sh:24-32`, `:62-67`; `bench/spikes/unified-rt/run-m1-policy-v2.sh:1-10`, `:18-25`; equivalent inventory in `bench/spikes/unified-rt/run-m1-bucket-matrix.sh:18-25`.
- **Linux:** ACL acceptance **refuses** under GNU chmod (`+a`/`-N` are BSD ACL syntax); handover also assumes `/usr/sbin/chown` and preprovisioned rig sudoers/accounts. BSD `stat -f '%N\t%m\t%z'` does not mean that format on GNU stat and the inventory command **refuses**/produces unsuitable output. These experiments are not Linux runtime prerequisites.
- **Windows:** native runs **refuse** because Bash/zsh, those Unix commands/ACL semantics, paths/accounts and macmon are not native equivalents. Compatibility-shell presence alone does not implement NTFS permissions or native process measurement.
- **How known:** Read only. These acceptance scripts require ownership changes and operator accounts; they were not executed. No such changes were made by this census.
- **Fix shape:** keep genuinely Apple-only ANE/Metal experiments explicitly scoped; if a cross-platform installer/acceptance path needs the same property, use POSIX ACLs (`setfacl`/`getfacl`) on Linux, Windows security descriptors/`icacls`, native stat/file APIs and provisioned ownership policy. Hard links themselves work on supported Linux/Windows filesystems; changing ownership of a shared inode is not solved by an OS label.

## Unresolved findings — explicit work remaining

These carry severity and separate platform status even where the evidence cannot establish a binary works/refuses verdict. None is a hidden assertion that the runtime is fine.

### U1 — Shared config-home and logger resolution are outside this checkout

**Medium; potentially Critical if config-home or logger initialization refuses native startup.**

- **Locations/evidence (Read):** `crates/synapse-module/src/lib.rs:243-255`, `:16605-16640`; dependency declarations `Cargo.toml:86-89`. User config delegates to `cortexkit-store-types` and logging to `cortexkit-log`, published dependencies, not local implementations.
- **Linux:** user-config tests exercise XDG/HOME and rejection of a relative result (`crates/synapse-module/src/lib.rs:19901-19920`, `:19960-19977`); actual default logger/root resolution in a service environment is **unresolved**, not proven degraded. Explicit `SYNAPSE_CONFIG_PATH` bypasses the config-home resolver.
- **Windows:** config error text names APPDATA/USERPROFILE, but neither comment nor a generic path test proves the dependency's native known-folder precedence. Logger initialization precedes Synapse's HOME lease failure; it could be an earlier refusal. Native defaults are **unresolved**. F1 is the independently established Synapse-owned HOME problem.
- **Fix/closure shape:** inspect and integration-test the pinned resolver/logger implementations with only native environment variables, multiple drive roots, services and non-ASCII homes. Adopt the same root policy across launcher, config, log, cache, store and lease.

### U2 — Store/lease permissions and Windows pipe authorization have not been established

**High security finding to close before claiming a hardened multi-user install.**

- **Locations/evidence (Read):** `crates/synapse-module/src/store.rs:7-8` delegates to `cortexkit_store::open_sqlite`; singleton `crates/synapse-module/src/lib.rs:377-399`; cache `crates/synapse-core/src/cache.rs:119-122`, `:159-166`, `:464-477`, `:539-546`; pipe `crates/synapse-core/src/worker_transport/windows.rs:45-49`; dependency declarations `Cargo.toml:86-88`.
- **Linux:** actual SQLite/lease owner/mode/lock policy is **unresolved** because the published implementations were not inspected here. Synapse's own cache/socket directory creation has no explicit private mode; effective isolation depends on umask and parent permissions. This is not proof the dependency no-ops leases. The test documents flock on Unix and LockFileEx on Windows (`crates/synapse-module/tests/skeleton_e2e.rs:3572`), but installed two-user ownership/recovery behavior is not a test result from this census.
- **Windows:** lease/store DACL enforcement, shared/exclusive locks and crash-release behavior are **unresolved**, not assumed Unix-mode no-ops. Named-pipe creation does not set an explicit DACL/security descriptor in this checkout; Tokio/Windows defaults, logon-session scope and rejection of remote clients require inspection/native tests. A nonce authenticates HELLO; it is not a substitute for protecting the endpoint/store.
- **Fix/closure shape:** inspect pinned native branches; run two-account ownership/ACL, shared-reader/exclusive-writer, stale-holder and crash-release tests on both OSes. Define restrictive DACLs/owner checks on Windows and private parent/mode/ownership policy on Linux. Verify a hostile precreated cache/lease/socket/pipe fails safely, without touching ambient resources.

### U3 — Open-handle rename/delete, links and network-filesystem contracts need native acceptance

**Medium reliability finding.**

- **Locations/evidence (Read):** cache publication `crates/synapse-core/src/cache.rs:217-225`, `:618-624`, deletion `:690-707`; hard-link alias `crates/synapse-core/src/dev_binary.rs:19-59`; cleanup comment `crates/synapse-module/tests/common/mod.rs:86-89`.
- **Linux:** local-filesystem rename/unlink/advisory-lock behavior has conventional implementations; **unresolved** on NFS/SMB, cross-volume overrides and restrictive mounts. Certification deliberately **refuses** cross-volume aliases; development aliases may copy only on EXDEV. These are explicit policies, not CPU/GPU gaps.
- **Windows:** NTFS same-volume hard links are represented in code, but FAT/exFAT support, share modes on open model/store files, replacement/delete while workers are alive, ACL inheritance and crash cleanup are **unresolved** for a real install. Tests explicitly acknowledge delayed Windows SQLite-handle closure. No general assertion that all Windows renames fail is justified.
- **Fix/closure shape:** state supported local-filesystem contracts, preserve the same-file certification invariant, test open-reader/worker publication and GC plus crash recovery, and handle native sharing violations with bounded observable retry/refusal. Symlink creation is not interchangeable with hard-link creation; Windows symlinks can require privilege/Developer Mode.

### U4 — Shipped llama runtime dependencies, native loader floors and GPU inference are not proved

**High: packaged accelerator inference and llama GPU decode have not been demonstrated by this census.**

- **Locations/evidence (Read):** candidate build/archives `.github/workflows/release-candidate.yml:60-80`, `:180-191`, `:241-258`; inventory `bench/parity/release-assets.json:94-100`, `:139-145`; llama smoke gap `crates/synapse-release-checks/smoke.py:19-53`; optional GPU tests `crates/synapse-worker-cuda/tests/gpu_parity.rs:1-25`, `crates/synapse-worker-vulkan/tests/gpu_parity.rs:58-59`.
- **Linux:** native CUDA/Vulkan worker source paths exist; actual supported-device inference, llama GPU decode, libc/libstdc++ compatibility on other distributions, llama's combined CUDA/Vulkan dependencies and extracted-artifact loader closure are **unresolved**. Linux CUDA's explicit runtime omission is already F7, not just an unknown.
- **Windows:** real named-pipe/CUDA/Vulkan implementations exist; actual CUDA/Vulkan decode/inference, MSVC runtime availability, combined llama loader dependencies, GPU driver/device variants and sidecar-only execution are **unresolved**. Ordinary compilation does not settle them.
- **Fix/closure shape:** package dependency closure and supported driver/OS floors; execute the extracted release artifacts on real Linux and Windows CUDA **and** Vulkan hardware, covering LOAD, embedding, rerank, generate, unload, restart and timeouts. Separate no-device refusals from full inference. This census intentionally ran no GPU work.

### U5 — Secure native Windows module launch-nonce handoff needs an end-to-end contract

**Medium security/launch-parity finding, not a demonstrated launch failure.**

- **Locations/evidence (Read):** `crates/synapse-module/src/lib.rs:228-234`, `:323-335`; stripping `crates/synapse-core/src/child_process.rs:18-19`; Unix-only handoff test `crates/synapse-module/tests/skeleton_e2e.rs:461-488`.
- **Linux:** the SDK reads/closes/caches the inherited fd-3 nonce before worker spawning; the source includes a module-side fd-only registration/serving test. Daemon-side validation/descriptor inheritance and production launcher behavior are **unresolved in this census**, not claimed passing from the profile unit tests.
- **Windows:** the module comment states the SDK always falls back to the `SUBC_LAUNCH_NONCE` environment copy because there is no descriptor handoff. This is **degraded secret transport relative to a private inherited descriptor**, but no authentication bypass or refusal was established. Native handle handoff, environment exposure and daemon-side supervision validation are **unresolved** in the published SDK/daemon, outside this checkout.
- **Fix/closure shape:** define either an audited Windows env-based security contract or a restricted inherited anonymous-pipe/handle handoff with single-reader close/cache semantics. Test actual supervised Windows startup, missing identity/nonce, denied handoff, retries, and absence of daemon credentials in workers/probes. Do not add a second independent nonce reader in the module.

### U6 — Deployment/signing references and install layout are not a checked-in portable installer

**Medium installation/support finding.**

- **Locations/evidence (Read/inventory):** tracked `scripts/` list; archive recipes `.github/workflows/release-candidate.yml:235-264`; promotion `.github/workflows/release.yml:123-131`; required daemon floor `.github/workflows/release-candidate.yml:266-280`.
- **Linux:** candidate artifacts target `linux-x64`; archive extraction is not a proven install/service/upgrade path. Worker sibling placement, runtime-library provisioning, executable mode, permissions and minimum daemon version on an end-user host are **unresolved**. No Linux ARM64 candidate is declared.
- **Windows:** artifacts target `windows-x64`; native installation, `.exe` sibling placement **with runtime DLLs**, service environment, update of running images, ACLs and Authenticode/trust policy are **unresolved**. No Windows ARM64 candidate is declared. The owner ruling establishes OS support, not evidence for additional architectures.
- **Important context correction:** this baseline's tracked `scripts/` contains **no codesign, lldb or BSD `stat -f` call and no release/deploy/placement shell script implementing those operations**. The release path is the two workflows and Python release checks; BSD inventory actually lives in the experimental bench scripts cited in F17. Do not report an absent script as a confirmed failing current runtime path. External/private installer or older deployment/signing/debugging recipes remain an unresolved portability finding until their sources are identified.
- **Fix/closure shape:** supply a documented/tested per-OS install/upgrade contract with required daemon floor, paths, worker dependency closure and trust/signature policy. Review any external deployment code explicitly. Use Linux package/signature and Windows Authenticode equivalents where required; lldb is debugging tooling, not an inferred inference dependency.

## Platform feature inventory and equivalents

“Implemented” here means a real code branch exists, not that this census certified it on hardware. These equivalents describe the required capability, not permission to introduce CPU lanes.

| Feature Synapse relies on | macOS mechanism | Linux equivalent / current state | Windows equivalent / current state |
| --- | --- | --- | --- |
| Accelerator discovery/admission | Metal default device; Apple-specific ANE floor/placement | CUDA driver/runtime device capabilities and Vulkan physical-device/features/memory enumeration exist in workers, but catalog discovery is empty (F3) | `nvcuda.dll` plus CUDA runtime/device APIs; `vulkan-1.dll` physical-device enumeration exist; catalog still empty (F3) |
| Owned inference | In-process owned Metal and supervised ANE workers | Supervised owned-CUDA/owned-Vulkan, feature/toolkit enabled; explicit profile preloads are represented, real inference unresolved (U4) | Same owned CUDA/Vulkan lanes over pipes, real inference unresolved (U4) |
| Decode | llama.cpp Metal worker; parked owned decode is not the current route to port | llama.cpp CUDA/Vulkan builds required; default CPU and candidate identity/selection gaps F5/F6 | llama.cpp CUDA/Vulkan, with same F5/F6 gaps; a native pipe runner exists |
| Worker spawning/supervision | Tokio `Command`, kill-on-drop, HELLO, child kill/reap, crash bookkeeping | Same host implementation, `--socket`; Unix process groups/cgroups/pidfds would be stronger tree/lifecycle facilities if required | Same host implementation, `--pipe`; native process handles/Job Objects are the equivalents for tree-wide lifecycle containment |
| Worker IPC | AF_UNIX socket | AF_UNIX; complete socket-path limit, private runtime directory and stale endpoint policy needed | Native named pipes (`NamedPipeServer`, blocking client File); first-instance/name/ACL policy differs (F10/U2), runtime directory ignored |
| File locks/leases | File lease dependency/advisory lock; direct ANE inherited lane-lock fd | `flock`/equivalent advisory locks; same shared-reader/exclusive-writer lifecycle needed; pinned dependency behavior unresolved (U2) | `LockFileEx` and crash-closed handles are the equivalent, not Unix chmod; pinned implementation/ACLs unresolved (U2) |
| File/IPC permissions | POSIX modes/ownership, Apple ACLs, private dirs | POSIX modes/umask/owner and POSIX ACLs; rig BSD ACL syntax is not portable (F17) | Security descriptors/DACLs/owner, trusted parent dirs and named-pipe security; don't rely on ignored POSIX mode bits (U2) |
| Signing/trust | Mach-O codesign/notarization/Gatekeeper may be required by an external installer | Signed package/repository or detached artifact signatures plus digests; no Gatekeeper/codesign clone is needed | Authenticode/timestamping and installed publisher/trust policy where required; checksums alone are not equivalent publisher identity (U6) |
| Daemon launch nonce | SDK fd-3 inherited pipe, single read/close/cache | Same Unix handoff, stripped from children | SDK environment fallback today; restricted inherited handle/pipe is the native capability equivalent (U5) |
| Worker nonce | `--nonce` and HELLO exact match | Same, bound to socket handshake | Same, bound to named-pipe handshake; separate from daemon nonce |
| Config/data/cache homes | Current shared config resolver plus `.local/share` data defaults | XDG config/data/cache/state/runtime conventions; current cache/lease ignore XDG data home (F1/U1) | Known-folder/APPDATA/LOCALAPPDATA/USERPROFILE policy; current lease/cache/fallback store require HOME (F1), shared config is unresolved (U1) |
| Filesystem identities/verification | Device/inode/mtime/ctime and digests | Same Unix stamp path | Native volume/file ID plus reliable change metadata/open-handle snapshot; currently stamp is None (F9) |
| Symlinks/hard links | Same-volume hard links for executable aliases; explicit symlink controls in Apple artifact paths/tests | Same-volume `link`; symlink via native API, EXDEV handling; certification refuses copies | NTFS hard links preserve `.exe`; symlinks/reparse points have distinct privilege/ACL rules; alias sidecar layout F12, filesystem acceptance U3 |
| Temp/runtime directories | `std::env::temp_dir`; short/private roots needed for sockets | Native TMPDIR/default temp, ideally user-owned XDG_RUNTIME_DIR for IPC; shared fallback today (F10) | Native temp/known-folder APIs; named pipes have no filesystem runtime root; do not infer pipe isolation from a private temp path |
| Process probes | `sw_vers`, sysctl; operator `ps`; parked decode uses kill(pid,0) | Native uname/proc/sysfs, pidfd/creation identity; runtime still shells out to uname; DMI/GPU utilities for certification (F2/F8/F14) | Native OS/CIM/process handles/creation time; runtime incorrectly uses uname, live certify uses PowerShell/CIM, sampler requires ps (F2/F14/F16) |
| Subprocess deadlines | Profile collector polls/kills own probe under 2s budget; host request timeouts | Same profile/host budgets; kill only owned processes, define bounded reap/tree policy if needed | Same host budgets; use native handles/Job Objects for owned descendants if tree termination is required; live certify's utility waits remain unbounded (F14) |
| Packaging/delivery | Darwin ARM64 candidate, separate Swift ANE worker | Linux x64 ZIPs; ELF userspace dependency closure not shipped for CUDA (F7); native install/upgrade U6 | Windows x64 ZIPs; owned-CUDA DLL/manifest/license closure exists; install/upgrade/trust and llama closure U4/U6 |

### Evidence for implemented platform branches and intentional specialization

- **CUDA is not macOS-only:** `crates/synapse-engine-cuda/build.rs:18-99` selects nvcc/nvcc.exe, platform library directories and PIC only off Windows. `crates/synapse-engine-cuda/src/cuda.rs:59-100` dynamically loads the native driver on Unix/Windows. Feature-enabled worker load probes before model opening (`crates/synapse-worker-cuda/src/main.rs:368-391`). A missing/too-old driver or GPU below the supported floor is an intentional refusal, not an argument for CPU fallback.
- **Vulkan is not Windows-only:** `crates/synapse-worker-vulkan/src/runtime.rs:27-123` loads `libvulkan.so.1` on Linux and `vulkan-1.dll` on Windows, enumerates adapters and inspects features/heaps. Admission rejects software devices and unsupported vendors/floors (`crates/synapse-worker-vulkan/src/admission.rs:73-119`). LOAD requires a manifest-owned Vulkan profile and operation/package binding (`crates/synapse-worker-vulkan/src/protocol.rs:54-100`). That is a real implementation below the missing catalog integration, not an inferred hardware pass.
- **llama has both transports:** `crates/synapse-worker-llama/src/main.rs:18-48` includes a runner for macOS, other Unix and Windows; `crates/synapse-worker-llama/src/runner.rs:219-257` connects by socket or named pipe. Its default/offload/backend contract is the defect, not a missing Windows main.
- **Apple workers intentionally refuse non-Apple execution:** ANE returns a macOS-only error except for its version probe (`crates/synapse-worker-ane/src/main.rs:28-34`); Swift build returns early off macOS (`crates/synapse-worker-ane/build.rs:16-18`). ANE-direct exits 2 with `ane_private_api_unavailable` (`crates/synapse-worker-ane-direct/src/main.rs:13-16`). Parked owned decode returns a Metal-only error except for version (`crates/synapse-worker-decode/src/main.rs:26-31`). Owned Metal load also has an unsupported-platform branch (`crates/synapse-engine-owned/src/lib.rs:623-634`). These are deliberate implementation specializations, **not** requirements to emulate private ANE/Metal APIs on Linux/Windows. The product defect is failure to substitute its promised CUDA/Vulkan/llama lanes.
- **Generic supervision is shared:** `crates/synapse-module/src/worker_host/mod.rs:956-1059`, `:1062-1110`, `:1162-1171`, `:2297-2313` spawn, handshake, timeout, kill and reap on both transports. No Linux/Windows no-op supervisor was found in these branches. Kill-on-drop/child kill is not a demonstrated process-tree kill; Unix process groups and Windows Job Objects would require an explicit tree-lifecycle contract. Host cleanup waits after killing without a separate reap deadline, unlike the profile collector's deliberate abandonment after its deadline (`crates/synapse-core/src/machine_profile.rs:307-321`). This is shared behavior, not established OS-specific failure.
- **Windows connect differs without being absent:** `crates/synapse-core/src/worker_transport/windows_client.rs:17-45` retries pipe open for up to 30s, then does a blocking HELLO/ACK; Unix workers connect directly and block on ACK. Host handshake deadlines and child termination are the protection against a stalled worker. Native Windows stalled-ACK/termination acceptance remains part of U4; the open retry is not a CPU or inference fallback.
- **Nonce roles must not be confused:** daemon launch credentials are stripped from both synchronous/asynchronous child commands (`crates/synapse-core/src/child_process.rs:6-19`); worker start passes a fresh worker nonce. Worker HELLO checks nonce/version/engine and optional binding on Unix and Windows (`crates/synapse-core/src/worker_transport/unix.rs:163-218`, `crates/synapse-core/src/worker_transport/windows.rs:158-213`). Worker nonce generation uses time, PID and a counter (`crates/synapse-module/src/worker_host/mod.rs:2435-2443`), not a CSPRNG, **equally on all OSes**. That shared hardening concern is not counted as a Linux/Windows-only divergence; endpoint access controls still need U2.

## What CI actually exercises (and what it does not)

This is a reading of the checked-in workflows/tests, **not a claim that a particular remote CI run passed**.

1. **Ordinary matrix:** `.github/workflows/tests.yml:43-62` declares Ubuntu 24.04 and Windows 2025, with macOS commented out pending real Apple GPU hardware. The explicit package set is synapse-core, synapse-module, synapse-worker-llama, owned-decode-worker, synapse-worker-decode, synapse-opctl, synapse-engine-cuda and synapse-worker-cuda. It omits the Vulkan and ANE worker packages as explicit ordinary test targets; don't promote that list to whole-workspace/native-accelerator coverage.
2. **Tests/lint:** `.github/workflows/tests.yml:247-272` runs Clippy `--all-targets`, then partitions nextest between `skeleton_e2e` and the rest. Windows nextest uses release CRT-compatible builds. The fixture-skip guard checks the e2e transcript (`:345-377`). That establishes intended test execution discipline, not real GPU inference.
3. **Deterministic module behavior:** tests launch a real module and in-process daemon and exercise routing, store/jobs, envelopes, certification/refusal, remote-provider mocks and lifecycle. `embed_query_preloaded_minilm_returns_vectors_and_envelope` explicitly asserts engine `test-deterministic`, with fixed vector bins (`crates/synapse-module/tests/skeleton_e2e.rs:1112-1173`). The label “MiniLM” does **not** make this a production MiniLM accelerator run. `crates/synapse-module/Cargo.toml:62-67` enables self `test-support` in test builds, allowing test catalogs/runnable-set overrides that release builds never see.
4. **Defaults are masked by the harness:** `crates/synapse-module/tests/common/mod.rs:110-121` supplies explicit config, lease root and XDG data home; the test daemon supplies storage. This does not prove native installed default roots (F1/U1). Tests inherit their runner's tools/environment; passing under a Unix-equipped Windows CI environment does not establish an installed native uname contract (F2).
5. **Decode fallback evidence is a mock:** `substitutable_owned_refusal_falls_back_to_llama_with_lane_provenance` points `worker_bin` at `synapse-worker-timeout-mock`, writes fixture bytes rather than a real GGUF, and asserts the string `fallback` (`crates/synapse-module/tests/skeleton_e2e.rs:3744-3816`). It proves routing/provenance, not CUDA/Vulkan llama decode.
6. **Real protocol coverage is useful but narrower:** the Unix-only timeout test covers long LOAD versus short EMBED budgets with a mock worker (`crates/synapse-module/tests/it/worker_host_timeout.rs:1-71`); it does not compile as Windows coverage. The Unix fd-only nonce test establishes the module side, and explicitly says its test daemon has no supervisor (`crates/synapse-module/tests/skeleton_e2e.rs:461-488`). Windows transport has a real pipe framing test (`crates/synapse-core/src/worker_transport/windows.rs:273-334`); this is not an installed GPU LOAD. Separate Vulkan protocol tests support both transports (`crates/synapse-worker-vulkan/tests/protocol_v2.rs:9-110`) but are not an explicit ordinary matrix package.
7. **Build gates are not inference gates:** normal CI explicitly builds llama **CPU** on Linux and Windows (`.github/workflows/tests.yml:274-335`). Manual workflow-dispatch gates build Linux llama CUDA and Windows llama Vulkan (`:386-476`, `:480-640`), and Windows owned-CUDA with DLL packaging/probe checks (`:644-791`). The use of CPU build gates does not authorize CPU product lanes. Builds/hashes and structured no-driver refusal are useful evidence of toolchain/loader behavior, not supported-device inference. The ignored CUDA parity test is Unix+CUDA-gated; Vulkan parity is feature-gated/ignored for real hardware. They were not run by this census.
8. **Candidate and release gates:** candidate builds enable accelerators and run `candidate_hosted_protocol_matrix` (`.github/workflows/release-candidate.yml:227-234`), which does module `--version`, worker HELLO/PING, unsupported request and SHUTDOWN, **no model LOAD/inference** (`crates/synapse-core/tests/candidate_workers.rs:19-139`). Final tag release validates inventory and 32 certification records (`.github/workflows/release.yml:33-34`) and runs packaged no-GPU smoke on three OSes (`:41-110`). That smoke installs Linux runtime libraries/Windows Vulkan loader and checks features/bindings/refusal codes (`crates/synapse-release-checks/smoke.py:19-53`), not the real installed module's default environment or llama generation. Record validation can be evidence of required external hardware runs, but this checkout/census does not contain an observed completed set to cite as a GPU pass (U4).

## Priority and first install blockers

**Priority order for remediation:**

- **Critical established:** F1/F2 (native Windows startup), F3/F4 (Linux and Windows primary catalog download/local serving).
- **High established:** F5/F6 (forbidden CPU/default and decode backend/artifact mismatch), F7 (Linux CUDA loader closure), F8 (accelerator identity absent from runtime certification). **High unresolved:** U2 (permission/authorization guarantees), U4 (actual packaged CUDA/Vulkan inference/decode). F12 is high when a sidecar-only Windows certification rig is the release gate.
- **Medium/conditional:** F9–F15, U1/U3/U5/U6. Close native startup/root uncertainties before performance work. F13 concerns source-install default features; it does not override the feature-enabled release recipe.
- **Low/ancillary:** F16/F17 for an end-user install; raise their priority only if those tools are made a release/install acceptance dependency.

### What blocks a Linux install first

On a normal Linux x64 host with HOME, uname and the required daemon available, machine-profile collection and singleton lease-root resolution can succeed. The first **primary local-service** barrier is the production catalog: detection finds no backend (F3), and the frozen catalog/constructor cannot route CUDA/Vulkan even if discovery is fixed (F4). This is **not** the intentional refusal on a GPU-less machine: it also applies to supported hardware. A manually configured owned-CUDA profile can bypass that catalog path, but an archive-only CUDA install next hits missing CUDA 13 userspace libraries (F7); a default source build instead hits disabled engines (F13). Decode separately needs F5/F6, not a CPU fallback. A minimal/service environment missing HOME/uname can fail earlier (F1/F2). Native end-to-end GPU acceptance remains U4.

### What blocks a Windows install first

In a native Windows account/service with only native profile variables, the first demonstrable **Synapse-owned** startup barrier is HOME-only singleton lease resolution, before serving (F1); published logger initialization happens even earlier and is U1. With roots explicitly supplied, profile collection still requires uname and refuses the whole initialization, including remote-only service (F2). Once both startup problems are fixed, empty detection and the Metal-only catalog block local CUDA/Vulkan service (F3/F4). A developer shell plus environment overrides can conceal those problems, but is not a Windows installation contract. Decode also needs the CUDA/Vulkan worker/default/artifact repair (F5/F6), and native ACL, DLL-layout and packaged hardware acceptance must close U2/U4/U6. A correct no-supported-GPU `backend_unavailable` result should survive all of these repairs unchanged.
