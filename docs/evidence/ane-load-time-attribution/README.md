# GTE ModernBERT ANE load-time attribution

Measured on 2026-09-11 on the same `Mac17,6` M5 Max, macOS 27.0 build `26A5425a`, model snapshot, rotation seed, and coremltools 9.0 environment as the accepted 8192-token parity run. `EVIDENCE.json` contains the exact timings, sizes, hashes, placement counts, memory observations, and guard settings.

## Verdict: the answer depends on keeping the compiled path stable

The 735-second spike path loaded a source `.mlpackage`. That path pays the slow work **every time** because coremltools automatically compiles each `MLModel` object to a different temporary `.mlmodelc` path:

| Source package | First load, same process | Second load, same process | Fresh process |
|---:|---:|---:|---:|
| 1024 | 36.641 s | 33.599 s | 33.698 s |
| 8192 | 601.565 s | 725.656 s | 670.302 s |

All three 8192 framework calls were slow: 595.645, 719.824, and 663.859 seconds inside `MLModel loadContentsOfURL`. The generated paths had three different UUIDs. The new values bracket the earlier 735.348-second observation closely enough to reproduce the mechanism; load time itself has substantial run-to-run variance.

The specialization is nevertheless a **one-time, cross-process cost** when the source package is explicitly compiled once and the exact `.mlmodelc` path is reused:

| Stable compiled model | First load, same process | Second load, same process | Fresh process |
|---:|---:|---:|---:|
| 1024 | 32.881 s | 0.272 s | 0.307 s |
| 8192 | 579.699 s | 2.162 s | 1.786 s |

The second 8192 load was 268 times faster than the first. A fresh process was 325 times faster, so the cache is not process-local. This is not merely an in-memory reuse effect.

**Operational answer:** direct `.mlpackage` loading pays every time; stable `.mlmodelc` loading pays once. An optimization campaign should first persist the compiled 8192 model at a stable versioned install path and reuse that exact path and compute configuration. That turns the long ANE preparation into an install or first-run cost. Only if install-time preparation cannot absorb roughly 10–12 minutes should the graph's first-specialization cost become the next campaign target.

## Cache location and invalidation

Apple describes the specialized asset as an OS-managed disk cache linked to the full compiled-model path and configuration. The cache survives application launches and may survive reboots, but Apple exposes neither its filesystem URL nor a public clear API. The relevant lifecycle and “prepare and cache” versus “cached” states are documented in [Explore Core ML tools](https://developer.apple.com/videos/play/wwdc2023/10049/) and the coremltools guide to [using a compiled model](https://apple.github.io/coremltools/docs-guides/source/model-prediction.html#why-use-a-compiled-model).

A safe 1024 invalidation control compiled the unchanged package to a second app-owned `.mlmodelc` path. The previous stable path loaded in 0.296 seconds from a fresh process; the new path took 32.113 seconds. Changing the path therefore restored the slow path without deleting private caches or restarting Core ML services.

The exact private cache directory remains intentionally unattributed. `fs_usage` required root on this host, and `xcrun xctrace` was unavailable in the installed command-line developer tools. No privilege escalation or private-cache deletion was attempted. A filesystem guess would not be durable evidence.

This path identity matters to the existing worker. `crates/synapse-worker-ane/swift/ane_worker.swift` loads a directory artifact in place, preserving a stable path, but expands a file artifact beneath a new `synapse-ane-<UUID>` temporary directory on each load. A production package must therefore be materialized once to a stable versioned directory before worker startup; repeatedly extracting an archive to a UUID path would reproduce the cold path by construction.

## Phase attribution

The experiment first called `MLModel.compileModel` through `coremltools.models.utils.compile_model`, then loaded the resulting fixed `.mlmodelc` with `CompiledMLModel`. At 8192:

- source `.mlpackage` to `.mlmodelc`: **5.909 s**;
- first fixed-path `CPU_AND_NE` load: **579.699 s wall**, of which Core ML reported **579.688 s** inside `MLModel loadContentsOfURL`;
- cached fixed-path load from a fresh process: **1.786 s wall**, **1.775 s** inside Core ML;
- cache-miss-specific Core ML envelope: **577.914 s**.

Thus 99.0% of explicit compile-plus-cold-load time was inside the Core ML runtime load call, not source-package compilation or Python overhead. The cache removes almost all of that envelope. Combined with the ANE placement plan and the `.all` control below, this attributes the superlinear cost to Core ML's ANE prepare-and-cache path rather than raw package deserialization.

Core ML does not expose a public timing split within that envelope. ANE compiler work, weight transformation/layout, and internal allocation remain **unattributed relative to one another**. The cached load still performs compiled-model deserialization and allocation, but those phases cannot be timed independently with the available API. The Core ML Instrument could label prepare-and-cache versus cached and expose data-copy activity on a full Xcode installation; it does not provide a weight-packing or ANE-residency breakdown.

Memory observations are bounds, not a phase split. During the stable 8192 pair, owned-process RSS peaked at 2.004 GB, minimum system-available memory was 43.148 GB, and maximum wired memory was 16.236 GB. The direct-package pair peaked at 3.350 GB owned RSS. The loaded model's process RSS rose from 303 MB to 856 MB on the first stable load. macOS can reclaim and reclassify memory during a long load, so these samples do not identify ANE-resident bytes.

## Scaling and package sizes

The same 149,014,272 float16 parameters are present at every fixed sequence length. The packages below all use unchanged Hadamard rotation seed 0 and the same accepted input/reference identities. The 2048 and 4096 packages were regenerated only to obtain exact on-disk size and lowered-operation counts; no parity threshold or model math changed.

| Tokens | Logical package size | Allocated size | MIL operations | Earlier direct load |
|---:|---:|---:|---:|---:|
| 1024 | 310,601,744 B (296.213 MiB) | 310,616,064 B | 30,397 | 36.756 s |
| 2048 | 318,010,810 B (303.279 MiB) | 318,017,536 B | 52,265 | 152.322 s |
| 4096 | 332,866,915 B (317.447 MiB) | 332,877,824 B | 96,001 | 190.534 s |
| 8192 | 362,794,575 B (345.988 MiB) | 362,807,296 B | 183,473 | 735.348 s |

From 1024 to 8192, parameter count is constant, package bytes rise only 1.168 times, lowered operation count rises 6.036 times, and the earlier load rises 20.006 times. Weight count and raw bytes therefore do not explain the curve. Attention dispatch count grows linearly with query-tile count, while the tensors in eight global-attention layers grow with sequence length. The non-smooth 2048/4096 step and opaque compiler make a fitted exponent unjustified, but the evidence locates the scaling in graph/device specialization, not parameter count.

## Compute unit control

`MLComputePlan` was loaded separately with `.cpuAndNeuralEngine` and `.all`. It is anticipated preferred-placement evidence, not a runtime dispatch trace.

At 1024, both modes took about 32 seconds cold and preferred the same 11,660 operations on ANE plus 13 small operations on CPU. `.all` gave no useful load-time or placement change.

At 8192, the fixed-path `.all` load took 32.112 seconds instead of 579.688 seconds, a 94.5% reduction, but every one of the 75,093 non-constant operations became GPU-preferred. That includes all 90 convolutions, 16,896 einsums, and 8,448 softmaxes. `CPU_AND_NE` instead preferred 75,024 operations on ANE and 69 on CPU. The fast `.all` result is therefore a placement change to GPU, not an ANE optimization, and does not satisfy an ANE lane objective.

## Safety and reproduction

All hardware work ran one child process at a time after the resident ANE worker was cleared and the measurement slot was authorized. The driver checked for a new `ck-synapse-worker-ane` immediately before and after each timed load. No competing worker appeared. Existing safeguards required 32 GiB available at preflight and aborted below 16 GiB available, above 50% wired memory, above 32 GiB owned RSS, or after the child timeout. No guard fired.

The durable driver is `bench/spikes/ane-modernbert-full-context/attribute_load.py`. Its `run` command performs, in order, two direct-package loads in one process, a direct-package load in a fresh process, explicit compilation to a persistent `.mlmodelc`, two fixed-path loads in one process, a fixed-path load in a fresh process, the 1024 versioned-path invalidation control, `.all` loading, and both placement plans. Build the placement helper first:

```bash
cd bench/spikes/ane-modernbert-full-context
./build_placement.sh
../../../.venv/bin/python attribute_load.py run \
  --stage 1024=/path/to/seq1024/gte-modernbert.mlpackage \
  --stage 8192=/path/to/seq8192/gte-modernbert.mlpackage \
  --artifacts /tmp/modernbert-load-attribution \
  --placement-binary .build/modernbert-placement \
  --measurement-slot-authorized "recorded non-overlapping slot"
```

`MLModel.load_duration_in_nano_seconds` (and the equivalent Core ML proxy timing on `CompiledMLModel`) measures the framework load call. `run_stages.run_guarded` records wall time, owned-child peak RSS, minimum system-available memory, maximum wired memory, timeouts, and abort reasons.
