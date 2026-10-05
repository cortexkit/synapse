# Bounded in-process reclaim alternatives (development)

Mac17,6 (M5 Max), OS build 26A434. Production's Core ML lane remained resident per task context and was untouched. Loads below are the unitless macOS 1-, 5-, and 15-minute averages. The five initial controls took 1,029 seconds total; the follow-up allocated-client control also finished within its 1,200-second cap. Combined hardware time was below the one-hour budget.

`list_reclaim_selectors.py` enumerated selectors and Objective-C type encodings without invoking them. The full inventory is embedded in both JSON reports. Applicable release selectors actually present were `_ANEInMemoryModel unloadWithQoS:error:` and `purgeCompiledModel`, plus `_ANEClient doUnloadModel:options:qos:error:`, `unloadModel:options:qos:error:`, `purgeCompiledModel:` and `purgeCompiledModelMatchingHash:`. The descriptor exposed no purge/unload selector. No client reset/invalidate selector was present. Real-time-only unload and purgeability-level setters were not guessed into ordinary model use; no undocumented enum values were supplied.

Each fresh process admitted GTE@128,256,512,1024,2048, for 110 executables. It then released GTE@128 (22 executables), except the all-object control which released every shape. After the release path it attempted a new 22-layer GTE@4096 shape, then retried after five seconds if that failed. Every control loaded only 14 replacement layers, failed at layer 14, rolled them back, and then failed the retry at layer 0. **None restored enough capacity for one complete 22-layer shape.** Every control observed 124 normal unload calls with BOOL true and NSError absent.

| Release path | Start load 1/5/15 | Additional observations | Last failed attempt finished after release |
|---|---|---|---:|
| Drop returned shared-client references | 13.85 / 15.30 / 16.70 | Shared singleton has no reset selector; returned retained reference released | 45.57 s |
| Fresh `_ANEClient new` | 10.27 / 13.41 / 15.71 | Returned nil; did not fabricate a client | 52.14 s |
| `_ANEInMemoryModel purgeCompiledModel` after normal unload | 31.41 / 18.16 / 17.00 | Invoked on each evicted model; void return gives no completion confirmation | 55.43 s |
| Shared-client model and hash purge after unload | 24.54 / 21.39 / 18.66 | Invoked both verified purge selectors; void return | 48.12 s |
| Release every resident executable/model/IOSurface object | 13.14 / 18.67 / 18.14 | Zero fully resident executables remain, but still no full-shape replacement | 84.03 s |
| Fresh allocated client, `initWithRestrictedAccessAllowed:false` | 29.38 / 24.51 / 20.79 | Client distinct from shared; both client unload selectors returned true with no NSError | 42.74 s |

The last control used inherited NSObject allocation and an initializer actually listed by the runtime, after `new` returned nil. It did not enable restricted access. The compiled model getter and hash getter were also enumerated, not invented. All objects explicitly retained by these diagnostic probes were released before replacement admission; this did not drain unscoped Objective-C autoreleased temporaries. A successful unload or purge call is not a confirmation of released hardware capacity.

At the end of these six controls, confirmed owning-process exit was the only observed successful reclaim boundary. The following autorelease-pool controls **supersede that process-only interpretation**: successful unloads were insufficient when Objective-C temporary ownership was not drained.

## Autorelease-pool finding

Start macOS 1-/5-/15-minute load averages (unitless): **12.59 / 12.63 / 13.92**. In one process, compile the same 110 executables inside an Objective-C autorelease pool, drop every resident shape, then drain that pool. A new 22-layer GTE@4096 shape **loads completely**, starting 0.306 seconds after release began and finishing after 59.761 seconds (including compilation). The whole hardware control took 149 seconds. No process exit or client replacement occurred. See `reclaim-autorelease-development.json`.

A second control places each compile/inference/eviction in its own scoped pool while retaining executables across commands. Start macOS 1-/5-/15-minute load averages (unitless): **33.70 / 21.12 / 17.17**. All five GTE shapes remain loaded (110 executables); a sibling GTE@256 inference has finite output. Evict only GTE@128 inside a pool and drain it. Replacement GTE@4096 then loads all 22 layers, beginning after 0.151 seconds and finishing after 75.060 seconds (including compile and sibling verification). The retained sibling runs again with byte-identical output. The full control took 295 seconds. **Single-shape eviction does return usable capacity with scoped autorelease hygiene**, and retained executables survive pool boundaries. See `reclaim-scoped-single-evict-development.json`.

These observations point to un-drained Objective-C temporary ownership, not an unavoidable per-process lifetime quota. Each pool strategy has only one completed fresh-process run so far; these are two different controls, not a repeatability study. The reported 0.151/0.306-second intervals are release-and-pool-drain durations before starting a successful compilation, not direct measurements of the instant the hardware allocator returns capacity. Neither successful first attempt used a separate cooldown. Only one retained sibling (GTE@256) was checked for finite, byte-identical output; this does not establish reference-model parity for all retained shapes. Six selector/reference-release controls plus these two controls stayed under the one-hour hardware budget. Additional admission policy remains pending a ruling because shape-level reclamation now demonstrably works.

Reproduction uses the worker test executable selected as `DRIVER` in `capacity-development.md`:

```
python3 crates/synapse-worker-ane-direct/tests/run_reclaim_paths.py --driver "$DRIVER" --timeout 3600 --methods drop-client-references fresh-client model-purge client-purge release-all --out crates/synapse-worker-ane-direct/evidence/reclaim-paths-development.json
python3 crates/synapse-worker-ane-direct/tests/run_reclaim_paths.py --driver "$DRIVER" --timeout 1200 --methods fresh-allocated-client --out crates/synapse-worker-ane-direct/evidence/reclaim-fresh-allocated-client-development.json
python3 crates/synapse-worker-ane-direct/tests/run_reclaim_paths.py --driver "$DRIVER" --timeout 1200 --methods autorelease-all --out crates/synapse-worker-ane-direct/evidence/reclaim-autorelease-development.json
python3 crates/synapse-worker-ane-direct/tests/run_reclaim_paths.py --driver "$DRIVER" --timeout 1200 --methods scoped-single-evict --out crates/synapse-worker-ane-direct/evidence/reclaim-scoped-single-evict-development.json
```

Launchers unset TMPDIR, select the required Xcode DEVELOPER_DIR and log before/after load beside every run. Serving does not invoke the explicitly experimental vendor diagnostic helpers.

## Fresh-process repeats after disk recovery

The disk cleanup removed ignored packages. Rebuilt all four packages using the documented parity converter from worktree-local copies of the pinned HF snapshots: all 16 manifest-listed files copied with symlinks dereferenced and SHA-256 verified; all four converted-package digests matched manifest pins. No parent checkout or production process was modified.

Mac17,6, OS build26A434. The operator reported production's Core ML GTE128/256/512 models resident on the same Neural Engine; these were not stopped or modified, so measurements reflect a shared device. Start load columns are unitless macOS 1-/5-/15-minute averages. Each row is a new process; initial GTE residency is five shapes (128,256,512,1024,2048), Qwen residency is four (128,256,512,1024). Evict128; replace with GTE4096 or Qwen2048. Every row's retained sibling at256 produced finite byte-identical output before/after.

| Control | Initial executables | Start load1/5/15 | Release-to-attempt start ms | Release-to-complete ms | Replacement loaded/error |
| --- | ---: | --- | ---: | ---: | --- |
| GTE pooled repeat1 |110|17.78/19.10/25.52|35.858|70228.393|22 / none|
| GTE pooled repeat2 |110|24.13/21.66/25.37|86.669|73019.327|22 / none|
| Qwen pooled |112|28.77/27.30/27.16|245.841|90221.922|28 / none|
| GTE eviction without pool (compile/inference still pooled) |110|29.11/30.31/29.05|47.187|68743.140|22 / none|

The GTE pooled result is now repeatable across three fresh processes (including the original control); Qwen succeeded once. No process/client replacement or cooldown was used. Timings include replacement compilation and sibling verification, not a direct allocator-return measurement.

**The eviction-only negative control unexpectedly succeeded.** Removing only the eviction pool does not reproduce the previous failure when compile/inference are pooled. This falsifies the claim that an eviction's own autorelease pool is necessary for the observed capacity return. It does not contradict draining Objective-C autoreleased temporary objects after each operation or the successful retained-shape reclamation, but exact root-cause placement needs an additional isolated control (compile without pools, inference/eviction with pools). No implementation follows these repeats yet.

Data: `reclaim-scoped-gte-repeats-development.json`, `reclaim-scoped-qwen-development.json`, and `reclaim-eviction-no-pool-development.json`. The four hardware runs completed serially in984seconds. The first launch attempt did not execute any hardware because the ignored file containing the test executable's path was missing after cleanup; reruns used the freshly rebuilt test executable directly.
