# Unload reclamation and compiler concurrency (development)

Mac17,6 (M5 Max), OS build 26A434, debug tests. GTE denotes gte-modernbert-base. Loads are the unitless macOS 1-, 5-, and 15-minute load averages, in that order. Production Core ML lane remained resident per run context and was untouched. The binding exposes no private hardware instance/peak counter. Callback counts below are successfully loaded executables held until the process finishes, not an ANE allocator census.

## Reclaim after eviction

Start load 1/5/15: **20.15 / 22.15 / 22.69**. A fresh process admitted GTE@128,256,512,1024,2048, for 110 executables. Removing GTE@128 invoked 22 unloads and left 88 fully resident executables. Replacement GTE@4096 has the same 22-layer charge:

| Delay after preceding attempt | Actual start after eviction | Finished after eviction | Loaded before rollback | Result |
|---:|---:|---:|---:|---|
| 0 ms | 0.001 ms | 36.806 s | 14 | resource failure at layer 14 |
| 50 ms | 36.863 s | 39.550 s | 0 | resource failure at layer 0 |
| 200 ms | 39.760 s | 42.681 s | 0 | resource failure at layer 0 |
| 1 s | 43.691 s | 46.646 s | 0 | resource failure at layer 0 |
| 5 s | 51.646 s | 54.769 s | 0 | resource failure at layer 0 |

All 124 unload log lines observed in this process reported BOOL true and no NSError (22 evicted layers, 14 rolled-back partial layers, 88 final resident layers). **No capacity recovery was observed through 54.769 seconds**, despite successful unload replies. These timings include graph construction and compilation; they do not establish millisecond hardware reclamation latency. Small configured waits are intervals between attempts, not isolated probes at those absolute offsets.

## Reclaim across owner process exit

Start load: **17.64 / 20.96 / 22.14**. An owner process retained 110 GTE executables. A second fresh process failed to admit GTE@128 at layer 11 while the owner lived. The owner then released its models and exited normally; the contender's first retry started 0.001 ms after observing the exit marker, loaded all 22 layers, and completed in 7.788 seconds with no error. Owner exit therefore restored usable capacity by the first retry, but the compile/load duration prevents asserting submillisecond reclamation.

The owner emitted 110 observed successful unload lines and the contender 32, all without NSError. The paired launcher combines libtest stdout and stderr; an initial unload line can be prefixed by libtest's test-name output and omitted by the line-start parser. Counts in this combined-log control are observed lower bounds, not a complete unload census. The direct diagnostic itself logs every BOOL/NSError result.

## Concurrent compilation controls

All graphs were constructed before a thread barrier. Each thread retained its successfully loaded executables through completion of every thread. For first-layer controls, distinct widths are 128,256,...,128×N; these are standalone layer graphs, not full supervisor shapes or replicas. Eight full supported shapes would exceed the measured ceiling, so only two/four complete GTE shapes were tested.

| Control | Concurrent jobs | Loaded executables | Start load 1/5/15 | Resource failures |
|---|---:|---:|---|---:|
| complete GTE shapes@128,256 | 2 | 44 | 12.94 / 17.97 / 20.79 | 0 |
| complete GTE shapes@128,256,512,1024 | 4 | 88 | 18.45 / 18.67 / 20.91 | 0 |
| first-layer graphs | 2 | 2 | 30.80 / 25.39 / 23.38 | 0 |
| first-layer graphs | 4 | 4 | 27.19 / 24.85 / 23.22 | 0 |
| first-layer graphs | 8 | 8 | 26.31 / 24.84 / 23.27 | 0 |
| first-layer graphs | 16 | 16 | 20.12 / 23.49 / 22.82 | 0 |
| first-layer graphs | 32 | 32 | 24.73 / 24.25 / 23.12 | 0 |

Every observed unload in these standalone controls returned true with no NSError. No compiler-concurrency failure threshold was observed up to 32 concurrent first-layer graphs, or four full shape compiles. This does not prove absence at higher concurrency or for Qwen graphs.

## Interpretation so far

Unload success does **not** demonstrate resource reclamation suitable for immediate replacement admission. These controls support process-lifetime resource retention or delayed cleanup beyond the tested interval; they do not expose Apple's internal counter or prove an unbounded lifetime quota. Process exit is the only tested action that restored capacity after a failed admission. A small eviction cooldown or compile serialization alone is not supported by the reclaim/concurrency data. A serialized full-stress experiment is still required before proposing additional production behavior.

## Reproduction

Build with `env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo test --locked -p synapse-worker-ane-direct --no-run --message-format=json`. Set `DRIVER` to the `executable` field of the `compiler-artifact` record whose target name is `ck-synapse-worker-ane-direct` and whose `profile.test` is true, then run:

```
python3 crates/synapse-worker-ane-direct/tests/run_turnover_capacity.py --driver "$DRIVER" --timeout 1800 --out crates/synapse-worker-ane-direct/evidence/turnover-concurrency-development.json
```

The launcher records before/after load for every experiment, unsets TMPDIR, selects the required Xcode DEVELOPER_DIR, and enables opt-in `ANE_UNLOAD_DIAGNOSTICS`. The vendor checksum and provenance document cover the diagnostic patch. The native unload API's return type and scheduling/retry policy are unchanged.

## Serialized admission stress

With `ANE_STRESS_SERIALIZE_ADMISSION=1`, the ignored test holds a lane-wide mutex for each complete shape admission (including its resource retry). The mutex and environment toggle exist only under `cfg(test)`; production policy was not changed. Start load: **11.64 / 19.26 / 22.60**, end load: **8.93 / 13.46 / 18.90**. The run took 314.02 seconds and **failed**: 4/37 requests completed, 33 failed, 4 shapes were admitted and 4 evicted, 33 resource-exhaustion events, no leased evictions or nonresident inference. Request errors name layers 18 and 0. Both reranker-pool checks failed. Peak logical occupancy was four shapes (two per model); weighted sample reservations never exceeded 100. The un-serialized budget run also peaked at 100 reserved executables and completed only four requests. See `stress-serialized-admission-development.json` for the full report.

Reproduction, with the module test driver built from this source and `MODULE_DRIVER` set to its executable:

```
env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer ANE_STRESS_SERIALIZE_ADMISSION=1 python3 crates/synapse-worker-ane-direct/tests/run_stress.py --driver "$MODULE_DRIVER" --wait-for-load --timeout 14400 --out crates/synapse-worker-ane-direct/evidence/stress-serialized-admission-development.json
```

## Proposed additional policy (not implemented)

Neither compiler concurrency nor short eviction cooldown explains the demonstrated failure. Retain the approved 100-slot headroom budget, but investigate **cumulative process-epoch charges**, module-wide across live worker epochs: successful and conservatively reserved failed compilations continue to consume budget after shape eviction, until the owning process is confirmed exited. Reuse already compiled shapes where possible. To reclaim a worker epoch, wait until all of that worker's leases and pending exchanges drain, confirm the old process has exited, start a fresh worker, and reload its pinned packages before admitting replacement shapes. Never recycle a leased worker or release charges merely on an `EVICTED` acknowledgment.

The existing generic `AneWorkerChannel::restart` only drops the stream and invokes a connector; it does not confirm owner exit or reload pinned packages. The real hardware test wrapper reloads after reconnect, but its factory does not wait for the old child on that path. A safe production recycle contract therefore requires explicit exit confirmation and package restoration, not simply invoking the current recovery helper. These measurements support owner-lifetime retention as a working policy basis; they do not identify Apple's private allocator or prove a cumulative quota mechanism. No additional production policy was changed pending a ruling.
