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

The last control used inherited NSObject allocation and an initializer actually listed by the runtime, after `new` returned nil. It did not enable restricted access. The compiled model getter and hash getter were also enumerated, not invented. All probe objects were released before replacement admission. A successful unload or purge call is not a confirmation of released hardware capacity.

The only measured reclaim boundary that restored full-shape capacity remains **confirmed owning-process exit**, as the earlier paired control demonstrates. Proceed with process epochs, not a client-level reset or a short cooldown. These observations do not identify Apple's allocator implementation or claim a lifetime quota beyond the measured retention behavior.

Reproduction uses the worker test executable selected as `DRIVER` in `capacity-development.md`:

```
python3 crates/synapse-worker-ane-direct/tests/run_reclaim_paths.py --driver "$DRIVER" --timeout 3600 --methods drop-client-references fresh-client model-purge client-purge release-all --out crates/synapse-worker-ane-direct/evidence/reclaim-paths-development.json
python3 crates/synapse-worker-ane-direct/tests/run_reclaim_paths.py --driver "$DRIVER" --timeout 1200 --methods fresh-allocated-client --out crates/synapse-worker-ane-direct/evidence/reclaim-fresh-allocated-client-development.json
```

Launchers unset TMPDIR, select the required Xcode DEVELOPER_DIR and log before/after load beside every run. Serving does not invoke the explicitly experimental vendor diagnostic helpers.
