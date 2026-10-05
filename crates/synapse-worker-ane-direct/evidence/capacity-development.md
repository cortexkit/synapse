# Direct ANE capacity measurements (development)

Machine: Mac17,6 (M5 Max). `sw_vers`: 27.0.1, build 26A434. Debug worker tests, Rust 1.99.0. Only this machine was measured. Production's Core ML lane remained resident according to the run context; `ck-synapse` and `ck-synapse-worker-ane-swift` processes were observed and not modified. Without a no-Core-ML control these measurements cannot establish whether Core ML consumes the same pool.

Packages already existed in this worktree. All four converted package SHA-256 digests matched the manifest; no conversion or snapshot copying was necessary. JSON evidence contains those digests, exact MIL and packed-weight byte counts, failures, and loads before/after each run. Source payload bytes are compiler inputs, **not** opaque executable allocation sizes. Slight MIL size differences reflect emitter ordering.

## Sequential mixes, repeated in fresh processes

G = gte-modernbert-base, Q = qwen3-embedding-0.6b. Alternating and all-four mixes walk the ladder round-robin, without replicas. R is the fully admitted executable count; P is the partial next shape's successfully loaded layers before rollback. R + P is their sum at failure, before rollback. Submitted totals include the failing layer; resident totals exclude failed admission.

| Mix / run | Start load 1/5/15 | Fully resident shapes | R + P | Next failure | Resident payload bytes | Submitted payload bytes |
|---|---|---|---|---|---:|---:|
| G / 1 | 23.93 / 19.45 / 18.88 | G@128,256,512,1024,2048 | 110 | none (lower bound) | 1,178,664,119 | 1,178,664,119 |
| G / 2 | 14.78 / 22.20 / 23.03 | same | 110 | none (lower bound) | 1,178,664,076 | 1,178,664,076 |
| Q / 1 | 24.02 / 21.88 / 19.97 | Q@128,256,512,1024 | 112 + 6 | Q@2048 layer 6 | 3,630,410,886 | 3,897,177,702 |
| Q / 2 | 25.28 / 24.17 / 23.75 | same | 112 + 3 | Q@2048 layer 3 | 3,630,410,901 | 3,782,849,068 |
| alternating / 1 | 27.90 / 26.16 / 22.25 | G,Q@128 and G,Q@256 | 100 + 18 | G@512 layer 18 | 2,227,577,876 | 2,426,522,560 |
| alternating / 2 | 43.92 / 33.93 / 28.15 | same | 100 + 15 | G@512 layer 15 | 2,227,577,923 | 2,395,085,361 |
| all four / 1 | 24.89 / 30.13 / 25.04 | all four@128 | 100 + 15 | G@256 layer 15 | 2,219,528,312 | 2,383,522,477 |
| all four / 2 | 28.83 / 33.25 / 29.46 | same | 100 + 15 | G@256 layer 15 | 2,219,528,372 | 2,383,522,493 |

Every failure was `ane_resources_exhausted`, no ANE resources, underlying `0x5`. Whole-shape ceilings repeated exactly; partial failure ceilings varied from 115 to 118. Executable count is a substantially better predictor than payload bytes, which varied from 2.38 to 3.90 billion at failure.

## Scope and large widths

Two fresh processes simultaneously admitted G shapes and retained successful shapes until both reports were written. Start load: 26.56 / 25.42 / 27.06. Each retained G@128 and G@256 (44 executables each). G@512 failed after 14 partial layers in one process and 13 in the other. Timestamped compile callbacks show **115 combined successfully loaded executables at the first failure** (58 + 57), far below the per-process single-run ceiling. This supports a system-wide shared pool, not a per-worker budget. The second callback occurred 3.128 seconds later; by then the first failed shape was rolling back, so its apparent combined count must not be treated as another stable capacity measurement. ANE unloading can lag Rust rollback.

G@4096 and G@8192 admitted first, followed by G@128,256,512: 110 fully resident executables. G@1024 then failed at layer 5, for 115 loaded executables before rollback. Start load: 19.62 / 23.77 / 26.33. Submitted payload total: 1,419,358,166 bytes. This matches the small-width failure count and supports charging each large G executable one slot. Qwen large widths were not independently measured; layer charging remains a heuristic with a runtime resource-exhaustion backstop.

## Earlier stress-count discrepancy

`State::sample` and `resident_shapes()` in the module supervisor count every slot key, including `Admitting` reservations and `Evicting` slots until acknowledgment. Consequently the earlier stress high-water mark of eight shapes is **not** evidence that ~200 physical executables loaded successfully: pending shapes can fail at their first layer while another admission samples all eight slot keys. Inventory is only stored after admission succeeds. These statistics describe budget occupancy, not a hardware executable census.

## Admission direction

Use a module-wide budget of **100 executables**, reserving a shape's full manifest layer count before sending its admission, including pending compiles and evictions until acknowledged. Retain the four-shape per-model cap. At least 15 slots separate this budget from the lowest observed hardware failure. Evict only unleased LRU shapes; wait with a bounded deadline if every candidate is leased, then return the existing transient refusal. Keep `ane_resources_exhausted` as the runtime backstop for unrelated system ANE occupancy and unmeasured hardware.

## Reproduction

Build with `env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo test --locked -p synapse-worker-ane-direct --no-run --message-format=json`. Set `DRIVER` to the `executable` field of its `compiler-artifact` record whose target name is `ck-synapse-worker-ane-direct` and whose `profile.test` is true, then run:

```
python3 crates/synapse-worker-ane-direct/tests/run_capacity.py --driver "$DRIVER" --batch 1 --timeout 1800 --out crates/synapse-worker-ane-direct/evidence/capacity-development-batch-1.json
python3 crates/synapse-worker-ane-direct/tests/run_capacity.py --driver "$DRIVER" --batch 2 --timeout 1800 --out crates/synapse-worker-ane-direct/evidence/capacity-development-batch-2.json
python3 crates/synapse-worker-ane-direct/tests/run_scope_capacity.py --driver "$DRIVER" --timeout 1800 --out crates/synapse-worker-ane-direct/evidence/capacity-development-scope-large.json
```

The launchers unset TMPDIR and set DEVELOPER_DIR to `/Applications/Xcode.app/Contents/Developer`. The production admission helper retains the same signature and delegates to private `admit_with_limit(shape, os, limit, compiler)` with named `PRODUCTION_SHAPE_LIMIT=4`. Only the ignored measurement test passes 64. A production-path unit test rejects a fifth shape; temporarily raising the constant to 5 makes that test fail while the other ten nonignored unit tests stay green.

## Budget implementation and first stress rerun

The approved 100-executable module-wide reservation budget is committed with five new supervisor tests. The reservation and comparison mutations each fail only the selected policy test; the pinned-layer-count control stays green. See `budget-mutations-development.json` for exact mutation evidence.

The full 37-request hardware stress **still failed** after waiting for a quiet start: load 15.10 / 19.45 / 21.30, rising to 25.97 / 24.93 / 22.92 at end. It ran for 567.29 seconds; 4 requests completed and 33 failed at compile layer 0 with no ANE resources (`0x5`). There were 4 admissions, 4 evictions, 34 resource-exhaustion events, no leased evictions, and no nonresident inference. Peak logical occupancy was 4 shapes, at most 100 reserved executables; the reranker pool checks failed. `stress-executable-budget-development.json` preserves the full report without a platform UUID.

Static reservation alone is therefore **not demonstrated sufficient** for concurrent or turnover-heavy workloads. Unload completion and concurrent compiler behavior require separate controls before selecting an additional production policy; the runtime transient refusal remains essential.
