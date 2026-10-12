# Windows Vulkan catalog lanes on a ROG Ally X

This is the first certification run of the `vulkan-windows-amd` row on real
hardware. Each cell goes through the same steps a user's install does: the
candidate downloads the model through the catalog from a byte-pinned local
mirror (`hf_endpoint`), loads the lane by its catalog id with the self-check,
and then runs the full parity fixtures. The
candidate is the release `ck-synapse.exe` and `ck-synapse-worker-vulkan.exe`,
run on the runner's default config (`crates/synapse-certify-runner/src/assets.rs`
`config`) and queue budget. No tolerance, timeout or budget was changed.

## Machine

| field | value |
| --- | --- |
| system | ASUS ROG Ally X, `ROG Ally X RC72LA_RC72LA` (record `system_product_name`) |
| OS | Windows 11 Home 10.0.26200 (build 26200.9457) |
| CPU | AMD Ryzen Z1 Extreme, 16 logical processors |
| memory | 24 GB shared with the GPU (23.2 GB visible to Windows) |
| GPU | AMD Radeon Graphics, integrated, vendor 0x1002, device 0x15bf |
| Vulkan | API 1.4.334, driver 2.0.373, `AMD proprietary driver 25.30.27.03 (LLPC)`, WDDM 32.0.23027.3001 |
| GPU UUID | stored only as `sha256("synapse-certify/gpu/v1" NUL uuid)` = `3aec480e7d8e6d043c584cbd36249f6feb9bd8d59d23d665ae0cf28d734a167a`, the same value as the records' `machine.gpu.uuid_sha256` |
| toolchain | rustc/cargo 1.96.1, Vulkan SDK 1.4.357.0 (installer sha256 `81f474711e9042f4cd22b31b2f7a8870db2e428b21586fb43dd80150be97310d`, the release workflow's pin), Python 3.12 |

`runs/6fbf7209/vulkaninfo-summary.txt` is `vulkaninfo --summary` with both
UUIDs replaced by hashes. The machine is the user's own, not a rental, so
`SYNAPSE_CERTIFY_INSTANCE_ID` was not set.

## Result

The candidate and runner were built in release mode from a clean checkout of
`6fbf7209`. That commit is `origin/master` `3f627d2f` plus a runner fix for
Windows verbatim paths (see "Windows defect found and fixed" below). The ten
gates are the parity evaluator's `complete`, `cosine`, `coverage`,
`coverage_8192`, `dimension`, `finite`, `ranking`, `score`, `semantics` and
`unit_norm`, as recorded in each record's `parity.gates`. Records are in
`docs/evidence/certification/6fbf72096f4e809530804b5104a8cf0998187d77/vulkan-windows-amd/`.

| cell (lane id) | certification | fixtures | fingerprint = pin | 8192 served | 8193 refused before the worker | 512-token row | 8192-token row |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gte-modernbert-base-vulkan | **passed**, record written | 17/17, all 10 gates true; min cosine 0.9999979, max norm error 8.1e-7 | yes, `9d281846…2182` | yes, 1 worker request, not truncated | yes, `sequence_too_long`, 0 worker requests | 535 ms (533–556) | 15.6 s (15.60–15.63) |
| gte-reranker-modernbert-base-vulkan | **passed**, record written | 125/125, all 10 gates true; max abs score error 2.83e-3, tau 1.0 on pool-10 and pool-100 | yes, `160e6ae7…6c99` | yes, 1 worker request, not truncated | yes, `sequence_too_long`, 0 worker requests | 544 ms (544–547) | 15.3 s (15.32–15.41) |
| qwen3-embedding-0.6b-vulkan | **refused**, no record | 16/17 completed (min cosine 0.9999998, max element error 9.8e-5, token ids all match); `long-8192` never completes | yes, `9a33500f…aeb9` (`models.list` and every served response) | **no**: `engine_crashed at timeout: request exceeded 30000 ms` after 30.6 s and 32.2 s | yes, `sequence_too_long`, 0 worker requests (catalog-probe) | 2.65 s (2.64–2.65) | **> 30 s**, killed by the worker limit |
| qwen3-reranker-0.6b-vulkan | **refused**, no record | 60/125 completed in the best run (max abs score error 5.9e-6, tau 1.0 on what completed); `long-8192` and the 64-candidate pool-100 request never complete | yes, `be2a9be4…47d7` | **no**: the same timeout after 30.7 s and 30.3 s | yes, `sequence_too_long`, 0 worker requests (catalog-probe) | 2.62 s (2.62–2.62) | **> 30 s**, killed by the worker limit |

The 512 and 8192 columns come from `tools/timing-probe.rs`: single requests by
lane id with the runner's exact parameters on a warm lane, three repeats (one
for a Qwen3 8192 request), median with the range in brackets. Each run of the
catalog probe also timed every fixture request once (`runs/6fbf7209/probe.*.jsonl`):
537 ms / 15.0 s (gte embed) and 560 ms / 17.5 s (gte rerank). An earlier
timing-probe run on the `3f627d2f` candidate, whose `ck-synapse-worker-vulkan.exe`
has the same sha256 as the one certified here,
measured 593 ms / 18.7 s for gte embed. Some of that spread is the handheld's
power and thermal state. Cold first request, including load and self-check:
2.8 s, 4.4 s, 14.9 s, 18.1 s.

Not run: the four `cuda-windows-nvidia` cells and the four
`vulkan-windows-nvidia` cells. The Ally has no NVIDIA GPU.

### Runner output, verbatim

Passing cells print the record on stdout. It is byte-for-byte the committed
JSON (checked for gte-modernbert-base). The runner's stderr is:

```
{"runner_commit":"6fbf72096f4e809530804b5104a8cf0998187d77","runner_path":"C:\\Users\\ufuka\\synapse-cert\\synapse\\target\\release\\ckdev-synapse-certify.exe","runner_sha256":"673165a34a58b23ee947e9224ce3895d9cccd705872e6e00badc47331e6fd435"}
```

followed by the candidate's own log lines. For qwen3-embedding-0.6b-vulkan
and qwen3-reranker-0.6b-vulkan the runner
exited 1 with empty stdout, wrote no record, and ended stderr with:

```
certification_refused: candidate error: {"error":{"class":"transient","code":"engine_crashed","message":"engine_crashed at timeout: request exceeded 30000 ms","retry_after_ms":250,"safe_to_retry_same_request":true},"module_generation":1}
```

Runner wall times: gte-modernbert-base 27.5 s, gte-reranker-modernbert-base
45.0 s, qwen3-embedding-0.6b 73.5 s, qwen3-reranker-0.6b 71.3 s
(`runs/6fbf7209/cells-and-artifact-sha256.txt`).

### The Qwen3 failures are time, not numbers

The runner stops at the first candidate error and prints only that error. The
8193 check never runs for these cells. `tools/catalog-probe.rs` repeats the
runner's session (catalog download from the mirror, lane id, the same request
grouping, parameters and evaluator, and the 8193 composition with the
`certify.observations` worker counter on either side), but it keeps going after
a failed request.

- `long-8192` exceeds the module's fixed 30 s worker request timeout on both
  models. On the RTX 5070 Ti the 8192 row also timed out
  (`docs/evidence/catalog-backends/README.md`). The RTX 5090 run reported at
  2.55 s for this row is not in this checkout, so it is not compared here. A
  Qwen3 512-token row takes 2.6 s here, about 6x the 5070 Ti's 0.42 s.
- **The reranker also fails on a second request, not only at 8192.** The
  runner packs pool-100 into a 64-candidate request (7,490 tokens) and a
  36-candidate request (4,230 tokens). On a fresh worker that had not timed
  out (`probe.qwen3-reranker-0.6b.skip8192.jsonl`), the 64-candidate request
  timed out at 30.2 s and the 36-candidate request took 27.4 s. The same
  64-candidate request took 7.5 s on the 5070 Ti.
- After two worker timeouts the module quarantines the lane. The next request
  is answered in 10 ms with `model/config is quarantined after repeated worker
  crashes` (`probe.qwen3-reranker-0.6b.jsonl`). That is why the full-order
  probe completed only 24 of 125 reranker cases.
- Every completed case is well inside tolerance, and the token ids the backend
  saw equal the fixture's `input_ids` on every case. Over completed cases the
  evaluator reports only `incomplete_output`.
- Fixing this needs a maintainer decision: a faster Qwen3 Vulkan path on
  integrated GPUs, a different long-request timeout policy, or not shipping the
  Qwen3 Vulkan lanes on this class of GPU. Nothing was changed for it here.

## Windows defect found and fixed: verbatim paths in the runner

With candidate and runner built from `3f627d2f` (`origin/master` at the time), every
Windows cell failed in 0.6 s, before the module registered:

```
synapse boot failed after HELLO_ACK: storage: synapse store: storage backend: unable to open database file: \\?\C:\Users\ufuka\synapse-cert\synapse\crates\synapse-certify-runner\.live\run-17900-1791824935826353100\data/cortexkit/synapse/store.db
certification_refused: candidate exited before registration: {"error":{"code":"unknown_module","message":"module_id 'synapse' is not registered"}}
```

- **Cause.** The runner canonicalized `--checkout` with
  `std::fs::canonicalize`, which on Windows always returns a verbatim
  `\\?\C:\…` path. It passes `<scratch>/data` to subc-daemon's
  `StorageConfig::Sqlite`. subc-daemon 0.34.0 builds the module store path as
  `format!("{}/cortexkit/{module_id}/store.db")`, and inside a verbatim path a
  forward slash is not a separator.
- **Fix.** Commit `6fbf7209` (made on this branch with the maintainer's
  approval) uses
  `dunce::canonicalize` in `synapse-certify-runner` for:
  - assets, checkout and mirror;
  - the mirror root and each served file. Using dunce on both sides keeps the
    containment check comparing paths of the same form.

  A new unit test, `containment_check_serves_inside_and_refuses_link_escaping_root`,
  passes on Linux (26 tests) and on this ROG Ally X (24 tests; the other two are
  Unix-only). With the `starts_with(root)` check deleted it fails on both, and
  on Windows it then serves the file reached through the junction.
- **Linux.** Linux runs never saw this, because Linux has no verbatim paths.

**The shipped module does not have this verbatim-path defect.**

- `catalog_probe.rs:53` (`cached_probe`) canonicalizes the worker path to the
  verbatim form. The path is only spawned and used as a cache key. On the Ally
  that probe is what makes `models.catalog` report the Vulkan lanes
  `runnable: true` (`detect_worker_backend`, compiled for non-macOS targets
  and tests). The runner
  refuses a non-runnable lane, so every cell above reaching download and load
  shows the verbatim worker path spawns correctly on Windows.
- `catalog_probe.rs:78` is `#[cfg(test)]`.
- `catalog/mod.rs:1339` is inside `mod tests` (from line 1285), in an ANE test.
  Production `local_file_url` callers receive model-cache paths built from
  `CORTEXKIT_MODEL_CACHE` or `LOCALAPPDATA`, which are not canonicalized.
- A user's daemon gets its data home from `LOCALAPPDATA`, not from a
  canonicalized path.
- `tools/timing-probe.rs` starts the candidate the way a user's daemon does
  (plain paths, worker from `SYNAPSE_OWNED_VULKAN_WORKER_BIN`). It downloaded,
  loaded and served all four lanes.

**Second runner gap, not fixed.** On Windows the runner does not isolate the
model cache. It sets `XDG_DATA_HOME`, which Windows ignores, so the candidate
would write to `%LOCALAPPDATA%\cortexkit\models`. `tools/run-cell.sh` sets
`CORTEXKIT_MODEL_CACHE` to a fresh directory for each cell, so every download
went through the mirror and the user's cache was never touched (it does not
exist on the Ally).

## Smart App Control

Smart App Control is on (`VerifiedAndReputablePolicyState = 1`). It refuses
some freshly linked executables at random with `An Application Control policy
has blocked this file. (os error 4551)`. Between 19:02 and 19:38 Ally local time (UTC+2, 2026-10-12),
`Microsoft-Windows-CodeIntegrity/Operational` logged 21 refusals:

- **Candidate and runner builds at `3f627d2f`:** 6 build scripts
  (`icu_properties_data` three times, `portable-atomic`, `rustls`,
  `subc-daemon`). The rebuild at `6fbf7209` had no build-script refusals.
- **Scratch drivers:** 8 build scripts and proc-macro DLLs.
- **First launch of a freshly linked executable:** 7 refusals.
  - `ck-synapse.exe` once per build, and `ckdev-synapse-certify.exe` once.
    Each was refused on its first launch and ran on the next.
  - A scratch probe binary was refused on four launches in a row, until it
    was relinked.

`tools/build-retry.sh` reruns cargo after cleaning only the refused package.
That relink gives the binary a new PE timestamp and PDB GUID, and the next
attempt has always been allowed. Nothing was signed, no exclusion was added and
SAC was not changed. Users building synapse from source on Windows with SAC on
will hit the same refusals, and on the first launch of their own build.

## Exact commands

Every command below ran over `ssh ally` in Git Bash, under `C:\Users\ufuka\synapse-cert`:

```
git clone https://github.com/cortexkit/synapse synapse    # origin/master 3f627d2f
# 6fbf7209 arrived as a git bundle and was checked out clean (git status --porcelain empty)
vulkansdk-1.4.357.0.exe --root C:\Users\ufuka\synapse-cert\VulkanSDK\1.4.357.0 --accept-licenses --default-answer --confirm-command install copy_only=1
python mirror.py C:/Users/ufuka/synapse-cert/synapse C:/Users/ufuka/synapse-cert/mirror   # all 8 files: size and sha256 equal the catalog
export VULKAN_SDK='C:\Users\ufuka\synapse-cert\VulkanSDK\1.4.357.0'
bash build-retry.sh logs/b build --release --locked -p synapse-module -p synapse-worker-vulkan --features vulkan --bin ck-synapse --bin ck-synapse-worker-vulkan
bash build-retry.sh logs/r build --locked --release -p synapse-certify-runner
cp target/release/ck-synapse.exe target/release/ck-synapse-worker-vulkan.exe ../assets/windows-x64/
python scripts/check-release-candidate.py spirv --worker ../assets/windows-x64/ck-synapse-worker-vulkan.exe   # exit 0
../assets/windows-x64/ck-synapse.exe certify source                                                          # 6fbf72096f4e809530804b5104a8cf0998187d77
CORTEXKIT_MODEL_CACHE=<fresh dir> target/release/ckdev-synapse-certify.exe run --row vulkan-windows-amd --model <model> --assets C:\Users\ufuka\synapse-cert\assets --checkout C:\Users\ufuka\synapse-cert\synapse --mirror C:\Users\ufuka\synapse-cert\mirror
```

`tools/run-cell.sh`, `tools/run-probe.sh` and `tools/run-timing.sh` are the
wrappers that were run. The candidate build at `3f627d2f` took 2 min 05 s, and
the runner 48 s. The rebuild at `6fbf7209` was incremental: 66 s and 13 s.

Certified artifacts, built at `6fbf7209` (sha256): `ck-synapse.exe`
`178bb776c0b7755c4b8626f48452134c291dd58bc9247759af7d28b9503f9963`,
`ck-synapse-worker-vulkan.exe`
`91e9763f92133ec11e7fd31e61567910ff74b29b2038e61f822456f06b1f8280` (identical
at `3f627d2f`), `ckdev-synapse-certify.exe`
`673165a34a58b23ee947e9224ce3895d9cccd705872e6e00badc47331e6fd435`.

## Files

- `runs/6fbf7209/`:
  - `probe.<model>.jsonl`: catalog-probe in fixture order;
  - `probe.<model>.skip8192.jsonl`: the Qwen3 cells without the 8192 case;
  - `cells-and-artifact-sha256.txt`;
  - `vulkaninfo-summary.txt`.
- `tools/`: the scripts and two scratch drivers. They were built outside the
  synapse workspace on the Ally, with subc crates pinned to this repository's
  versions. catalog-probe used the checkout's `Cargo.lock` and its
  `synapse-parity` and `synapse-certify` crates.
- Raw logs (runner and candidate stderr, build logs, probe stderr) stay on the
  Ally under `C:\Users\ufuka\synapse-cert\logs`.

**Records are tied to the fix commit.** `6fbf7209` is a branch commit. The
runner refuses unless candidate and runner come from the same clean commit, so
these records stay valid only if that commit lands unrewritten.

**Clean-up.** No process started for this run is still running, the
per-cell model caches were deleted, and `%LOCALAPPDATA%\cortexkit` holds only
its earlier `aft` entry. The mirror, the assets, the SDK copy and the logs
remain under `C:\Users\ufuka\synapse-cert`.
