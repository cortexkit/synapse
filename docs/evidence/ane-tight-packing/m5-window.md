# M5 Max Neural Engine window — 2026-10-13

## Preparation and fixed contract

Approved window: **02:00–02:50 UTC on 2026-10-13** (04:00 Madrid).
Source: current `origin/master` fetched before preparation,
`f4143a9ad82ea3a44cb69278dc3ab6035a933088`. No production binary is replaced.
The machine identifies itself as Apple M5 Max. Initial 1/5/15-minute load was
9.57 / 11.45 / 15.36; subsequent native build starts were below the 30 load1
preparation ceiling. Toolchain: rustc 1.99.0 (b940084d7 2026-09-28), Cargo
1.99.0 (5f94df478 2026-08-27), Xcode 27.0 (27A266a), Python 3.9.6.
Native builds use `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`
and unset `TMPDIR`; executable test drivers and private binaries are ad-hoc
signed before execution.

The packed comparison uses the **same committed module-serving driver** as the
[M4 rerun](README.md#third-module-serving-rerun-incomplete-stopped-for-paging):
6,341 exported AFT chunks, 127 outer calls, two in flight, an eight-row engine
quantum, and serial/packed alternating for three repetitions each. Each arm has
1,000 queries under two bulk callers. The 0.80× median-export gate, bit identity,
observed dispatch, and query p50/p99 baseline + max(10%, 5 ms) gates are unchanged.
Packing stays off in production. Source changes in the example only adapt the
busy-process fence to an explicitly unloaded production daemon on a shared Mac
and record load1/5/15 before each arm, including an arm that does not finish.

Approved safety adaptations:

- Allow exactly the production `ck-synapse` PID that the supervisor unloads;
  never another daemon or any worker, even with the same allowlisted PID.
- Fence Synapse/ANE processes, not unrelated `ckdev-subc`, Broca or Prefrontal
  fleets. Those processes are not stopped and may contribute CPU noise.
- Unload **both** `qwen3-embedding-0.6b-ane` and
  `gte-modernbert-base-ane-fp16` via supported singular **`model.unload`**.
  Require unloaded management states, no direct/Swift ANE workers from any
  parent, and two full minutes with no lane admission/refusal/completion/load
  events. If traffic reloads either lane, repeat supported unload and wait;
  abort at 02:15Z without measuring if quietness never holds.
- AFT has no unattended pause. Its indexing and searches can reload the lane.
  The supervisor watches production logs and external workers throughout each
  stage. Any observed production ANE activity voids the current arm and stops
  private work. Production logs do not expose every arrival; the report does
  not equate absence of log events alone with absence of requests.
- Preserve the normal machine-wide ANE lock and default `HOME`. All private
  store/config/data/lease roots remain isolated. No production process is
  signaled. Stop private work at 02:50Z regardless of progress, then restore
  both lanes and record each successful embedding's UTC time, fingerprint and
  actual vector dimension (Qwen `5f7ac7cd`/1024, GTE `5a2374bc`/768).

The parent confirmed that the intended stress procedure is the **unchanged
37-request** `crates/synapse-worker-ane-direct/tests/run_stress.py` and committed
`ane-direct-stress.schema.json`, labeled **development hardware evidence, not
release certification**. Current release workflows have no direct-ANE stress
job. Run it second with a prebuilt release libtest driver and `--wait-for-load`;
do not bypass its load1 < 16/load5 < 20 gates, 600-second admission deadline,
assertions or runner timeout. The window deadline remains an additional outer
stop, not a relaxed test timeout. A load-blocked run is not run, never passed.

## Reproduction

Native builds (serialized; check load1 ≤ 30 first):

```sh
unset TMPDIR
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cargo build --release --locked -p synapse-module --features test-support \
  --bin ck-synapse --example ane_packing_serving
cargo build --release --locked -p synapse-worker-ane-direct
cargo test --release --locked -p synapse-module --lib --no-run --message-format=json
```

Extract the one `synapse_module` test executable from Cargo's artifact messages.
Ad-hoc sign that driver, the two release binaries and the example with
`codesign --force --sign -`. Test-support is required by the existing M4 driver
for its private catalog; these binaries are never deployed.

The four stress packages were independently regenerated without ANE inference
with the committed Rust `parity-manifest convert` command and cached pinned HF
snapshots. All four file SHA-256 values match `bench/parity/models.json`:

| Package | SHA-256 |
| --- | --- |
| gte-modernbert-base | `8a31e3d67a89412b4314136f5ec3489a46dacf438480010139c7d6c2dc045375` |
| gte-reranker-modernbert-base | `2ecb952868e28ed5a45c3f5c057e0f6d46df384ba540e1f05113462fd0d6a1de` |
| qwen3-embedding-0.6b | `1af6d091d7d5f21a998af0f438333d9af051c82af8b27bd2439c60efa09bd5db` |
| qwen3-reranker-0.6b | `cc02357cfc81a92562b251914bed5c8d3960b7edf4814ee73de9f21611e22959` |

Use the following supervisor command with explicit private paths:

```sh
python3 docs/evidence/ane-tight-packing/m5-window.py --dry-run \
  --out "$UNIQUE_RAW_DIRECTORY" --assets "$RELEASE_ASSETS" \
  --driver "$RELEASE_LIBTEST_DRIVER" --weights "$PINNED_QWEN_CHECKPOINT" \
  --input "$AFT_EXPORT_JSONL" --packages "$CONVERTED_PACKAGES"
```

`--dry-run` verifies signatures, every package/workload/checkpoint pin, current
production metadata, process names and readable logs. It does not unload or
embed. `--self-test` exercises synthetic quiet/abort, worker contamination,
deadline, log rotation, vector validation and identity-fenced private cleanup.
For the approved window use `--run` and a different unique raw directory; it
waits once for 02:00Z if launched at 01:58Z. Never run this dated script on a
later night without a new approval and a separately reviewed window definition.
Raw reports, logs and private roots stay under `/tmp`, not build directories.
Compact committed records and raw-file checksums follow after the window.

Preparation verification is retained in `m5-window-preparation.json`: 13 signed
release example tests, 12 supervisor tests (native Python 3.9.6 and Linux
Python 3.14.4), six existing reducer tests, two hardware-free stress schema/timing
tests, strict native example Clippy, scoped formatting, and all input/package
pins passed. Python diagnostics are current and clean; Rust analyzer coverage
was pending, so native Clippy and release compilation are the authoritative
Rust checks. The initial dry run caught a historical-tail partial-record bug;
it was fixed, regression-tested, and the complete non-ANE dry run then passed.
Both silent safety fences were mutation-tested: widening the production PID
exception failed only its named example test (12 others passed), and ignoring
external workers failed only the supervisor contamination test (11 others
passed). Both files were staged before mutation and restored to empty unstaged
diffs afterward. No mutation remains. An isolated `/tmp` Python virtualenv
contains jsonschema 4.25.1 solely for validating the committed stress schema;
no manifest or lockfile changed.

The bounded background sleep was launched at 21:05Z with an explicit seven-hour
process cap. It wakes once at 01:58Z, then invokes the unattended supervisor;
there is no wall-clock status polling loop.

## Outcome

**Aborted at the required quietness deadline; both experiments NOT RUN.** The
supervisor began at 02:00:00.002Z and aborted at **02:15:00.062Z** after
900.060 seconds. No 120-second quiet interval was obtained. The preflight load
at 01:58Z was 11.64 / 9.98 / 9.90. Neither private module-serving arm nor the
37-request stress runner started; there are zero completed serial/packed arms,
no export ratio, no vector-identity/query-latency measurement, and no stress
completion/resource-failure verdict. An exit-zero supervisor means the planned
abort and restoration completed, **not** that either acceptance gate passed.

Supported management unload initially returned `unloaded` for GTE at
02:00:00.194Z and Qwen at 02:00:00.352Z. GTE stayed unloaded during the check.
Qwen kept reloading with production traffic: 118 unload responses were recorded,
29 explicitly saying `unloaded`; 89 had no flat `state` in the returned value
and are not counted as confirmed unloads. Those full response bodies were not
retained. The supervisor required worker absence and unloaded management states
independently of these response labels. At the abort snapshot both lanes were
unloaded and **no direct or Swift ANE workers remained**, but only about 24
seconds had elapsed since the last explicit Qwen unload. The quiet interval
was not shortened to use the remaining window.

An independently captured production-log slice, **02:00Z inclusive to 02:15Z
exclusive**, has 460 Qwen lane events: **230 distinct admitted jobs** (209
`embed.batch`, 21 `embed.query`), 212 `job done` and **18 `job failed`, all with
`code=shape_compiling`**. GTE has no lane events in that slice. These are
production observations during unload/reload churn, not benchmark failures or
a complete request-arrival audit. The first admission was at 02:00:43.842Z and
the last at 02:13:48.339Z. The longest gap between observed lane events,
including the window edges, was **100.633 seconds** (02:11:56.076Z to
02:13:36.709Z), below the required 120 seconds. Thus the log itself does not
supply a qualifying quiet interval. No attempt was made to bypass the chip-busy
condition or continue measuring after the abort.

### Restoration times for AFT retry review

| Production lane | First successful restoration embedding (UTC, 2026-10-13) | Fingerprint prefix | Actual dimensions | Attempts |
| --- | --- | --- | ---: | ---: |
| Qwen direct ANE | **02:15:16.429Z** | `5f7ac7cd` | 1024 | 2 |
| GTE Core ML ANE | **02:15:19.700Z** | `5a2374bc` | 768 | 4 |

The supervisor validated the **returned vectors**, including finite values,
not merely lane metadata. Both management states were `ready` by
02:15:19.708Z. Production daemon PID 91767 was unchanged and never signaled;
a subsequent process snapshot showed its direct and Swift ANE children running
again. **No private daemon or worker was started or remained**; cleanup
signaled no process. Restoration happened immediately after the 02:15Z abort,
not at the nominal 02:50Z end.

### Retained evidence and disposition

`m5-window-result.json` is the compact result and artifact manifest. Raw files
remain under `/tmp/m5-ane-window-20261013/`, outside the repository and all
regenerable build directories:

| Raw file | Bytes | SHA-256 |
| --- | ---: | --- |
| `window.json` | 30,266 | `fdbc5ebbfdbef51cf7ff8410072729b859beab87d2da3fcb26a797a2a6b547e1` |
| `supervisor.log` | 21,618 | `ace1d49d80941c5e2baef068b9e9167d4ec5bca7b59d3b796c5d42fe9ec34f08` |
| `production-lane-window.log` | 78,905 | `9730dac2d2e3792e9034d81f1b7208caaa320a93deff77619680c97115f5db50` |

No benchmark raw report exists because neither driver was launched. The signed
packed-driver alias is retained in `/tmp/m5-ane-window-20261013/`; its checksum
is in `m5-window-result.json`. The supervisor source was
`561e9391aeeeb40fb2666e3a0a6ce83cd92f3621`, based on fetched master
`f4143a9ad82ea3a44cb69278dc3ab6035a933088`. The machine owner approved the
example-only adaptation that permits one unloaded production daemon, fences
Synapse/ANE clients instead of unrelated development fleets, and records
per-arm load. Prepared native asset hashes are in `m5-window-preparation.json`
and the raw window record. The Rust/Python acceptance logic, workload, timeouts
and tolerances were not changed to produce a result. The stress schema was
validated during preparation but there is no new stress report to validate.

Packing remains off by default. **Neither benchmark ran**, so this abort
produced no new performance or stress pass/fail verdict. It neither confirms
nor refutes the M4's partial **packed export wall time / serial export wall
time = 0.61×** observation. It also does not test the newer 100-live-executable
budget, scoped Objective-C autorelease pools, or the worker shape-compilation
lane fix under the committed 37-request stress workload described above. A
future approved window needs an effective pause of production embedding
callers: production batch/query requests were still active during the
**2026-10-13 UTC** window.
