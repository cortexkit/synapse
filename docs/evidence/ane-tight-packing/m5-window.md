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

## Outcome

Pending the approved window; no Neural Engine measurements have run during
preparation. Neither benchmark has a verdict yet.
