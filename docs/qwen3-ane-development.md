# Qwen3 0.6B ANE catalog development

## Status on the M5 Max

The catalog now offers `gte-modernbert-base-ane`,
`qwen3-embedding-0.6b-ane`, and `qwen3-reranker-0.6b-ane`. They reference the
existing profiles in `bench/parity/models.json`, convert the original pinned
checkpoint with the certification converter, and use the same stored-config,
worker supervision, token grammar, and sealed parity self-check as profile
preloads. There is no second copy of the manifest profiles. The three existing
Metal declarations and fingerprints are unchanged, and Metal remains the first
backend when an unqualified catalog id is selected.

This is **not a shipping certification**. During this development session the
one-minute load was never below 16: observed values included 25.36, 33.39, 54.34,
52.24, 36.06, 39.23, and finally 41.20 (final 1/5/15-minute load:
41.20/30.66/34.58). The actual comparison runner was invoked and refused at that
last load before launching a daemon or worker.
Hardware certification, Qwen 8192 shape admission, latency comparison, and
throughput measurement were therefore not run. No accuracy, boundary,
placement, latency ratio, or throughput pass is claimed. The original-weight
conversion/configuration test passed for all three catalog ANE lanes; it does
not execute an accelerator and is not evidence of hardware accuracy or capacity.

The 8192 ceiling remains unchanged. Qwen charges 28 executables per shape
against the shared live budget of 100, but that count is not proof that an
8192 executable actually loads. The next quiet-window measurement must establish
this before a shipping decision. If it fails, report the largest independently
loaded shape and its compile/load time; do not reduce the ceiling.

Release evidence remains required. In particular, adding ANE file memberships
changes installation-manifest digests, and the catalog evidence gate now binds
the direct worker's source tree too. Existing release evidence must not be
treated as certification of these new lanes. Development certification records
contain the Mac's hardware UUID and must remain outside git pending the owner's
publication decision.

## Quiet-window commands

Run from a clean, committed checkout on this Mac. All hardware commands unset
`TMPDIR` and select Xcode. Do not point any command at the production daemon.
The candidate producer, worker, and comparison harness create their own daemon
and `ckdev-*` hard links.

```sh
set -eu
backup="$HOME/Backups/synapse-cert-dev"
mkdir -p "$backup"
export SYNAPSE_QWEN_WEIGHTS="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
reranker_weights="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473"

quiet() {
  python3 - <<'PY'
import os
load = os.getloadavg()
print("load_1_5_15", load, flush=True)
if load[0] >= 16:
    raise SystemExit("quiet window refused; no hardware run")
PY
}

env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  CARGO_BUILD_JOBS=1 cargo build --release --locked \
  -p synapse-module -p synapse-worker-ane-direct
```

### 8192 capacity, before certification

The configuration test can prepare digest-verified converted packages without
using accelerators. It checks the original checkpoint and tokenizer pins and
rebuilds each lane through both install configuration and certification preload
configuration. The packages are ignored build artifacts, not alternate weights.

```sh
env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  CARGO_BUILD_JOBS=1 SYNAPSE_PINNED_HF_CACHE="$HOME/.cache/huggingface/hub" \
  ANE_TEST_PACKAGES="$PWD/target/ane-direct-packages" \
  cargo test --release --locked -p synapse-module --lib \
  catalog_original_snapshots_rebuild_ane_lane_pins -- --ignored --nocapture

for model in qwen3-embedding-0.6b qwen3-reranker-0.6b; do
  quiet > "$backup/$model-capacity-load.log"
  env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
    CARGO_BUILD_JOBS=1 ANE_TEST_MODEL="$model" ANE_TEST_SHAPE=8192 \
    ANE_TEST_PACKAGES="$PWD/target/ane-direct-packages" \
    cargo test --release --locked -p synapse-worker-ane-direct \
    fresh_process_qwen_single_shape_admission -- --ignored --nocapture \
    > "$backup/$model-capacity-8192.log" 2>&1
done
```

The probe records its starting load, compile/load milliseconds, final resident
shape, and whether all 28 executables loaded. Each invocation is a fresh process.
If 8192 fails, stop the certification/benchmark sequence, and repeat the same
probe separately with `ANE_TEST_SHAPE=4096`, then 2048, 1024, 512, 256, and 128
as needed to identify the largest shape that loads. Keep the failure log as well
as the successful shape's time. Give each invocation a bounded external runner
timeout (for example 1200 seconds), rather than abandoning a live compile.

### Hardware certification

This is the command from [direct-ane-serving.md](direct-ane-serving.md), with
Qwen's original checkpoint substituted. Embedding runs first, then reranking.
The command writes a record under the checkout; immediately move that record
to the private backup directory, including when the record reports a drop.

```sh
source_commit="$(git rev-parse HEAD)"
scratch="$(mktemp -d)"
ln "$PWD/target/release/ck-synapse" "$scratch/ckdev-synapse"
for model in qwen3-embedding-0.6b qwen3-reranker-0.6b; do
  weights="$SYNAPSE_QWEN_WEIGHTS"
  if [ "$model" = qwen3-reranker-0.6b ]; then weights="$reranker_weights"; fi
  quiet > "$backup/$model-certification-load.log"
  set +e
  env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
    "$scratch/ckdev-synapse" certify run \
    --row ane-m5 --model "$model" \
    --assets "$PWD/target/release" --checkout "$PWD" --weights "$weights" \
    > "$backup/$model-$source_commit.stdout.json" \
    2> "$backup/$model-$source_commit.stderr.log"
  result=$?
  set -e
  record="$PWD/docs/evidence/certification/$source_commit/ane-m5/$model.json"
  if [ -f "$record" ]; then mv "$record" "$backup/$model-$source_commit.record.json"; fi
  if [ "$result" -ne 0 ]; then exit "$result"; fi
done
rm "$scratch/ckdev-synapse"
rmdir "$scratch"
```

Report the records' accuracy cases, 8192 acceptance and 8193 refusal, and
placement gates **verbatim**. Do not replace them with the earlier single-fixture
parity measurements. The certifier's latency series is not an interleaved
comparison, so run the separate comparison below for the owner's 3× decision.

### Warm latency and AFT-shaped throughput

```sh
quiet > "$backup/qwen-comparison-load.log"
env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  CARGO_BUILD_JOBS=1 SYNAPSE_COMPARE_ASSETS="$PWD/target/release" \
  SYNAPSE_COMPARE_OUT="$backup/qwen-ane-metal-production-comparison.json" \
  cargo run --release --locked -p synapse-module --example qwen_ane_compare
```

The harness refuses before launching anything unless the one-minute load is
below 16. It verifies the original checkpoint's pinned files, creates a private
daemon, and serves both profiles through the production candidate. At each of
128 and 512 composed tokens, it performs an untimed warmup per arm followed by
nine samples per arm, alternating ANE/Metal then Metal/ANE. It checks the actual
response token counts and reports medians and ANE/Metal ratios. Only the 512-token
ratio decides whether latency is no more than 3× Metal; hardware accuracy is an
independent requirement.

For throughput it generates code-like chunks with 100–128 composed tokens,
warms each arm, then runs twenty 64-row `embed.batch` calls per arm as ten pairs
with two calls in flight. It reports sustained rows/minute over those pairs and
nearest-rank per-call p50/p90, excluding the warmup. Every timed request records
load before and after; a window that becomes busy fails rather than producing a
claimed quiet result. Output goes to the private backup directory, not git.
The generator was checked against the original pinned tokenizer without
accelerator inference: its 64 rows compose to 7670 tokens, inside the 8192-token
inline budget. This validates the workload, not its hardware throughput.

The direct ANE worker **runs rows one at a time**: its batch handler loops over
the flattened row slices and calls `Model::run` once per row. Each row executes
the layer chain separately. A 64-row call is not a multi-row ANE dispatch, and
two concurrent calls do not make those layer executables batch-vectorized.
