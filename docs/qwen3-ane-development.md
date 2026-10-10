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

The configuration test prepares the direct worker's converted weight packages
without running inference on an accelerator. It checks SHA-256 checksums of the
original model and tokenizer files, converts the weights using the rules in
`bench/parity/models.json`, and checks each resulting package's recorded checksum.
It also compares catalog installation settings with the startup-loading settings
used for hardware certification. The saved packages are ignored build artifacts
derived from those exact checkpoints, not replacement model checkpoints.

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

The probe records its starting load, time spent compiling and loading, which
sequence length remains loaded, and whether all 28 Neural Engine programs can
stay loaded together: one compiled program for each of Qwen's 28 transformer
layers, for a single fixed sequence length. Each invocation starts a new process
so it does not retain programs for sequence lengths tested earlier.
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
cargo build --locked --release -p synapse-certify-runner
for model in qwen3-embedding-0.6b qwen3-reranker-0.6b; do
  weights="$SYNAPSE_QWEN_WEIGHTS"
  if [ "$model" = qwen3-reranker-0.6b ]; then weights="$reranker_weights"; fi
  quiet > "$backup/$model-certification-load.log"
  set +e
  env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
    "$PWD/target/release/ckdev-synapse-certify" run \
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
```

The runner and candidate must have the same clean build commit. For releases,
point `--assets` at the extracted candidate instead of rebuilding its binaries;
candidate and worker execution remains hard-link-only on the same filesystem.

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

## AFT head-to-head

`aft_embed_headtohead` replays the 6,341 exported document chunks in the metadata's
127-batch order, with two requests in flight and immediate replenishment on
completion. Texts are sent as-is, with no query instruction prefix. Each arm
warms the first batch once, then times the entire replay including that batch.
The ANE decision bar is **wall time at most 2× Bionic**, freeing the GPU even if
ANE is slower. Metal is a reference, not the decision baseline.

This driver **does not refuse on load**: it records one-minute load at each
arm's start and end, plus samples every 30 seconds during warmup/replay. Busy
shared-machine results are rough measurements, not quiet-window certification.
The Synapse arms verify the pinned original checkpoint and share the existing
comparison's private daemon, production candidate, profiles, and `ckdev-*` hard
links. They never discover or contact the production daemon. The aggregate
inline token budget accommodates 64 full-context rows without changing the
model's per-row limit. Any inline diversion, provider error/refusal, wrong row
count/dimension, or non-finite vector fails the run; Synapse vectors must also
be L2-normalized within 1e-3. Bionic normalization is recorded but not required.

Build the candidate/worker as above and build the examples. Set
`SYNAPSE_QWEN_WEIGHTS` to the pinned checkpoint directory from the setup above.
Run from the checkout root:

```sh
env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  CARGO_BUILD_JOBS=1 cargo build --release --locked \
  -p synapse-module -p synapse-worker-ane-direct
env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  CARGO_BUILD_JOBS=1 cargo build --release --locked -p synapse-module --examples

: "${SYNAPSE_QWEN_WEIGHTS:?set the pinned Qwen checkpoint directory}"
export SYNAPSE_COMPARE_ASSETS="$PWD/target/release"
export SYNAPSE_HEADTOHEAD_INPUT="$HOME/.local/share/cortexkit/synapse/aft-headtohead/engram.jsonl"
export SYNAPSE_HEADTOHEAD_OUT="$HOME/Backups/synapse-cert-dev/aft-headtohead.json"

# Pause AFT's own embedding fills before Bionic; the driver cannot verify this.
env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  SYNAPSE_HEADTOHEAD_ARMS=bionic target/release/examples/aft_embed_headtohead

env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  SYNAPSE_HEADTOHEAD_ARMS=ane target/release/examples/aft_embed_headtohead

env -u TMPDIR DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  SYNAPSE_HEADTOHEAD_ARMS=metal target/release/examples/aft_embed_headtohead
```

**AFT must pause its fills during the Bionic arm.** Bionic defaults to
`http://localhost:1234/v1/embeddings`, using
`text-embedding-qwen3-embedding-0.6b`; override the endpoint with
`SYNAPSE_HEADTOHEAD_BIONIC_URL`. The input and its adjacent
`engram.jsonl.meta.json` are both required. The plan must cover every chunk seq
exactly once, in batches of 1–64 rows. There is no synthetic-workload fallback.

All three commands update the same JSON file, retaining other arms only when
the input and metadata SHA-256 hashes match. Run them sequentially, not
concurrently against the same output. Use a fresh output filename for a new
measurement window; rerunning an arm replaces only that arm. Alternatively set
`SYNAPSE_HEADTOHEAD_ARMS=bionic,ane,metal` (the default) for one back-to-back run.
Keep the output outside git. The stdout summary includes `ane_wall / bionic_wall`
and the 2× decision when both arms succeeded. Errors/refusals are counted by
code and cause a nonzero exit after writing the arm's report.

The JSON retains vectors for 64 fixed seqs evenly spread across the file and
reports their median/min cosine for ANE vs Metal and ANE vs Bionic once both
arms are present. These are liveness sanity checks, **not quality gates**;
expected cosines are above 0.999 and about 0.99 respectively. No hardware arms
were run while developing the driver. Pure tests can be run without model
assets or accelerator inference:

```sh
cargo test --locked -p synapse-module --example aft_embed_headtohead
```
