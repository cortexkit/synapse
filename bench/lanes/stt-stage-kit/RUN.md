# Owner-run matched Parakeet stages (M5)

This kit measures, not optimizes, the existing serving math. **Only the hardware
owner runs `measure`, during an explicitly paused Neural Engine window.** It does
not stop other processes, change `TMPDIR`, or bypass the lane lock. Both native
processes take the production nonblocking lock at
`$HOME/Library/Caches/ck-synapse/ane-direct.lock`; there is no repository lock file.
Each process releases it on exit. The pause must cover the entire sequence,
including gaps between processes. A busy lane is a refusal, not permission to
substitute another lock.

## Before the paused window: assets, build, sign, seal

Run from the repository root on the M5 with full Xcode installed. Use the same
sealed prepared WAVs, pinned snapshot, tensor mirror and already converted ANE
package as the signed certification harness. The tensor mirror is the
content-addressed store of reference encoder tensors and decode traces used by
the parity validators; see the [reference asset setup](../../parity/reference/README.md)
and the [signed harness guide](../../../crates/synapse-worker-ane-direct/README.md).
The package **must already exist**;
the preparation step hashes it, so conversion cannot silently move into the
window. No attested candidate binary or asset is rebuilt by `measure`.

Set these existing locations (the snapshot revision is fixed):

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export ANE_PARAKEET_SNAPSHOT="$HOME/.cache/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
export ANE_PARAKEET_MIRROR="$PWD/bench/parity/.cache-out"
export ANE_PARAKEET_WAVS="$PWD/target/scratch/parakeet-wavs"
export ANE_PARAKEET_PACKAGE="$PWD/target/scratch/ane-parakeet-kit/parakeet.safetensors"
test -f "$ANE_PARAKEET_PACKAGE"
export KIT="$PWD/bench/lanes/stt-stage-kit"
export FLUID="$PWD/bench/lanes/stt-fluidaudio"
export EVIDENCE="$PWD/.cortexkit/internal-docs/investigations/ane-parakeet-speed/matched-$(date -u +%Y%m%dT%H%M%SZ)"
```

Use the shared benchmark's pinned **66a239a32609d3f90b50a9634e4b2820ab977011**
runner, not the newer **16780f04d2ece24f67d0fc00efccd0296491c1a8** source revision.
The assets remain at **7dd20fe6b1797d35f5e3307e8b1732d9a178edfe**. This establishes
a new reproducible comparison and makes no claim about earlier benchmark binaries.
Python needs the existing
benchmark's `huggingface_hub` and `soundfile` dependencies (install in an existing
benchmark virtual environment, not the system Python).

```sh
export FLUID_MODEL="$(python3 - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('FluidInference/parakeet-tdt-0.6b-v3-coreml',
    revision='7dd20fe6b1797d35f5e3307e8b1732d9a178edfe',
    allow_patterns=['Preprocessor.mlmodelc/**', 'Encoder.mlmodelc/**',
        'Decoder.mlmodelc/**', 'JointDecisionv3.mlmodelc/**', 'parakeet_vocab.json']))
PY
)"
xcrun --find metal
bash crates/synapse-worker-ane-direct/tests/run_parakeet.sh build
export ANE_EXECUTABLE="$(< target/scratch/ane-parakeet-kit/executable)"
swift package --package-path "$FLUID" resolve
python3 "$FLUID/prepare-timing.py" "$FLUID/.build/checkouts/FluidAudio"
swift build --package-path "$FLUID" --configuration release --disable-automatic-resolution -Xswiftc -DSTAGE_KIT
export FLUID_EXECUTABLE="$FLUID/.build/release/stt-fluidaudio"
codesign --force --sign - --identifier ckdev-stt-fluidaudio-stage "$FLUID_EXECUTABLE"
codesign --verify --strict "$ANE_EXECUTABLE"
codesign --verify --strict "$FLUID_EXECUTABLE"

# Existing parity-normalized benchmark scorer: build now, never during the window.
cargo build --locked --release --manifest-path bench/lanes/stt-bias/parity-score/Cargo.toml
export SCORER="$PWD/bench/lanes/stt-bias/parity-score/target/release/stt-bias-parity-score"
mkdir -p target/stt-bias
export MANIFEST="$PWD/target/stt-bias/stage-manifest-$(date -u +%Y%m%dT%H%M%SZ).jsonl"
python3 bench/lanes/stt-bias/parity.py prepare --wavs "$ANE_PARAKEET_WAVS" --manifest "$MANIFEST"
python3 "$KIT/prepare.py" --manifest "$MANIFEST" --wavs "$ANE_PARAKEET_WAVS" \
  --fluid-model "$FLUID_MODEL" --snapshot "$ANE_PARAKEET_SNAPSHOT" \
  --package "$ANE_PARAKEET_PACKAGE" --ane-executable "$ANE_EXECUTABLE" \
  --fluid-executable "$FLUID_EXECUTABLE" --scorer "$SCORER" --out "$EVIDENCE"
```

`prepare-timing.py` requires a pristine dependency checkout at the exact pin. It
validates every original file and replacement context **before writing any file**.
Do not reset a dirty checkout to make it pass. Use a fresh benchmark build
directory/check-out if already prepared. `timing-patch.json` contains the exact
timing-only replacements; `StageKit.swift` is copied into that dependency. All
hooks are behind `#if STAGE_KIT`; ordinary benchmark builds and the shipped worker
do not include the added stage instrumentation. The Swift configuration still
requests CPU+ANE exactly as the shared runner does; no GPU override is introduced.

`prepared.json` seals source revision/status, binary, converted package, snapshot,
model asset and WAV hashes, chip/OS, precision and the immutable request order.
Preparation reuses the existing full-corpus WAV validator (1200 IDs, hashes,
FLOAT mono 16 kHz, sample counts). The signed harness independently validates
fixture seals and model assets. Keep `prepared.json` and all receipts private:
they contain local paths and transcripts.

## During the owner-paused window: one command, no builds

Pause other Neural Engine users through the normal owner arrangement. Do not kill
unrelated processes. The harness admits at most four corpus rungs (96 ANE
executables), warming only admitted shapes. Other clients must not retain extra
embedding executables beyond the shared 100-program budget; record any retained
ones in the owner's window notes. This harness cannot inventory another process.

```sh
unset TMPDIR
python3 "$KIT/run.py" measure "$EVIDENCE" --owner-paused-neural-engine
```

The driver verifies sealed inputs again, runs the **three exact first-load
reference clips separately** (and stops if they fail), then the unchanged 1200
WAV cases in ascending valid-frame order. Synapse is run counter-disabled then
`ANE_PARAKEET_HOST_TIMING=1`; FluidAudio follows in the same order, counter-disabled
then enabled. Both use one unmeasured same-clip warmup before each primary call.
Only primary warm calls enter latency statistics; no first-load clips, warmups,
precomputed-tensor fixtures (sealed mel/encoder arrays instead of corpus WAV inputs)
or observation repeats are extra corpus clips. No per-TDT
step logging is added. Every measured clip is serialized only after completion.
Warmup calls are journalled with a distinct invocation/population ID and excluded;
Fluid warmup model stages are unmeasured (only their original request wall is retained).

Synapse's run limit remains **81,850 ms**, not the 600-second certification limit.
Corpus structural/accuracy checks and all ten per-split WER gates are unchanged.
The stage-only mode excludes the synthetic long-duration ladder fixtures used to
test padding/capacity limits, because they are not among the 1200 corpus WAVs; it
cannot issue a certified maximum. A collect-all corpus accuracy failure is retained with its nonzero exit
code and does not erase measured timings. A runtime failure leaves incomplete
records: `compare.py` refuses such a population. No command retries hardware.
The stdout/stderr, full accuracy receipts, case journals, per-shape admission
inventory/cache hits, and all exit codes remain under `$EVIDENCE`.

## After the window: accuracy and the single comparison table

The owner may resume other Neural Engine users before these CPU-only commands:

```sh
python3 "$KIT/run.py" score "$EVIDENCE"
python3 "$KIT/compare.py" --request "$EVIDENCE/request.json" \
  --output "$EVIDENCE/compare.json" --tolerance-ms 1.0 \
  "$EVIDENCE/synapse-off.stages.jsonl" "$EVIDENCE/synapse-on.stages.jsonl" \
  "$EVIDENCE/fluid-off.stages.jsonl" "$EVIDENCE/fluid-on.stages.jsonl" \
  > "$EVIDENCE/compare.md"
```

The scorer is the existing `stt-common`/`synapse-parity` normalization and micro-WER
implementation, not an alternate Python normalization. It uses scratch files
under `target/stt-bias` because the existing bridge requires that boundary, then
copies the complete per-split scores into retained evidence. Fluid benchmark WER
is reported, **not certified**. Both WER against human ground truth and WER against
the sealed NVIDIA fp32 reference transcripts stay visible; the latter measures
text disagreement, not the NVIDIA reference's own error against ground truth.
Read the Synapse receipt's WER/gate failures alongside these scores.

### Reading the table and its limits

* Join keys are run ID, engine, instrumentation mode, sealed clip ID, primary
  invocation ID and segment ID. Duplicate/missing primaries fail closed.
  Individual segments are summed **before** per-clip median, nearest-rank p95 and
  arithmetic mean. Fluid window/recovery invocations are task-locally accumulated
  into one pre-aggregated clip record with the actual segment count. Overlapping
  windows may execute concurrently: segment-summed model work is **not** request
  latency. The original Fluid WAV→text request timer is a separate table row.
* Synapse's `model_total_ms` is exactly the surrounding `inference_seconds`
  boundary. The test-only encoder timer encloses the synchronous 24-program loop,
  including dispatch/wait and progress callbacks: **not pure ANE silicon time**.
  Existing predictor/joint/readback counters are reused. Greedy bookkeeping is
  the inclusive greedy duration minus predictor and joint, never counted twice.
  Remaining model work measures validation, input packing and decoder allocation.
  Serialization, autorelease cleanup and timer gaps remain in the signed residual.
* Fluid model work starts after preprocessor prediction, before encoder-input
  preparation, and ends at the raw TDT hypothesis, before text formatting. Its
  encoder/predictor/joint timers enclose the actual Core ML prediction calls,
  including runtime dispatch/readback costs. `other_unsplit_ms` is the derived
  remainder, **not** an independently measured readback/bookkeeping split. Its
  counts include recovery and overlapping-window work; they need not match ours.
* Synapse observation mode is ON: the signed harness's `inference_seconds`
  includes unprojected encoder readback and decoder trace capture. Fluid
  observation mode is OFF. The extra Synapse encoder
  readback/trace cost remains visible, not normalized away. Synapse frontend,
  IPC/queue and output formatting are **unmeasured**, not zero.
  Fluid frontend, window merging and output formatting are not separately timed;
  only its original inclusive request wall is retained outside the model boundary.
* Loaded `MLModel.configuration.computeUnits` for preprocessor, encoder (and any
  ladder functions), predictor and joint is recorded, as are the configurations
  seen by the timed calls. `cpuAndGPU` or `all` means GPU is permitted and must be
  labelled a different resource policy; no value is normalized to ANE. Allowed
  units alone never prove the actual device placement. The default kit requests
  `cpuAndNeuralEngine` for encoder/predictor/joint and upstream forces CPU frontend.
* The report prints instrumented vs counter-disabled warm **model totals** and
  their percent difference for each engine. Fluid's baseline retains the two
  segment-boundary clocks needed to measure its model total; per-model clocks and
  counters are off. Sequential A/B passes also include thermal/scheduling drift;
  the observed difference is an overhead disclosure, not a pure causal estimate.
  No full-run totals including warmups/loading are substituted for warm inference.
* Synapse checks `model_total ≈ encoder + readback + predictor + joint +
  bookkeeping + remaining_model` on each invocation and reports the residual
  distribution. The default absolute tolerance is **1 ms per segment**, a starting
  timer budget, not a hardware-calibrated guarantee. Inspect the actual residuals
  and disclose any tolerance adjustment; do not enlarge it to hide missing work.
  Fluid's unsplit remainder is not claimed as an independent residual proof.
* Mean stage differences are additive for Synapse up to its residual. Component
  medians are not additive. The output includes the actual components of the
  matched clip nearest each engine's total median; never subtract unrelated
  medians to infer an encoder time. Report the measured gap without assuming a
  particular value in advance.

## CPU-only kit checks (no hardware execution)

```sh
python3 -m unittest discover -s bench/lanes/stt-stage-kit -p 'test_*.py' -v
python3 -m unittest discover -s bench/lanes/stt-fluidaudio -p 'test_*.py' -v
```

The worker's macOS filtered Rust tests cover record identity/absence and the
existing CPU timing/parity switch. They never admit or execute an ANE program.
