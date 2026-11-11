# CUDA transcription hardware kit

Run from this checkout on both a real NVIDIA Windows machine and a real NVIDIA
Linux machine. Compilation and a driverless test run are not certification.
Use Rust >=1.89, the CUDA 13.2 toolkit/runtimes and the native host C++ compiler
(MSVC on Windows). Keep these commands serialized with other native Cargo builds.

## Inputs

1. Fetch `nvidia/parakeet-tdt-0.6b-v3` at
   `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` into the standard Hugging Face hub
   cache, verifying the model files against `bench/parity/checkpoints/`.
2. Produce the unchanged, pinned CUDA package with the existing converter:

   ```sh
   cargo run --locked --manifest-path bench/parity/Cargo.toml --bin parity-manifest -- convert --profile parakeet-tdt-0.6b-v3.owned-cuda --hf-cache <hub-cache> --out <package.safetensors>
   ```

   The expected package SHA-256 is
   `55144098605bbd64592686c407a31b09f16bca33b02c3e3b04b46bbe705287a3`.
   LOAD independently verifies the digest, profile metadata, tensor shapes and dtype.
3. Place the 26 external tensors under
   `bench/parity/.cache-out/parakeet-tdt-0.6b-v3/`, each named by its SHA-256 and
   extension as in the committed fixture seals. Set the mirror variable below to
   `bench/parity/.cache-out` (the parent of that model directory). Every consumed
   artifact is checked against its committed SHA-256 and byte length by the evaluator.
4. In the pinned reference Python environment (requirements are in
   `bench/parity/reference/requirements-parakeet.txt`), prepare the corpus WAVs:

   ```sh
   python crates/synapse-worker-cuda/scripts/prepare-transcribe-audio.py --cache target/scratch/parakeet-audio-cache --out target/scratch/cuda-speech-wavs
   ```

   The helper fetches the committed corpus URLs, checks archive and member SHA-256,
   and runs the reference audio recipes. It never loads the neural reference model
   or regenerates any reference output. Tensor cases use their sealed mel inputs;
   other cases use the shared frontend over the prepared 16 kHz WAVs.

## Run

Set these environment variables using absolute paths (PowerShell `$env:NAME=...`
on Windows, `export NAME=...` on Linux):

```text
SYNAPSE_CUDA_SPEECH_PACKAGE=<package.safetensors>
SYNAPSE_CUDA_SPEECH_MIRROR=<checkout>/bench/parity/.cache-out
SYNAPSE_CUDA_SPEECH_AUDIO=<checkout>/target/scratch/cuda-speech-wavs
SYNAPSE_CUDA_SPEECH_EVIDENCE=<checkout>/.cortexkit/internal-docs/investigations/cuda-speech-<machine>
```

Preserve `nvidia-smi` and `nvcc --version` alongside the test log, including device,
driver, operating system, commit and the exact command. Arrange an otherwise idle
device: the reported peak is total device-used memory, including any other process.

```sh
cargo test --locked --release -p synapse-worker-cuda --features cuda --bin ck-synapse-worker-cuda cuda_transcribe_certifies_tensor_trace_wer_and_observation_invariance -- --ignored --nocapture
```

The single opt-in test runs the production LOAD and TRANSCRIBE handlers and parses
their real framed responses on either OS. It checks the load revision, response
order and record lengths. Against the sealed fp32 reference, it requires a minimum
per-frame encoder cosine of 0.999 and checks the greedy trace. A first differing
trace step is exempt only if its frame index matches and each differing argmax
has a reference top-1 minus top-2 margin below `tau_margin` from the
`parakeet-tdt-0.6b-v3.owned-cuda` profile. Trace-length differences fail, and emitted
triples must always agree with the engine's own trace. The test permits at most a
0.5 percentage-point word error rate (WER) increase in each corpus split. On every tensor
fixture within the selected duration and encoder-frame limits, it also requires bit-identical
`(token_id, frame_index, duration)` triples and valid encoder-frame counts with the
request's certification-only `observe` flag on and off. Each such fixture also runs
twice with `observe` off to detect run-to-run nondeterminism separately from an
observation mismatch. It checks exact
completed-work progress counts and <120 s progress gaps, printing completed work
rather than timer heartbeats. It records the selected maximum, largest certified
duration-ladder clip, peak device-used bytes and longest progress gap in
`cuda-transcribe-summary.json`.

Default maximum is the sealed 600000 ms target. If hardware cannot reach it, select
an exact sealed ladder duration using `SYNAPSE_CUDA_SPEECH_MAX_MS`; it must still
cover every non-ladder fixture. The ladder consists of deterministic concatenated
corpus clips whose sample counts test one frame below, at, and above the encoder
input-shape boundaries of 128, 256, 512, 1024, 2048, 4096 and 8192 frames, plus the
600-second target. Above-limit ladder clips must be refused before encoder work and
are recorded separately from graded fixtures. Do not call
a smaller run a 600-second certification, relax tolerances, or change reference
seals to pass.

Keep the two machines' logs and JSON summaries outside `target/`, and return them
to the change reviewer for inspection before landing.

The test writes `cuda-transcribe-grading.json` before the final pass/fail check.
It grades every fixture independently for diagnostics, retaining all failures
rather than stopping at the first bad trace. For each trace failure or exemption,
it records the fixture, first differing step, frame index, affected head, reference
top-two ids/logits and margin, and engine top-two ids/logits. It also records
per-split reference and engine word edits, total ground-truth words, WER and the
delta in percentage points, regardless of trace failures. Encoder statistics cover
all valid frames with a stored reference tensor: per-fixture and overall minimum,
mean, maximum, linearly interpolated percentiles and count below cosine 0.999.
Each encoder fixture also lists every frame below 0.999, with its zero-based
index, valid encoder length, distance from the last valid frame, cosine, reference
and engine L2 norms, and the engine/reference norm ratio. These rows distinguish
padding-boundary errors from discrepancies throughout a clip or ill-conditioned
cosines caused by near-zero encoder rows.

`cuda-transcribe-summary.json` is preserved even when final grading fails. Its
`grading_passed` and `grading_error` fields identify that outcome; a failed run
records `largest_executed_ladder_ms`, but leaves `largest_certified_ladder_ms` null.
The canonical parity evaluator still determines success using the unchanged
profile thresholds. A diagnostic report does not turn a failed gate into a pass.

## Measurement-only all-F32 comparison

The sealed CUDA profile, package digest and frozen encoder, trace and WER gates
are unchanged. The hardware tests alone accept `SYNAPSE_CUDA_SPEECH_PRECISION=f32`.
That switch calls a separate, explicitly unpinned engine load method: it verifies
the measured package's full SHA-256, original profile metadata, tensor shapes and
homogeneous F32 storage, but does not claim that its digest is production-approved.
Ordinary worker LOAD always uses the sealed F16 package and cannot select this
override through its runtime config or environment.

Create a scratch converter input, then run the existing v1 converter against the
same pinned HF cache used for the F16 package. `HF_CACHE` below is the directory
containing the pinned model's `models--nvidia--parakeet-tdt-0.6b-v3/snapshots/` tree.
The Python helper changes only `fp32_tensors` in an ignored manifest copy. The
copy retains the original approved F16 package digest as conversion input, not
as approval for the F32 output. The converter prints the new digest; the harness
records it and independently verifies the package bytes.

```sh
python3 crates/synapse-worker-cuda/scripts/prepare-f32-measurement.py
cargo run --locked --release --manifest-path bench/parity/Cargo.toml --no-default-features --bin parity-manifest -- convert --dir target/scratch/cuda-speech-f32 --profile parakeet-tdt-0.6b-v3.owned-cuda --hf-cache "$HF_CACHE" --out target/scratch/cuda-speech-f32/model.safetensors
```

Retain the existing mirror/audio/maximum/toolchain environment from the normal
kit. Set a different evidence directory for each variant so the second cannot
overwrite the first. Run the same tests back to back; the shell syntax below is
for Bash (including Git Bash on Windows):

```sh
export SYNAPSE_CUDA_SPEECH_PRECISION=f16
export SYNAPSE_CUDA_SPEECH_PACKAGE="$F16_PACKAGE"
export SYNAPSE_CUDA_SPEECH_EVIDENCE="$F16_EVIDENCE"
cargo test --locked --release -p synapse-worker-cuda --features cuda --bin ck-synapse-worker-cuda cuda_transcribe_ladder_1023_is_repeatable_and_observation_read_only -- --ignored --nocapture
cargo test --locked --release -p synapse-worker-cuda --features cuda --bin ck-synapse-worker-cuda cuda_transcribe_certifies_tensor_trace_wer_and_observation_invariance -- --ignored --nocapture

export SYNAPSE_CUDA_SPEECH_PRECISION=f32
export SYNAPSE_CUDA_SPEECH_PACKAGE="$PWD/target/scratch/cuda-speech-f32/model.safetensors"
export SYNAPSE_CUDA_SPEECH_EVIDENCE="$F32_EVIDENCE"
cargo test --locked --release -p synapse-worker-cuda --features cuda --bin ck-synapse-worker-cuda cuda_transcribe_ladder_1023_is_repeatable_and_observation_read_only -- --ignored --nocapture
cargo test --locked --release -p synapse-worker-cuda --features cuda --bin ck-synapse-worker-cuda cuda_transcribe_certifies_tensor_trace_wer_and_observation_invariance -- --ignored --nocapture
```

Both reports include `measurements`: precision, the measurement-only flag, actual
package digest/size, pedantic math/no-TF32 declaration, model load time, total
fixture time and `fixture_groups`. Groups separate `ladders` (fixture IDs starting
with `ladder-`, including rung-boundary and maximum-duration cases) from
`short_clips` (all other fixtures, including silence, resample and hop cases). Each group
records fixture/refusal counts, wall time including frontend/fixture reads and
the repeated plain checks, and `observed_request_s` for one observed request per
fixture, including expected refusals but excluding frontend and repeat work.
Use the latter for the closest engine-side comparison; neither timer is a
GPU-event-only measurement. SHA-256 verification and grading are outside group
times. Peak device usage has the same sampling semantics as the normal kit.

The F32 context keeps only float weight views and consumes float operands for
linear, query/key score, relative-position and probability/value GEMMs, with explicit
`CUBLAS_COMPUTE_32F_PEDANTIC`, pedantic math and atomics disabled. It does not
round activations or weights through half and does not enable TF32. The F16
context retains its existing half operands and arithmetic. Every frozen gate
and the repeatability/observation checks still run; a new variant is a measurement,
not a production profile approval or hardware certification until its gates pass.

## Narrow repeatability diagnostic

With the same package and tensor-mirror environment variables, the following test
runs only `ladder-1023`, the sealed concatenated-corpus clip with 1023 valid encoder
frames: two requests with `observe` off, followed by two with it on. It compares
emitted triples and valid-frame counts across the requests and checks that the two
copied encoder tensors and joint traces (including top-two logits) are bit-identical.
It needs no prepared WAVs or evidence output directory, and it is not a substitute
for full certification.

```sh
cargo test --locked --release -p synapse-worker-cuda --features cuda --bin ck-synapse-worker-cuda cuda_transcribe_ladder_1023_is_repeatable_and_observation_read_only -- --ignored --nocapture
```
