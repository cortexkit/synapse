# CUDA transcription hardware kit

Run from this checkout on both a real NVIDIA Windows machine and a real NVIDIA
Linux machine. Compilation and a driverless test run are not certification.
Use Rust >=1.89, the CUDA 13.2 toolkit/runtimes and the native host C++ compiler
(MSVC on Windows). Keep these commands serialized with other native Cargo builds.

## Inputs

1. Fetch `nvidia/parakeet-tdt-0.6b-v3` at
   `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` into the standard Hugging Face hub
   cache, verifying the model files against `bench/parity/checkpoints/`.
2. Produce the pinned all-F32 CUDA package with the existing v1 converter:

   ```sh
    cargo run --locked --release --manifest-path bench/parity/Cargo.toml --bin parity-manifest -- convert --profile parakeet-tdt-0.6b-v3.owned-cuda --hf-cache <hub-cache> --out <package.safetensors>
   ```

   The expected package SHA-256 is
    `1f89f0a76ef7f1bc738eaaac4befd6bc2dd94cf5c3bad2ae2b98885657ecddec`
    (2,508,311,520 bytes). An already converted package with that digest can be
    reused. There is no F16 loader or measurement-only digest override.
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

Before the full-corpus run, check and test the native CUDA build. Clippy's two
allowed lints cover existing token-model code, not the transcription path:

```sh
cargo check --locked -p synapse-engine-cuda -p synapse-worker-cuda --features cuda --all-targets
cargo clippy --locked -p synapse-engine-cuda -p synapse-worker-cuda --features cuda --all-targets -- -D warnings -A clippy::too_many_arguments -A clippy::arc_with_non_send_sync
cargo test --locked -p synapse-engine-cuda --features cuda transcribe
cargo test --locked -p synapse-worker-cuda --features cuda transcribe
```

Then execute the full corpus with the F32 package and the default 600,000 ms
maximum (remove any previous `SYNAPSE_CUDA_SPEECH_MAX_MS` override):

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

## Numeric profile and timing

Both reports include `measurements`: F32 precision, actual
package digest/size, pedantic math/no-TF32 declaration, model load time, total
fixture time and `fixture_groups`. Groups separate `ladders` (fixture IDs starting
with `ladder-`, including rung-boundary and maximum-duration cases) from
`short_clips` (all other fixtures, including silence, resample and hop cases). Each group
records fixture/refusal counts, wall time including frontend/fixture reads and
the repeated plain checks, and `observed_request_s` for one observed request per
fixture, including expected refusals but excluding frontend and repeat work.
Use the latter for the closest engine-side comparison; neither timer is a
GPU-event-only measurement. SHA-256 verification and grading are outside group
times. `peak_device_used_bytes` is the largest device-wide used memory
(`cudaMemGetInfo` total minus free) sampled after each weight upload and at
encoder layer boundaries. It includes other processes on the GPU, so run the
kit on an otherwise idle device.

The CUDA context keeps only float weight views and consumes float operands for
linear, query/key score, relative-position and probability/value GEMMs, with explicit
`CUBLAS_COMPUTE_32F_PEDANTIC`, pedantic math and atomics disabled. It does not
round activations or weights through half and does not enable TF32. The package
storage and declared compute dtype are F32. Every frozen gate and the
repeatability/observation checks still run unchanged; compilation alone does not
certify the lane.

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
