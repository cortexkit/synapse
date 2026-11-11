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
0.5 percentage-point word error rate increase in each corpus split. On every tensor
fixture within the selected duration and encoder-frame limits, it also requires bit-identical
`(token_id, frame_index, duration)` triples and valid encoder-frame counts with the
request's certification-only `observe` flag on and off. It checks exact
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
