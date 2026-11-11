# Vulkan transcription hardware kit

Run from this checkout on narli (native Windows, NVIDIA RTX 3080), a rented
NVIDIA Linux machine, and an Apple GPU through the explicit MoltenVK diagnostic.
Compilation and driverless tests are not certification. There is no macOS Vulkan
production lane: its LOAD/probe still refuse Apple adapters.

Use Rust >=1.89 and Vulkan SDK **1.4.357.0**. Serialize Cargo/native builds. On
Windows set `VULKAN_SDK=C:\Users\ufuka\synapse-cert\VulkanSDK\1.4.357.0`.
On Linux the `with-linux-sdk.sh` wrapper below fetches the official LunarG SDK,
checks SHA-256 `0f09bf6a0625e346bf004be70b92907e934a4c76606b323441b2baf3a5a0e66d`,
and selects its `1.4.357.0/x86_64` directory. An existing explicit `VULKAN_SDK`
is used unchanged; the build checks its compiler version.

## Inputs

1. Fetch the pinned Hugging Face snapshot (with `huggingface_hub` installed in
   the reference Python environment):

   ```sh
   python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/parakeet-tdt-0.6b-v3', revision='541d1f99c6b0c3cd0b11a95167540bb8edefd82b', cache_dir='target/scratch/hf-hub', allow_patterns=['*.json','model.safetensors'])"
   cargo run --locked --manifest-path bench/parity/Cargo.toml --bin parity-manifest -- convert --profile parakeet-tdt-0.6b-v3.owned-vulkan --hf-cache target/scratch/hf-hub --out target/scratch/parakeet-vulkan.safetensors
   ```

   The unchanged package SHA-256 is
   `0ae0837a03fca46e34f6900dcbb9cb2d73f36dc2ed911a27a595937587bf28b2`.
   Production LOAD verifies the package SHA, profile metadata, shapes, F16
   floating tensors and lossless I64 BatchNorm counters before uploading.

2. Place the 26 external tensors under
   `bench/parity/.cache-out/parakeet-tdt-0.6b-v3/`, each named by its committed
   SHA-256 and extension. The mirror variable points to `bench/parity/.cache-out`,
   not the model subdirectory. Every artifact is checked against its committed
   SHA-256 and byte length before use.

3. Use the pinned reference Python environment from
   `bench/parity/reference/requirements-parakeet.txt` and prepare the corpus:

   ```sh
   python crates/synapse-worker-vulkan/scripts/prepare-transcribe-audio.py --cache target/scratch/parakeet-audio-cache --out target/scratch/parakeet-wavs
   ```

   The helper checks archive/member hashes and runs the reference audio recipes,
   without loading the neural model or replacing reference outputs. Tensor
   cases consume sealed mel inputs. Other cases use the shared speech frontend
   over these 16 kHz WAVs; the test checks each WAV against `inputs.json` first.
   Already prepared WAVs and their index may be transferred together unchanged.

## Hardware run

Set absolute paths (`$env:NAME='...'` in PowerShell; `export NAME=...` in bash):

```text
SYNAPSE_VULKAN_SPEECH_PACKAGE=<checkout>/target/scratch/parakeet-vulkan.safetensors
SYNAPSE_VULKAN_SPEECH_MIRROR=<checkout>/bench/parity/.cache-out
SYNAPSE_VULKAN_SPEECH_AUDIO=<checkout>/target/scratch/parakeet-wavs
SYNAPSE_VULKAN_SPEECH_EVIDENCE=<checkout>/.cortexkit/internal-docs/investigations/vulkan-speech-<machine>
```

NVIDIA Windows (native Rust/MSVC):

```sh
cargo test --locked --release -p synapse-worker-vulkan --features vulkan --lib vulkan_transcribe_certifies_tensor_trace_wer_and_observation_invariance -- --ignored --nocapture
```

NVIDIA Linux:

```sh
bash crates/synapse-worker-vulkan/scripts/with-linux-sdk.sh cargo test --locked --release -p synapse-worker-vulkan --features vulkan --lib vulkan_transcribe_certifies_tensor_trace_wer_and_observation_invariance -- --ignored --nocapture
```

MoltenVK correctness, with the macOS SDK installed and `VULKAN_SDK` set:

```sh
export VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
export SYNAPSE_MOLTENVK_LOADER="$VULKAN_SDK/lib/libvulkan.1.dylib"
cargo test --locked --release -p synapse-worker-vulkan --features moltenvk-diagnostic --lib vulkan_transcribe_certifies_tensor_trace_wer_and_observation_invariance -- --ignored --nocapture
```

The diagnostic uses a test-only Apple device constructor and the pinned package;
NVIDIA runs use the production LOAD. All runs use the production TRANSCRIBE
handler, encoder, GPU encoder projection, shared greedy loop and host joint.
The test parses the real response frames in order, checks record lengths,
revision binding, the encoder/trace/per-split word-error-rate gates, and bit-identical
triples and valid frames with observation on/off for every tensor fixture.
Every stored encoder frame must have cosine similarity >=0.999. A trace
divergence is exempt only for the reference profile's small argmax margins at
the same frame; a shifted frame or trace-length difference always fails.
Each corpus split's word-error rate may exceed the reference by at most 0.5
absolute percentage points, using the shared evaluator's pinned normalizer.
Encoder/trace checks run after each utterance and preserve every fixture's
finding, including failures, in `vulkan-fixture-findings.jsonl`. Each trace
finding reports the first differing decision (the decision used by the grading
rule), the affected head, reference margin, and both reference/engine top-2
pairs and logits. Later divergence after that first decision is not counted as
another independent margin failure. Word-error rates are aggregated over entire
splits. The final JSON is written even if numerical grading fails, and the test
still fails unless the shared evaluator accepts the run.
Completed work produces exactly 24 encoder progress units plus one per 125
decoded frames. Progress gaps must remain below 120 s; no timer heartbeats exist.

The single arena inventory at the target maximum of 600000 ms is
**2,021,900,544 device-local bytes**, with **134,217,728 bytes** as the largest
bound storage range. The test records the driver's actual arena allocation and
requires it to be at most **2,021,900,544 bytes**. These are allocation requirements,
not a claim that hardware certification has passed. Preserve `vulkaninfo`,
`nvidia-smi` where available, OS, SDK version, command, commit and the test log
alongside the JSON summary; use an otherwise idle GPU for external peak-memory
measurements. Transfer buffers may also consume host-visible memory.

Duration-ladder fixtures concatenate and trim fixed corpus clips to probe the
encoder-frame rung boundaries and the 600-second target; other fixtures test
the corpus, silence, resampling and hop boundaries.
The summary `vulkan-transcribe-summary.json` records graded/refused counts,
split WER, margin exemptions, observation comparisons, longest progress gap,
arena bytes, maximum duration, all encoder/trace/WER failures, and the largest
**actually passed** ladder. `largest_certified_ladder_ms` is null unless the
complete run passes; `largest_passed_ladder_ms` records individual ladder results.
Do not describe the target maximum as certified until this complete run passes.
`SYNAPSE_VULKAN_SPEECH_FILTER=<fixture-id-prefix>` is available for a quick
numerical investigation; such summaries have `complete: false` and do not
certify a lane. A smaller production maximum requires updating the profile and
its inventory floors, still covering every non-ladder fixture, then rerunning.

For a targeted numerical investigation, optionally set
`SYNAPSE_VULKAN_SPEECH_DEBUG_ENCODER` to a scratch fp32 encoder file whose basename
starts with its full SHA-256 and whose frame count matches the filtered fixture.
It is hash-checked and used only for diagnostics, never to replace the sealed
trace or change a gate. On the first failing trace, a filtered run (or one with
this debug tensor) repeats observed/plain inference and writes
`vulkan-trace-failure.json`, the engine encoder and its projected output. The JSON
includes both heads' top-2 logits/margins, repeated encoder/triple equality,
reference encoder max-abs at the differing frame, GPU projection error against
CPU f64 dot products using the package's fp16 weights, and joint replay with
those diagnostic projections. A regenerated reference may have small
cross-architecture differences; record that provenance separately. These CPU
calculations exist only in the test harness and do not alter production inference.

## Separating weight storage from GPU arithmetic

The speech shader widens fp16 weights at each load and uses fp32 activations,
shared tiles, dot products, LayerNorm reductions, attention scores, softmax and
score-times-value accumulation. The arena upload itself retains fp16 bytes.
Do not infer half-precision accumulation from an encoder mismatch alone.

In the pinned reference Python environment, generate diagnostic CPU controls:

```sh
HF_HUB_OFFLINE=1 "$REFERENCE_PYTHON" \
  crates/synapse-worker-vulkan/scripts/diagnose-transcribe-storage.py \
  --hf-cache "$HF_CACHE" --mirror bench/parity/.cache-out \
  --audio target/scratch/parakeet-wavs \
  --vulkan-summary "$PREVIOUS_EVIDENCE/vulkan-transcribe-summary.json" \
  --out target/scratch/storage-reference --mixed
```

The script reads the immutable sealed mel/encoder/trace artifacts and verifies
all hashes; corpus cases without stored mel use hash-checked prepared WAVs.
It first captures fp32 CPU outputs for `ladder-1025`, `ladder-4096`,
`silence-30s` and the shortest stored-tensor LibriSpeech case. It then rounds
only persistent floating checkpoint tensors through fp16, leaving computed
position frequencies and all arithmetic fp32. It captures final encoders for
all in-envelope tensor fixtures and every trace-failed fixture listed in the
`--vulkan-summary` report. Every selected encoder is replayed through the
fp16-storage joint; the report's trace-failed cases are also replayed through
the original fp32 joint. Full-shadow traces are saved for comparison, not made
into new seals.

With `--mixed`, three additional diagnostic modes keep original fp32 tensors
for layers 21–23, all attention projection weights, or all feed-forward weights.
They run on the union of `trace_failures` and `encoder_failures` in the
`--vulkan-summary` report, not the complete corpus; their results cannot certify
a new profile. `--reuse-captures` reuses
existing SHA-checked encoder files when only additional replays are needed.
CPU per-layer/final metrics and trace findings are in `storage-summary.json`.

Copy the `fp32/` and `f16-storage/` children of `storage-reference` to the GPU host
(selective-mode directories are not needed), then set
`SYNAPSE_VULKAN_SPEECH_LAYER_REFERENCE` to its root before the normal opt-in test.
For each matching fixture the test compares subsampling and layer outputs
against available fp32/shadow captures. It writes `<fixture>-layer-errors.jsonl`
with max-abs error, cosine quantiles, every final encoder frame's cosine and
norms, and failing frame indices measured both from the start and valid end.
`silence-30s` includes per-frame norms at every layer. The final summary adds
whole-trace agreement with the CPU storage-shadow pipeline: frame index, token
argmax and duration argmax at every evaluation, plus each emitted token's
`[token_id, frame_index, duration]` triple.
This instrumented run includes extra GPU downloads and CPU comparisons; report
it as a diagnostic, not a clean performance measurement. No reference, numerical
gate, kernel revision or production computation changes when captures are enabled.

Keep logs/JSON outside `target/` and attach all three hardware results to the
change review before landing. Do not relax tolerances or reference seals.
