# In-process Metal transcription

`transcribe::OwnedMetalTranscribeEngine` implements the core `TranscribeEngine`
trait for the pinned Parakeet TDT 0.6B v3 checkpoint. It does not reuse or change
the embedding engine's graph revision, identity, buckets or model state.

Load a `safetensors-package` artifact with these `RuntimeConfig` values:

| Key | Value |
| --- | --- |
| `model_path` | Hugging Face package directory containing `config.json` and `model.safetensors` |
| `max_encoder_frames` | Required positive certified lane maximum, at most 7500 |
| `family` | Optional; if present, `parakeet-tdt` |
| `dtype` | Optional; if present, `f32` |
| `execution` | Optional; if present, `explicit` |

Input is normalized, un-rung-padded f32 mel features, 128 bins per frame, with
the shared front end's invalid tail zeroed. The engine admits the valid encoder
count before GPU allocation and selects the smallest of 128, 256, 512, 1024,
2048, 4096 and 8192 that contains it. The module still owns audio-duration
admission, audio processing and detokenization.

Subsampling, all 24 encoder layers with full relative-position attention, and
the encoder's 1024-to-640 joint projection execute as explicit MPSGraph GPU
executables. One reusable layer graph binds each layer's own weights. Only one
rung's plans stay resident; weights remain resident across requests. Masks match
transformers 5.16.1 with `synapse-parakeet-mask-fix-v1`: after every subsampling
convolution, before the encoder's depthwise convolution, and on both query and
key attention rows. Fully masked attention probabilities are zeroed after the
finite-sentinel softmax.

The sequential two-layer LSTM predictor and joint head run on the host using
Accelerate BLAS, driven by `synapse_speech::greedy`. Only valid projected rows
are retained for the joint. `transcribe_with_progress` connects completed-work
events to the caller's watchdog: one per encoder layer and one per 125 decoded
encoder frames, with no timer heartbeats. `TranscribeEngine::transcribe` uses
the same computation with a no-op progress sink. When `TranscribeRequest.observe`
is enabled, the output retains encoder tensors and decode traces for certification.
It copies the same results and never selects a different graph or decoder path.

## Certified envelope

`METAL_TRANSCRIBE_KERNEL_REVISION = owned-metal-transcribe-v1` was graded with
the fp32 reference pinned by SHA256 in
[`bench/parity/fixtures/index.json`](../../bench/parity/fixtures/index.json)
through **600,000 ms**, **7500 valid encoder frames**,
padded to **8192**, with full attention. These are exposed as
`CERTIFIED_MAX_DURATION_MS` and `CERTIFIED_MAX_ENCODER_FRAMES` for catalog
integration; they are not token limits. Larger ladder fixtures are refused
before GPU work. Certification is evidence for this numerical envelope, not a
latency promise under GPU contention on every Mac.

The opt-in integration test grades every fixture within that envelope using
the parity crate's encoder, trace and per-split word error rate (WER) gates. It verifies every
artifact hash, all prepared WAV hashes, exact progress event sequences and
observe on/off equality for every in-envelope tensor fixture. The two stub
joint unit tests exercise the same engine decode adapter and the shared loop
on macOS without requiring a model or GPU work.

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
CARGO_TARGET_DIR="$PWD/target" \
SYNAPSE_METAL_SPEECH_GPU=1 \
SYNAPSE_METAL_SPEECH_MODEL=/path/to/pinned/hf/package \
SYNAPSE_METAL_SPEECH_AUDIO=/path/to/prepared/wavs \
SYNAPSE_METAL_SPEECH_EVIDENCE=/path/to/c6.jsonl \
cargo test --locked --release -p synapse-engine-owned \
  --test transcribe_metal transcribe_metal_c6_parity -- --ignored --nocapture
```

The audio directory contains `inputs.json`, mapping fixture ids to relative
`path` and `sha256`, plus WAVs for cases without stored mel tensors. External
tensor files are identified by the index's size and SHA256 records and resolve
below `bench/parity/.cache-out/`. Set
`SYNAPSE_METAL_SPEECH_CASE` to an id substring for a short diagnostic run. The
test's diagnostic monitor prints at least every 30 seconds and stops at a
110-second completed-work gap or 48 GiB resident memory. Those diagnostic lines
are not engine progress events. Per-fixture JSONL and the final summary record
timing, cosine, WER, memory and the certified ladder maximum. The trace-exemption
field records whether the first differing head decision qualified for the
reference's low-logit-margin exception; it does not exempt the encoder gate.
