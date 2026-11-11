# synapse-speech

Portable processing for the pinned `nvidia/parakeet-tdt-0.6b-v3` model. The crate
has no engine, module, store or daemon dependency and can be checked for
`aarch64-apple-ios` without an Apple SDK or native tokenizer library.

`assets/tokenizer.json` is an unmodified NVIDIA Parakeet tokenizer from
[`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`, licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Its SHA-256 is
`bd321b096832a3f270bd3b2a88823957920f1a5c5ada71114a26ea729d0cbe91`.

- `decode_wav` validates the entire RIFF container before returning finite mono
  PCM. It accepts PCM16, float32 and their WAVE_FORMAT_EXTENSIBLE subformats,
  8–192 kHz, 1–8 channels, and at most 256 MiB. PCM16 uses `/32768`; floats are
  not clipped; downmix is an arithmetic mean.
- `resample` implements the reference's fp64 Hann-windowed sinc with six zero
  crossings, 0.99 lower-Nyquist cutoff, ceiling sample count and zero edges.
- `Frontend::process_wav` validates, resamples and computes the frontend's own
  row-major `[mel_frames, 128]` tensor, without padding to an engine's
  preallocated input shape. Count functions
  take the number of **resampled 16-kHz** mono samples. The centered extra mel
  frame is zero after normalization. Very short recordings with no valid mel
  frames return a zero tensor; a single valid frame uses zero variance rather
  than dividing by zero. Normalization retains the pinned fp32 cascade-sum
  rounding, even for digital silence, whose reference features are not zero.
- `greedy` drives an engine's `Joint`. `reset` initializes a zero-state blank
  predictor; `evaluate` supplies both finite logit heads; `accept` advances the
  predictor only for emitted tokens. The engine retains all neural state and
  projected encoder output. Observation is passive, and progress is reported
  only after each 125 completed encoder frames.
- `Tokenizer::pinned` uses the bundled, SHA-256-checked tokenizer vocabulary and
  pinned Parakeet decoding behavior: adjacent identical ids collapse before pad
  ids are removed and Metaspace is decoded. Other special ids are not skipped.
  Word groups use the original token pieces, with fractional-millisecond timing
  clamped to the supplied audio duration.

`SPEECH_FRONTEND_REVISION` covers WAV decode, scaling, downmix, resampling and
features. `DECODE_RULE_REVISION` covers greedy policy, detokenization and timing.
The per-frame cap is the reference's bounded adaptation, not the unbounded
Transformers TDT rule. It forces progress only after the tenth zero-advance
evaluation; a positive duration on that evaluation does not get an extra frame.

## Verification

Ordinary offline tests validate rejection of unsupported or malformed WAVs and
non-finite samples, observation, progress, frame counts, words and `no_speech`,
and constants against the pinned processor, generation and tokenizer configs.
Two stub joints test the per-frame evaluation cap: one always emits duration 0;
the other emits duration 1 on the tenth evaluation. All 1,247 reference traces
are also replayed. Reference artifacts are sealed by SHA-256 and byte length in
`bench/parity/fixtures/index.json`; tests verify those seals before use:

```sh
cargo test --locked -p synapse-speech --profile ci-test --test core --test reference
cargo clippy --locked -p synapse-speech --all-targets -- -D warnings
cargo check --locked -p synapse-speech --target aarch64-apple-ios
```

Two explicit cache-dependent tests additionally cover all 79 stored encoder
outputs and all 79 frontend tensors (including 16 resampled WAVs). Missing or
corrupt artifacts fail; the tests never silently skip individual fixtures:

```sh
SYNAPSE_SPEECH_AUDIO=/absolute/path/to/prepared-wavs \
SYNAPSE_SPEECH_MIRROR=/absolute/path/to/ladder-mirror \
cargo test --locked -p synapse-speech --profile ci-test --test reference -- --ignored --nocapture
```

The mirror is the `bench/parity/.cache-out` handoff described in the reference
README. Prepared WAVs are named `<fixture-id>.wav`, generated from each tensor
fixture's `input` recipe by `bench/parity/reference/generate_parakeet.py` using
verified corpus source members. Non-resampled cases contain the reference's
16-kHz PCM as float32 WAV; resampled cases retain their sealed high-rate stereo
PCM16/float32 WAV bytes, so the production resampler is exercised. The tests
check resampled features against **their own** sealed log-mel at `tau_feat`;
`tau_resample` is only the separate native-16-kHz sanity comparison. These are
the profile's maximum absolute feature differences: `tau_feat = 0.001` compares
the implementation with its reference tensor, while `tau_resample = 1.546491`
bounds the measured change from a native recording to its resampled version.

Replay supplies the sealed top-two decision heads to the real generic loop and
checks exact traces, including row indices into the stored encoder output. It
certifies loop policy, not neural predictor arithmetic or hardware kernels;
those belong to the engine implementations and hardware certification.
