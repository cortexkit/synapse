# synapse-speech

Portable processing for the pinned `nvidia/parakeet-tdt-0.6b-v3` and
`Qwen/Qwen3-ASR-1.7B` models. The crate has no engine, module, store or daemon
dependency. Its default Parakeet-only build can be checked for
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

## Qwen3-ASR frontend and prompts

The `qwen` module has an independent frontend; Parakeet's feature computation
and constants are unchanged. Qwen uses 16-kHz PCM, a periodic 400-sample Hann
window, 200-sample reflection padding, 160-sample hop, 128 Slaney-normalized mel
bins, base-10 log floor `1e-10`, a waveform-wide eight-log-unit clamp, and
`(log_mel + 4) / 4`. The final short-time Fourier transform (STFT) frame is
dropped: both actual and valid frame counts are `floor(samples / 160)`.
Reflection padding, like the pinned torch
implementation, requires more than 200 input samples. Audio tokens are
`13 * (frames / 100) + ceil((frames % 100) / 8)`, not `ceil(frames / 8)`.
The [frozen frontend rules](../../bench/parity/reference/qwen3-asr-reference.json)
specify the mel scale and Slaney filter-area normalization.

Enable the **`qwen` Cargo feature** for `qwen::Tokenizer` and prompt rendering.
Without it, neither the encoder nor its tokenizer/regex dependencies are linked.
The unmodified Apache-2.0 `vocab.json` and `merges.txt` assets come from
`Qwen/Qwen3-ASR-1.7B` revision `7278e1e70fe206f11671096ffdd38061171dd6e5`;
`qwen::VOCAB_SHA256` and `qwen::MERGES_SHA256` enforce their original byte seals.
Encoding uses Unicode Normalization Form C (NFC) and byte-level byte-pair encoding
(BPE), with special-token parsing disabled for caller text. Non-special added
tokens, including `<asr_text>`, keep
their pinned ordinary-text ids; only the renderer can insert chat/audio framing.
The bundled, checksum-checked `pretokenizer.json` records the actual reference
processor's split: qwen-asr 0.0.6 loads the processor with
`fix_mistral_regex=True`, replacing plain Qwen2's regex with a case-aware split.
For example, it separates `YouTube` where plain Qwen2 does not. Template markers
are inserted as ids.

`Tokenizer::render(PromptRequest)` counts the full forwarded context, including
comma-space separators and the newline before a kept caller prompt. It drops
the prompt first, then trailing terms until the context fits `bias_limit`, and
returns the exact ids, placeholder span and `bias_applied` accounting. The limit
is supplied by the caller; this crate does not select or certify a serving bias
limit. `LANGUAGES`, `language_name` and `language_code` share all 30 pinned model
names between hint rendering and output parsing; the served catalog separately
narrows its declared languages to the Parakeet word-timing companion's set.

`Tokenizer::parse` accepts generated ids only and returns transcript text and a
contract language code, never raw model output. With no hint, a language line
and `<asr_text>` marker are required; `None` and unknown language names report no
language. With a valid hint, the continuation needs no marker and reports the
table's canonical base code, including for regional hints. Unsupported hints
map to `invalid_params` with field `language`; malformed output has a distinct
`InvalidAsrOutput` error.

`qwen::align_words` is available without the tokenizer feature. It splits Qwen
text on whitespace and uses `synapse-wer-norm-v1` only for edit-distance matching,
leaving spelling and punctuation unchanged. Matches and substitutions borrow
Parakeet times; unaligned runs divide the gap evenly, including zero-length gaps
between overlapping anchors. Empty-normalizing words never align. Bad companion
times are refused rather than changed. The serving caller must treat a companion
with no words as `no_speech` and skip Qwen; nonempty text without companion words
is an internal-invariant error.

```sh
cargo test --locked -p synapse-speech --features qwen --test qwen
cargo test --locked -p synapse-speech --no-default-features --test qwen
cargo clippy --locked -p synapse-speech --features qwen --all-targets -- -D warnings
```

The ordinary tests compare every executable sealed corpus case's prompt ids,
span, frame/token counts and detected language with the reference, plus two
small sealed mel tensors. The additional `qwen-prompts.json` goldens were generated
with the pinned template and qwen-asr 0.0.6's processor under Transformers 4.57.6,
including its regex patch and `split_special_tokens=True` for context only. They
cover caller prompts, literal markers, Unicode, hints, trimming, and the
case-sensitive split. The feature-off test inspects Cargo's
normal dependency graph to keep the Parakeet-only dependency boundary explicit.

For all 6,559 executable cases (1,556 distinct PCM tensors), reconstruct the
public corpus recipes into a mirror containing `<recording-key>.pcm` and
`<mel-sha256>.mel`, both row-major little-endian f32. The recording keys are from
`bench/parity/reference/qwen3-asr-corpus.json`, and the mel seals and exact input
PCM digests are in `bench/parity/fixtures/qwen3-asr-1.7b/traces/`. The single
over-duration refusal has no mel/prompt execution. Missing or corrupt mirror
files fail, never silently skip:

```sh
SYNAPSE_QWEN_PARITY_INPUTS=/absolute/path/to/public-pcm-mel-mirror \
cargo test --locked -p synapse-speech --test qwen qwen_all_mel_reference -- --ignored --nocapture --test-threads=1
```

The all-mel test checks every fixture's exact counts and seals; identical PCM
must name the same mel seal and is computed only once. This is frontend and
prompt parity, not encoder, decoding, name-recall or hardware certification.
