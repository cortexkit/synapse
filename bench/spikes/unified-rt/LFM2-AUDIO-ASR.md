# LFM2-Audio ASR on the owned runtime

## Scope and pins

This wave adds whole-utterance speech input to the existing owned LFM2 decode
path. It implements the DSP frontend, noncausal FastConformer, audio projector,
modality splice, and greedy text decode. It does **not** implement streaming,
the depthformer, Mimi, audio token embeddings, or audio output.

The validation inputs are:

- [`LiquidAI/LFM2-Audio-1.5B@c798aad30dc3cd72e72970beab51326b8443bd94`](https://huggingface.co/LiquidAI/LFM2-Audio-1.5B/tree/c798aad30dc3cd72e72970beab51326b8443bd94);
- `model.safetensors` LFS SHA-256
  `d0cae5b6a1cbc308472535d6fa310fe446bb9ea601934a14db2366040a9fa129`
  (2,940,723,992 bytes);
- [`Liquid4All/liquid-audio@84cdb243859aaa53db660bc3f4718b54133336bd`](https://github.com/Liquid4All/liquid-audio/tree/84cdb243859aaa53db660bc3f4718b54133336bd),
  package version 1.2.0.

The oracle venv used Python 3.12.12, PyTorch 2.13.0, torchaudio 2.11.0,
Transformers 5.13.1, Accelerate 1.14.0, librosa 0.11.0, NumPy 2.4.6, and
soundfile 0.14.0. `lfm2-audio-reference-requirements.txt` records exact package
pins. `reference_lfm2_audio.py` verifies the source revision, checkpoint
snapshot directory, and (unless explicitly skipped) the weight hash.

The audio checkpoint's nested `lfm` config and all backbone tensors were
compared with the text-only LFM2-1.2B layout. Hidden size, 16-layer
`C,C,A,C,C,A,C,C,A,C,A,C,A,C,A,C` layout, head shapes, effective 8192 MLP
width, convolution shapes, tied embedding logits, and norm/RoPE settings all
match. The serialization deltas are only the nested config, the `lfm.` tensor
prefix, and the absence of a separate generation config. The existing loader
now accepts those two layouts while detection routes nested audio checkpoints
to `lfm2-audio` and leaves plain `lfm2` text-only.

## Owned path

### DSP frontend

`lfm2_audio.rs` reads mono 16-bit, wider integer, or float WAV at exactly 16
kHz and computes:

1. pre-emphasis `y[0] = x[0]`, `y[n] = x[n] - 0.97*x[n-1]`;
2. centered STFT with 512-point FFT, 400-sample **non-periodic** Hann window,
   160-sample hop, and constant-zero padding;
3. one-sided power spectrum and a 128-bin Slaney-normalized mel filterbank from
   0 through 8 kHz;
4. `log(mel + 2^-24)`;
5. per-feature sample standard deviation normalization over the valid frames,
   adding `1e-5` to the standard deviation, then zeroing the extra centered-STFT
   frame.

Dither remains configured as `1e-5` for checkpoint validation but is never
applied in inference. RustFFT is the only DSP dependency; no learned frontend
or Python is used by the owned path.

### FastConformer and projector

The learned audio tower loads running batch-normalization statistics and owns:

- a 256-channel 2D convolution followed by two depthwise-stride-2 plus
  pointwise stages, for total time/frequency factor 8, then a 4096-to-512
  linear projection;
- 17 noncausal Conformer layers at width 512, with two 2048-wide SiLU FFNs,
  eight 64-wide heads, Transformer-XL relative-position attention, and a
  kernel-9 depthwise convolution module;
- LayerNorm(512), Linear(512, 2048), exact GELU, and Linear(2048, 2048).

Relative positions are `[L-1, ..., 0, ..., -(L-1)]`. The attention computes
`(Q+u)K^T + rel_shift((Q+v)P^T)`, scales by `1/sqrt(64)`, and attends over the
entire utterance. FFN residuals use the Conformer factor 0.5. Batch norm is
strictly inference-mode and uses serialized `running_mean` and `running_var`.

All dense products use the selected owned `KernelProvider`. On Metal fp32 this
routes the pre-encoder pointwise products, every Conformer projection, the
projector, and the LFM2 backbone through MPSGraph. FFT, depthwise convolutions,
normalization, relative-score assembly, softmax, and pointwise activations are
explicit shared Rust code. The Metal result is therefore a correctness gate
for Metal dense operators in the new tower, not a claim that the entire audio
encoder is GPU-resident.

### Chat splice and decode

The pinned ASR example uses the exact system prompt `Perform ASR.`. A
source-level trap is that `ChatState.add_audio()` inserts **no token ID 128**
for audio input. It keeps a shorter text-token stream and inserts
`AUDIO_IN` entries into a longer modality stream. `_prefill` scatters ordinary
text embeddings and projected audio frames into that modality order, with no
learned modality embedding. The owned runtime mirrors this arrangement rather
than treating `<|audio_start|>` as an input placeholder.

The resulting continuous and token embeddings enter the existing LFM2 cache
one absolute position at a time. Audio and text consequently share the same
RoPE position counter and the same short-convolution history. Greedy generation
uses the tied text embedding matrix and terminates on ID 7; ID 130 is also
accepted as a text-end guard for ASR-only callers. The reference examples all
terminated on ID 7.

The CLI consumes JSONL rows shaped `{ "id": "...", "path": "clip.wav" }`
through `--asr-audio`. Relative WAV paths resolve from the manifest directory.
A manifest is an offline batch, but clips are currently encoded and decoded
whole and independently, so no cross-clip padding enters either convolution
history or projection. True padded multi-clip execution remains future work;
when added it must mask encoder padding before projection and left-pad the LFM2
splice.

## Correctness gates

Correctness ran on an Apple M5 Max development host (128 GiB, macOS 26.5.2).
`prepare_lfm2_audio_clips.py` generated 20 deterministic English clips with
macOS `say` (Samantha, rate 180) and converted them with `afconvert` to mono
16 kHz PCM WAV. These are correctness fixtures, not an accuracy benchmark.
Every clip was passed independently through the pinned Python oracle and owned
runtime.

| Gate | Result |
| --- | --- |
| Mel, owned Rust vs liquid-audio fp32 | **Passed 20/20** (the requested first ten plus ten more). Global maximum absolute deviation `0.00016486645`, below `1e-3`. |
| FastConformer + projector, CPU/Accelerate fp32 vs liquid-audio fp32 | **Passed every frame on 20/20**. Global minimum cosine `0.999999999994049`, above `0.9999`. |
| Greedy ASR, CPU/Accelerate fp32 vs liquid-audio fp32 | **20/20 token-exact**, including terminal ID 7. |
| FastConformer + projector, Metal MPSGraph fp32 vs liquid-audio fp32 | **Passed every frame on 20/20**. Global minimum cosine `0.999999999990515`. |
| Greedy ASR, Metal MPSGraph fp32 vs owned CPU fp32 | **20/20 token-exact**; both paths were also 20/20 exact against liquid-audio. |

No first-divergence or WER fallback was needed. The serialized checkpoint is
BF16; “fp32” means fp32 arithmetic after exact BF16-to-fp32 weight conversion,
not recovery of pre-serialization fp32 parameters.

Four representative results:

| Clip | Greedy transcript |
| --- | --- |
| `say-01` | `The quick brown fox jumps over the lazy dog.` |
| `say-05` | `The library closes at six o'clock on Friday evening.` |
| `say-13` | `The weather forecast predicts light rain tomorrow morning.` |
| `say-17` | `She planted rosemary, mint, and basil in the garden.` |

Indicative wall times from the correctness executable, not timed claims:

- CPU/Accelerate fp32: 2.21 s cold load, 1.40 s total frontend/encoder for 20
  clips, and 69.20 s total LFM2 prefill/decode;
- Metal MPSGraph fp32 with lazy execution: 2.47 s cold load, 2.97 s total
  frontend/encoder, and 149.81 s total prefill/decode.

The scalar whole-utterance orchestration and repeated per-clip cache setup are
intentionally unoptimized.

## Reproduction

```bash
MODEL="$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2-Audio-1.5B/snapshots/c798aad30dc3cd72e72970beab51326b8443bd94"

uv venv /tmp/lfm2-audio-ref --python 3.12 --seed
uv pip install --python /tmp/lfm2-audio-ref/bin/python \
  -r bench/spikes/unified-rt/lfm2-audio-reference-requirements.txt

git clone https://github.com/Liquid4All/liquid-audio /tmp/liquid-audio
git -C /tmp/liquid-audio checkout 84cdb243859aaa53db660bc3f4718b54133336bd

/tmp/lfm2-audio-ref/bin/python \
  bench/spikes/unified-rt/prepare_lfm2_audio_clips.py \
  --out-dir target/lfm2-audio-clips \
  --manifest target/lfm2-audio-inputs.jsonl

/tmp/lfm2-audio-ref/bin/python \
  bench/spikes/unified-rt/reference_lfm2_audio.py \
  --model "$MODEL" --inputs target/lfm2-audio-inputs.jsonl \
  --out target/lfm2-audio-reference.jsonl \
  --versions-out target/lfm2-audio-reference-versions.json \
  --liquid-audio-source /tmp/liquid-audio --max-new-tokens 64

cargo run --release -p spike-unified-rt --bin spike-unified-rt -- \
  --model "$MODEL" --tokenizer "$MODEL/tokenizer.json" \
  --asr-audio target/lfm2-audio-inputs.jsonl \
  --asr-reference target/lfm2-audio-reference.jsonl \
  --asr-artifacts-out target/lfm2-audio-owned-cpu-artifacts.jsonl \
  --max-new-tokens 64 --decode-cache-bucket 512 \
  --device cpu --dtype f32 --out target/lfm2-audio-owned-cpu.json

# Metal correctness gate
cargo run --release -p spike-unified-rt --bin spike-unified-rt -- \
  --model "$MODEL" --tokenizer "$MODEL/tokenizer.json" \
  --asr-audio target/lfm2-audio-inputs.jsonl \
  --asr-reference target/lfm2-audio-reference.jsonl \
  --max-new-tokens 64 --decode-cache-bucket 512 \
  --device metal --dtype f32 --execution lazy \
  --out target/lfm2-audio-owned-metal.json
```

## Traps found

1. `torch.hann_window(..., periodic=False)` and `pad_mode="constant"` are both
   observable. The common periodic-window/reflect-padding STFT defaults are
   wrong here.
2. The frontend reports valid length `floor(samples / 160)`, one less than the
   raw centered-STFT width. `ChatState.add_audio()` discards that length, uses
   the raw width, and therefore feeds one zero mel frame into the encoder.
   Matching the mathematically cleaner valid length breaks parity.
3. The log guard is additive `2^-24`; it is not a clamp.
4. The special token `<|audio_start|>` (128) switches generated text into the
   audio-output path, but ASR audio input is out-of-band and does not serialize
   token 128. The modality stream, not a token or learned embedding, reserves
   audio-input positions.
5. Pointwise Conv1d/Conv2d tensors retain singleton kernel axes in safetensors.
   They must be viewed as `[out, in]` for provider GEMM without changing stored
   order.
6. Batch-normalization running statistics are BF16 checkpoint values and must
   be loaded. Recomputing batch statistics changes every downstream frame.
7. The audio checkpoint's LFM tensors are prefixed `lfm.` and its config is
   nested. Architecture-name substring detection alone previously misrouted it
   into the text-only family and then failed while parsing the top-level config.
8. Audio vectors consume ordinary absolute LFM2 positions. Resetting RoPE at
   the text suffix or excluding audio from short-convolution state changes the
   first generated token.

## Next wave

Streaming needs an explicit chunk contract for centered DSP, the pre-encoder,
17 noncausal layers, relative attention, and both LFM2 cache types. It cannot be
implemented by feeding arbitrary multi-token chunks into the existing cached
short-convolution branch. The next wave should also add real padded multi-clip
batches with encoder masks and left-padded LFM2 splices, then move the explicit
FFT/depthwise/normalization/relative-score operations into resident Metal
plans.

Audio output is a separate wave: it requires token 128 mode switching, eight
codebooks, depthformer inference, codebook offsets and embeddings, Mimi decode,
and coordinated rollback of text KV, short-convolution, and audio generation
state. None of those components are loaded or executed by this ASR path.
