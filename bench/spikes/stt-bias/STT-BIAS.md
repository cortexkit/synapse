# Context-biased STT spike

This spike adds two *per-request* bias arms to the owned LFM2-Audio ASR path:

1. a vocabulary-bearing system-message prefix before the existing out-of-band
   audio splice; and
2. a soft token trie that adds a finite logit bonus at the existing pre-commit
   decode point.

With bias flags disabled, the runtime keeps the `Perform ASR.` chat prompt and
applies no logit bonus. `--asr-prompt-bias` and `--asr-trie-delta` opt into the
arms described below.

## Evaluation corpus

[`evalkit/terms.jsonl`](evalkit/terms.jsonl) has 120 terms: 80 technical terms
from public technical documentation, and 40 ordinary control words.
[`evalkit/generate_utterances.py`](evalkit/generate_utterances.py) deterministically
builds three carrier sentences per term (360 source utterances, seed `20260719`).
[`evalkit/synthesize_audio.py`](evalkit/synthesize_audio.py) uses macOS `say`
(Samantha and Daniel) and `afconvert` to make mono, signed-16-bit, 16 kHz WAVs.
The full two-voice corpus is roughly 70 MB, so `evalkit/audio/` is ignored and
regenerated rather than committed.

Select one generated carrier sentence (a sentence containing the target term)
per term with a reproducible seed, and balance the voices for evaluation. Rows
marked excluded stay in the audio manifest but are omitted from term and WER
metrics because the bootstrap TTS cannot speak their written form reliably.
Term matching is case-insensitive and
literal, with word boundaries. Case fidelity separately requires the original
spelling in the transcript.

Control requests deliberately receive the full included technical vocabulary.
Their metric is therefore a useful false-insertion check: an otherwise ordinary
control transcript counts as an insertion if it contains any included technical
term.

## Bias safety

Biasing is experimental. Finite logit bonuses can prevent EOS, so bound the
generation length and validate term accuracy, WER and false insertions on an
independent evaluation selection. Do not enable the combined arm for product
traffic; prefix and trie bias are not interchangeable with a certified ASR path.

## TTS-excluded terms

These rows remain reproducible corpus members but are excluded from term-exact,
case, and WER aggregates. The reason is a macOS `say` pronunciation mismatch,
not an ASR judgment:

- `cuBLASLt` — mixed-case CUDA library name is spelled rather than spoken stably.
- `MPSGraph` — acronym and camel-case suffix vary by voice.
- `subc` — spoken as an ordinary word rather than the project name.
- `PAQ` — acronym pronunciation is not stable.
- `tok/s` — slash unit differs between voices.
- `RMSNorm` — acronym-plus-word pronunciation is unstable.
- `GGUF` — all-caps model format is read as a variable letter sequence.
- `mlx` — lowercase acronym is treated as an ordinary word by one voice.
- `fp16`, `BF16`, `F16`, `Float32` — alphanumeric precision labels expand inconsistently.
- `RoPE` — intended acronym conflicts with the ordinary word “rope”.
- `KV cache` — initialism pronunciation is inconsistent.
- `LFM2`, `M1`, `Q8_0` — hardware/model notation is not rendered as stable lexical speech.
- `JSONL`, `ASR`, `WAV`, `AIFF`, `MPS`, `SIMD`, `GEMM`, `STFT`, `RMS` — all-caps forms are letter sequences rather than stable word targets.
- `CUDA`, `ONNX Runtime`, `GGML`, `FAISS`, `GELU`, `SiLU` — intended acronym pronunciation is not preserved by the bootstrap voices.
- `llama.cpp` — dotted project name is not rendered consistently.
- `Qwen3`, `ModernBERT`, `MiniLM`, `SwiGLU`, `RustFFT`, `MPSGraphContext` — mixed alphanumeric or camel-case API/model names are not intelligible or stable.

The machine-readable, per-term reason is authoritative in
[`evalkit/terms.jsonl`](evalkit/terms.jsonl).

## Runtime interface

ASR JSONL rows may add request-local fields:

```json
{"id":"code-view","path":"clip.wav","bias_terms":["memcpy","MPSGraph"],"bias_prompt":"Transcribe faithfully; use vocabulary only when spoken."}
```

`--asr-prompt-bias` serializes those terms and optional prompt as system text
before audio. `--asr-trie-delta 2|4|6` builds a trie over bare and
space-prefixed tokenizer encodings for original, lowercase, uppercase, and
capitalized term spellings. A path remains active for the last 16 committed
transcript tokens by default (`--asr-trie-window`); only its next token receives
the finite bonus. It is deliberately a soft bonus, never a mask.

`--asr-bias-terms terms.jsonl` and `--asr-bias-prompt` add a global context to
the per-row fields for batch experiments. `evalkit/prepare_arm_manifest.py`
uses target-only context for technical rows and the full technical vocabulary
for controls.

## Reproduce

See [`evalkit/README.md`](evalkit/README.md) for deterministic synthesis,
manifest creation, arm execution, scoring, and Markdown rendering commands.
The evaluation command uses CPU only.
