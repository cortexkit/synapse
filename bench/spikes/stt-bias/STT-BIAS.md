# Context-biased STT spike

This spike adds two *per-request* bias arms to the owned LFM2-Audio ASR path:

1. a vocabulary-bearing system-message prefix before the existing out-of-band
   audio splice; and
2. a soft token trie that adds a finite logit bonus at the existing pre-commit
   decode point.

No context flags leave the `Perform ASR.` chat prompt unchanged. The post-change
CPU parity gate against the pinned liquid-audio reference remains **20/20
token-exact** (mel maximum absolute difference `0.00016486645`, encoder minimum
cosine `0.999999999994049`).

## Evaluation corpus

[`evalkit/terms.jsonl`](evalkit/terms.jsonl) has 120 terms: 80 technical terms
from public technical documentation, and 40 ordinary control words.
[`evalkit/generate_utterances.py`](evalkit/generate_utterances.py) deterministically
builds three carrier sentences per term (360 source utterances, seed `20260719`).
[`evalkit/synthesize_audio.py`](evalkit/synthesize_audio.py) uses macOS `say`
(Samantha and Daniel) and `afconvert` to make mono, signed-16-bit, 16 kHz WAVs.
The full two-voice corpus is roughly 70 MB, so `evalkit/audio/` is ignored and
regenerated rather than committed.

The recorded CPU measurement selects carrier slot 1 for every term and
alternates Samantha and Daniel: 120 requests total. It scores 41 non-excluded
technical terms and 40 controls; 39 technical rows are retained in the audio
manifest but omitted from term and WER metrics because the bootstrap TTS cannot
speak their written form reliably. Term matching is case-insensitive and
literal, with word boundaries. Case fidelity separately requires the original
spelling in the transcript.

Control requests deliberately receive the full included technical vocabulary.
Their metric is therefore a useful false-insertion check: an otherwise ordinary
control transcript counts as an insertion if it contains any included technical
term.

## Results

Raw summaries are in [`results.json`](results.json). Times are aggregate CPU
wall time for the 120-request measurement; added prompt tokens are aggregate
text-token positions relative to the no-context prompt.

| Arm | Term-exact (class A) | False insertion (class B) | WER | Case fidelity | Added prompt tokens | Prefill / decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 48.8% (20/41) | 0.0% (0/40) | 6.9% | 26.8% | 0 | 339.46 s / 66.07 s |
| prompt prefix | 68.3% (28/41) | 0.0% (0/40) | 69.4% | 53.7% | 9,097 | 739.62 s / 81.47 s |
| trie boost, delta 6 | 90.2% (37/41) | 0.0% (0/40) | 4.7% | 39.0% | 0 | 352.35 s / 70.01 s |
| combined, delta 2 | 90.2% (37/41) | 0.0% (0/40) | 106.1% | 68.3% | 9,097 | 584.17 s / 75.50 s |

The prompt prefix improves exact spelling and case, but the long technical
vocabulary drives several control utterances into generic assistant completions.
That is why its WER is unacceptable despite zero literal technical-term
insertions. It also adds 9,097 prompt tokens across this measurement (75.8 per
request); the scalar CPU prefill implementation makes that cost visible here.

The trie has no prompt tokens. Its own candidate lookup took 9.1 ms total for
120 requests at delta 6. It is the only currently useful arm: it improves both
term exactness and WER, though it still does not preserve case often enough.

The selected combined row uses conservative delta 2, not the standalone delta
6: delta 4 and delta 6 both produced a non-terminating vocabulary-biased decode
when combined with the prefix. Even the stable delta-2 combination has severe
WER regression, so **do not enable the combined arm for product traffic**.

### Trie delta development sweep

Delta was tuned on a separate deterministic development selection of 24
non-excluded rows (12 technical and 12 controls), using the same alternating
voice policy.

| Delta | Term-exact | False insertion | WER | Case fidelity |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 50.0% (6/12) | 0.0% (0/12) | 9.4% | 33.3% |
| 4 | 75.0% (9/12) | 0.0% (0/12) | 7.9% | 25.0% |
| 6 | 83.3% (10/12) | 0.0% (0/12) | 6.9% | 25.0% |

No completed delta-2/4/6 run inserted a vocabulary item into a class-B control
utterance. A deliberately high delta of 12 failed more seriously: it did not
emit EOS within 64 tokens on the first evaluation request. The combined arm
failed to emit EOS at delta 6 within 64 tokens, at delta 4 within 64 tokens,
and again at delta 4 even after raising the cap to 128. This is the observed
high-bias failure boundary; the evaluator still records explicit 0.0%
class-B false insertion through delta 6 rather than inferring safety from the
absence of a failure.

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
The evaluation command uses CPU only. It does not access the campaign-locked
M1.
