# STT context-bias evaluation kit

`terms.jsonl` contains 120 terms: 80 technical spelling-sensitive terms collected
from this repository and CortexKit sibling vocabulary, and 40 ordinary control
words. Each row records its class and source. Terms marked `excluded` remain in
the generated corpus but are omitted from term and WER scoring because macOS
`say` cannot produce a stable pronunciation for that spelling. This avoids
confusing a TTS pronunciation failure with an ASR context-bias failure.

## Reproduce the corpus

```bash
python3 bench/spikes/stt-bias/evalkit/generate_utterances.py \
  --out bench/spikes/stt-bias/evalkit/utterances.jsonl

# The full corpus has 360 carrier sentences. Generate two 16 kHz mono WAV
# variants per sentence; audio is ignored because the set is about 70 MB.
python3 bench/spikes/stt-bias/evalkit/synthesize_audio.py \
  --utterances bench/spikes/stt-bias/evalkit/utterances.jsonl \
  --out-dir target/stt-bias/audio \
  --manifest target/stt-bias/all-audio.jsonl \
  --voices Samantha,Daniel
```

The synthesizer writes an AIFF with `say` and converts it with
`afconvert -f WAVE -d LEI16@16000 -c 1`, so every produced clip is mono,
signed-16-bit, 16 kHz WAV. It is deliberately a bootstrap corpus, not a
pronunciation oracle. The excluded list was produced by spot-checking the
ambiguous acronym, punctuation, and mixed-alphanumeric forms with both voices.

For the committed results, select one seeded carrier per term, synthesize both
voices, and alternate them so the 120-request run is balanced by voice:

```bash
python3 bench/spikes/stt-bias/evalkit/select_utterances.py \
  --utterances bench/spikes/stt-bias/evalkit/utterances.jsonl \
  --out target/stt-bias/one-carrier.jsonl
python3 bench/spikes/stt-bias/evalkit/synthesize_audio.py \
  --utterances target/stt-bias/one-carrier.jsonl \
  --out-dir target/stt-bias/audio \
  --manifest target/stt-bias/two-voice.jsonl --voices Samantha,Daniel
python3 bench/spikes/stt-bias/evalkit/select_audio_manifest.py \
  --manifest target/stt-bias/two-voice.jsonl \
  --out target/stt-bias/eval-audio.jsonl
```

## Run an arm

`prepare_arm_manifest.py` attaches bias context per request. Technical utterances
receive their expected term. Control utterances receive the full included
technical vocabulary, making false vocabulary insertion measurable. The owned
runtime consumes `bias_terms` and `bias_prompt` in an ASR input JSONL row; it
only changes the model when a bias flag is present.

```bash
MODEL="$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2-Audio-1.5B/snapshots/c798aad30dc3cd72e72970beab51326b8443bd94"
python3 bench/spikes/stt-bias/evalkit/run_eval.py \
  --model "$MODEL" --audio-manifest target/stt-bias/eval-audio.jsonl \
  --out-dir target/stt-bias/results --arm trie --delta 6 --device cpu
```

Use `--arm baseline`, `prompt`, `trie`, or `combined`. `trie` and `combined`
accept `--delta`; tune `2`, `4`, and `6` on a separate dev selection before
choosing a stable combined setting. `score.py` records case-insensitive
term-exact accuracy, case fidelity, WER, and the class-B false-insertion rate.
It also records prompt-token and prefill/decode timing from the owned runtime.

```bash
python3 bench/spikes/stt-bias/evalkit/render_results.py \
  --scores target/stt-bias/results/baseline-score.json \
           target/stt-bias/results/prompt-score.json \
           target/stt-bias/results/trie-delta-6-score.json \
           target/stt-bias/results/combined-delta-2-score.json \
  --out target/stt-bias/results/table.md
```
