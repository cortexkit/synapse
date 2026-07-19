# STT voice test

Minimal localhost bench for comparing **baseline** vs **trie δ6** ASR on a
phrase you speak (or a WAV you POST). Audio never leaves the machine.

## Commands

```bash
# 1. Build the owned ASR spike (once)
cargo build --release -p spike-unified-rt

# 2. Serve the UI (binds 127.0.0.1:4799; override with --port)
python3 tools/stt-voice-test/serve.py
```

Open http://127.0.0.1:4799/ — record a phrase, edit bias terms if needed, stop
to upload. The server writes a temp WAV, builds a one-row ASR JSONL manifest,
and runs `target/release/spike-unified-rt` twice (no bias flags, then
`--asr-trie-delta 6`). Results show side by side with decode times and term
highlights.

Requires a local Hugging Face snapshot of `LiquidAI/LFM2-Audio-1.5B` under
`~/.cache/huggingface/hub` (same resolution as the STT-bias eval kit). Pass
`--model PATH` to override. Device defaults to Metal on macOS with CPU fallback
(`--device cpu|metal|auto`).
