# ANE stateless speculative decode

This spike measures a **stateless fixed-window** Qwen3-0.6B causal language
model on Apple Silicon. It is a feasibility measurement, not a
serving path. The graph accepts left-padded `input_ids` and an
`attention_mask`, re-encodes the complete window, and returns logits for the
last `K` positions. There is no mutable KV state.

## Contents

- `convert_qwen3_to_coreml.py` — `torch.export`-only conversion for windows
  32/64/128 and `K` 1/4/8. The Qwen3 decoder body, final RMSNorm, and tied
  `lm_head` are included; the hidden state is sliced before the final matmul.
  `--unroll-k K` instead exports K explicit greedy passes, shifts the fixed
  window, and appends each argmax token inside one graph; its output is a
  `[1, K]` `token_ids` array.
- `ane_spec_decode.swift` — Core ML compile, timed-run, placement, and greedy
  argmax commands.
- `build_runner.sh` — builds the Swift runner for macOS 14.4+.
- `measure_spec_decode.py` — runs the 200-call warm latency matrix, 30-second
  macmon power windows, and 20-prompt x 8-step CPU-fp32 greedy parity check.
- `measure_phase_b.py` — drives the Metal-step verifier with the persistent
  ANE JSONL draft server, checks the 20 pinned fixtures plus depth-470, and
  emits baseline/speculative latency and break-even metrics.

## Setup and measurement

Use compatible Apple Silicon hardware with exclusive Neural Engine access.
The model snapshot must already be available locally; these commands do not
download it. Do not interrupt another hardware user to obtain a measurement slot.

```bash
cd bench/spikes/ane-spec-decode
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
./build_runner.sh
```

Convert the complete last-K matrix; allow space for each model package:

```bash
MODEL="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<revision>"
for W in 32 64 128; do
  for K in 1 4 8; do
    .venv/bin/python convert_qwen3_to_coreml.py \
      --model "$MODEL" --window "$W" --last-k "$K" \
      --out "artifacts/models/qwen3-w${W}-k${K}.mlpackage" \
      --report-json "artifacts/conversion-w${W}-k${K}.json"
  done
done
```

Convert the W32/K4 autoregressive package:

```bash
.venv/bin/python convert_qwen3_to_coreml.py \
  --model "$MODEL" --window 32 --unroll-k 4 \
  --out artifacts/models/qwen3-w32-unroll-k4.mlpackage \
  --report-json artifacts/conversion-w32-unroll-k4.json
```

Run 200 warm calls for both compute-unit settings and sample ANE rails for
approximately 30 seconds per cell:

```bash
.venv/bin/python measure_spec_decode.py \
  --model "$MODEL" \
  --windows 32 64 128 --last-k 1 4 8 \
  --models-dir artifacts/models \
  --out results/phase-a-raw.json \
  --calls 200 --warmup 20 --power-seconds 30
```

For the unrolled timing cell, use the same harness with the autoregressive package:

```bash
.venv/bin/python measure_spec_decode.py \
  --model "$MODEL" --windows 32 --unroll-k 4 \
  --models-dir artifacts/models --out results/phase-c-raw.json \
  --calls 200 --warmup 20 --power-seconds 30
```

This also compares the unrolled K4 output with four resident K1 calls on 20
initial windows. `measure_spec_decode.py` records a failed conversion, compile,
runtime, or missing power sample as a hole. It never turns a failed cell into a
zero rate. The `CPU_ONLY` rows intentionally report zero ANE share and zero ANE
watts.

## Composition measurement

Build the Swift runner, compile the W32/K4 package, then use the
owned Metal-step binary on compatible hardware with exclusive ANE access:

```bash
python3 measure_phase_b.py \
  --target /path/to/spike-unified-rt \
  --model "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<revision>" \
  --ane-model artifacts/models/qwen3-w32-k4.mlmodelc \
  --out results/phase-b-raw.json
```

The ANE `serve` command consumes one JSONL request containing up to 32 recent
token IDs and returns four drafts. It performs four sequential stateless calls,
using only the final-position argmax of the K4 package on each call. The K4
package's other output positions are logits for already-supplied input tokens,
not independent future proposals. This preserves the autoregressive dependency;
report drafting compute and IPC costs separately from last-K output throughput.
