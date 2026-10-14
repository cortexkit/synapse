# ANE encoder spike

This folder contains the fixed-bucket Core ML proof paths for MiniLM, GTE ModernBERT, and Qwen3-Embedding-0.6B on Apple Silicon ANE.

## Contents

- `convert_minilm_to_coreml.py` — converts the HF MiniLM encoder to a fixed-shape fp16 `.mlpackage` using export only.
- `convert_modernbert_to_coreml.py` — converts fixed-bucket GTE ModernBERT embedder or reranker packages with pooling/head inside Core ML and verifies eager/export/Core ML smoke parity.
- `convert_qwen3_to_coreml.py` — converts fixed-bucket Qwen3-Embedding-0.6B with causal attention, terminal-EOS pooling, and the Conv2d ANE layout using `torch.export` only.
- `prep_tokenized_jsonl.py` — turns corpus JSONL into fixed-bucket `{id,input_ids,attention_mask}` JSONL.
- `prep_qwen3_tokenized_jsonl.py` — creates left-padded Qwen3 inputs with the model-config terminal EOS in the final position.
- `ort_reference.py` — ONNX Runtime fp32 reference over the same pretokenized inputs.
- `qwen3_fp32_reference.py` — regenerates only the bucket-specific Qwen fp32 rows whose active tokens differ from the frozen reference.
- `ane_coreml.swift` — generalized Swift CLI with two subcommands:
  - `compile` — `.mlpackage`/`.mlmodel` -> permanent `.mlmodelc`
  - `run` (`embed` alias) — load `.mlmodelc`, select graph/mean/CLS pooling or pair scoring, emit JSONL, and optionally dump MLComputePlan placement
- `build_runner.sh` — builds the Swift CLI to `.build/ane-coreml`.

## Requirements

- macOS on Apple Silicon; macOS 14.4+ for `MLComputePlan`
- Python 3.12 and `uv` for conversion and input preparation
- local Hugging Face snapshots for `sentence-transformers/all-MiniLM-L6-v2`,
  `Qdrant/all-MiniLM-L6-v2-onnx` (model and tokenizer), and
  `Qwen/Qwen3-Embedding-0.6B` when using the Qwen converter/reference

Conversion and inference may run on different compatible machines. The Swift
runner needs compiled models and prepared JSONL inputs, not a Python environment.
Reserve exclusive Neural Engine access before hardware measurements; do not
interrupt another hardware user to obtain it.

## Local setup

```bash
cd bench/spikes/ane-minilm
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
./build_runner.sh
```

## Qwen3-Embedding-0.6B fixed-bucket workflow

Qwen3 requires `transformers` 4.51.3, so use its separate requirements file
rather than changing the MiniLM conversion environment. The converter supports
`torch.export` only; trace-built packages are not supported.

```bash
cd bench/spikes/ane-minilm
uv venv --python 3.12 .venv-qwen3
uv pip install --python .venv-qwen3/bin/python -r requirements-qwen3.txt

for bucket in 128 256 512; do
  .venv-qwen3/bin/python convert_qwen3_to_coreml.py \
    --seq-len "$bucket" \
    --out "/tmp/ane-wave2/models/qwen3-seq${bucket}.mlpackage" \
    --report-json "/tmp/ane-wave2/reports/qwen3-seq${bucket}.json"
  .venv-qwen3/bin/python prep_qwen3_tokenized_jsonl.py \
    --bucket "$bucket" --input /tmp/qwen3-corpus-400.jsonl \
    --output "/tmp/ane-wave2/data/qwen3-400-b${bucket}.jsonl"
done
```

Use `qwen3_fp32_reference.py` only when a shorter bucket changes active
Qwen tokens. Reuse the checked frozen ORT fp32 reference for rows whose active
token sequence is unchanged; do not compare a truncated candidate to a
longer-token reference vector.

## Convert the 256 and 512 bucket variants

The converter accepts only `--frontend export`; trace conversion is intentionally unsupported.

```bash
cd bench/spikes/ane-minilm
source .venv/bin/activate

python convert_minilm_to_coreml.py \
  --seq-len 256 \
  --out /tmp/ane-encoder/models/all-MiniLM-L6-v2-seq256.mlpackage \
  --report-json /tmp/ane-encoder/models/all-MiniLM-L6-v2-seq256.report.json

python convert_minilm_to_coreml.py \
  --seq-len 512 \
  --out /tmp/ane-encoder/models/all-MiniLM-L6-v2-seq512.mlpackage \
  --report-json /tmp/ane-encoder/models/all-MiniLM-L6-v2-seq512.report.json
```

## Build the Swift CLI

```bash
cd bench/spikes/ane-minilm
./build_runner.sh
```

The binary lands at:

```bash
bench/spikes/ane-minilm/.build/ane-coreml
```

## Prepare pretokenized inputs locally

The prep step strips the tokenizer's baked-in `Fixed(128)` padding policy, truncates to the requested bucket, and pads manually to the bucket with a correct attention mask.

```bash
cd bench/spikes/ane-minilm
source .venv/bin/activate

python prep_tokenized_jsonl.py \
  --bucket 256 \
  --input /path/to/corpus.jsonl \
  --text-field text \
  --limit 1000 \
  --output /tmp/ane-encoder/data/sample-1000-b256.jsonl

python prep_tokenized_jsonl.py \
  --bucket 512 \
  --input /path/to/corpus.jsonl \
  --text-field text \
  --limit 1000 \
  --output /tmp/ane-encoder/data/sample-1000-b512.jsonl
```

## Produce the ORT fp32 reference locally

```bash
cd bench/spikes/ane-minilm
source .venv/bin/activate

python ort_reference.py \
  --model /path/to/minilm-onnx/model.onnx \
  --input /tmp/ane-encoder/data/sample-1000-b256.jsonl \
  --output /tmp/ane-encoder/reference/ort-1000-b256.jsonl \
  --stats-out /tmp/ane-encoder/reference/ort-1000-b256.stats.json

python ort_reference.py \
  --model /path/to/minilm-onnx/model.onnx \
  --input /tmp/ane-encoder/data/sample-1000-b512.jsonl \
  --output /tmp/ane-encoder/reference/ort-1000-b512.jsonl \
  --stats-out /tmp/ane-encoder/reference/ort-1000-b512.stats.json
```

## Optional local smoke / parity check

```bash
./bench/spikes/ane-minilm/.build/ane-coreml compile \
  --model /tmp/ane-encoder/models/all-MiniLM-L6-v2-seq256.mlpackage \
  --out /tmp/ane-encoder/models/all-MiniLM-L6-v2-seq256.mlmodelc

./bench/spikes/ane-minilm/.build/ane-coreml embed \
  --model /tmp/ane-encoder/models/all-MiniLM-L6-v2-seq256.mlmodelc \
  --input /tmp/ane-encoder/data/sample-1000-b256.jsonl \
  --output /tmp/ane-encoder/results/coreml-1000-b256.jsonl \
  --stats-out /tmp/ane-encoder/results/coreml-1000-b256.stats.json \
  --placement-out /tmp/ane-encoder/results/coreml-1000-b256.placement.json \
  --batch-size 8

cargo build --release -p synapse-bench
./target/release/synapse-bench parity \
  --reference /tmp/ane-encoder/reference/ort-1000-b256.jsonl \
  --candidate /tmp/ane-encoder/results/coreml-1000-b256.jsonl \
  --k 10 \
  --stride 4
```

## Notes

- The Swift runner writes stats JSON with `cold_load_s`, `infer_wall_s`, `docs_per_s`, and `tokens_per_s`.
- `placement-*.report.json` uses `dispatchable_device_share` as the meaningful residency number. `unknown` ops are Core ML `const` nodes with no runtime dispatch.
