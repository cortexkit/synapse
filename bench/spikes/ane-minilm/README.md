# ANE MiniLM spike

This folder contains the fixed-bucket Core ML proof path for `sentence-transformers/all-MiniLM-L6-v2` on Apple Silicon ANE.

## Contents

- `convert_minilm_to_coreml.py` — converts the HF MiniLM encoder to a fixed-shape fp16 `.mlpackage` using export only.
- `convert_modernbert_to_coreml.py` — converts fixed-bucket GTE ModernBERT embedder or reranker packages with pooling/head inside Core ML and verifies eager/export/Core ML smoke parity.
- `prep_tokenized_jsonl.py` — turns corpus JSONL into fixed-bucket `{id,input_ids,attention_mask}` JSONL.
- `ort_reference.py` — ONNX Runtime fp32 reference over the same pretokenized inputs.
- `ane_coreml.swift` — generalized Swift CLI with two subcommands:
  - `compile` — `.mlpackage`/`.mlmodel` -> permanent `.mlmodelc`
  - `run` (`embed` alias) — load `.mlmodelc`, select graph/mean/CLS pooling or pair scoring, emit JSONL, and optionally dump MLComputePlan placement
- `build_runner.sh` — builds the Swift CLI to `.build/ane-coreml`.
- `SPIKE.md` — original MiniLM results and verdict.
- `ANE-WAVE1.md` — ModernBERT wave-1 conversion, parity, placement, power, throughput, and latency evidence.

## Requirements

Local conversion / prep machine:

- macOS on Apple Silicon
- Python 3.12
- `uv`
- Hugging Face cache entries already present locally:
  - `sentence-transformers/all-MiniLM-L6-v2`
  - `Qdrant/all-MiniLM-L6-v2-onnx` (for `model.onnx` + `tokenizer.json`)

Bench host requirements:

- macOS 14.4+ (for `MLComputePlan`)
- no Xcode project required
- no Python required on the remote box if you rsync the prepared JSONL and ORT reference vectors from the local machine

## Local setup

```bash
cd bench/spikes/ane-minilm
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
./build_runner.sh
```

## Convert the 256 and 512 bucket variants

The converter accepts only `--frontend export`. Trace conversion is intentionally unavailable because the trace-built seq512 variant missed parity catastrophically.

```bash
cd bench/spikes/ane-minilm
source .venv/bin/activate

python convert_minilm_to_coreml.py \
  --seq-len 256 \
  --out ~/bench-tools/ane-spike/models/all-MiniLM-L6-v2-seq256.mlpackage \
  --report-json ~/bench-tools/ane-spike/models/all-MiniLM-L6-v2-seq256.report.json

python convert_minilm_to_coreml.py \
  --seq-len 512 \
  --out ~/bench-tools/ane-spike/models/all-MiniLM-L6-v2-seq512.mlpackage \
  --report-json ~/bench-tools/ane-spike/models/all-MiniLM-L6-v2-seq512.report.json
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
  --input corpus/aft-chunks.jsonl \
  --text-field embed_text \
  --limit 1000 \
  --output ~/bench-tools/ane-spike/data/aft-1000-b256.jsonl

python prep_tokenized_jsonl.py \
  --bucket 512 \
  --input corpus/aft-chunks.jsonl \
  --text-field embed_text \
  --limit 1000 \
  --output ~/bench-tools/ane-spike/data/aft-1000-b512.jsonl
```

## Produce the ORT fp32 reference locally

```bash
cd bench/spikes/ane-minilm
source .venv/bin/activate

python ort_reference.py \
  --model ~/.cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/manual/model.onnx \
  --input ~/bench-tools/ane-spike/data/aft-1000-b256.jsonl \
  --output ~/bench-tools/ane-spike/reference/ort-1000-b256.jsonl \
  --stats-out ~/bench-tools/ane-spike/reference/ort-1000-b256.stats.json

python ort_reference.py \
  --model ~/.cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/manual/model.onnx \
  --input ~/bench-tools/ane-spike/data/aft-1000-b512.jsonl \
  --output ~/bench-tools/ane-spike/reference/ort-1000-b512.jsonl \
  --stats-out ~/bench-tools/ane-spike/reference/ort-1000-b512.stats.json
```

## Optional local smoke / parity check

```bash
./bench/spikes/ane-minilm/.build/ane-coreml compile \
  --model ~/bench-tools/ane-spike/models/all-MiniLM-L6-v2-seq256.mlpackage \
  --out ~/bench-tools/ane-spike/models/all-MiniLM-L6-v2-seq256.mlmodelc

./bench/spikes/ane-minilm/.build/ane-coreml embed \
  --model ~/bench-tools/ane-spike/models/all-MiniLM-L6-v2-seq256.mlmodelc \
  --input ~/bench-tools/ane-spike/data/aft-1000-b256.jsonl \
  --output ~/bench-tools/ane-spike/results/coreml-1000-b256.jsonl \
  --stats-out ~/bench-tools/ane-spike/results/coreml-1000-b256.stats.json \
  --placement-out ~/bench-tools/ane-spike/results/coreml-1000-b256.placement.json \
  --batch-size 8

cargo build --release -p synapse-bench
./target/release/synapse-bench parity \
  --reference ~/bench-tools/ane-spike/reference/ort-1000-b256.jsonl \
  --candidate ~/bench-tools/ane-spike/results/coreml-1000-b256.jsonl \
  --k 10 \
  --stride 4
```

## M1 bench-box workflow

### 1) Sync the bundle

Build locally, then rsync the source, binaries, model packages, prepared JSONL, and ORT references to the M1 box:

```bash
ssh [bench-host-alias] 'rm -rf ~/bench-tools/ane-spike && mkdir -p ~/bench-tools/ane-spike/{src,bin,models,data,reference,results}'

rsync -av bench/spikes/ane-minilm/ [bench-host-alias]:~/bench-tools/ane-spike/src/
rsync -av bench/spikes/ane-minilm/.build/ane-coreml [bench-host-alias]:~/bench-tools/ane-spike/bin/
rsync -av target/release/synapse-bench [bench-host-alias]:~/bench-tools/ane-spike/bin/
rsync -av ~/bench-tools/ane-spike/models/ [bench-host-alias]:~/bench-tools/ane-spike/models/
rsync -av ~/bench-tools/ane-spike/data/ [bench-host-alias]:~/bench-tools/ane-spike/data/
rsync -av ~/bench-tools/ane-spike/reference/ [bench-host-alias]:~/bench-tools/ane-spike/reference/
```

### 2) Compile the `.mlpackage` models on the M1 box

The remote host does not need `coremlcompiler`; the Swift CLI's `compile` subcommand calls `MLModel.compileModel(at:)` directly.

```bash
ssh [bench-host-alias] '
set -euo pipefail
cd ~/bench-tools/ane-spike
bin/ane-coreml compile --model models/all-MiniLM-L6-v2-seq256.mlpackage --out models/all-MiniLM-L6-v2-seq256.mlmodelc
bin/ane-coreml compile --model models/all-MiniLM-L6-v2-seq512.mlpackage --out models/all-MiniLM-L6-v2-seq512.mlmodelc
'
```

### 3) Placement reports on the M1 box

```bash
ssh [bench-host-alias] '
set -euo pipefail
cd ~/bench-tools/ane-spike
head -n 1 data/aft-1000-b256.jsonl > data/aft-1-b256.jsonl
head -n 1 data/aft-1000-b512.jsonl > data/aft-1-b512.jsonl
bin/ane-coreml embed \
  --model models/all-MiniLM-L6-v2-seq256.mlmodelc \
  --input data/aft-1-b256.jsonl \
  --output results/placement-b256.jsonl \
  --stats-out results/placement-b256.stats.json \
  --placement-out results/placement-b256.report.json \
  --batch-size 1 > /dev/null
bin/ane-coreml embed \
  --model models/all-MiniLM-L6-v2-seq512.mlmodelc \
  --input data/aft-1-b512.jsonl \
  --output results/placement-b512.jsonl \
  --stats-out results/placement-b512.stats.json \
  --placement-out results/placement-b512.report.json \
  --batch-size 1 > /dev/null
'
```

### 4) Locked timed runs + powermetrics on the M1 box

Prime `sudo` with the operator-provided password, then keep the lock only for the timed embed runs themselves:

```bash
ssh [bench-host-alias] '
set -euo pipefail
cd ~/bench-tools/ane-spike
cleanup() {
  if [ -n "${PM_PID:-}" ]; then
    kill "$PM_PID" >/dev/null 2>&1 || true
    wait "$PM_PID" 2>/dev/null || true
  fi
  rmdir [bench-user-home]/bench.lock >/dev/null 2>&1 || true
}
trap cleanup EXIT
until mkdir [bench-user-home]/bench.lock 2>/dev/null; do sleep 30; done
for bucket in 256 512; do
  echo "<operator-provided sudo password>" | sudo -S -p "" true >/dev/null
  rm -f "results/powermetrics-b${bucket}.txt"
  sudo -n powermetrics -i 500 -s ane_power,gpu_power,cpu_power > "results/powermetrics-b${bucket}.txt" 2>/dev/null &
  PM_PID=$!
  bin/ane-coreml embed \
    --model "models/all-MiniLM-L6-v2-seq${bucket}.mlmodelc" \
    --input "data/aft-1000-b${bucket}.jsonl" \
    --output "results/coreml-1000-b${bucket}.jsonl" \
    --stats-out "results/coreml-1000-b${bucket}.stats.json" \
    --batch-size 8 > /dev/null
  kill "$PM_PID" >/dev/null 2>&1 || true
  wait "$PM_PID" 2>/dev/null || true
  unset PM_PID
  bin/synapse-bench parity \
    --reference "reference/ort-1000-b${bucket}.jsonl" \
    --candidate "results/coreml-1000-b${bucket}.jsonl" \
    --k 10 \
    --stride 4 \
    > "results/parity-1000-b${bucket}.json"
done
'
```

### 5) Pull the results back

```bash
rsync -av [bench-host-alias]:~/bench-tools/ane-spike/results/ /tmp/ane-m1-results/
```

## Notes

- The Swift runner writes stats JSON with `cold_load_s`, `infer_wall_s`, `docs_per_s`, and `tokens_per_s`.
- `placement-*.report.json` uses `dispatchable_device_share` as the meaningful residency number. `unknown` ops are Core ML `const` nodes with no runtime dispatch.
- The M1 measurement box used for the committed spike did not have local Python / `uv` / developer tools; that is why the remote instructions rsync prepared inputs and ORT references instead of recreating them on-host.
- Do **not** record the operator-provided sudo password in git-tracked files or result docs.
