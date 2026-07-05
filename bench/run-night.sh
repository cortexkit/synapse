#!/usr/bin/env bash
# THE night run: every lane, sequentially, idle-gated, on the full AFT corpus.
# Fresh results directory per run so no stale JSON mixes into the table.
#
# Usage:  nohup bash bench/run-night.sh > /tmp/night-run.log 2>&1 & disown
# Or via the harness power wrapper as usual — this script calls it per lane.
#
# Preconditions checked up front (fail fast, before waiting for idle):
# - all release binaries built
# - all model snapshots present
# - LMStudio serving on :1234 with the qwen3 embedding model (optional lane)
# - ts-embed deps installed (bun install ran)
# - mlx-minilm venv at /tmp/synapse-mlx-minilm-venv
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BENCH=./target/release/synapse-bench
RESULTS=bench/results/night-$(date +%Y%m%d)
CORPUS=bench/data/corpus-v2.jsonl
PROMPTS=bench/data/microllm-prompts-v1.jsonl
WAIT_MAX=${WAIT_MAX:-43200}
WAIT_STEP=60
mkdir -p "$RESULTS"

# --- Preconditions -----------------------------------------------------------
fail=0
need() { [ -e "$1" ] || { echo "MISSING: $1" >&2; fail=1; }; }
need target/release/lane-ort-embed
need target/release/lane-mlx
need target/release/lane-llama
need target/release/lane-burn
need target/release/lane-wrap-embed
need target/release/synapse-bench
need "$CORPUS"
need "$PROMPTS"
need bench/lanes/ts-embed/node_modules
need /tmp/synapse-mlx-minilm-venv/bin/python

SNAP_ONNX=$HOME/.cache/huggingface/hub/models--onnx-community--Qwen3-Embedding-0.6B-ONNX/snapshots/c25a394dd583836952667c12f008335071b3f43d
SNAP_MLX_EMBED=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/* 2>/dev/null | head -1)
SNAP_MLX_MICROLLM=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/* 2>/dev/null | head -1)
SNAP_GGUF_EMBED=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/* 2>/dev/null | head -1)
SNAP_GGUF_LLM=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots/* 2>/dev/null | head -1)
SNAP_MINILM=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/* 2>/dev/null | head -1)
SNAP_MINILM_GGUF=$(ls -d "$HOME"/.cache/huggingface/hub/models--second-state--All-MiniLM-L6-v2-Embedding-GGUF/snapshots/* 2>/dev/null | head -1)
SNAP_LFM=$(ls -d "$HOME"/.cache/huggingface/hub/models--LiquidAI--LFM2.5-230M-GGUF/snapshots/* 2>/dev/null | head -1)
for v in SNAP_ONNX SNAP_MLX_EMBED SNAP_MLX_MICROLLM SNAP_GGUF_EMBED SNAP_GGUF_LLM SNAP_MINILM SNAP_MINILM_GGUF SNAP_LFM; do
  [ -n "${!v}" ] || { echo "MISSING snapshot: $v" >&2; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "preconditions failed; fix before the night run" >&2; exit 1; }

LMSTUDIO_UP=0
curl -sf http://127.0.0.1:1234/v1/models >/dev/null 2>&1 && LMSTUDIO_UP=1
echo "preconditions OK (lmstudio_up=$LMSTUDIO_UP); results -> $RESULTS"

# --- Runner ------------------------------------------------------------------
run() {
  local name="$1"; shift
  local waited=0
  echo "=== lane: $name ($(date +%H:%M:%S)) ==="
  until $BENCH power --out "$RESULTS/$name.measure.json" -- "$@"; do
    waited=$((waited + WAIT_STEP))
    if [ "$waited" -ge "$WAIT_MAX" ]; then
      echo "gave up waiting for idle for $name after ${WAIT_MAX}s" >&2
      return 1
    fi
    echo "machine busy; retrying $name in ${WAIT_STEP}s (waited ${waited}s)"
    sleep "$WAIT_STEP"
  done
}

# --- Workload A: Qwen3-Embedding-0.6B (quality-upgrade model) ----------------
run ort-cpu-embed \
  ./target/release/lane-ort-embed \
  --model "$SNAP_ONNX/onnx/model.onnx" --tokenizer "$SNAP_ONNX/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/ort-cpu-embed.json" \
  --vectors-out "$RESULTS/ort-cpu-embed-vectors.jsonl" \
  --pooling last --max-length 512 \
  --model-label "Qwen3-Embedding-0.6B@onnx-fp32"

run mlx-embed \
  ./target/release/lane-mlx embed \
  --model "$SNAP_MLX_EMBED" --tokenizer "$SNAP_MLX_EMBED/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/mlx-embed.json" \
  --reference "$RESULTS/ort-cpu-embed-vectors.jsonl" \
  --model-label "Qwen3-Embedding-0.6B@mlx-bf16"

run llama-metal-embed \
  ./target/release/lane-llama embed \
  --model "$SNAP_GGUF_EMBED/Qwen3-Embedding-0.6B-f16.gguf" \
  --tokenizer "$SNAP_MLX_EMBED/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/llama-metal-embed.json" \
  --reference "$RESULTS/ort-cpu-embed-vectors.jsonl" \
  --model-label "Qwen3-Embedding-0.6B@gguf-f16"

if [ "$LMSTUDIO_UP" -eq 1 ]; then
  run wrap-lmstudio-embed \
    ./target/release/lane-wrap-embed \
    --base-url http://127.0.0.1:1234 --model text-embedding-qwen3-embedding-0.6b \
    --lane wrap-lmstudio --corpus "$CORPUS" \
    --out "$RESULTS/wrap-lmstudio-embed.json" \
    --reference "$RESULTS/ort-cpu-embed-vectors.jsonl" \
    --tokenizer "$SNAP_MLX_EMBED/tokenizer.json" \
    --rss-process-names "LM Studio Helper,lms" \
    --model-label "Qwen3-Embedding-0.6B@lmstudio"
else
  echo "skip wrap-lmstudio-embed: not reachable" >&2
fi

# AFT spike replication: Qwen3-Embedding 4-bit DWQ via Python mlx-embeddings
# (their 2026-06 spike measured 22.8k tok/s on code chunks with this config).
run mlx-qwen-dwq-embed \
  /tmp/synapse-mlx-minilm-venv/bin/python bench/lanes/mlx-minilm/main.py \
  --model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
  --corpus "$CORPUS" --out "$RESULTS/mlx-qwen-dwq-embed.json" \
  --vectors-out "$RESULTS/mlx-qwen-dwq-vectors.jsonl" \
  --model-label "Qwen3-Embedding-0.6B@mlx-4bit-dwq"

# --- Workload A floor: all-MiniLM-L6-v2 across every engine ------------------
run ort-cpu-minilm-embed \
  ./target/release/lane-ort-embed \
  --model "$SNAP_MINILM/model.onnx" --tokenizer "$SNAP_MINILM/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/ort-cpu-minilm-embed.json" \
  --vectors-out "$RESULTS/ort-cpu-minilm-embed-vectors.jsonl" \
  --pooling mean --max-length 512 \
  --model-label "all-MiniLM-L6-v2@ort-cpu-fp32"

run llama-metal-minilm-embed \
  ./target/release/lane-llama embed \
  --model "$SNAP_MINILM_GGUF/all-MiniLM-L6-v2-ggml-model-f16.gguf" \
  --tokenizer "$SNAP_MINILM/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/llama-metal-minilm-embed.json" \
  --reference "$RESULTS/ort-cpu-minilm-embed-vectors.jsonl" \
  --pooling mean \
  --model-label "all-MiniLM-L6-v2@gguf-f16-metal"

run llama-cpu-minilm-embed \
  ./target/release/lane-llama embed \
  --model "$SNAP_MINILM_GGUF/all-MiniLM-L6-v2-ggml-model-f16.gguf" \
  --tokenizer "$SNAP_MINILM/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/llama-cpu-minilm-embed.json" \
  --reference "$RESULTS/ort-cpu-minilm-embed-vectors.jsonl" \
  --pooling mean --gpu-layers 0 \
  --model-label "all-MiniLM-L6-v2@gguf-f16-cpu"

run burn-wgpu-embed \
  ./target/release/lane-burn \
  --model "$SNAP_MINILM/model.onnx" --tokenizer "$SNAP_MINILM/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/burn-wgpu-embed.json" \
  --reference "$RESULTS/ort-cpu-minilm-embed-vectors.jsonl" \
  --pooling mean \
  --model-label "all-MiniLM-L6-v2@burn-wgpu-f32"

run ts-transformersjs-q8-embed \
  bun bench/lanes/ts-embed/main.mjs --engine transformersjs --dtype default \
  --corpus "$CORPUS" --out "$RESULTS/ts-transformersjs-q8-embed.json" \
  --model-label "Xenova/all-MiniLM-L6-v2@transformersjs-q8"

run ts-transformersjs-fp32-embed \
  bun bench/lanes/ts-embed/main.mjs --engine transformersjs --dtype fp32 \
  --corpus "$CORPUS" --out "$RESULTS/ts-transformersjs-fp32-embed.json" \
  --model-label "Xenova/all-MiniLM-L6-v2@transformersjs-fp32"

run ts-ort-node-embed \
  bun bench/lanes/ts-embed/main.mjs --engine ort-node \
  --corpus "$CORPUS" --out "$RESULTS/ts-ort-node-embed.json" \
  --model-label "Qdrant/all-MiniLM-L6-v2-onnx@ort-node-fp32"

run mlx-minilm-embed \
  /tmp/synapse-mlx-minilm-venv/bin/python bench/lanes/mlx-minilm/main.py \
  --corpus "$CORPUS" --out "$RESULTS/mlx-minilm-embed.json" \
  --model-label "all-MiniLM-L6-v2@mlx-bf16"

# --- Workload B: micro-LLM one-shots -----------------------------------------
run mlx-microllm \
  ./target/release/lane-mlx microllm \
  --model "$SNAP_MLX_MICROLLM" --tokenizer "$SNAP_MLX_MICROLLM/tokenizer.json" \
  --prompts "$PROMPTS" --out "$RESULTS/mlx-microllm.json"

run llama-metal-microllm \
  ./target/release/lane-llama microllm \
  --model "$SNAP_GGUF_LLM/Qwen3-0.6B-Q8_0.gguf" \
  --prompts "$PROMPTS" --out "$RESULTS/llama-metal-microllm.json" \
  --model-label "Qwen3-0.6B@gguf-q8_0"

run llama-metal-microllm-lfm \
  ./target/release/lane-llama microllm \
  --model "$SNAP_LFM/LFM2.5-230M-Q8_0.gguf" \
  --prompts "$PROMPTS" --out "$RESULTS/llama-metal-microllm-lfm.json" \
  --model-label "LFM2.5-230M@gguf-q8_0"

echo "night run complete ($(date +%H:%M:%S)); results in $RESULTS/"
