#!/usr/bin/env bash
# THE night run: every lane, sequentially, idle-gated, on the full AFT corpus.
# Rev 2 (saturation audit): burn 16M attention units, sorted length-uniform
# batching in burn/mlx lanes, mlx-python 32k/256 defaults. ort keeps the
# 9-thread production policy ON PURPOSE (the saturated-machine number is a
# separate column, not the default).
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
LLAMA_BIN=${LLAMA_SERVER_BIN:-/opt/zerobrew/bin/llama-server}
MLX_PY=${MLX_PYTHON:-/tmp/synapse-mlx-minilm-venv/bin/python}
POTION_PY=${POTION_PYTHON:-bench/lanes/potion/.venv/bin/python}
WAIT_MAX=${WAIT_MAX:-43200}
WAIT_STEP=60
mkdir -p "$RESULTS"

find_snapshot() {
  local pattern="$1"
  local found
  found=$(ls -d $pattern 2>/dev/null | head -n 1 || true)
  if [ -z "$found" ]; then
    return 1
  fi
  printf '%s\n' "$found"
}

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
need "$MLX_PY"
need "$POTION_PY"

SNAP_ONNX=$(find_snapshot "$HOME/.cache/huggingface/hub/models--onnx-community--Qwen3-Embedding-0.6B-ONNX/snapshots/*" || true)
SNAP_MLX_EMBED=$(find_snapshot "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*" || true)
SNAP_MLX_MICROLLM=$(find_snapshot "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*" || true)
SNAP_GGUF_EMBED=$(find_snapshot "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/*" || true)
SNAP_GGUF_LLM=$(find_snapshot "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots/*" || true)
SNAP_MINILM=$(find_snapshot "$HOME/.cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/*" || true)
SNAP_MINILM_GGUF=$(find_snapshot "$HOME/.cache/huggingface/hub/models--second-state--All-MiniLM-L6-v2-Embedding-GGUF/snapshots/*" || true)
SNAP_LFM=$(find_snapshot "$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-230M-GGUF/snapshots/*" || true)
SNAP_GTE_MODERNBERT=$(find_snapshot "$HOME/.cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots/*" || true)
SNAP_NOMIC_MODERNBERT=$(find_snapshot "$HOME/.cache/huggingface/hub/models--nomic-ai--modernbert-embed-base/snapshots/*" || true)
SNAP_NOMIC_MLX=$(find_snapshot "$HOME/.cache/huggingface/hub/models--mlx-community--nomicai-modernbert-embed-base-bf16/snapshots/*" || true)
SNAP_GTE_GGUF=$(find_snapshot "$HOME/.cache/huggingface/hub/models--keisuke-miyako--gte-modernbert-base-gguf/snapshots/*" || true)
SNAP_NOMIC_GGUF=$(find_snapshot "$HOME/.cache/huggingface/hub/models--keisuke-miyako--modernbert-embed-base-gguf-q8_0/snapshots/*" || true)
SNAP_JINA=$(find_snapshot "$HOME/.cache/huggingface/hub/models--jinaai--jina-embeddings-v5-text-nano-retrieval/snapshots/*" || true)
SNAP_QWEN_QUANTS=$(find_snapshot "$HOME/.cache/huggingface/hub/models--mradermacher--Qwen3-Embedding-0.6B-GGUF/snapshots/*" || true)
SNAP_QWEN_MLX_8BIT=$(find_snapshot "$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3-Embedding-0.6B-8bit/snapshots/*" || true)

for v in SNAP_ONNX SNAP_MLX_EMBED SNAP_MLX_MICROLLM SNAP_GGUF_EMBED SNAP_GGUF_LLM SNAP_MINILM SNAP_MINILM_GGUF SNAP_LFM SNAP_GTE_MODERNBERT SNAP_NOMIC_MODERNBERT SNAP_NOMIC_MLX SNAP_GTE_GGUF SNAP_NOMIC_GGUF SNAP_JINA SNAP_QWEN_QUANTS SNAP_QWEN_MLX_8BIT; do
  [ -n "${!v}" ] || { echo "MISSING snapshot: $v" >&2; fail=1; }
done

need "$SNAP_GGUF_EMBED/Qwen3-Embedding-0.6B-f16.gguf"
need "$SNAP_GGUF_EMBED/Qwen3-Embedding-0.6B-Q8_0.gguf"
need "$SNAP_GTE_MODERNBERT/onnx/model.onnx"
need "$SNAP_GTE_MODERNBERT/tokenizer.json"
need "$SNAP_GTE_MODERNBERT/config.json"
need "$SNAP_NOMIC_MODERNBERT/onnx/model.onnx"
need "$SNAP_NOMIC_MODERNBERT/tokenizer.json"
need "$SNAP_NOMIC_MODERNBERT/config.json"
need "$SNAP_NOMIC_MLX/config.json"
need "$SNAP_NOMIC_MLX/tokenizer.json"
need "$SNAP_NOMIC_MLX/model.safetensors.index.json"
need "$SNAP_GTE_GGUF/gte-modernbert-base-F16.gguf"
need "$SNAP_NOMIC_GGUF/modernbert-embed-base-Q8_0.gguf"
need "$SNAP_JINA/onnx/model.onnx"
need "$SNAP_JINA/onnx/model.onnx_data"
need "$SNAP_JINA/tokenizer.json"
need "$SNAP_JINA/config.json"
need "$SNAP_JINA/v5-nano-retrieval-F16.gguf"
need "$SNAP_JINA/v5-nano-retrieval-Q8_0.gguf"
need "$SNAP_QWEN_QUANTS/Qwen3-Embedding-0.6B.Q4_K_M.gguf"
need "$SNAP_QWEN_QUANTS/Qwen3-Embedding-0.6B.Q6_K.gguf"
need "$SNAP_QWEN_MLX_8BIT/config.json"
need "$SNAP_QWEN_MLX_8BIT/tokenizer.json"
need "$SNAP_QWEN_MLX_8BIT/model.safetensors.index.json"
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
  ./target/release/lane-llama embed --server-binary "$LLAMA_BIN" \
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
  "$MLX_PY" bench/lanes/mlx-minilm/main.py \
  --model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ \
  --corpus "$CORPUS" --out "$RESULTS/mlx-qwen-dwq-embed.json" \
  --vectors-out "$RESULTS/mlx-qwen-dwq-vectors.jsonl" \
  --model-label "Qwen3-Embedding-0.6B@mlx-4bit-dwq"

# Full-corpus DWQ quality report (cosine + top-k rank overlap vs fp32).
# Not idle-gated: pure math, no performance measurement.
$BENCH parity \
  --reference "$RESULTS/ort-cpu-embed-vectors.jsonl" \
  --candidate "$RESULTS/mlx-qwen-dwq-vectors.jsonl" \
  --k 10 --stride 50 > "$RESULTS/dwq-parity-report.json" || echo "dwq parity report failed" >&2

# --- Workload A floor: all-MiniLM-L6-v2 across every engine ------------------
run ort-cpu-minilm-embed \
  ./target/release/lane-ort-embed \
  --model "$SNAP_MINILM/model.onnx" --tokenizer "$SNAP_MINILM/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/ort-cpu-minilm-embed.json" \
  --vectors-out "$RESULTS/ort-cpu-minilm-embed-vectors.jsonl" \
  --pooling mean --max-length 512 \
  --model-label "all-MiniLM-L6-v2@ort-cpu-fp32"

run llama-metal-minilm-embed \
  ./target/release/lane-llama embed --server-binary "$LLAMA_BIN" \
  --model "$SNAP_MINILM_GGUF/all-MiniLM-L6-v2-ggml-model-f16.gguf" \
  --tokenizer "$SNAP_MINILM/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/llama-metal-minilm-embed.json" \
  --reference "$RESULTS/ort-cpu-minilm-embed-vectors.jsonl" \
  --pooling mean \
  --model-label "all-MiniLM-L6-v2@gguf-f16-metal"

run llama-cpu-minilm-embed \
  ./target/release/lane-llama embed --server-binary "$LLAMA_BIN" \
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
  --pooling mean --attention-units 16000000 \
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
  "$MLX_PY" bench/lanes/mlx-minilm/main.py \
  --corpus "$CORPUS" --out "$RESULTS/mlx-minilm-embed.json" \
  --model-label "all-MiniLM-L6-v2@mlx-bf16"

run potion-static-embed \
  "$POTION_PY" bench/lanes/potion/main.py \
  --corpus "$CORPUS" --out "$RESULTS/potion-static-embed.json" \
  --vectors-out "$RESULTS/potion-static-embed-vectors.jsonl" \
  --model-label "potion-code-16M@model2vec-static"

# --- Model matrix: ModernBERT-class + Jina + Qwen3 quant lanes ---------------
run ort-cpu-gte-modernbert-embed \
  ./target/release/lane-ort-embed \
  --model "$SNAP_GTE_MODERNBERT/onnx/model.onnx" --tokenizer "$SNAP_GTE_MODERNBERT/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/ort-cpu-gte-modernbert-embed.json" \
  --vectors-out "$RESULTS/ort-cpu-gte-modernbert-embed-vectors.jsonl" \
  --pooling cls --max-length 512 \
  --model-label "gte-modernbert-base@onnx-fp32"

run llama-metal-gte-modernbert-embed \
  ./target/release/lane-llama embed --server-binary "$LLAMA_BIN" \
  --model "$SNAP_GTE_GGUF/gte-modernbert-base-F16.gguf" \
  --tokenizer "$SNAP_GTE_MODERNBERT/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/llama-metal-gte-modernbert-embed.json" \
  --reference "$RESULTS/ort-cpu-gte-modernbert-embed-vectors.jsonl" \
  --pooling cls \
  --model-label "gte-modernbert-base@gguf-f16"

run ort-cpu-nomic-modernbert-embed \
  ./target/release/lane-ort-embed \
  --model "$SNAP_NOMIC_MODERNBERT/onnx/model.onnx" --tokenizer "$SNAP_NOMIC_MODERNBERT/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/ort-cpu-nomic-modernbert-embed.json" \
  --vectors-out "$RESULTS/ort-cpu-nomic-modernbert-embed-vectors.jsonl" \
  --pooling mean --max-length 512 --prefix-document "search_document: " \
  --model-label "modernbert-embed-base@onnx-fp32"

run llama-metal-nomic-modernbert-embed \
  ./target/release/lane-llama embed --server-binary "$LLAMA_BIN" \
  --model "$SNAP_NOMIC_GGUF/modernbert-embed-base-Q8_0.gguf" \
  --tokenizer "$SNAP_NOMIC_MODERNBERT/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/llama-metal-nomic-modernbert-embed.json" \
  --reference "$RESULTS/ort-cpu-nomic-modernbert-embed-vectors.jsonl" \
  --pooling mean --prefix-document "search_document: " \
  --model-label "modernbert-embed-base@gguf-q8_0"

run ort-cpu-jina-v5-nano-embed \
  ./target/release/lane-ort-embed \
  --model "$SNAP_JINA/onnx/model.onnx" --tokenizer "$SNAP_JINA/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/ort-cpu-jina-v5-nano-embed.json" \
  --vectors-out "$RESULTS/ort-cpu-jina-v5-nano-embed-vectors.jsonl" \
  --pooling last --max-length 512 \
  --model-label "jina-embeddings-v5-text-nano-retrieval@onnx-fp32"

run llama-metal-jina-v5-nano-embed \
  ./target/release/lane-llama embed --server-binary "$LLAMA_BIN" \
  --model "$SNAP_JINA/v5-nano-retrieval-F16.gguf" \
  --tokenizer "$SNAP_JINA/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/llama-metal-jina-v5-nano-embed.json" \
  --reference "$RESULTS/ort-cpu-jina-v5-nano-embed-vectors.jsonl" \
  --pooling last \
  --model-label "jina-embeddings-v5-text-nano-retrieval@gguf-f16"

run llama-metal-qwen-q4-k-m-embed \
  ./target/release/lane-llama embed --server-binary "$LLAMA_BIN" \
  --model "$SNAP_QWEN_QUANTS/Qwen3-Embedding-0.6B.Q4_K_M.gguf" \
  --tokenizer "$SNAP_MLX_EMBED/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/llama-metal-qwen-q4-k-m-embed.json" \
  --reference "$RESULTS/ort-cpu-embed-vectors.jsonl" \
  --min-parity 0.0 --pooling last \
  --model-label "Qwen3-Embedding-0.6B@gguf-q4_k_m"

run llama-metal-qwen-q6-k-embed \
  ./target/release/lane-llama embed --server-binary "$LLAMA_BIN" \
  --model "$SNAP_QWEN_QUANTS/Qwen3-Embedding-0.6B.Q6_K.gguf" \
  --tokenizer "$SNAP_MLX_EMBED/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/llama-metal-qwen-q6-k-embed.json" \
  --reference "$RESULTS/ort-cpu-embed-vectors.jsonl" \
  --min-parity 0.0 --pooling last \
  --model-label "Qwen3-Embedding-0.6B@gguf-q6_k"

run llama-metal-qwen-q8-0-embed \
  ./target/release/lane-llama embed --server-binary "$LLAMA_BIN" \
  --model "$SNAP_GGUF_EMBED/Qwen3-Embedding-0.6B-Q8_0.gguf" \
  --tokenizer "$SNAP_MLX_EMBED/tokenizer.json" \
  --corpus "$CORPUS" --out "$RESULTS/llama-metal-qwen-q8-0-embed.json" \
  --reference "$RESULTS/ort-cpu-embed-vectors.jsonl" \
  --min-parity 0.0 --pooling last \
  --model-label "Qwen3-Embedding-0.6B@gguf-q8_0"

run mlx-qwen-8bit-embed \
  "$MLX_PY" bench/lanes/mlx-minilm/main.py \
  --model "$SNAP_QWEN_MLX_8BIT" \
  --corpus "$CORPUS" --out "$RESULTS/mlx-qwen-8bit-embed.json" \
  --vectors-out "$RESULTS/mlx-qwen-8bit-vectors.jsonl" \
  --model-label "Qwen3-Embedding-0.6B@mlx-8bit"

# Full-corpus 8-bit MLX quality report (cosine + top-k rank overlap vs fp32).
# Not idle-gated: pure math, no performance measurement.
$BENCH parity \
  --reference "$RESULTS/ort-cpu-embed-vectors.jsonl" \
  --candidate "$RESULTS/mlx-qwen-8bit-vectors.jsonl" \
  --k 10 --stride 50 > "$RESULTS/mlx-qwen-8bit-parity-report.json" || echo "mlx qwen 8bit parity report failed" >&2

# --- Workload B: micro-LLM one-shots -----------------------------------------
run mlx-microllm \
  ./target/release/lane-mlx microllm \
  --model "$SNAP_MLX_MICROLLM" --tokenizer "$SNAP_MLX_MICROLLM/tokenizer.json" \
  --prompts "$PROMPTS" --out "$RESULTS/mlx-microllm.json"

run llama-metal-microllm \
  ./target/release/lane-llama microllm --server-binary "$LLAMA_BIN" \
  --model "$SNAP_GGUF_LLM/Qwen3-0.6B-Q8_0.gguf" \
  --prompts "$PROMPTS" --out "$RESULTS/llama-metal-microllm.json" \
  --model-label "Qwen3-0.6B@gguf-q8_0"

run llama-metal-microllm-lfm \
  ./target/release/lane-llama microllm --server-binary "$LLAMA_BIN" \
  --model "$SNAP_LFM/LFM2.5-230M-Q8_0.gguf" \
  --prompts "$PROMPTS" --out "$RESULTS/llama-metal-microllm-lfm.json" \
  --model-label "LFM2.5-230M@gguf-q8_0"

echo "night run complete ($(date +%H:%M:%S)); results in $RESULTS/"
