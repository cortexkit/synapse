#!/usr/bin/env bash
# Serial, idle-gated bench matrix. Safe to start on a busy machine: each lane
# waits (up to WAIT_MAX) for the idle gate before measuring, so results are
# never contaminated by other workloads.
set -euo pipefail
cd "$(dirname "$0")/.."

BENCH=./target/release/synapse-bench
RESULTS=bench/results
# corpus-v2 = AFT's real chunk export (byte-exact embed_text, 15,271 chunks),
# converted from corpus/aft-chunks.jsonl. v1 (line-chunked) is integration-only.
CORPUS=bench/data/corpus-v2.jsonl
PROMPTS=bench/data/microllm-prompts-v1.jsonl
WAIT_MAX=${WAIT_MAX:-14400}   # max seconds to wait for idle per lane (4h)
WAIT_STEP=60

wait_for_idle_and_run() {
  local name="$1"; shift
  local waited=0
  echo "=== lane: $name ==="
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

find_snapshot() {
  local pattern="$1"
  local found
  found=$(ls -d $pattern 2>/dev/null | head -n 1 || true)
  if [ -z "$found" ]; then
    return 1
  fi
  printf '%s\n' "$found"
}

SNAP_ONNX=$HOME/.cache/huggingface/hub/models--onnx-community--Qwen3-Embedding-0.6B-ONNX/snapshots/c25a394dd583836952667c12f008335071b3f43d
SNAP_MLX_EMBED=$(find_snapshot "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*")
SNAP_MLX_MICROLLM=${SNAP_MLX_MICROLLM:-$(find_snapshot "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*" || true)}

# Lane 1: ort-cpu reference (workload A)
wait_for_idle_and_run ort-cpu-embed \
  ./target/release/lane-ort-embed \
  --model "$SNAP_ONNX/onnx/model.onnx" \
  --tokenizer "$SNAP_ONNX/tokenizer.json" \
  --corpus "$CORPUS" \
  --out "$RESULTS/ort-cpu-embed.json" \
  --vectors-out "$RESULTS/ort-cpu-embed-vectors.jsonl" \
  --pooling last --max-length 512 \
  --model-label "Qwen3-Embedding-0.6B@onnx-fp32"

# Lane 2: mlx-rs / Metal embedding (workload A)
wait_for_idle_and_run mlx-embed \
  ./target/release/lane-mlx embed \
  --model "$SNAP_MLX_EMBED" \
  --tokenizer "$SNAP_MLX_EMBED/tokenizer.json" \
  --corpus "$CORPUS" \
  --out "$RESULTS/mlx-embed.json" \
  --vectors-out "$RESULTS/mlx-embed-vectors.jsonl" \
  --reference "$RESULTS/ort-cpu-embed-vectors.jsonl" \
  --model-label "Qwen3-Embedding-0.6B@mlx-bf16"

# Lane 3: mlx-rs / Metal micro-LLM one-shot (workload B).
# Leave SNAP_MLX_MICROLLM empty to skip until the bf16 safetensors snapshot is cached.
if [ -n "$SNAP_MLX_MICROLLM" ]; then
  wait_for_idle_and_run mlx-microllm \
    ./target/release/lane-mlx microllm \
    --model "$SNAP_MLX_MICROLLM" \
    --tokenizer "$SNAP_MLX_MICROLLM/tokenizer.json" \
    --prompts "$PROMPTS" \
    --out "$RESULTS/mlx-microllm.json"
else
  echo "skip mlx-microllm: cache Qwen/Qwen3-0.6B bf16 safetensors and set SNAP_MLX_MICROLLM if auto-detect does not find it" >&2
fi

SNAP_GGUF_EMBED=$(find_snapshot "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/*")
SNAP_GGUF_LLM=$(find_snapshot "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots/*")
SNAP_MINILM=$(find_snapshot "$HOME/.cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/*")

# Lane 4: llama-server / Metal embedding (workload A)
wait_for_idle_and_run llama-metal-embed \
  ./target/release/lane-llama embed \
  --model "$SNAP_GGUF_EMBED/Qwen3-Embedding-0.6B-f16.gguf" \
  --tokenizer "$SNAP_MLX_EMBED/tokenizer.json" \
  --corpus "$CORPUS" \
  --out "$RESULTS/llama-metal-embed.json" \
  --vectors-out "$RESULTS/llama-metal-embed-vectors.jsonl" \
  --reference "$RESULTS/ort-cpu-embed-vectors.jsonl" \
  --model-label "Qwen3-Embedding-0.6B@gguf-f16"

# Lane 5: llama-server / Metal micro-LLM one-shot (workload B)
wait_for_idle_and_run llama-metal-microllm \
  ./target/release/lane-llama microllm \
  --model "$SNAP_GGUF_LLM/Qwen3-0.6B-Q8_0.gguf" \
  --prompts "$PROMPTS" \
  --out "$RESULTS/llama-metal-microllm.json" \
  --model-label "Qwen3-0.6B@gguf-q8_0"

# Lane 5b: llama-server / Metal micro-LLM, LFM2.5-230M (tok/s comparison point
# for the micro-LLM class; same prompts/contract as 5)
SNAP_LFM=$(find_snapshot "$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-230M-GGUF/snapshots/*")
wait_for_idle_and_run llama-metal-microllm-lfm \
  ./target/release/lane-llama microllm \
  --model "$SNAP_LFM/LFM2.5-230M-Q8_0.gguf" \
  --prompts "$PROMPTS" \
  --out "$RESULTS/llama-metal-microllm-lfm.json" \
  --model-label "LFM2.5-230M@gguf-q8_0"

# Lane 6a: ort-cpu MiniLM (floor pair for burn, which could not import Qwen3)
wait_for_idle_and_run ort-cpu-minilm-embed \
  ./target/release/lane-ort-embed \
  --model "$SNAP_MINILM/model.onnx" \
  --tokenizer "$SNAP_MINILM/tokenizer.json" \
  --corpus "$CORPUS" \
  --out "$RESULTS/ort-cpu-minilm-embed.json" \
  --vectors-out "$RESULTS/ort-cpu-minilm-embed-vectors.jsonl" \
  --pooling mean --max-length 512 \
  --model-label "all-MiniLM-L6-v2@ort-cpu-fp32"

# Lane 6b: burn wgpu/Metal MiniLM (compare against 6a, same model)
wait_for_idle_and_run burn-wgpu-embed \
  ./target/release/lane-burn \
  --model "$SNAP_MINILM/model.onnx" \
  --tokenizer "$SNAP_MINILM/tokenizer.json" \
  --corpus "$CORPUS" \
  --out "$RESULTS/burn-wgpu-embed.json" \
  --reference "$RESULTS/ort-cpu-minilm-embed-vectors.jsonl" \
  --pooling mean \
  --model-label "all-MiniLM-L6-v2@burn-wgpu-f32"

# Lane 7: wrapped LMStudio (workload A). Requires LMStudio running with
# the qwen3 embedding model loaded; skipped when unreachable.
if curl -sf http://127.0.0.1:1234/v1/models >/dev/null 2>&1; then
  wait_for_idle_and_run wrap-lmstudio-embed \
    ./target/release/lane-wrap-embed \
    --base-url http://127.0.0.1:1234 \
    --model text-embedding-qwen3-embedding-0.6b \
    --lane wrap-lmstudio \
    --corpus "$CORPUS" \
    --out "$RESULTS/wrap-lmstudio-embed.json" \
    --reference "$RESULTS/ort-cpu-embed-vectors.jsonl" \
    --tokenizer "$SNAP_MLX_EMBED/tokenizer.json" \
    --rss-process-names "LM Studio Helper,lms" \
    --model-label "Qwen3-Embedding-0.6B@lmstudio"
else
  echo "skip wrap-lmstudio-embed: LMStudio not reachable on :1234" >&2
fi

echo "matrix complete; results in $RESULTS/"
