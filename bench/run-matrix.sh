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

# Further lanes are appended as their binaries land:
# - llama-metal-embed (A) + llama-metal-microllm (B)
# - burn-wgpu-embed (A)
# - wrap-lmstudio-embed (A), wrap-ollama-embed (A)

echo "matrix complete; results in $RESULTS/"
