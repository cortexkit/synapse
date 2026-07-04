#!/usr/bin/env bash
# Serial, idle-gated bench matrix. Safe to start on a busy machine: each lane
# waits (up to WAIT_MAX) for the idle gate before measuring, so results are
# never contaminated by other workloads.
set -euo pipefail
cd "$(dirname "$0")/.."

BENCH=./target/release/synapse-bench
RESULTS=bench/results
CORPUS=bench/data/corpus-v1.jsonl
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

SNAP_ONNX=$HOME/.cache/huggingface/hub/models--onnx-community--Qwen3-Embedding-0.6B-ONNX/snapshots/c25a394dd583836952667c12f008335071b3f43d

# Lane 1: ort-cpu reference (workload A)
wait_for_idle_and_run ort-cpu-embed \
  ./target/release/lane-ort-embed \
  --model "$SNAP_ONNX/onnx/model.onnx" \
  --tokenizer "$SNAP_ONNX/tokenizer.json" \
  --corpus "$CORPUS" \
  --out "$RESULTS/ort-cpu-embed.json" \
  --pooling last --max-length 512 \
  --model-label "Qwen3-Embedding-0.6B@onnx-fp32"

# Further lanes are appended as their binaries land:
# - mlx-embed (workload A) + mlx-microllm (workload B)
# - llama-metal-embed (A) + llama-metal-microllm (B)
# - burn-wgpu-embed (A)
# - wrap-lmstudio-embed (A), wrap-ollama-embed (A)

echo "matrix complete; results in $RESULTS/"
