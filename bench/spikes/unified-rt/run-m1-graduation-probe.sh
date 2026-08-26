#!/bin/zsh
set -euo pipefail

ROOT=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/bench-tools/graduation-probe
BIN="$ROOT/bin/spike-unified-rt"
LLAMA_LANE="$ROOT/bin/lane-llama"
LLAMA_SERVER=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/bench-tools/bin/llama-server-wrap.sh
PARITY="$ROOT/bin/synapse-bench"
MLX_PY="$ROOT/venvs/mlx-embeddings/bin/python"
MLX_SCRIPT="$ROOT/scripts/mlx-embed.py"
DATA=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/bench-tools/unified-rt-serving/data
RESULTS="$ROOT/results/raw"
PACKAGES="$ROOT/packages"
MACMON=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/bench-tools/bin/macmon
LOCK=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/bench.lock
MINILM_CORPUS="$ROOT/data/minilm-corpus-400.jsonl"

MINILM=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
MINILM_MLX=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/.cache/huggingface/hub/models--mlx-community--all-MiniLM-L6-v2-bf16/snapshots/b6691709eacd8f0afcc3faace288cf50e611f3aa
MINILM_GGUF=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/.cache/huggingface/hub/models--second-state--All-MiniLM-L6-v2-Embedding-GGUF/snapshots/544f204f2eaa2d71361ffc74d6df7170285b286a/all-MiniLM-L6-v2-ggml-model-f16.gguf
MODERNBERT=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/.cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots/e7f32e3c00f91d699e8c43b53106206bcc72bb22
MODERNBERT_GGUF=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/.cache/huggingface/hub/models--keisuke-miyako--gte-modernbert-base-gguf/snapshots/529959733131e37a8282ef5e03a35d185d236b55/gte-modernbert-base-F16.gguf
QWEN3=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
QWEN3_GGUF=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-f16.gguf

mkdir -p "$RESULTS" "$PACKAGES" "$ROOT/data"
/usr/bin/head -n 400 "$DATA/minilm-corpus-1000.jsonl" >"$MINILM_CORPUS"

for prerequisite in "$BIN" "$LLAMA_LANE" "$LLAMA_SERVER" "$PARITY" "$MLX_PY" "$MLX_SCRIPT" \
  "$MACMON" "$MINILM/model.safetensors" "$MINILM/tokenizer.json" "$MINILM_MLX/model.safetensors" \
  "$MINILM_GGUF" "$MODERNBERT/model.safetensors" "$MODERNBERT/tokenizer.json" "$MODERNBERT_GGUF" \
  "$QWEN3/model.safetensors" "$QWEN3/tokenizer.json" "$QWEN3_GGUF" \
  "$MINILM_CORPUS" "$DATA/ort-minilm-1000-vectors.jsonl" \
  "$DATA/modernbert-corpus-400.jsonl" "$DATA/modernbert-ort-400-vectors.jsonl" \
  "$DATA/qwen3-corpus-400.jsonl" "$DATA/qwen3-ort-400-vectors.jsonl"; do
  [[ -e "$prerequisite" ]] || { print -u2 "missing prerequisite: $prerequisite"; exit 1; }
done

acquire_lock() {
  while true; do
    if ! mkdir "$LOCK" 2>/dev/null; then
      print -u2 "benchmark lock busy; retrying in 5 minutes"
      sleep 300
      continue
    fi
    if pgrep -f Runner.Worker >/dev/null; then
      print -u2 "CI Runner.Worker active; releasing lock and retrying in 5 minutes"
      rmdir "$LOCK"
      sleep 300
      continue
    fi
    break
  done
}

log_command() {
  local log=$1
  shift
  printf '+ ' >"$log"
  printf '%q ' "$@" >>"$log"
  printf '\n' >>"$log"
}

run_locked() {
  local stem=$1
  shift
  local log="$RESULTS/$stem.log"
  acquire_lock
  trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT INT TERM HUP
  log_command "$log" "$@"
  "$@" >>"$log" 2>&1
  rmdir "$LOCK"
  trap - EXIT INT TERM HUP
}

run_powered() {
  local stem=$1
  shift
  local log="$RESULTS/$stem.log"
  local macmon_log="$RESULTS/$stem-macmon.jsonl"
  local macmon_err="$RESULTS/$stem-macmon.log"
  local macmon_pid=""

  acquire_lock
  trap '[[ -n "$macmon_pid" ]] && kill "$macmon_pid" 2>/dev/null || true; rmdir "$LOCK" 2>/dev/null || true' EXIT INT TERM HUP
  log_command "$log" "$@"
  : >"$macmon_log"
  "$MACMON" -i 100 pipe >>"$macmon_log" 2>"$macmon_err" &
  macmon_pid=$!
  local ready=0
  for _ in {1..100}; do
    if [[ -s "$macmon_log" ]]; then
      ready=1
      break
    fi
    sleep 0.1
  done
  (( ready == 1 )) || { print -u2 "macmon did not produce a sample for $stem"; exit 1; }

  python3 -c 'import time; print(time.time())' >"$RESULTS/$stem-start-epoch.txt"
  local exit_code=0
  "$@" >>"$log" 2>&1 || exit_code=$?
  python3 -c 'import time; print(time.time())' >"$RESULTS/$stem-end-epoch.txt"
  kill "$macmon_pid" 2>/dev/null || true
  wait "$macmon_pid" 2>/dev/null || true
  macmon_pid=""
  rmdir "$LOCK"
  trap - EXIT INT TERM HUP
  (( exit_code == 0 )) || return "$exit_code"
}

owned_family() {
  local family=$1
  local dtype=$2
  local model=$3
  local corpus=$4
  local reference=$5
  local package_dir="$PACKAGES/$family-exact"

  rm -rf "$package_dir"
  mkdir -p "$package_dir"
  run_locked "owned-$family-prime" \
    "$BIN" --model "$model" --tokenizer "$model/tokenizer.json" --corpus "$corpus" \
    --reference "$reference" --limit 400 --out "$RESULTS/owned-$family-prime.json" \
    --dtype "$dtype" --device metal --package-cache "$package_dir" --shapes exact --passes 1 \
    --model-label "$family-owned-$dtype-exact-prime"

  for repeat in 1 2; do
    run_powered "owned-$family-hit-r$repeat" \
      "$BIN" --model "$model" --tokenizer "$model/tokenizer.json" --corpus "$corpus" \
      --reference "$reference" --limit 400 --out "$RESULTS/owned-$family-hit-r$repeat.json" \
      --dtype "$dtype" --device metal --package-cache "$package_dir" --shapes exact --passes 1 \
      --model-label "$family-owned-$dtype-exact-hit-r$repeat"
  done

  run_powered "owned-$family-steady" \
    "$BIN" --model "$model" --tokenizer "$model/tokenizer.json" --corpus "$corpus" \
    --reference "$reference" --limit 400 --out "$RESULTS/owned-$family-steady.json" \
    --dtype "$dtype" --device metal --package-cache "$package_dir" --shapes exact --passes 3 \
    --model-label "$family-owned-$dtype-exact-steady"
}

llama_family() {
  local family=$1
  local model=$2
  local tokenizer=$3
  local corpus=$4
  local reference=$5
  local pooling=$6

  for repeat in 1 2; do
    local stem="llama-$family-r$repeat"
    run_powered "$stem" \
      "$LLAMA_LANE" embed --server-binary "$LLAMA_SERVER" --model "$model" --tokenizer "$tokenizer" \
      --corpus "$corpus" --out "$RESULTS/$stem.json" --vectors-out "$RESULTS/$stem-vectors.jsonl" \
      --reference "$reference" --pooling "$pooling" --model-label "$family-llama-metal-f16-r$repeat"
    "$PARITY" parity --reference "$reference" --candidate "$RESULTS/$stem-vectors.jsonl" \
      --k 10 --stride 1 >"$RESULTS/$stem-parity.json"
  done
}

mlx_family() {
  local family=$1
  local model=$2
  local corpus=$3
  local reference=$4

  for repeat in 1 2; do
    local stem="mlx-$family-r$repeat"
    run_powered "$stem" \
      "$MLX_PY" "$MLX_SCRIPT" --model "$model" --corpus "$corpus" --limit 400 \
      --out "$RESULTS/$stem.json" --vectors-out "$RESULTS/$stem-vectors.jsonl" \
      --model-label "$family-mlx-bf16-r$repeat"
    "$PARITY" parity --reference "$reference" --candidate "$RESULTS/$stem-vectors.jsonl" \
      --k 10 --stride 1 >"$RESULTS/$stem-parity.json"
  done
}

if [[ "${ONLY_LLAMA_MINILM:-0}" == 1 ]]; then
  llama_family minilm "$MINILM_GGUF" "$MINILM/tokenizer.json" "$MINILM_CORPUS" "$DATA/ort-minilm-1000-vectors.jsonl" mean
  exit 0
fi

owned_family minilm f16 "$MINILM" "$MINILM_CORPUS" "$DATA/ort-minilm-1000-vectors.jsonl"
llama_family minilm "$MINILM_GGUF" "$MINILM/tokenizer.json" "$MINILM_CORPUS" "$DATA/ort-minilm-1000-vectors.jsonl" mean
mlx_family minilm "$MINILM_MLX" "$MINILM_CORPUS" "$DATA/ort-minilm-1000-vectors.jsonl"

owned_family qwen3 f16 "$QWEN3" "$DATA/qwen3-corpus-400.jsonl" "$DATA/qwen3-ort-400-vectors.jsonl"
llama_family qwen3 "$QWEN3_GGUF" "$QWEN3/tokenizer.json" "$DATA/qwen3-corpus-400.jsonl" "$DATA/qwen3-ort-400-vectors.jsonl" last
mlx_family qwen3 "$QWEN3" "$DATA/qwen3-corpus-400.jsonl" "$DATA/qwen3-ort-400-vectors.jsonl"

owned_family gte-modernbert f32 "$MODERNBERT" "$DATA/modernbert-corpus-400.jsonl" "$DATA/modernbert-ort-400-vectors.jsonl"
llama_family gte-modernbert "$MODERNBERT_GGUF" "$MODERNBERT/tokenizer.json" "$DATA/modernbert-corpus-400.jsonl" "$DATA/modernbert-ort-400-vectors.jsonl" cls
