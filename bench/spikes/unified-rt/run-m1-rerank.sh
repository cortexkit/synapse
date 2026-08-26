#!/bin/zsh
set -euo pipefail

ROOT=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/bench-tools/unified-rt-serving
BIN="$ROOT/bin/spike-unified-rt"
DATA="$ROOT/data"
RESULTS="$ROOT/results/m1-rerank"
PACKAGES="$ROOT/packages/m1-rerank"
MACMON=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/bench-tools/bin/macmon
LOCK=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/bench.lock

MODEL=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/.cache/huggingface/hub/models--Alibaba-NLP--gte-reranker-modernbert-base/snapshots/f7481e6055501a30fb19d090657df9ec1f79ab2c
REQUESTS="$DATA/cosqa-rerank-1x50-repeated.jsonl"
REFERENCE="$DATA/cosqa-rerank-1x50-repeated-reference.jsonl"

mkdir -p "$RESULTS" "$PACKAGES"

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

run_rerank() {
  local stem=$1
  local limit=$2
  local command=(
    "$BIN"
    --model "$MODEL"
    --tokenizer "$MODEL/tokenizer.json"
    --rerank-requests "$REQUESTS"
    --reference "$REFERENCE"
    --scores-out "$RESULTS/$stem-scores.jsonl"
    --out "$RESULTS/$stem.json"
    --limit "$limit"
    --dtype f32
    --device metal
    --execution explicit
    --package-cache "$PACKAGES"
    --shapes bucketed
    --bucket-policy 1
    --max-length 512
    --model-label "gte-reranker-modernbert-f32-m1-$stem"
  )
  log_command "$RESULTS/$stem.log" "${command[@]}"
  "${command[@]}" >>"$RESULTS/$stem.log" 2>&1
}

run_prime() {
  (
    local stem=prime
    rm -rf "$PACKAGES"
    mkdir -p "$PACKAGES"
    acquire_lock
    trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT INT TERM HUP
    run_rerank "$stem" 20
  )
}

run_powered() {
  (
    local repeat=$1
    local stem="hit-r$repeat"
    local macmon_pid=""

    acquire_lock
    trap '[[ -n "$macmon_pid" ]] && kill "$macmon_pid" 2>/dev/null || true; rmdir "$LOCK" 2>/dev/null || true' EXIT INT TERM HUP
    : >"$RESULTS/$stem-macmon.jsonl"
    "$MACMON" -i 100 pipe >>"$RESULTS/$stem-macmon.jsonl" 2>"$RESULTS/$stem-macmon.log" &
    macmon_pid=$!
    local monitor_ready=0
    for _ in {1..100}; do
      if [[ -s "$RESULTS/$stem-macmon.jsonl" ]]; then
        monitor_ready=1
        break
      fi
      sleep 0.1
    done
    (( monitor_ready == 1 )) || {
      print -u2 "macmon did not produce a sample within 10 seconds"
      exit 1
    }
    sleep 2

    python3 -c 'import time; print(time.time())' >"$RESULTS/$stem-start-epoch.txt"
    local exit_code=0
    run_rerank "$stem" 20 || exit_code=$?
    python3 -c 'import time; print(time.time())' >"$RESULTS/$stem-end-epoch.txt"
    kill "$macmon_pid" 2>/dev/null || true
    wait "$macmon_pid" 2>/dev/null || true
    macmon_pid=""
    rmdir "$LOCK"
    trap - EXIT INT TERM HUP
    (( exit_code == 0 )) || return "$exit_code"
  )
}

for prerequisite in "$BIN" "$MACMON" "$MODEL/model.safetensors" "$MODEL/tokenizer.json" "$REQUESTS" "$REFERENCE"; do
  [[ -e "$prerequisite" ]] || {
    print -u2 "missing prerequisite: $prerequisite"
    exit 1
  }
done

run_prime
run_powered 1
run_powered 2
