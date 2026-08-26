#!/bin/zsh
set -euo pipefail

ROOT=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/bench-tools/unified-rt-serving
BIN="$ROOT/bin/spike-unified-rt"
DATA="$ROOT/data"
RESULTS="$ROOT/results"
PACKAGES="$ROOT/packages"
LOCK=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/bench.lock

MINILM=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
MODERNBERT=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/.cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots/e7f32e3c00f91d699e8c43b53106206bcc72bb22
QWEN3=${SYNAPSE_BENCH_ROOT:?set SYNAPSE_BENCH_ROOT}/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3

mkdir -p "$RESULTS" "$PACKAGES"

run_locked() {
  local family=$1
  local dtype=$2
  local cache_state=$3
  local repetition=$4
  local model=$5
  local corpus=$6
  local reference=$7
  local package_dir="$PACKAGES/$family-$dtype"
  local stem="$RESULTS/$family-$dtype-$cache_state-run$repetition"

  (
    while true; do
      if ! mkdir "$LOCK" 2>/dev/null; then
        echo "benchmark lock busy; retrying in 5 minutes" >&2
        sleep 300
        continue
      fi
      if pgrep -f Runner.Worker >/dev/null; then
        echo "CI Runner.Worker active; releasing lock and retrying in 5 minutes" >&2
        rmdir "$LOCK"
        sleep 300
        continue
      fi
      break
    done
    trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT INT TERM HUP

    if [[ "$cache_state" == miss ]]; then
      rm -rf "$package_dir"
      mkdir -p "$package_dir"
    fi

    print -r -- "+ $BIN --model $model --tokenizer $model/tokenizer.json --corpus $corpus --reference $reference --limit 400 --out $stem.json --dtype $dtype --device metal --package-cache $package_dir --model-label $family-$dtype-m1-$cache_state-run$repetition"
    /usr/bin/time -p "$BIN" \
      --model "$model" \
      --tokenizer "$model/tokenizer.json" \
      --corpus "$corpus" \
      --reference "$reference" \
      --limit 400 \
      --out "$stem.json" \
      --dtype "$dtype" \
      --device metal \
      --package-cache "$package_dir" \
      --model-label "$family-$dtype-m1-$cache_state-run$repetition"
  ) >"$stem.log" 2>&1
}

run_cell() {
  local family=$1
  local model=$2
  local corpus=$3
  local reference=$4
  local dtype
  for dtype in f32 f16; do
    run_locked "$family" "$dtype" miss 1 "$model" "$corpus" "$reference"
    run_locked "$family" "$dtype" miss 2 "$model" "$corpus" "$reference"
    run_locked "$family" "$dtype" hit 1 "$model" "$corpus" "$reference"
    run_locked "$family" "$dtype" hit 2 "$model" "$corpus" "$reference"
  done
}

if [[ "${SKIP_MINILM:-0}" != 1 ]]; then
  run_cell minilm "$MINILM" "$DATA/minilm-corpus-1000.jsonl" "$DATA/ort-minilm-1000-vectors.jsonl"
fi
run_cell gte-modernbert "$MODERNBERT" "$DATA/modernbert-corpus-400.jsonl" "$DATA/modernbert-ort-400-vectors.jsonl"
run_cell qwen3-embedding-0.6b "$QWEN3" "$DATA/qwen3-corpus-400.jsonl" "$DATA/qwen3-ort-400-vectors.jsonl"
