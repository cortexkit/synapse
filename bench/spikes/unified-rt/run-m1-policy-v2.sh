#!/bin/zsh
set -euo pipefail

ROOT=[bench-user-home]/bench-tools/unified-rt-serving
BIN="$ROOT/bin/spike-unified-rt"
DATA="$ROOT/data"
RESULTS="$ROOT/results/m1-policy-v2"
PACKAGES="$ROOT/packages/m1-policy-v2"
MACMON=[bench-user-home]/bench-tools/bin/macmon
LOCK=[bench-user-home]/bench.lock

MINILM=[bench-user-home]/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
MODERNBERT="$ROOT/models/gte-modernbert-base-embedding"
QWEN3=[bench-user-home]/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3

mkdir -p "$RESULTS" "$PACKAGES"

inventory() {
  local package_dir=$1
  local output=$2
  if [[ -d "$package_dir" ]]; then
    find "$package_dir" -exec stat -f '%N\t%m\t%z' {} + | LC_ALL=C sort >"$output"
  else
    : >"$output"
  fi
}

package_count() {
  local package_dir=$1
  if [[ -d "$package_dir" ]]; then
    find "$package_dir" -type d -name '*.mpsgraphpackage' | wc -l | tr -d ' '
  else
    print 0
  fi
}

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

run_locked() {
  local family=$1
  local dtype=$2
  local cache_state=$3
  local model=$4
  local corpus=$5
  local reference=$6
  local limit=$7
  local package_dir="$PACKAGES/$family-bucketed-v2"
  local stem="$RESULTS/$family-bucketed-v2-$cache_state"

  (
    acquire_lock
    local macmon_pid=""
    trap '[[ -n "$macmon_pid" ]] && kill "$macmon_pid" 2>/dev/null || true; rmdir "$LOCK" 2>/dev/null || true' EXIT INT TERM HUP

    if [[ "$cache_state" == miss ]]; then
      rm -rf "$package_dir"
      mkdir -p "$package_dir"
    elif [[ ! -d "$package_dir" ]]; then
      print -u2 "HIT requested without package directory: $package_dir"
      exit 1
    fi

    local before="$stem-packages-before.tsv"
    local after="$stem-packages-after.tsv"
    local current="$stem-packages-current.tsv"
    local previous="$stem-packages-previous.tsv"
    local stable=0

    if [[ "$cache_state" == hit ]]; then
      inventory "$package_dir" "$before"
    fi

    print -r -- "+ $BIN --model $model --tokenizer $model/tokenizer.json --corpus $corpus --reference $reference --limit $limit --out $stem.json --dtype $dtype --device metal --package-cache $package_dir --shapes bucketed --bucket-policy 2 --passes 3 --model-label $family-$dtype-m1-bucketed-v2-$cache_state" >"$stem.log"
    : >"$stem-macmon.jsonl"
    "$MACMON" -i 100 pipe >>"$stem-macmon.jsonl" 2>"$stem-macmon.log" &
    macmon_pid=$!
    local monitor_ready=0
    for _ in {1..100}; do
      if [[ -s "$stem-macmon.jsonl" ]]; then
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

    python3 -c 'import time; print(time.time())' >"$stem-start-epoch.txt"
    /usr/bin/time -p "$BIN" \
      --model "$model" \
      --tokenizer "$model/tokenizer.json" \
      --corpus "$corpus" \
      --reference "$reference" \
      --limit "$limit" \
      --out "$stem.json" \
      --dtype "$dtype" \
      --device metal \
      --package-cache "$package_dir" \
      --shapes bucketed \
      --bucket-policy 2 \
      --passes 3 \
      --model-label "$family-$dtype-m1-bucketed-v2-$cache_state" \
      >>"$stem.log" 2>&1 &
    local benchmark_pid=$!

    if [[ "$cache_state" == miss ]]; then
      : >"$previous"
      while kill -0 "$benchmark_pid" 2>/dev/null; do
        inventory "$package_dir" "$current"
        if [[ "$(package_count "$package_dir")" == 10 ]] && cmp -s "$current" "$previous"; then
          stable=$((stable + 1))
          if (( stable >= 3 )); then
            cp "$current" "$before"
            break
          fi
        else
          stable=0
        fi
        cp "$current" "$previous"
        sleep 0.1
      done
    fi

    local benchmark_status=0
    wait "$benchmark_pid" || benchmark_status=$?
    python3 -c 'import time; print(time.time())' >"$stem-end-epoch.txt"
    kill "$macmon_pid" 2>/dev/null || true
    wait "$macmon_pid" 2>/dev/null || true
    macmon_pid=""
    if (( benchmark_status != 0 )); then
      if [[ -f "$stem.json" ]] && python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["padding_waste_gate_passed"] is not False)' "$stem.json"; then
        print 'passed=false' >"$stem-padding-waste-gate.txt"
      else
        exit "$benchmark_status"
      fi
    else
      print 'passed=true' >"$stem-padding-waste-gate.txt"
    fi

    [[ -f "$before" ]] || {
      print -u2 "failed to capture package inventory before inference"
      exit 1
    }
    inventory "$package_dir" "$after"
    if cmp -s "$before" "$after"; then
      print 'unchanged=true' >"$stem-package-mutation.txt"
    else
      print 'unchanged=false' >"$stem-package-mutation.txt"
      diff -u "$before" "$after" >>"$stem-package-mutation.txt" || true
      exit 1
    fi
    rm -f "$current" "$previous"
  )
}

run_locked minilm f16 miss "$MINILM" "$DATA/minilm-corpus-1000.jsonl" "$DATA/ort-minilm-1000-vectors.jsonl" 400
run_locked minilm f16 hit "$MINILM" "$DATA/minilm-corpus-1000.jsonl" "$DATA/ort-minilm-1000-vectors.jsonl" 400
run_locked gte-modernbert f32 miss "$MODERNBERT" "$DATA/modernbert-corpus-400.jsonl" "$DATA/modernbert-ort-400-vectors.jsonl" 400
run_locked gte-modernbert f32 hit "$MODERNBERT" "$DATA/modernbert-corpus-400.jsonl" "$DATA/modernbert-ort-400-vectors.jsonl" 400
run_locked qwen3-embedding-0.6b f16 miss "$QWEN3" "$DATA/qwen3-corpus-400.jsonl" "$DATA/qwen3-ort-400-vectors.jsonl" 400
run_locked qwen3-embedding-0.6b f16 hit "$QWEN3" "$DATA/qwen3-corpus-400.jsonl" "$DATA/qwen3-ort-400-vectors.jsonl" 400
run_locked gte-modernbert-mc f32 miss "$MODERNBERT" "$DATA/mc-corpus.jsonl" "$DATA/mc-exact-vectors.jsonl" 11293
run_locked gte-modernbert-mc f32 hit "$MODERNBERT" "$DATA/mc-corpus.jsonl" "$DATA/mc-exact-vectors.jsonl" 11293
