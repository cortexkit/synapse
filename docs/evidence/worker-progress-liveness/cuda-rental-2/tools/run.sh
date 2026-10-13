#!/bin/bash
set -euo pipefail
source /root/env.sh
export CK_LOG=info
mkdir -p /root/raw /root/records
models=(gte-modernbert-base gte-reranker-modernbert-base qwen3-embedding-0.6b qwen3-reranker-0.6b)
for revision in progress baseline; do
  if [[ "$revision" == baseline ]]; then checkout=/root/baseline; rows=(cuda-linux-nvidia); else checkout=/root/synapse; rows=(cuda-linux-nvidia vulkan-linux-nvidia); fi
  for row in "${rows[@]}"; do
    for model in "${models[@]}"; do
      name="$revision.$row.$model"
      printf 'CERTIFY %s %s\n' "$name" "$(date -u +%FT%TZ)"
      set +e
      CORTEXKIT_MODEL_CACHE="/root/cache-$name" "/root/runner-$revision" run --row "$row" --model "$model" --assets "/root/assets-$revision" --checkout "$checkout" --mirror /root/mirror > "/root/raw/$name.stdout.json" 2> "/root/raw/$name.stderr.log"
      rc=$?
      set -e
      printf '%s\n' "$rc" > "/root/raw/$name.exit"
      printf 'CERTIFY_RESULT %s %s\n' "$name" "$rc"
    done
  done
done
python3 -m http.server 8765 --bind 127.0.0.1 --directory /root/mirror > /root/raw/timing-mirror.log 2>&1 &
mirror_pid=$!
trap 'kill "$mirror_pid" 2>/dev/null || true' EXIT
for backend in cuda vulkan; do
  for model in "${models[@]}"; do
    if [[ "$backend" == cuda ]]; then revisions=(baseline progress); else revisions=(progress); fi
    for revision in "${revisions[@]}"; do
      name="$revision.$backend.$model"
      printf 'TIMING %s %s\n' "$name" "$(date -u +%FT%TZ)"
      set +e
      /root/build-target-timing/release/timing-probe "/root/assets-$revision/linux-x64/ck-synapse" "/root/assets-$revision/linux-x64/ck-synapse-worker-$backend" "/root/synapse/bench/parity/fixtures/$model/$model.ref-v1.transformers-5.16.1.seed-0.json" "$model" "/root/timing-$name" http://127.0.0.1:8765 6 "$backend" 512 8192 > "/root/raw/timing.$name.jsonl" 2> "/root/raw/timing.$name.stderr.log"
      rc=$?
      set -e
      printf '%s\n' "$rc" > "/root/raw/timing.$name.exit"
      printf 'TIMING_RESULT %s %s\n' "$name" "$rc"
    done
  done
done
printf 'RUNS_DONE\n'
