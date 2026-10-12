#!/bin/bash
# Run one Windows Vulkan certification cell exactly as the release instructions
# describe, with the model cache pinned to a fresh per-cell directory.
# Usage: run-cell.sh <model>
model=$1
base=/c/Users/ufuka/synapse-cert
cd "$base/synapse"
cache="C:\\Users\\ufuka\\synapse-cert\\model-cache\\$model-$(date +%s)"
echo "=== vulkan-windows-amd $model $(date -u +%FT%TZ) cache=$cache"
start=$(date +%s.%N)
CORTEXKIT_MODEL_CACHE="$cache" target/release/ckdev-synapse-certify.exe run \
  --row vulkan-windows-amd --model "$model" \
  --assets 'C:\Users\ufuka\synapse-cert\assets' \
  --checkout 'C:\Users\ufuka\synapse-cert\synapse' \
  --mirror 'C:\Users\ufuka\synapse-cert\mirror' \
  > "$base/logs/vulkan-windows-amd.$model.stdout.json" 2> "$base/logs/vulkan-windows-amd.$model.stderr.log"
status=$?
end=$(date +%s.%N)
echo "exit=$status seconds=$(python -c "print(round($end-$start,1))") $(date -u +%FT%TZ)"
