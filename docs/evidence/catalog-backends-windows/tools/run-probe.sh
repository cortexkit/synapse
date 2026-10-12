#!/bin/bash
# Run catalog-probe for one Vulkan cell against a loopback mirror of the
# byte-pinned files. Usage: run-probe.sh <model>
model=$1
base=/c/Users/ufuka/synapse-cert
cd "$base"
port=$((20000 + RANDOM % 20000))
python -m http.server "$port" --bind 127.0.0.1 --directory 'C:\Users\ufuka\synapse-cert\mirror' > "logs/probe.$model.mirror.log" 2>&1 &
server=$!
sleep 2
scratch="C:\\Users\\ufuka\\synapse-cert\\probe\\$model-$(date +%s)"
echo "=== probe $model $(date -u +%FT%TZ) scratch=$scratch"
CATALOG_PROBE_SKIP_8192=${SKIP:-} catalog-probe/target/release/catalog-probe.exe \
  'C:\Users\ufuka\synapse-cert\assets\windows-x64\ck-synapse.exe' \
  'C:\Users\ufuka\synapse-cert\assets\windows-x64\ck-synapse-worker-vulkan.exe' \
  'C:\Users\ufuka\synapse-cert\synapse' "$model" "$scratch" "http://127.0.0.1:$port" \
  > "logs/probe.$model${SKIP:+.skip8192}.jsonl" 2> "logs/probe.$model${SKIP:+.skip8192}.stderr.log"
echo "probe_exit=$? $(date -u +%FT%TZ)"
kill $server; wait $server 2>/dev/null
