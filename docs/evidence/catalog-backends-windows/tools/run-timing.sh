#!/bin/bash
# Time single fixture rows on one Vulkan lane through the catalog path.
# Usage: run-timing.sh <model> <repeats> <len>...
model=$1; shift
base=/c/Users/ufuka/synapse-cert
cd "$base"
port=$((20000 + RANDOM % 20000))
python -m http.server "$port" --bind 127.0.0.1 --directory "C:\\Users\\ufuka\\synapse-cert\\mirror" > "logs/timing.$model.mirror.log" 2>&1 &
server=$!
sleep 2
fixture=$(python -c "import json;i=json.load(open('synapse/bench/parity/fixtures/index.json'));print([v['path'] for v in i.values() if v['model']=='$model'][0])")
scratch="C:\\Users\\ufuka\\synapse-cert\\timing\\$model-$(date +%s)"
timing-probe/target/release/timing-probe.exe \
  'C:\Users\ufuka\synapse-cert\assets\windows-x64\ck-synapse.exe' \
  'C:\Users\ufuka\synapse-cert\assets\windows-x64\ck-synapse-worker-vulkan.exe' \
  "C:\\Users\\ufuka\\synapse-cert\\synapse\\bench\\parity\\$fixture" \
  "$model" "$scratch" "http://127.0.0.1:$port" "$@" > "logs/timing.$model.jsonl" 2> "logs/timing.$model.stderr.log"
echo "probe_exit=$?"
kill $server; wait $server 2>/dev/null
cat "logs/timing.$model.jsonl"
