#!/bin/bash
source /root/env.sh
for m in qwen3-embedding-0.6b qwen3-reranker-0.6b; do
  rm -rf /root/probe-runs/$m; mkdir -p /root/probe-runs
  echo "=== $m $(date -u +%FT%TZ)"
  timeout 1200 /root/probe/target/release/parity-probe vulkan-linux-nvidia $m /root/assets /root/synapse /root/weights/$m /root/probe-runs/$m > /root/logs/probe.vulkan-linux-nvidia.$m.jsonl 2> /root/logs/probe.vulkan-linux-nvidia.$m.err
  echo "exit=$?"
done
echo PROBE_DONE
