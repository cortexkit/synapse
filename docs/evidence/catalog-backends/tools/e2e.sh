#!/bin/bash
source /root/env.sh
cd /root/e2e && cargo build --release 2>&1 | tail -1
for spec in "gte-modernbert-base cuda" "gte-modernbert-base vulkan" "gte-reranker-modernbert-base cuda" "gte-reranker-modernbert-base vulkan" "qwen3-embedding-0.6b cuda" "qwen3-embedding-0.6b vulkan" "qwen3-reranker-0.6b cuda" "qwen3-reranker-0.6b vulkan"; do
  set -- $spec
  rm -rf /root/e2e-runs/$1-$2
  echo "=== $1 $2 $(date -u +%FT%TZ)"
  timeout 1500 /root/e2e/target/release/catalog-e2e /root/assets /root/synapse $1 $2 /root/e2e-runs/$1-$2 > /root/logs/e2e.$1.$2.jsonl 2> /root/logs/e2e.$1.$2.err
  echo "exit=$?"
  grep pin_check /root/logs/e2e.$1.$2.jsonl | cut -c1-600
done
echo E2E_DONE
