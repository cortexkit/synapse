#!/bin/bash
source /root/env.sh
export SYNAPSE_CERTIFY_INSTANCE_ID=vast-55175162
mkdir -p /root/logs
for row in "$@"; do
 for model in gte-modernbert-base gte-reranker-modernbert-base qwen3-embedding-0.6b qwen3-reranker-0.6b; do
  [ -n "${ONLY:-}" ] && [ "$ONLY" != "$model" ] && continue
  echo "=== $row $model $(date -u +%FT%TZ)"
  start=$(date +%s)
  /root/synapse/target/release/ckdev-synapse-certify run --row $row --model $model --assets /root/assets --checkout /root/synapse --weights /root/weights/$model > /root/logs/$row.$model.stdout.json 2> /root/logs/$row.$model.stderr.log
  echo "exit=$? seconds=$(( $(date +%s) - start ))"
 done
done
echo CERT_DONE
