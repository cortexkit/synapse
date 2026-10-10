#!/bin/bash
set -euo pipefail
source /root/env.sh
cd /root/synapse
date -u +%FT%TZ
time cargo build --release --locked -p synapse-module -p synapse-worker-cuda -p synapse-worker-vulkan --features cuda --features vulkan --bin ck-synapse --bin ck-synapse-worker-cuda --bin ck-synapse-worker-vulkan
time cargo build --locked --release -p synapse-certify-runner
git status --porcelain
date -u +%FT%TZ
echo BUILD_DONE
