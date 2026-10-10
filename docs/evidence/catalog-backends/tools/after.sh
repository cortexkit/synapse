#!/bin/bash
set -euo pipefail
source /root/env.sh
cd /root/synapse
git checkout -q fix2
git rev-parse HEAD
git status --porcelain | wc -l
cargo test -p synapse-engine-cuda --lib a_package_file_without 2>&1 | grep -E "^test |test result"
cargo build --release --locked -p synapse-module -p synapse-worker-cuda -p synapse-worker-vulkan --features cuda --features vulkan --bin ck-synapse --bin ck-synapse-worker-cuda --bin ck-synapse-worker-vulkan 2>&1 | tail -1
cargo build --locked --release -p synapse-certify-runner 2>&1 | tail -1
mkdir -p /root/assets
for b in ck-synapse ck-synapse-worker-cuda ck-synapse-worker-vulkan; do ln -f target/release/$b /root/assets/$b; done
/root/assets/ck-synapse certify source
sha256sum /root/assets/* target/release/ckdev-synapse-certify
git status --porcelain | wc -l
cd /root/probe && cargo build --release 2>&1 | tail -1
echo AFTER_DONE
