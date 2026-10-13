#!/bin/bash
set -euo pipefail
source /root/env.sh
mkdir -p /root/raw /root/records
rustc --version
cargo --version
cd /root/synapse
if [[ ! -d /root/baseline ]]; then
  git worktree add --detach /root/baseline 5f6d7419fc42a5ebf6ed1b8eab2115b6cc84fe5a
fi
for revision in baseline progress; do
  export CARGO_TARGET_DIR="/root/build-target-$revision"
  if [[ "$revision" == baseline ]]; then checkout=/root/baseline; else checkout=/root/synapse; fi
  cd "$checkout"
  printf 'BUILD %s\n' "$revision"
  git rev-parse HEAD
  test -z "$(git status --porcelain)"
  time cargo build --release --locked -p synapse-module -p synapse-worker-cuda -p synapse-worker-vulkan --features cuda --features vulkan --bin ck-synapse --bin ck-synapse-worker-cuda --bin ck-synapse-worker-vulkan
  time cargo build --locked --release -p synapse-certify-runner
  mkdir -p "/root/assets-$revision/linux-x64"
  for binary in ck-synapse ck-synapse-worker-cuda ck-synapse-worker-vulkan; do
    cp "$CARGO_TARGET_DIR/release/$binary" "/root/assets-$revision/linux-x64/$binary"
  done
  cp "$CARGO_TARGET_DIR/release/ckdev-synapse-certify" "/root/runner-$revision"
  sha256sum /root/assets-$revision/linux-x64/* "/root/runner-$revision"
  "/root/assets-$revision/linux-x64/ck-synapse" certify source
  "/root/assets-$revision/linux-x64/ck-synapse-worker-cuda" --probe-floor
  "/root/assets-$revision/linux-x64/ck-synapse-worker-vulkan" --probe-floor
  test -z "$(git status --porcelain)"
done
mkdir -p /root/timing-probe/src
cp /root/timing-probe.rs /root/timing-probe/src/main.rs
cp /root/timing-probe.Cargo.toml /root/timing-probe/Cargo.toml
cp /root/synapse/Cargo.lock /root/timing-probe/Cargo.lock
cd /root/timing-probe
export CARGO_TARGET_DIR=/root/build-target-timing
cargo fetch --locked
cargo build --release --locked --offline
printf 'BUILDS_DONE\n'
