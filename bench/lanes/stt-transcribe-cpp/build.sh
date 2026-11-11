#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
backend=${1:?usage: build.sh metal|cuda|vulkan}
case "$backend" in
  metal|cuda) ;;
  vulkan)
    # Match Hex's shader subset; the pinned fork includes the ggml patch that
    # makes these settings effective. Keep the Rust toolchain's C runtime selection.
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DGGML_VULKAN_SHADER_TYPES=f32,f16,q8_0,q5_0 -DGGML_VULKAN_MUL_MAT_ID=OFF"
    ;;
  *) echo "unknown backend: $backend" >&2; exit 2 ;;
esac
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-target/stt-$backend}
cargo build --locked --release -p stt-transcribe-cpp --features "$backend"
