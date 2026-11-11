#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
backend=${1:?usage: build.sh metal|cuda|vulkan}
case "$backend" in
  metal) ;;
  cuda)
    # On Windows the pinned transcribe.cpp writes no CUDA libraries into its
    # static link manifest (its library_paths stays empty), so the final link
    # misses cudart, cuBLAS and the driver API. Name them here; CUDA_PATH is the
    # toolkit the CMake build used.
    case "$(uname -s)" in
      MINGW*|MSYS*|CYGWIN*)
        : "${CUDA_PATH:?set CUDA_PATH to the CUDA toolkit}"
        export RUSTFLAGS="${RUSTFLAGS:-} -L native=$(cygpath -m "$CUDA_PATH")/lib/x64 -l dylib=cudart -l dylib=cublas -l dylib=cuda"
        ;;
    esac
    ;;
  vulkan)
    # Match Hex's shader subset; the pinned fork includes the ggml patch that
    # makes these settings effective. Keep the Rust toolchain's C runtime selection.
    export CMAKE_ARGS="${CMAKE_ARGS:-} -DGGML_VULKAN_SHADER_TYPES=f32,f16,q8_0,q5_0 -DGGML_VULKAN_MUL_MAT_ID=OFF"
    # On Windows the link manifest names vulkan-1 but no folder to find it in.
    case "$(uname -s)" in
      MINGW*|MSYS*|CYGWIN*)
        : "${VULKAN_SDK:?set VULKAN_SDK to the Vulkan SDK}"
        export RUSTFLAGS="${RUSTFLAGS:-} -L native=$(cygpath -m "$VULKAN_SDK")/Lib"
        ;;
    esac
    ;;
  *) echo "unknown backend: $backend" >&2; exit 2 ;;
esac
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-target/stt-$backend}
cargo build --locked --release -p stt-transcribe-cpp --features "$backend"
