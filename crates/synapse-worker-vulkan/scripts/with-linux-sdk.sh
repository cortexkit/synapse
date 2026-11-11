#!/usr/bin/env bash
# Use the same immutable compiler release as the native Vulkan worker builds.
set -euo pipefail
if [[ -z "${VULKAN_SDK:-}" ]]; then
  archive="$PWD/target/scratch/vulkansdk-linux-x86_64-1.4.357.0.tar.xz"
  destination="$PWD/target/scratch/vulkan-sdk"
  export VULKAN_SDK="$destination/1.4.357.0/x86_64"
  if [[ ! -x "$VULKAN_SDK/bin/glslc" ]]; then
    mkdir -p "$destination"
    curl -fL --retry 2 https://sdk.lunarg.com/sdk/download/1.4.357.0/linux/vulkansdk-linux-x86_64-1.4.357.0.tar.xz -o "$archive"
    printf '%s  %s\n' 0f09bf6a0625e346bf004be70b92907e934a4c76606b323441b2baf3a5a0e66d "$archive" | sha256sum -c -
    tar -xJf "$archive" -C "$destination"
  fi
fi
exec "$@"
