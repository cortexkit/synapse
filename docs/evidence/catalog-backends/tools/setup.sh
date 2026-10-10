#!/bin/bash
set -euo pipefail
cd /root
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
source /root/.cargo/env
rustc --version; cargo --version
git clone -q https://github.com/cortexkit/synapse /root/synapse
cd /root/synapse
git checkout -q ff24e6715e1bb58ad6ba0c120b889ef7431ca6c3
git rev-parse HEAD; git status --porcelain | wc -l
T=/root/tk
OWNED_CUDA_REDIST_BASE=https://developer.download.nvidia.com/compute/cuda/redist
root="$T/cuda-13.2.1"
mkdir -p "$root" "$T/cuda-components"
while read -r name version digest; do
  archive="$name-linux-x86_64-$version-archive.tar.xz"
  curl --fail --location --retry 3 -sS "$OWNED_CUDA_REDIST_BASE/$name/linux-x86_64/$archive" -o "$T/$archive"
  component="$T/cuda-components/$name"
  python3 scripts/check-release-candidate.py extract --archive "$T/$archive" --sha256 "$digest" --destination "$component" --strip-components 1
  cp -a "$component/." "$root/"
done <<'PINS'
cuda_nvcc 13.2.78 1dc73b03f1d74081866986ca406f7b5981d14306ccbb03127ba45401f36e1862
cuda_crt 13.2.78 17e731ba749765c2e2e325e3bffecee94226c866cf59f13ba3792aa4f2b1bb31
libnvvm 13.2.78 9c665ebec40d0dec4df1858d5a0129972b00351dd674e783d30405cf712d925d
cuda_cccl 13.2.75 1801f304d92085a327ab46db58264d7f7f48cce80f5825637c405c3a2410dadd
cuda_cudart 13.2.75 9502ab2c7824e5ef3b554d290b9f564c5026994f469b99c72451a53578f386b9
libcublas 13.4.0.1 eda8001ac6b9a6ad862d054aa9b502ff39684147f1e86d02cc0a58a5ce67f62e
PINS
ln -s lib "$root/lib64"
curl --fail --location --retry 3 -sS https://sdk.lunarg.com/sdk/download/1.4.357.0/linux/vulkansdk-linux-x86_64-1.4.357.0.tar.xz -o "$T/vulkan.tar.xz"
python3 scripts/check-release-candidate.py extract --archive "$T/vulkan.tar.xz" --sha256 0f09bf6a0625e346bf004be70b92907e934a4c76606b323441b2baf3a5a0e66d --destination "$T/vulkan"
cat > /root/env.sh <<EOS
source /root/.cargo/env
export CUDA_HOME=$root CUDA_PATH=$root PATH=$root/bin:\$PATH
export LD_LIBRARY_PATH=$T/cuda-components/cuda_cudart/lib:$T/cuda-components/libcublas/lib
export VULKAN_SDK=$T/vulkan/1.4.357.0/x86_64
EOS
source /root/env.sh
nvcc --version | tail -2
ls $VULKAN_SDK/bin/glslc
echo SETUP_DONE
