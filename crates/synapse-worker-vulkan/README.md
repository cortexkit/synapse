# Owned Vulkan worker

`ck-synapse-worker-vulkan` serves the four `owned-vulkan` manifest profiles over
worker protocol v2. The default build needs neither a Vulkan SDK nor a loader;
HELLO and PING do not attempt loader discovery.

Enable the GPU build with Vulkan SDK **1.4.357.0**, setting `VULKAN_SDK` to its
platform directory (containing `bin/glslc`, or `Bin/glslc.exe` on Windows):

```sh
cargo build -p synapse-worker-vulkan --features vulkan
ck-synapse-worker-vulkan --version
ck-synapse-worker-vulkan --probe-floor
ck-synapse-worker-vulkan --probe-floor --model qwen3-embedding-0.6b
```

The build checks the SDK path and compiler version. Both owned shaders use `-O`
and a fixed target environment: Vulkan 1.2 for plain kernels, Vulkan 1.3 for
cooperative matrices. Embedded SPIR-V is checked for source/debug instructions.
The kernel revision hashes the plain and cooperative binaries in that order;
the feature-disabled worker hashes an empty shader set. The manifest binding
hashes its embedded canonical JSON, not the pretty-printed source file.

LOAD takes `runtime_config.profile`, `runtime_config.operation` and the pinned
converted-package digest. Its artifact is a canonical safetensors file, or a
directory containing `model.safetensors`; no `config.json` is consulted. Profile,
operation and claimed digest checks precede device discovery. Device admission
precedes artifact opening; required head keys are checked from the header before
reading tensor data. A replacement LOAD releases the previous model arena first.

All transformer and readout stages run in owned GLSL. ModernBERT uses LayerNorm,
global/local RoPE and GEGLU; Qwen3 uses RMSNorm, per-head Q/K normalization,
causal grouped-query attention and SwiGLU. Online attention softmax avoids a
sequence-squared buffer. GTE rerank returns the classifier sigmoid; Qwen3 reads
only the manifest's yes/no rows. No request is truncated or given extra tokens.

One device-local arena contains the weights, reusable activations and results.
The shared floor function rounds each slice to 256 bytes. The worker processes
one sequence per sub-batch, retaining the original request's batch-longest
right-padding, and reuses that arena across all 256 sequences. Host transfers
use mapped coherent arena memory when available, otherwise non-device-local
staging memory. A JSON load report on stderr records the requested arena size,
driver `VkMemoryRequirements.size`, allocation count and selected heap. Driver
allocation padding is visible in that report rather than hidden in payload
accounting.

Cooperative GEMM requires API 1.3, `VK_KHR_cooperative_matrix`, its enabled feature,
the Vulkan memory-model feature and the supported subgroup 16×16×16
f16-input/f32-accumulator shape. Other devices
and other projection shapes use the plain kernel. Neither kernel is a wrapper
around another inference runtime.

Hosted protocol/floor tests need no GPU. Model-output parity, performance and
physical-allocation evidence still require certification on AMD/NVIDIA hardware;
a successful hosted smoke test is not evidence of model-output parity.

## Development-only GPU parity

`moltenvk-diagnostic` is non-default and exposes an explicit macOS-only test
constructor, not a production LOAD/probe override. It opens the SDK's real dylib,
enables portability enumeration/subset and uses the plain shader on Apple GPUs.
Default and `vulkan` builds cannot import that constructor (compile-fail doctest).

After verifying and converting the manifest-pinned checkpoints with
`bench/parity`'s `parity-manifest convert` command, run:

```sh
export VULKAN_SDK="$PWD/target/vulkan-sdk/1.4.357.0/macOS"
VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json" \
SYNAPSE_MOLTENVK_LOADER="$VULKAN_SDK/lib/libvulkan.1.dylib" \
SYNAPSE_VULKAN_TEST_PACKAGES="$PWD/target/gpu-parity/packages" \
cargo test -p synapse-worker-vulkan --features moltenvk-diagnostic \
  --test gpu_parity -- --ignored --nocapture
```

The ignored test executes the production Engine and shaders against CPU-fp32
fixtures, including 129/200-token rows, batch-longest padding and a ten-candidate
pool. It requires cosine >= 0.999, unit norm within 1e-3, fp16 score error <= 0.02,
gap-gated top-10 order and Kendall tau >= 0.99. Package directories must not
contain config.json. AMD/NVIDIA release certification remains separate.

Measured on MoltenVK plain compute (API 1.3; float16, 16-bit storage and subgroup
arithmetic supported; cooperative matrices unavailable):

```text
GPU_PARITY gte-modernbert-base min_cosine=0.999997843 rows=11 padded_width=200
GPU_PARITY gte-reranker-modernbert-base max_score_error=0.001476765 min_kendall_tau=1.000000000 rows=19
GPU_PARITY qwen3-embedding-0.6b min_cosine=0.999999780 rows=11 padded_width=200
GPU_PARITY qwen3-reranker-0.6b max_score_error=0.000006936 min_kendall_tau=1.000000000 rows=19
```
