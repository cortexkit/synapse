# Vulkan day 1: MiniLM on the Ally RDNA3 iGPU

## Verdict

`VK_KHR_cooperative_matrix` is required for a competitive raw-Vulkan MiniLM path on this RDNA3 class. Across two fresh owned-runtime processes, the cooperative path averaged **37,240.7 real tok/s**, while the tiled plain shader averaged **21,423.2 real tok/s**. Cooperative matrices were **1.738x / 73.83% faster** with effectively identical parity. The same-day llama.cpp-Vulkan incumbent averaged **34,295.7 real tok/s** over four fresh server processes, so the cooperative prototype was **8.59% faster**; the plain path was 37.4% slower than llama.cpp.

The hardware answers the consult's precision unknown favorably: the Ally exposes a subgroup-scoped `16x16x16` cooperative shape with f16 A/B and **fp32 C/result**. Both paths pass the frozen 400-row gates. Retain the plain shader as the extension-free and ragged-edge fallback, but do not treat it as the performance substrate for this GPU class.

This is a measurement verdict, not a production-graduation claim. The prototype is MiniLM-only, uses host-visible coherent UMA allocations, lacks timestamp-query stage attribution and concurrent arenas, and was measured on one AMD Windows driver.

## Architecture

`--device vulkan --dtype f16` enters the same family-keyed `block_forward` seam as Metal and CUDA. Vulkan is feature-gated (`--features vulkan`) and rejected for non-MiniLM families. The implementation has these properties:

- ash 0.38 drives Vulkan directly. ash was chosen over a C shim because it keeps the Windows build independent of Vulkan SDK headers/import libraries while exposing cooperative-matrix feature/property queries, descriptor sets, command buffers, and pipeline-cache bytes without another ownership boundary.
- Encoder weights and f16 biases are uploaded once and retained in Vulkan storage buffers. Layer-normalization scales and biases remain model-owned fp32 storage, avoiding the temporary-pointer lifetime failure documented in `F16-SERVING.md`.
- Each exact `(batch, sequence)` shape owns activation buffers, descriptor sets, pipelines, and one reusable whole-encoder command buffer. Eager shape discovery happens before `infer_wall_s`.
- Per batch, the host writes the input hidden states and mask, submits one encoder command buffer, waits once, and reads pooled fp32 vectors once. Hidden states remain device-resident through all six encoder layers and mean-pool/L2.
- Both GEMM shaders consume f16 A/B and accumulate to fp32. Cooperative properties require fp32 result storage for the fp32 accumulator configuration, so both paths write a transient fp32 GEMM result before the same shared conversion/fusion kernel produces the next f16 activation. This keeps GEMM selection as the A/B variable.
- The shared non-GEMM shaders implement QKV bias/layout, scale+mask+softmax, context layout, residual+bias+layer normalization with fp32 statistics/parameters, bias+GELU, and mean-pool/L2.
- The plain path is a coalesced `16x16` shared-memory tiled shader with no architecture-specific optimization.
- The cooperative path uses one 64-lane subgroup per `16x16x16` tile. Full tiles use `coopMatMulAdd`; a plain edge dispatch handles incomplete M/N tiles. GEMMs whose K dimension is not divisible by 16 use the plain shader. Projections, MLPs, and QK use cooperative matrices; attention PV keeps its row-major B operand on the parity-certified plain path under both exact and bucketed policies.

## Cooperative-matrix property query — verbatim

The first rig action was a direct ash call to `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`, not an inference from extension presence. The output below is committed separately as [`results/vulkan-day1/cooperative-matrix-properties.json`](results/vulkan-day1/cooperative-matrix-properties.json).

```json
[
  {
    "device_name": "AMD Radeon Graphics",
    "api_version": "1.4.334",
    "driver_version_raw": 8388981,
    "cooperative_matrix": true,
    "cooperative_matrix_robust_buffer_access": true,
    "properties": [
      {
        "m": 16,
        "n": 16,
        "k": 16,
        "a_type": "FLOAT16",
        "b_type": "FLOAT16",
        "c_type": "FLOAT32",
        "result_type": "FLOAT32",
        "saturating_accumulation": false,
        "scope": "SUBGROUP"
      },
      {
        "m": 16,
        "n": 16,
        "k": 16,
        "a_type": "FLOAT16",
        "b_type": "FLOAT16",
        "c_type": "FLOAT16",
        "result_type": "FLOAT16",
        "saturating_accumulation": false,
        "scope": "SUBGROUP"
      },
      {
        "m": 16,
        "n": 16,
        "k": 16,
        "a_type": "FLOAT16",
        "b_type": "FLOAT16",
        "c_type": "FLOAT16",
        "result_type": "FLOAT16",
        "saturating_accumulation": true,
        "scope": "SUBGROUP"
      },
      {
        "m": 16,
        "n": 16,
        "k": 16,
        "a_type": "UINT8",
        "b_type": "UINT8",
        "c_type": "SINT32",
        "result_type": "SINT32",
        "saturating_accumulation": false,
        "scope": "SUBGROUP"
      },
      {
        "m": 16,
        "n": 16,
        "k": 16,
        "a_type": "UINT8",
        "b_type": "UINT8",
        "c_type": "SINT32",
        "result_type": "SINT32",
        "saturating_accumulation": true,
        "scope": "SUBGROUP"
      },
      {
        "m": 16,
        "n": 16,
        "k": 16,
        "a_type": "UINT8",
        "b_type": "SINT8",
        "c_type": "SINT32",
        "result_type": "SINT32",
        "saturating_accumulation": false,
        "scope": "SUBGROUP"
      },
      {
        "m": 16,
        "n": 16,
        "k": 16,
        "a_type": "UINT8",
        "b_type": "SINT8",
        "c_type": "SINT32",
        "result_type": "SINT32",
        "saturating_accumulation": true,
        "scope": "SUBGROUP"
      },
      {
        "m": 16,
        "n": 16,
        "k": 16,
        "a_type": "SINT8",
        "b_type": "UINT8",
        "c_type": "SINT32",
        "result_type": "SINT32",
        "saturating_accumulation": false,
        "scope": "SUBGROUP"
      },
      {
        "m": 16,
        "n": 16,
        "k": 16,
        "a_type": "SINT8",
        "b_type": "UINT8",
        "c_type": "SINT32",
        "result_type": "SINT32",
        "saturating_accumulation": true,
        "scope": "SUBGROUP"
      },
      {
        "m": 16,
        "n": 16,
        "k": 16,
        "a_type": "SINT8",
        "b_type": "SINT8",
        "c_type": "SINT32",
        "result_type": "SINT32",
        "saturating_accumulation": false,
        "scope": "SUBGROUP"
      },
      {
        "m": 16,
        "n": 16,
        "k": 16,
        "a_type": "SINT8",
        "b_type": "SINT8",
        "c_type": "SINT32",
        "result_type": "SINT32",
        "saturating_accumulation": true,
        "scope": "SUBGROUP"
      }
    ]
  }
]
```

## Correctness gates

The source fixture is the canonical 1,000-row corpus (`b7c8424f…`). A byte-preserving slice of its first 400 records hashes to the requested `5a9bfdc8c069657aa46cbb45bef91bc1a0ddc72602bfb96b189af31ba55f630c`. The frozen 1,000-row reference hashes to `7589eea5148562f6141c864d3357bab5dceb6881055afcf93b80efbdcae7d24d`; parity matches exactly 400 IDs.

| Path | Rows | Real tokens | Mean cosine | Mean top-10 overlap | Gate |
|---|---:|---:|---:|---:|---|
| plain tiled fp16/fp32 | 400 | 66,783 | `0.9999996795` | `0.998750` | PASS |
| cooperative fp16/fp32 | 400 | 66,783 | `0.9999996777` | `0.998750` | PASS |
| llama.cpp-Vulkan f16 GGUF | 400 | 66,783 | `0.9999803447` | not emitted by incumbent lane | cosine PASS |

The owned-runtime gates were cosine `>= 0.9999` and overlap `>= 0.995` on all three in-process passes in both fresh processes. The llama lane applies its cosine gate but does not implement rank-overlap reporting; it is an incumbent row, not one of the two required dual-GEMM gates.

## Dual-GEMM A/B and incumbent

Each owned-runtime process eagerly built the warmup plus five exact corpus shapes, then ran three corpus passes. Values below are pass 3. Throughput uses real tokens, while the owned runtime separately reports 76,507 padded execution tokens.

| Fresh process | Plain tok/s | Cooperative tok/s |
|---:|---:|---:|
| empty explicit `VkPipelineCache` | 21,271.9 | 37,288.9 |
| serialized-cache reload | 21,574.5 | 37,192.6 |
| **Mean** | **21,423.2** | **37,240.7** |

| llama.cpp-Vulkan fresh process | Real tok/s | Cold load |
|---:|---:|---:|
| 1 | 34,105.7 | 0.536 s |
| 2 | 34,142.3 | 0.519 s |
| 3 | 34,440.1 | 0.547 s |
| 4 | 34,494.6 | 0.525 s |
| **Mean / range** | **34,295.7 / 34,105.7–34,494.6** | **0.532 s mean** |

### Bucket-resident architecture check

A supplemental policy-v2 run eagerly built descriptor sets and whole-encoder command buffers for all ten bucket shapes (`16x64` through `8x512`) before inference. Both paths passed the same parity gates on every pass: plain delivered 21,947.1 tok/s at cosine `0.9999996795`, and cooperative delivered 37,140.9 tok/s at cosine `0.9999996779`; overlap was `0.998750` for both. Cooperative was 69.23% faster in this single bucketed cell.

The bucket runner executed 81,152 padded tokens for 66,783 real tokens, so policy-v2 waste was 17.71%. Both commands wrote complete results after parity passed and then exited nonzero on the harness's independent `<15%` serving-padding gate. This is a bucket-policy issue, not a Vulkan correctness failure; exact-shape results remain the primary same-harness A/B and incumbent comparison. Raw bucket outputs are committed as `plain-bucketed.json` and `cooperative-bucketed.json`.

The first three llama JSONs retain the lane's old hard-coded `llama-metal-embed` metadata even though the executed server was `C:\bench\bin\llama-vulkan\llama-server.exe` with `ggml-vulkan.dll`. The lane now accepts `--lane-label`; repeat 4 records `llama-vulkan-embed` directly.

### Real-token reconciliation

The staged tokenizer JSON contains fixed padding to 128. The old llama lane left that checkpoint padding enabled, reporting 69,596 tokens for the 400 rows while silently omitting pad IDs when it decoded request text. That was the documented padded-counter trap, not extra model work. The lane now calls `tokenizer.with_padding(None)` before counting and batching. All four published llama reruns and both owned paths therefore report the same **66,783 real tokens**. The pre-fix 36,645.4 tok/s sample is excluded.

## Pipeline-cache behavior

`VkPipelineCache` files were deleted before each empty-cache cell, then retrieved with `vkGetPipelineCacheData`, written to disk at context destruction, and supplied as `pInitialData` in the next fresh process.

| Path | Empty-cache pipeline creation, six shapes | Reload pipeline creation, six shapes | Cache bytes | Full eager cold load empty / reload |
|---|---:|---:|---:|---:|
| plain | 1.686 ms | 1.896 ms | 22,684 | 5.123 / 3.867 s |
| cooperative | 2.758 ms | 2.536 ms | 25,716 | 2.528 / 2.520 s |

The driver accepts and returns stable cache blobs, but explicit reload has **no practically meaningful pipeline-creation benefit** on this run: plain became 0.210 ms slower and cooperative saved only 0.222 ms. Pipeline creation was already sub-millisecond per shape in the empty-explicit-cache cells. AMD's separate Windows shader cache had been populated by correctness smokes and was not purged, so these numbers answer application `VkPipelineCache` behavior, not a machine-wide shader-cache purge. Full eager cold load includes model/tokenizer work, allocations, persistent weight upload, command recording, and one execution of every discovered shape; it must not be attributed to pipeline creation alone.

Other isolated host phases were loader `1.85–3.14 ms`, instance `53.34–56.30 ms`, device `15.05–15.89 ms`, and persistent-weight upload `51.45–53.19 ms` across the four published cells.

## Rig protocol

The full fingerprint is in [`results/vulkan-day1/rig-fingerprint.txt`](results/vulkan-day1/rig-fingerprint.txt). The rig was an ASUS ROG Ally X with Ryzen Z1 Extreme / RDNA3 integrated Radeon, 24.9 GB physical memory, Windows build 26200.8655, Vulkan 1.4.334, AMD Vulkan driver 2.0.373 (`25.30.27.03 LLPC`), display driver `32.0.23027.3001`, and the Windows **Turbo** power scheme.

Cell order was plain smoke, cooperative smoke, plain empty/reload, cooperative empty/reload, llama repeats 1–4, plain bucketed, a cooperative-PV diagnostic, then corrected cooperative bucketed. Published cells ran serially with no other lane/server process, and a 60-second idle interval separated cells/repeats. The close repeat ranges show no material late-cell thermal drift: cooperative changed -0.26% between processes and llama rose 1.14% from first to fourth. No driver crash or TDR occurred.

Two early smokes were force-stopped before measurement because reusing queried Vulkan feature structs while building the device-enable `pNext` chain created a host-side cycle and made `vkCreateDevice` spin. Distinct query/enable structures fixed the pointer-lifetime error. No queue had been submitted and those smokes are excluded.

## Commands executed

Property query and fixture verification:

```bat
C:\bench\vk-probe\target\release\vk-probe.exe > C:\bench\coop-properties.json
certutil -hashfile C:\bench\data\minilm-corpus-1000-official.jsonl SHA256
powershell -NoProfile -ExecutionPolicy Bypass -File C:\bench\slice-first400.ps1
certutil -hashfile C:\bench\data\minilm-corpus-first400.jsonl SHA256
certutil -hashfile C:\bench\data\ort-minilm-1000-vectors-official.jsonl SHA256
C:\Windows\System32\vulkaninfo.exe --summary
powercfg /getactivescheme
```

Native Ally build and gates:

```bat
cargo build --release --features vulkan --manifest-path bench\spikes\unified-rt\Cargo.toml --bin spike-unified-rt
cargo test --features vulkan --manifest-path bench\spikes\unified-rt\Cargo.toml
cargo clippy --features vulkan --bin spike-unified-rt --manifest-path bench\spikes\unified-rt\Cargo.toml -- -D warnings -A dead-code
cargo clippy --features vulkan --bin vulkan_probe --manifest-path bench\spikes\unified-rt\Cargo.toml -- -D warnings
cargo fmt --all -- --check
cargo test --manifest-path bench\lanes\llama\Cargo.toml
cargo clippy --all-targets --manifest-path bench\lanes\llama\Cargo.toml -- -D warnings
```

The Windows spike clippy invocation allows `dead_code` because macOS-only Metal test helpers remain parsed but unreachable on Windows; all other warnings are denied. The same feature build passed strict all-target clippy without that allowance on macOS.

SPIR-V was built locally and vendored because the Ally had no `glslc`:

```sh
brew install shaderc
for shader in bench/spikes/unified-rt/src/vulkan_shaders/*.comp; do
  glslc --target-env=vulkan1.3 "$shader" \
    -o "bench/spikes/unified-rt/src/vulkan_spv/$(basename "${shader%.comp}").spv"
done
```

Published owned-runtime command; change `plain`/`cooperative`, cache/output names, and remove the cache file for the empty-cache cell:

```bat
target\release\spike-unified-rt.exe ^
  --model C:\bench\model-minilm ^
  --tokenizer C:\bench\model-minilm\tokenizer.json ^
  --corpus C:\bench\data\minilm-corpus-1000-official.jsonl ^
  --reference C:\bench\data\ort-minilm-1000-vectors-official.jsonl ^
  --limit 400 --dtype f16 --device vulkan --vulkan-gemm cooperative ^
  --vulkan-pipeline-cache C:\bench\vk-cache-coop.bin ^
  --shapes exact --passes 3 --out C:\bench\vk-coop-cold.json
```

Published incumbent command; each repeat launched a fresh lane and server process:

```bat
target\release\lane-llama.exe embed ^
  --model C:\bench\models\minilm-f16.gguf ^
  --tokenizer C:\bench\model-minilm\tokenizer.json ^
  --corpus C:\bench\data\minilm-corpus-first400.jsonl ^
  --reference C:\bench\data\ort-minilm-1000-vectors-official.jsonl ^
  --min-parity 0.9999 --model-label minilm-f16-llama-vulkan ^
  --lane-label llama-vulkan-embed ^
  --server-binary C:\bench\bin\llama-vulkan\llama-server.exe ^
  --pooling mean --embd-normalize 2 --ctx-size 512 ^
  --batch-size 4096 --ubatch-size 1024 --gpu-layers 99 --parallel 1 ^
  --out C:\bench\llama-vulkan-r4.json
```

## Remaining risks and next measurements

1. Add Vulkan timestamp queries around GEMM, attention, pointwise, and pool stages. Whole-command wall timing proves the A/B but does not isolate where the 73.8% gain lands.
2. Diagnose cooperative row-major-B attention PV before enabling it. An initial all-cooperative bucket smoke produced cosine `0.25989876`; keeping PV on the plain shader restored `0.9999996779`. The failed smoke is excluded from performance tables, and the safe fallback may understate cooperative upside.
3. Replace per-buffer coherent UMA allocations with a suballocated device arena and add concurrent per-shape arenas before production serving.
4. Repeat on another RDNA3 Windows device/driver and a non-AMD Vulkan implementation. The shader hard-requires subgroup size 64 only for the cooperative pipeline and rejects other sizes rather than silently misexecuting.
5. Measure a machine-wide cold shader cache in a disposable driver-cache environment. The application cache result here is valid, but the AMD internal cache was already warm from correctness smokes.
6. Add rank-overlap reporting to the llama lane if incumbent parity is promoted from a comparison row to a certification gate.
