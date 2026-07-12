# Vulkan wave 2: ModernBERT and Qwen3 on Ally RDNA3

## Readout

Both owned Vulkan family graphs pass the frozen 400-row ORT gates in plain and cooperative modes on the Ally. Cooperative matrix is the clear owned-runtime choice, but neither family graduates on throughput against the same-day llama.cpp Vulkan incumbent. The Qwen3 llama.cpp F16 conversion also misses the campaign cosine gate by `0.000009` and is shown as a diagnostic incumbent, not a passing parity cell.

| Family / fresh process | Pass 1 tok/s | Pass 2 tok/s | Pass 3 tok/s | cosine | top-10 overlap | cold load |
|---|---:|---:|---:|---:|---:|---:|
| gte-modernbert plain | 2,280.0 | 2,151.9 | 2,107.9 | 0.9999989704 | 0.998500 | 9.149 s |
| gte-modernbert cooperative | 3,966.2 | 3,910.2 | 3,888.6 | 0.9999990203 | 0.997750 | 6.059 s |
| Qwen3-Embedding-0.6B plain | 498.3 | 497.9 | 551.8 | 0.9999989124 | 0.998750 | 39.473 s |
| Qwen3-Embedding-0.6B cooperative | 985.4 | 984.3 | 985.6 | 0.9999989402 | 0.998250 | 22.768 s |

| Family | owned cooperative, pass 3 | llama.cpp Vulkan | owned / llama | llama cosine | llama cold load |
|---|---:|---:|---:|---:|---:|
| gte-modernbert | 3,888.6 tok/s | 7,525.0 tok/s | 0.517x | 0.999956592 | 0.776 s |
| Qwen3-Embedding-0.6B | 985.6 tok/s | 1,621.6 tok/s | 0.608x | 0.999891079 (fails 0.9999) | 1.558 s |

Throughput uses 62,838 real tokens for ModernBERT and 46,716 for Qwen3. Bucket padding was 13.57% and 12.69%, respectively. No TDR occurred; Qwen3 did not require chunked dispatch. The runtime did not expose per-repeat clocks. The active Windows power scheme was `6fecc5ae-f350-48a5-b669-b472cb895ccf (Turbo)`. Runs were sequential fresh processes, but the requested 60-second thermal gaps were not reliably enforced; the per-pass wall-time drift above is therefore the thermal signal available for this wave.

## Family graph details

ModernBERT uses fp16 storage with fp32 GEMM and reduction accumulation. Global and local RoPE tables are precomputed per resident shape at theta 160,000 and 10,000. The local-128 restriction is an additive mask tensor whose contents are consumed by the scale/mask/softmax shader. Layer zero copies its input instead of applying attention LayerNorm. All linears are bias-free, GeGLU computes `gelu(first_half) * second_half`, and the existing host path takes CLS then L2-normalizes.

Qwen3 keeps RMSNorm and per-head q/k RMSNorm in the resident command buffer, applies theta-1,000,000 RoPE, and records GQA as two group-strided batched QK/PV calls sharing the eight KV heads. The context transpose dispatch is sized from the 2,048-wide query tensor rather than the 1,024 hidden width. Causal and key-padding masks are combined in softmax; SwiGLU, final RMSNorm, host last-token pooling, and L2 normalization complete the path. The 20-row exact smoke (which spans many batch positions) reached cosine `0.999998855` and overlap `1.0` before the 400-row gates.

## PV row-major-B investigation

The bounded review rechecked the cooperative shader against the Ally property dump and the SPIR-V load flags. Transposed weights and QK use cooperative row-major A plus column-major B as certified on day 1. PV's context operand is physically row-major and would require the row-major `gl_MatrixUseB` path that failed day-1 parity; offsets and GQA grouping do not alter that layout. No contradictory property appeared, so PV remains on the plain shader for all three families. Cooperative mode still applies to every transposed linear and QK GEMM. This avoids reopening the already isolated driver/layout failure without evidence of a corrected load form.

## Fixtures and model identity

All fixture hashes were verified on the Ally with `certutil -hashfile ... SHA256` before measurement:

| File | SHA-256 |
|---|---|
| ModernBERT corpus | `b4ff00f6d2d9f0652146b7438c2ecd421746bcead466cccf18ec79e45ff79aa8` |
| ModernBERT ORT vectors | `d1fb6aaf48c36c8ed7b06b9c69e6244f01393e085d32f49b15194671f7a44000` |
| Qwen3 corpus | `5a9bfdc8c069657aa46cbb45bef91bc1a0ddc72602bfb96b189af31ba55f630c` |
| Qwen3 ORT vectors | `cacee1f64d12704ea94cded9861f6aef903a018800b2e0a1ec67589c33c7cf46` |
| ModernBERT safetensors | `3e85899d5728cb7de79781c0c3acfb91ccef9f875f1f7e0b3c9f3dd4b6a724ba` |
| Qwen3 safetensors | `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd` |
| ModernBERT F16 GGUF | `2cda419ce09e87b1dd5294177a1c7d563e9536700e33652a9bb9e9c503c4e437` |
| Qwen3 embedding F16 GGUF | `421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340` |

Raw LaneResult files and the machine fingerprint are under `results/vulkan-wave2/`.
