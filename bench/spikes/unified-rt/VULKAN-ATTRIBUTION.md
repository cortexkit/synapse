# Vulkan stage attribution: the deep-family gap

## Verdict

**The roughly 2x deep-family gap is a large-linear-GEMM and memory-placement failure, not dispatch overhead: qkv/out/MLP GEMMs consume 81.50% of ModernBERT GPU time and 90.30% of Qwen3 GPU time while their weights are read from a non-device-local `HOST_VISIBLE | HOST_COHERENT` heap; barriers plus the descriptor/pipeline interval residual consume only 0.60% and 0.35%, respectively.** Qwen3 adds a repeatable depth cliff: the same linear shapes run 1.32x slower in layers 18–27 than in layers 1–17, close to where its separately allocated f16 layer weights pass roughly 512 MiB.

MiniLM is the control and validates the instrumentation. Its GEMMs account for 932.01 ms of a 1,262.17 ms GPU encoder pass (73.84%), so it is GEMM-dominated as expected. Timestamp support is native and well formed on this driver: `timestampComputeAndGraphics=true`, `timestampPeriod=10.0 ns`, and the compute queue exposes 64 valid timestamp bits.

The recommendation is a **conditional GO for one GEMM/memory-only optimization wave**, with a kill gate described below. Do not spend a wave on barriers, descriptor rebinding, pointwise fusion, or layout first: even perfect removal of all four is insufficient to close either incumbent gap.

## Method

The instrumentation is opt-in through `SYNAPSE_VULKAN_PROFILE=1`; `SYNAPSE_VULKAN_PROFILE_OUT` selects an NDJSON sink. Every dispatch in the resident encoder records three `VK_QUERY_TYPE_TIMESTAMP` values: immediately before the dispatch, immediately after it, and after its existing compute-to-compute memory barrier. Each wrapped tick delta is multiplied by the reported 10 ns `timestampPeriod` and divided by 1,000 before emission as `stage_us_mean`; the document tables rescale those raw microseconds to milliseconds for readability. An overall query pair measures the command-buffer span. Two adjacent terminal timestamps measure timestamp-command overhead without removing a dependency required by the graph. The first execution of each shape is retrieved but excluded as preload; the emitted means cover the three timed corpus passes. Host-visible output map/copy is timed separately as readback.

All ten serving buckets (`8x64`, `8x96`, `8x128`, `8x160`, `8x192`, `8x256`, `8x320`, `8x384`, `8x448`, `8x512`) were prebuilt exactly as in wave 2. The 400-row corpora submitted seven MiniLM buckets, seven ModernBERT buckets, and five Qwen3 buckets; unused larger buckets have no contribution to corpus attribution. There were 50 encoder command-buffer executions per corpus pass for every family. Tables below aggregate each stage over those 50 executions and divide the three-pass total by three. Thus every value is milliseconds per full 400-row serving pass, not milliseconds per dispatch.

The stage abbreviations are: `QKV` = GEMM-qkv, `QK` = GEMM-attn-scores, `SM` = softmax+mask, `PV` = GEMM-PV, `Out` = GEMM-out, `Up` = GEMM-mlp-up (both gate and up for Qwen3), `Down` = GEMM-mlp-down, `PW` = norms/activations/residual/pool, `L/T` = layout/transpose/RoPE layout, `R/B` = host readback, and `Bar` = the post-dispatch barrier intervals.

Raw profiles and result JSONs are in [`results/vulkan-attribution/`](results/vulkan-attribution/). The exact probe response is [`vulkan-attr-probe.json`](results/vulkan-attribution/vulkan-attr-probe.json); rig, memory, and thermal facts are in [`rig.txt`](results/vulkan-attribution/rig.txt).

## Whole-pass attribution

| Family | GPU span | QKV | QK | SM | PV | Out | Up | Down | PW | L/T | Bar | interval residual | host R/B | timed wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM, 6 layers | 1,262.165 | 205.993 | 72.747 | 143.069 | 101.064 | 56.453 | 219.444 | 276.311 | 92.113 | 62.576 | 32.184 | 0.212 | 4.456 | 1,503.759 |
| gte-modernbert, 22 layers | 14,670.028 | 4,883.151 | 256.157 | 642.335 | 627.518 | 851.382 | 4,659.445 | 1,562.432 | 604.588 | 494.883 | 87.514 | 0.623 | 640.087 | 16,115.133 |
| Qwen3-Embedding-0.6B, 28 layers | 46,875.361 | 8,706.851 | 489.359 | 601.807 | 1,221.205 | 6,148.745 | 17,585.925 | 9,887.720 | 904.150 | 1,167.749 | 160.635 | 1.215 | 701.699 | 48,806.111 |

`GPU span = dispatch stages excluding host R/B + Bar + interval residual`. The residual contains commands between timestamped dispatch bodies, including pipeline/descriptor binds and push constants. It is 0.017% of MiniLM, 0.004% of ModernBERT, and 0.003% of Qwen3. Descriptor rebinding is therefore counted but is not a material gap mechanism. Host wall additionally includes embedding preparation, queue/fence overhead, f16 decode, and family pooling outside this command buffer.

### Where the extra milliseconds live

The layer-count-only controls are MiniLM multiplied by `22/6` for ModernBERT and by `28/6` for Qwen3. Positive numbers are extra milliseconds beyond that control.

| Stage | ModernBERT extra vs 3.667x MiniLM | Qwen3 extra vs 4.667x MiniLM |
|---|---:|---:|
| QKV | +4,127.8 | +7,745.6 |
| QK | -10.6 | +149.9 |
| softmax+mask | +117.7 | -65.8 |
| PV | +257.0 | +749.6 |
| Out | +644.4 | +5,885.3 |
| MLP up | +3,854.8 | +16,561.9 |
| MLP down | +549.3 | +8,598.3 |
| pointwise | +266.8 | +474.3 |
| layout/transpose | +265.4 | +875.7 |
| barriers | -30.5 | +10.4 |
| **extra GPU span** | **+10,042.1** | **+40,985.3** |

The four large-linear classes (QKV, Out, Up, Down) explain 91.4% of ModernBERT's extra GPU time and 94.6% of Qwen3's. Pointwise plus layout explain only 5.3% and 3.3%. Barrier growth explains none of ModernBERT's excess and 0.03% of Qwen3's.

## Per-layer tables

### MiniLM

| Layer | QKV | QK | SM | PV | Out | Up | Down | PW | L/T | R/B | Bar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 64.976 | 19.452 | 24.961 | 17.099 | 9.382 | 35.081 | 49.603 | 14.151 | 12.086 | 0 | 6.827 |
| 1 | 27.942 | 10.747 | 23.356 | 16.959 | 9.248 | 33.503 | 36.298 | 14.128 | 9.662 | 0 | 5.073 |
| 2 | 28.908 | 10.621 | 23.297 | 16.900 | 9.225 | 35.156 | 46.256 | 14.413 | 9.611 | 0 | 5.104 |
| 3 | 28.102 | 10.650 | 23.768 | 16.821 | 9.336 | 34.788 | 51.847 | 14.590 | 10.025 | 0 | 5.118 |
| 4 | 27.659 | 10.656 | 23.639 | 16.611 | 9.912 | 46.843 | 45.967 | 14.359 | 10.692 | 0 | 5.040 |
| 5 | 28.407 | 10.621 | 24.048 | 16.673 | 9.349 | 34.073 | 46.339 | 14.550 | 10.500 | 0 | 4.995 |
| final/readback | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.922 | 0 | 4.456 | 0.026 |

### gte-modernbert

| Layer | QKV | QK | SM | PV | Out | Up | Down | PW | L/T | R/B | Bar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 444.293 | 14.246 | 28.171 | 37.383 | 52.043 | 233.048 | 94.790 | 29.678 | 21.415 | 0 | 5.062 |
| 1 | 241.278 | 11.907 | 36.693 | 34.257 | 45.634 | 215.929 | 85.418 | 30.199 | 23.550 | 0 | 3.921 |
| 2 | 215.325 | 11.409 | 32.353 | 31.248 | 42.347 | 213.123 | 73.549 | 27.833 | 23.883 | 0 | 3.896 |
| 3 | 211.980 | 11.520 | 28.603 | 29.205 | 39.417 | 211.309 | 55.543 | 27.664 | 22.907 | 0 | 3.917 |
| 4 | 209.301 | 11.446 | 30.933 | 27.846 | 37.683 | 210.105 | 77.012 | 27.459 | 22.242 | 0 | 3.922 |
| 5 | 209.088 | 11.410 | 28.996 | 27.529 | 37.594 | 209.792 | 59.704 | 28.756 | 23.051 | 0 | 3.891 |
| 6 | 208.982 | 11.516 | 26.331 | 28.066 | 37.347 | 214.896 | 82.025 | 27.051 | 21.966 | 0 | 3.923 |
| 7 | 207.617 | 11.669 | 30.346 | 27.598 | 37.899 | 213.886 | 77.028 | 26.911 | 22.383 | 0 | 3.888 |
| 8 | 206.720 | 11.585 | 29.564 | 27.759 | 37.431 | 214.365 | 74.080 | 27.663 | 22.884 | 0 | 3.917 |
| 9 | 207.454 | 11.521 | 26.616 | 27.325 | 37.027 | 208.960 | 72.459 | 26.726 | 22.671 | 0 | 3.915 |
| 10 | 211.724 | 11.473 | 31.133 | 27.897 | 37.304 | 209.851 | 68.549 | 26.627 | 23.010 | 0 | 3.890 |
| 11 | 208.982 | 11.402 | 29.008 | 27.385 | 37.287 | 208.384 | 70.547 | 26.759 | 22.730 | 0 | 3.899 |
| 12 | 214.149 | 11.608 | 25.607 | 27.282 | 36.910 | 204.663 | 67.911 | 26.924 | 21.752 | 0 | 3.904 |
| 13 | 212.438 | 11.562 | 29.896 | 27.382 | 37.584 | 208.397 | 80.305 | 26.937 | 22.426 | 0 | 3.922 |
| 14 | 212.841 | 11.414 | 31.382 | 27.301 | 37.484 | 208.000 | 67.003 | 26.318 | 21.723 | 0 | 3.868 |
| 15 | 211.881 | 11.525 | 26.000 | 27.584 | 37.427 | 207.739 | 70.639 | 27.078 | 22.353 | 0 | 3.908 |
| 16 | 208.354 | 11.339 | 29.286 | 27.449 | 37.109 | 207.617 | 59.488 | 28.348 | 22.239 | 0 | 3.911 |
| 17 | 207.393 | 11.536 | 30.382 | 27.290 | 37.067 | 209.884 | 52.020 | 26.332 | 22.178 | 0 | 3.906 |
| 18 | 208.959 | 11.609 | 25.559 | 27.511 | 37.093 | 208.325 | 57.059 | 27.120 | 21.765 | 0 | 3.879 |
| 19 | 208.715 | 11.406 | 29.900 | 27.462 | 37.095 | 213.877 | 73.001 | 25.545 | 22.453 | 0 | 3.926 |
| 20 | 208.187 | 11.570 | 30.474 | 27.304 | 37.442 | 214.143 | 75.425 | 25.924 | 22.744 | 0 | 3.929 |
| 21 | 207.490 | 11.485 | 25.103 | 27.457 | 37.156 | 213.152 | 68.876 | 27.932 | 22.557 | 0 | 3.902 |
| final/readback | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2.807 | 0 | 640.087 | 0.417 |

### Qwen3-Embedding-0.6B

| Layer | QKV | QK | SM | PV | Out | Up | Down | PW | L/T | R/B | Bar |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 525.641 | 15.392 | 26.070 | 54.452 | 231.028 | 587.611 | 296.553 | 38.527 | 38.612 | 0 | 6.536 |
| 1 | 317.890 | 15.510 | 21.440 | 42.198 | 179.663 | 564.166 | 296.873 | 31.618 | 38.953 | 0 | 5.703 |
| 2 | 277.249 | 15.915 | 20.293 | 41.780 | 177.817 | 541.427 | 291.191 | 30.911 | 37.721 | 0 | 5.667 |
| 3 | 271.879 | 16.338 | 20.521 | 41.396 | 185.397 | 567.376 | 291.875 | 30.482 | 38.748 | 0 | 5.670 |
| 4 | 295.289 | 16.728 | 20.553 | 42.113 | 197.229 | 544.129 | 289.097 | 30.541 | 37.904 | 0 | 5.665 |
| 5 | 291.036 | 17.845 | 21.030 | 43.634 | 203.622 | 591.991 | 305.666 | 30.739 | 36.734 | 0 | 5.675 |
| 6 | 298.270 | 17.540 | 21.363 | 43.975 | 180.318 | 556.160 | 298.181 | 31.236 | 39.511 | 0 | 5.673 |
| 7 | 300.604 | 17.134 | 21.411 | 43.861 | 175.201 | 575.558 | 303.290 | 31.226 | 40.256 | 0 | 5.704 |
| 8 | 290.870 | 18.396 | 21.474 | 43.479 | 203.047 | 553.748 | 294.693 | 31.955 | 40.429 | 0 | 5.706 |
| 9 | 288.381 | 17.148 | 21.531 | 44.106 | 181.207 | 603.235 | 294.520 | 32.062 | 40.810 | 0 | 5.699 |
| 10 | 289.807 | 16.311 | 21.407 | 44.008 | 182.350 | 582.535 | 317.867 | 32.272 | 41.318 | 0 | 5.692 |
| 11 | 295.829 | 18.661 | 21.305 | 43.957 | 187.010 | 571.196 | 299.776 | 31.205 | 39.894 | 0 | 5.691 |
| 12 | 301.810 | 17.682 | 21.318 | 43.884 | 183.820 | 558.105 | 295.550 | 31.985 | 43.035 | 0 | 5.693 |
| 13 | 307.746 | 16.788 | 21.467 | 45.308 | 203.796 | 565.632 | 293.471 | 31.614 | 39.886 | 0 | 5.700 |
| 14 | 300.730 | 17.515 | 21.703 | 44.781 | 182.943 | 573.493 | 299.606 | 32.548 | 38.725 | 0 | 5.689 |
| 15 | 294.508 | 18.268 | 21.357 | 43.713 | 175.772 | 568.543 | 309.018 | 32.476 | 41.856 | 0 | 5.698 |
| 16 | 298.607 | 20.047 | 21.567 | 43.179 | 190.442 | 548.614 | 302.077 | 31.979 | 39.985 | 0 | 5.706 |
| 17 | 304.911 | 19.151 | 21.476 | 44.863 | 192.030 | 567.342 | 290.127 | 32.211 | 41.004 | 0 | 5.689 |
| 18 | 301.419 | 18.368 | 21.622 | 43.777 | 182.361 | 720.426 | 448.085 | 32.838 | 41.887 | 0 | 5.701 |
| 19 | 308.057 | 17.544 | 21.296 | 42.595 | 334.879 | 860.464 | 569.074 | 33.182 | 46.048 | 0 | 5.697 |
| 20 | 318.193 | 17.928 | 21.113 | 42.650 | 212.445 | 704.415 | 381.480 | 32.171 | 48.332 | 0 | 5.706 |
| 21 | 296.339 | 17.523 | 21.161 | 43.289 | 326.659 | 712.063 | 320.679 | 33.213 | 43.832 | 0 | 5.687 |
| 22 | 310.996 | 17.178 | 21.745 | 42.064 | 203.682 | 737.343 | 454.026 | 31.744 | 44.198 | 0 | 5.683 |
| 23 | 304.948 | 18.233 | 21.355 | 42.344 | 230.073 | 698.797 | 437.918 | 32.210 | 46.528 | 0 | 5.713 |
| 24 | 329.525 | 18.139 | 21.454 | 42.637 | 215.599 | 691.757 | 363.869 | 32.970 | 45.489 | 0 | 5.697 |
| 25 | 308.512 | 18.025 | 21.465 | 42.965 | 345.376 | 658.521 | 370.885 | 32.442 | 43.987 | 0 | 5.688 |
| 26 | 323.904 | 15.912 | 21.893 | 41.834 | 234.994 | 816.241 | 529.084 | 33.348 | 44.761 | 0 | 5.705 |
| 27 | 353.901 | 18.141 | 21.414 | 42.363 | 449.982 | 765.040 | 643.192 | 32.101 | 47.308 | 0 | 5.699 |
| final/readback | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2.343 | 0 | 701.699 | 0.403 |

The Qwen3 late-layer effect is present in every exercised bucket, not one noisy shape. Averaged within each command buffer, large-linear time in layers 18–27 is 1.249x, 1.330x, 1.317x, 1.323x, and 1.324x that of layers 1–17 for sequence buckets 64, 96, 128, 160, and 192. The backend's first compatible storage type is memory type 1 on heap 0 (`HOST_VISIBLE | HOST_COHERENT`, not `DEVICE_LOCAL`); Vulkan reports memory type 2 on heap 1 as a host-visible, coherent, device-local alternative. This directly identifies the first experiment for a later optimization wave without changing math in this wave.

## Barrier and command-processor analysis

Each MiniLM and ModernBERT layer records 14 dispatches, barriers, and descriptor rebinds. Each Qwen3 layer records 21 because GQA uses two QK and two PV GEMMs and has separate q/k/v and gate/up projections. Final normalization/pooling adds one. The representative `8x160` layer-zero intervals are:

| Family | dispatch body, us | barriers | barrier interval, us | 14/21x empty timestamp baseline, us | reconstructed barrier-free layer, us | maximum saving |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM | 4,344.908 | 14 | 149.295 | 0.537 | 4,344.908 | 3.32% |
| ModernBERT | 16,383.253 | 14 | 104.293 | 0.503 | 16,383.253 | 0.63% |
| Qwen3 | 44,068.527 | 21 | 123.542 | 0.766 | 44,068.527 | 0.28% |

Executing a genuinely dependency-free copy of a layer would change the graph's memory visibility and could change its math, so it was not run. The table uses the mission's permitted empty-timestamp sandwich: actual barriers are isolated between timestamps and compared with adjacent timestamp pairs, then removed arithmetically from the same layer. It is a stronger bound than an unsafe no-barrier dispatch. Even perfect barrier elimination saves only 87.5 ms per ModernBERT pass and 160.6 ms per Qwen3 pass.

Descriptor/pipeline rebind cost is bounded by the overall interval residual after subtracting every dispatch and barrier interval: 0.623 ms for ModernBERT and 1.215 ms for Qwen3 per full pass. Layout dispatches cost 494.9 ms and 1,167.7 ms; pointwise dispatches cost 604.6 ms and 904.2 ms. None is the primary mechanism.

## llama.cpp-Vulkan structure

The Ally binary identifies itself as llama.cpp version 9580, commit `b4e3dc613`, built with Clang 19.1.5. The staged directory contains the binary distribution rather than the source checkout, so the exact revision was inspected from upstream source at that commit. The source-level fusion inventory is recorded here separately from graph scheduling:

| Path | Shares one dispatch/shader in `b4e3dc613` | Remains separate |
|---|---|---|
| ModernBERT projections | Packed `wqkv` is one GEMM and three metadata views; packed `ffn_up [E,2F]` is one GEMM. These are model/graph packing, not Vulkan pattern fusion. | QK, softmax, PV, output projection, GeGLU, down projection, residual, and ordinary LayerNorm affine work remain separate operations. |
| Qwen3 projections/MLP | Standard weights use separate Q/K/V GEMMs and separate gate/up GEMMs; one split-SwiGLU shader performs `SiLU(gate) * up`. | Output/down GEMMs and full-sequence residual adds remain separate. A packed QKV GGUF can select the common packed graph path, but that is weight-layout-dependent rather than shader fusion. |
| Qwen3 normalization/RoPE | Vulkan recognizes `RMS_NORM + MUL` and `RMS_NORM + MUL + ROPE`; the latter can extend through K-cache view/set-rows when its type, contiguity, head-width, and push-constant constraints hold. Q and K are still separate invocations. | ModernBERT ordinary `NORM -> MUL -> ADD` does not match these RMS patterns. |
| Attention | `FLASH_ATTN_EXT` can put QK, scale/mask, online softmax, and PV in one Vulkan shader family; split-K can add a reduction dispatch. | Without flash selection, the graph is QK GEMM -> one scale+mask softmax shader -> PV GEMM. The wave-2 command did not record flash selection, so this report does not assume it ran. |
| Matmul epilogues | `MUL_MAT + ADD` exists only for mat-vec (`ggml_nrows(mul) == 1`) under strict layout/type constraints. | It cannot fuse ordinary multi-token projection-plus-bias/residual work in these embedding cells. |
| Layout/copy | `VIEW`, `RESHAPE`, and `PERMUTE` are metadata and launch no shader. | `CONT` and `CPY` are separate copy shaders unless the K-cache extended RMS/RoPE pattern absorbs them. |

Exact-revision sources: [`build_attn_mha` flash selection](https://github.com/ggml-org/llama.cpp/blob/b4e3dc613baa92a3884d4151e3d631395c81934a/src/llama-graph.cpp#L2040-L2105), [Vulkan flash dispatch](https://github.com/ggml-org/llama.cpp/blob/b4e3dc613baa92a3884d4151e3d631395c81934a/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L9860-L9985), [RMS/RoPE fusion detection](https://github.com/ggml-org/llama.cpp/blob/b4e3dc613baa92a3884d4151e3d631395c81934a/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L15787-L15812), [fusion constraints and mat-vec-only add](https://github.com/ggml-org/llama.cpp/blob/b4e3dc613baa92a3884d4151e3d631395c81934a/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L15128-L15194), [packed QKV graph path](https://github.com/ggml-org/llama.cpp/blob/b4e3dc613baa92a3884d4151e3d631395c81934a/src/llama-graph.cpp#L1179-L1198), [FFN/GLU construction](https://github.com/ggml-org/llama.cpp/blob/b4e3dc613baa92a3884d4151e3d631395c81934a/src/llama-graph.cpp#L1259-L1385), and [GLU pipeline selection](https://github.com/ggml-org/llama.cpp/blob/b4e3dc613baa92a3884d4151e3d631395c81934a/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L10458-L10481).

The important comparison is not merely dispatch count. llama.cpp's graph still has separate norm, attention, and matrix nodes in these paths; its advantage can therefore be consistent with materially better large-matrix kernels and device-local tensor placement. Any shader epilogue fusion is bounded by the small non-GEMM stage costs above.

## Ranked optimization plan (not implemented)

Predictions below are explicit fractions of measured stage budgets. Overlapping candidates cannot be added as independent savings.

1. **Put persistent weights in a device-local arena and stage uploads.** This changes allocation/transfer policy, not math. It touches the 11,956.4 ms ModernBERT and 42,329.2 ms Qwen3 large-linear budgets. A conservative 10–35% stage reduction predicts **1.20–4.18 s** and **4.23–14.82 s** per pass; the hard Amdahl caps are 11.96 s and 42.33 s. The Qwen layer-18 cliff is the pass/fail discriminator.
2. **Specialize the cooperative large-linear kernel for these M/N/K regimes.** Increase work reuse/occupancy and reduce fp32 writeback traffic without changing accumulation order until parity proves otherwise. A 20–40% reduction of the same measured large-linear budgets predicts **2.39–4.78 s** for ModernBERT and **8.47–16.93 s** for Qwen3, capped by 11.96/42.33 s. This overlaps candidate 1 and must be measured after it.
3. **Combine Qwen3 q/k/v and gate/up projections to reuse the input activation.** Qwen3's QKV plus MLP-up budget is 26,292.8 ms. A 5–12% reduction predicts **1.31–3.16 s**, capped at 26.29 s. ModernBERT already uses combined QKV and MLP-up matrices, so this is Qwen-only.
4. **Keep final pool/L2 on the GPU and return only embeddings.** Host readback is 640.1 ms for ModernBERT and 701.7 ms for Qwen3. A 70–95% reduction predicts **0.45–0.61 s** and **0.49–0.67 s**, with hard caps of 0.640/0.702 s.
5. **Only after the GEMM gate, fuse adjacent layout/pointwise epilogues.** The combined budgets are 1,099.5 ms and 2,071.9 ms. A 25–50% reduction predicts **0.27–0.55 s** and **0.52–1.04 s**, capped by 1.10/2.07 s.
6. **Do not prioritize barriers or descriptor sets.** Their combined measured ceilings are **88.1 ms** for ModernBERT and **161.9 ms** for Qwen3, only 0.60% and 0.35% of GPU time.

At the measured serving shapes, llama.cpp's wave-2 throughput implies 8.351 s for ModernBERT and 28.809 s for Qwen3. The instrumented owned path took 16.115 s and 48.806 s, so it must remove 7.765 s (48.2%) and 19.998 s (41.0%) end to end. If large-linear GEMM were the only target, that requires 64.9% of ModernBERT's linear budget and 47.2% of Qwen3's. The measured budgets make Qwen3 plausible and ModernBERT difficult but not physically impossible; MiniLM already proves that the owned cooperative substrate can beat llama.cpp on this GPU.

**Go/no-go:** run exactly one device-local-arena plus large-GEMM wave. Continue only if the aggregate large-linear stages fall by at least 50% on Qwen3 and 55% on ModernBERT while frozen parity remains green; otherwise stop the owned deep-family Vulkan effort on this hardware class. Those gates do not by themselves guarantee a win, but missing them makes the remaining Amdahl budget inadequate.

## Runs, parity, and thermal discipline

Published cell order was MiniLM, ModernBERT, Qwen3, then the final MiniLM parity cell. Every cell was a fresh process and had a 60-second idle interval before launch. No TDR occurred. The runtime exposes no per-repeat GPU clock. Pass-one to pass-three wall drift was -1.33% for MiniLM (`1.5240 -> 1.5038 s`), +0.94% for ModernBERT (`15.9646 -> 16.1151 s`), and +2.29% for Qwen3 (`47.7137 -> 48.8061 s`); pass-two to pass-three drift was +0.10%, +0.81%, and +0.59%. Post-sequence ACPI zones were 35.05 C, 14.05 C, and 20.05 C, but firmware does not identify any as GPU junction, so wall drift is the only trustworthy thermal signal. The active scheme remained Turbo.

The final instrumentation-enabled 400-row MiniLM cell passed unchanged math at cosine `0.9999996779` and top-10 overlap `0.998750` (gates `>=0.9999` and `>=0.995`). It produced 66,783 real tokens and 77,312 padded tokens at the same policy-1 serving buckets. Timestamp writes and profile retrieval therefore did not perturb parity.

Representative commands (with a 60-second idle before each) were:

```bat
set SYNAPSE_VULKAN_PROFILE=1
set SYNAPSE_VULKAN_PROFILE_OUT=C:\bench\attr-modern.ndjson
target\release\spike-unified-rt.exe ^
  --model C:\bench\model-modernbert ^
  --tokenizer C:\bench\model-modernbert\tokenizer.json ^
  --corpus C:\bench\data\modernbert-corpus-400.jsonl ^
  --limit 400 --dtype f16 --device vulkan --vulkan-gemm cooperative ^
  --shapes bucketed --bucket-policy 1 --passes 3 ^
  --out C:\bench\attr-modern.json
```

Qwen3 substituted `model-qwen3`, `qwen3-corpus-400.jsonl`, and the Qwen output paths. MiniLM used `model-minilm` and `minilm-corpus-1000-official.jsonl`. The final parity command additionally supplied `--reference C:\bench\data\ort-minilm-1000-vectors-official.jsonl`.
