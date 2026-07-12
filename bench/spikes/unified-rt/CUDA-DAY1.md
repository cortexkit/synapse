# CUDA day 1: MiniLM on the RTX 4090 development rig

## Verdict

The MiniLM-only CUDA provider passes the frozen 400-row gate with f16 storage/operands and fp32 accumulation: mean cosine `0.9999995953` and mean top-10 overlap `0.999000`. Every exact-shape CUDA Graph was compared with the same uncaptured calls during plan construction and was bit-exact. The graph-on and graph-off corpus JSONL files were also byte-identical.

CUDA Graph submission is **not** the measured limiter. Across five fresh processes, graph-on averaged `481,984.5 tok/s` and graph-off averaged `498,156.3 tok/s`; graph-on was `3.25%` slower by the means, with wide and overlapping ranges caused by the rental board's active software power cap. The honest day-1 number is therefore the repeat range, not a peak prediction: graph-on `418,319–570,122 tok/s`, graph-off `391,312–569,347 tok/s`, labeled **dev-rig-4090**, not a consumer floor.

A stable fast repeat's CUDA-event profile attributed `54.4%` of device time to GEMMs, `22.5%` to fused scale+mask+softmax, `22.8%` to the remaining pointwise/transposition kernels, and `0.3%` to pooling. Nsight Compute counters were blocked by `ERR_NVGPUCTRPERM`, so this result does not invent a GEMM-utilization number. Per the council decision tree, the next implementation should remove/fuse QKV and context layout traffic, then reprofile on a counter-enabled host. Flash attention remains deferred until score/softmax traffic is shown to dominate after those layout fixes.

## Scope and architecture

This provider intentionally implements only all-MiniLM-L6-v2. ModernBERT, Qwen3, sparse attention, flash attention, graph updates, concurrent arenas, graph serialization, TF32, and pure-f16 accumulation remain out of scope.

`--device cuda --dtype f16` enters the existing family-keyed `block_forward` seam. The MiniLM typed context selects either the Metal backend or a CUDA backend; other families are rejected before context creation. The CUDA backend uses a small raw CUDA C++ FFI shim compiled by `build.rs`, matching the repository's Objective-C bridge architecture. This was preferred over a Rust CUDA wrapper because it exposes cuBLASLt algorithm objects, workspace sizes, CUDA stream capture, and CUDA events directly without adding a dependency or hiding pointer ownership.

For each exact `(batch, sequence)` shape, construction performs this order:

1. Upload model-owned f16 weights and biases once; retain layer-normalization scale and bias as fp32 device allocations.
2. Allocate a 256-byte-aligned, stable-address activation arena, compact mask, fp32 pooled output, and the maximum workspace selected for that shape.
3. Create descriptors and select one concrete cuBLASLt algorithm for each unique GEMM class outside capture.
4. Run the full sequence uncaptured and collect per-stage CUDA events.
5. Capture compute only on one stream, instantiate the graph, restore the original input, launch it, and require bit-exact pooled output against the uncaptured run.
6. Keep both paths. H2D input/mask copies occur before compute and D2H pooled-vector copies after compute on the same stream.

The corpus's exact shapes are eagerly constructed before `infer_wall_s`, so algorithm selection, warmup, capture, instantiate, first launch, and captured/uncaptured comparison are cold-load work. The graph retains 85 device calls per bucket: 14 calls per encoder layer across six layers, plus fused mean-pool/L2. Graph-on reduces those calls to one host graph submission per bucket but does not erase their device-side boundaries or score workspace traffic.

The retained kernels are deliberately stage-fused rather than one-kernel-per-op chains:

- one QKV bias plus BHSD transpose kernel after three projections;
- one fp32-accumulator scale+padding-mask+softmax kernel;
- one residual+bias+layer-normalization kernel with fp32 statistics and fp32 norm parameters;
- one bias+exact-GELU kernel;
- one mean-pool+L2 kernel before readback.

Dense materialized attention scores are f16 and bounded by sequence length 512. QK, PV, projection, and MLP GEMMs use `CUDA_R_16F` operands/output with `CUBLAS_COMPUTE_32F`.

## Correctness gate

| Path | Rows | Mean cosine | Mean top-10 overlap | Gate |
|---|---:|---:|---:|---|
| graph-on | 400 | `0.9999995953` | `0.999000` | PASS |
| graph-off | 400 | `0.9999995953` | `0.999000` | PASS |

Thresholds were cosine `>= 0.9999` and overlap `>= 0.995`. The run used 69,596 input tokens. Internal plan logs reported `captured_exact=true` for the warmup shape and all five corpus shapes. `cmp /tmp/cuda-graph-on-vectors.jsonl /tmp/cuda-graph-off-vectors.jsonl` also succeeded.

## Graph-on versus graph-off

Each row is a fresh process. Throughput excludes eager shape construction but includes CPU tokenization/embedding assembly, H2D input/mask, GPU compute, and D2H pooled vectors.

| Repeat | Graph-on tok/s | Graph-off tok/s |
|---:|---:|---:|
| 1 | 423,577.5 | 441,696.4 |
| 2 | 434,376.3 | 536,867.0 |
| 3 | 563,527.5 | 391,312.1 |
| 4 | 418,319.3 | 551,558.9 |
| 5 | 570,122.1 | 569,347.2 |
| Mean | **481,984.5** | **498,156.3** |
| Median | 434,376.3 | 536,867.0 |
| Range | 418,319–570,122 | 391,312–569,347 |

The variation tracks the fingerprinted P2/software-power-cap state; neither mode has a defensible repeatable win. The important result is negative: host graph submission did not close a stable gap, so adding more graph machinery is not the next optimization.

## Cold-load phase breakdown

The table is graph-on repeat 1. `cold_load_s` was `820.706 ms`. The residual includes safetensor/config/tokenizer/corpus loading, CPU embedding warmup, arena allocation, the final eager-preload execution of each newly verified plan, and host bookkeeping not isolated by the CUDA shim.

| Phase | Time |
|---|---:|
| CUDA context initialization | 116.920 ms |
| Stream + cuBLASLt handle | 10.679 ms |
| Persistent weight upload | 117.649 ms |
| Algorithm selection, six shapes | 80.315 ms |
| Full uncaptured warmups/profile | 86.130 ms |
| Graph capture | 0.629 ms |
| Graph instantiate | 0.833 ms |
| First captured launches + synchronization | 78.711 ms |
| Other load/host/eager-preload work | 328.840 ms |
| **Total cold load** | **820.706 ms** |

Capture and instantiation are negligible. Context startup, weight upload, first cuBLASLt heuristic initialization, warmup/first launch, and ordinary host loading dominate cold load.

## Stage profile and launch counts

Nsight Systems was unavailable. CUDA event timers surround retained stage groups in the uncaptured verification sequence. The representative values below are graph-off repeat 5, whose total and throughput matched the other fast repeats. Power-capped slow repeats are retained in the result set rather than discarded.

| Stage | Time | Device share |
|---|---:|---:|
| Projection + MLP GEMMs | 16.114 ms | 38.9% |
| Attention QK + PV GEMMs | 6.387 ms | 15.4% |
| Fused scale + mask + softmax | 9.314 ms | 22.5% |
| Bias/GELU, residual/norm, QKV/context transposes | 9.435 ms | 22.8% |
| Mean pool + L2 | 0.130 ms | 0.3% |
| **Profiled device total** | **41.380 ms** | **100%** |

Each shape has 85 retained calls and the five steady corpus buckets therefore execute 425 device calls. Graph-on makes five host graph launches; graph-off submits all 425 calls. Their overlapping throughput distributions show that launch submission is not the bottleneck.

`ncu --set basic` was attempted on a 20-row representative run, but the host returned `ERR_NVGPUCTRPERM`. Event timing is therefore the available profile; no tensor-core utilization, occupancy, or bandwidth percentage is claimed.

## Selected algorithms and workspaces

All rows use cuBLASLt `120803`, f16 operands/output, and `CUBLAS_COMPUTE_32F`. `HH` is hidden-to-hidden, `HI` hidden-to-intermediate, `IH` intermediate-to-hidden. Entries are `algorithm id / workspace bytes`. Algorithms were selected before capture and retained unchanged.

| Shape | Arena bytes | Max workspace | HH | HI | IH | QK | PV |
|---|---:|---:|---|---|---|---|---|
| 1x128 warmup | 2,166,784 | 786,432 | 21 / 0 | 21 / 0 | 21 / 786,432 | 21 / 0 | 21 / 0 |
| 148x163 | 427,867,360 | 0 | 30 / 0 | 6 / 0 | 30 / 0 | 24 / 0 | 24 / 0 |
| 110x189 | 381,708,496 | 0 | 5 / 0 | 6 / 0 | 5 / 0 | 24 / 0 | 24 / 0 |
| 90x210 | 356,533,696 | 0 | 30 / 0 | 6 / 0 | 30 / 0 | 23 / 0 | 23 / 0 |
| 51x243 | 243,600,904 | 1,164 | 30 / 0 | 6 / 0 | 5 / 1,164 | 24 / 0 | 24 / 0 |
| 1x300 | 6,311,296 | 921,600 | 21 / 0 | 6 / 0 | 6 / 921,600 | 23 / 0 | 23 / 0 |

An initial unaligned arena exposed a cuBLASLt `NOT_SUPPORTED` result for an odd-sized shape. Aligning every arena slice to 256 bytes fixed that descriptor/address contract. After alignment, every selected algorithm ran uncaptured, captured successfully, and produced exact output; no capture-specific algorithm instability was observed.

## Rig and reproducibility

The full fingerprint is in [`results/cuda-day1/rig-fingerprint.txt`](results/cuda-day1/rig-fingerprint.txt). Important values are RTX 4090, driver 595.80, VBIOS `95.02.3C.00.4E`, CUDA toolkit 12.8, cuBLASLt 120803, a 425 W requested power limit, 73 C, P2, software power cap active, and no other GPU process. The frozen reference SHA-256 is `7589eea5148562f6141c864d3357bab5dceb6881055afcf93b80efbdcae7d24d`; the corpus SHA-256 is `b7c8424f5b6bc5df61d96146a03642671789c1d41cbe37e82864117330996a10`.

Raw lane results and a machine-readable summary are in [`results/cuda-day1/`](results/cuda-day1/). Raw stderr profile logs remain on the measurement host under `/tmp/cuda-*.log`; all values transcribed here are also represented in `summary.json` where applicable.

## Commands executed

The repository bundle on the rig lacked sibling workspace path dependencies, so the two required workspace members (`bench/harness` and `bench/spikes/unified-rt`) were copied unchanged into `/work/cuda-spike` with an equivalent minimal workspace manifest. CUDA source was rsynced from the task worktree; the source hashes are in the fingerprint.

Build and gates:

```sh
source ~/.cargo/env
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cargo build --release --features cuda \
  --manifest-path bench/spikes/unified-rt/Cargo.toml
cargo test --features cuda --manifest-path bench/spikes/unified-rt/Cargo.toml
cargo clippy --features cuda --manifest-path bench/spikes/unified-rt/Cargo.toml -- -D warnings
cargo fmt --manifest-path bench/spikes/unified-rt/Cargo.toml -- --check
```

Measured graph-on command (graph-off changes only `--cuda-graphs false`, output names, and label):

```sh
./target/release/spike-unified-rt \
  --model /work/model --tokenizer /work/model/tokenizer.json \
  --corpus /work/data/minilm-corpus-1000-official.jsonl \
  --reference /work/data/ort-minilm-1000-vectors-official.jsonl \
  --limit 400 --out /tmp/cuda-graph-on.json \
  --vectors-out /tmp/cuda-graph-on-vectors.jsonl \
  --dtype f16 --device cuda --cuda-graphs true \
  --model-label minilm-f16-dev-rig-4090-graph-on
cmp /tmp/cuda-graph-on-vectors.jsonl /tmp/cuda-graph-off-vectors.jsonl
```

The same command was run in five fresh processes per mode. Fingerprint commands are listed verbatim in `rig-fingerprint.txt`. Profiler fallback command:

```sh
/usr/local/cuda/bin/ncu --set basic --target-processes all --csv \
  --log-file /tmp/cuda-ncu-basic.csv \
  ./target/release/spike-unified-rt \
  --model /work/model --tokenizer /work/model/tokenizer.json \
  --corpus /work/data/minilm-corpus-1000-official.jsonl \
  --limit 20 --out /tmp/cuda-ncu.json \
  --dtype f16 --device cuda --cuda-graphs false
# ERR_NVGPUCTRPERM: performance counters unavailable to the container
```


## Wave 2: layout fusion

Wave 2 executed the day-1 layout branch and pruned it. A transpose-equivalent column-major cuBLASLt descriptor made `CUBLASLT_EPILOGUE_BIAS` available for Q, K, and V while preserving physical BSH output. QK then consumed each BSH head through a strided batch, and PV used a matching output view to write context directly into BSH columns. The full candidate removed both explicit transpose kernels and reduced the fused arena for the largest corpus bucket from `427,867,360` to `316,703,968` bytes while preserving 256-byte slice alignment.

The direct views had a cost: QK and PV could no longer flatten `batch * heads` into one strided batch because the BSH offset between the last head of one batch item and the first head of the next item is not constant. They therefore ran one `batch`-wide cuBLASLt call per head. Retained calls rose from 85 to 205 per bucket, and the stable event profile regressed. Per the decision tree, all source changes were reverted; the raw measurements remain as negative evidence.

### Per-fusion runs

Each graph mode below is a fresh 400-row process. CUDA-event values come from the graph-off process and sum the uncaptured construction profiles for the warmup and five corpus buckets. Every run was P2 with the 425 W software power cap active. Post-run clocks show why cross-process deltas are not by themselves retention evidence.

| Candidate | Graph-off tok/s and state | Graph-on tok/s and state | Projection + MLP GEMMs | Attention GEMMs | Softmax | Pointwise + transposes | Pool | Device total | Result |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Baseline | 399,500.1 — P2, 2190 MHz, 423.35 W | 596,299.9 — P2, 2790 MHz, 314.08 W | 29.543 ms | 20.808 ms | 22.452 ms | 18.942 ms | 0.150 ms | 91.895 ms | retained control |
| QKV bias epilogue + direct BSH views | 424,940.4 — P2, 2580 MHz, 422.30 W | 573,456.7 — P2, 2790 MHz, 232.55 W | 32.154 ms | 25.790 ms | 22.059 ms | 10.774 ms | 0.147 ms | 90.924 ms | continue to context test |
| Context-direct PV only | 424,556.1 — P2, 2280 MHz, 424.03 W | 418,536.7 — P2, 2250 MHz, 423.20 W | 23.009 ms | 30.584 ms | 15.296 ms | 18.779 ms | 0.149 ms | 87.817 ms | independent branch; per-head PV cost visible |
| QKV + context direct views | 561,882.3 — P2, 2550 MHz, 422.52 W | 580,211.0 — P2, 2790 MHz, 185.95 W | 28.055 ms | 15.827 ms | 15.217 ms | 8.379 ms | 0.148 ms | 67.626 ms | recheck at stable clocks |

All candidates passed the 400-row gate. The QKV and full-layout candidates produced mean cosine `0.9999996269` and mean top-10 overlap `0.999000`; the context-only candidate produced mean cosine `0.9999995934` and overlap `0.999250`. Every exact-shape construction printed `captured_exact=true`, graph-on and graph-off both completed, and `cmp` found their vector JSONL files byte-identical.

The QKV bias descriptor selected the same algorithm ids and workspace sizes as the corresponding hidden-to-hidden GEMM for all six shapes. Capture therefore remained stable; no capture-contract blocker occurred. The first row-major bias descriptor had no supported CUDA 12.8 heuristic, so the measured candidate used the mathematically equivalent column-major transpose formulation.

### Stable profile and retention decision

The cap produced wide process-level variance even though every row remained P2. The retention comparison therefore uses two fast-clock CUDA-event profiles: the final reverted implementation's uncaptured profile from the graph-on process and the full candidate's `wave2-layout-fused-off-r6` profile. Both had 2790 MHz post-run SM clocks.

| Stage | Retained baseline | Full layout candidate | Delta |
|---|---:|---:|---:|
| Projection + MLP GEMMs | 16.152 ms | 21.143 ms | +4.991 ms |
| Attention QK + PV GEMMs | 6.388 ms | 8.670 ms | +2.282 ms |
| Fused scale + mask + softmax | 9.345 ms | 9.000 ms | -0.345 ms |
| Bias/GELU, residual/norm, layout kernels | 9.434 ms | 5.761 ms | -3.673 ms |
| Mean pool + L2 | 0.131 ms | 0.133 ms | +0.002 ms |
| **Profiled device total** | **41.450 ms** | **44.707 ms** | **+3.257 ms (+7.86%)** |

The candidate removed `3.673 ms` of pointwise/layout work, but fragmenting QK/PV into per-head batches added `7.273 ms` to GEMM stages. The stable device total regressed `7.86%`, so both layout changes were reverted. The existing fused residual+bias+layer-normalization and bias+GELU chains were not changed because the profile gave no independent reason to split or rewrite them.

### Retained throughput range

The final source is byte-for-byte the day-1 CUDA implementation. The table includes five wave-2 fresh processes per mode; state was sampled immediately after each process. Low sampled power can mean the process had already completed, so SM clock and the active power-cap flag are more useful than the instantaneous wattage alone.

| Repeat | Graph-off tok/s and state | Graph-on tok/s and state |
|---:|---|---|
| 1 | 399,500.1 — P2, 2190 MHz, 423.35 W | 596,299.9 — P2, 2790 MHz, 314.08 W |
| 2 | 428,017.5 — P2, 2175 MHz, 423.83 W | 551,886.0 — P2, 2790 MHz, 130.89 W |
| 3 | 567,336.9 — P2, 2790 MHz, 172.72 W | 563,667.5 — P2, 2790 MHz, 147.88 W |
| 4 | 505,764.9 — P2, 2790 MHz, 133.37 W | 428,655.0 — P2, 2310 MHz, 352.65 W |
| 5 | 591,995.8 — P2, 2790 MHz, 394.20 W | 464,831.9 — P2, 2790 MHz, 154.26 W |
| **Range** | **399,500–591,996 tok/s** | **428,655–596,300 tok/s** |

The full candidate's exploratory ranges were graph-off `472,415–599,246 tok/s` across six processes and graph-on `420,184–580,414 tok/s` across five. Every throughput sample, including its P-state, post-run SM clock, and power reading, is recorded in [`wave2-summary.json`](results/cuda-day1/wave2-summary.json). Because those ranges overlap and the board remained software-power-capped, they do not override the stable CUDA-event regression.

### Wave-2 verdict

Flash-attention promotion does **not** trigger. After reverting the failed layout branch, GEMMs again account for `54.4%` of the stable device profile, softmax/score traffic `22.5%`, pointwise plus transposes `22.8%`, and pooling `0.3%`. The next branch is GEMM layout and algorithm work: preserve a single `batch * heads` attention batch while eliminating layout traffic, likely through a genuinely fused projection/output path rather than per-head cuBLASLt views. Flash attention remains behind evidence that score/softmax traffic dominates after a non-regressing layout strategy.

Raw `wave2-*.json` files, the machine-readable delta table, power annotations, and the updated rig fingerprint are in [`results/cuda-day1/`](results/cuda-day1/). Nsight Compute remains blocked by `ERR_NVGPUCTRPERM`; no counter-derived utilization claim is made.
