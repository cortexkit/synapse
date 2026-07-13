# x86 CPU floor day 1: MiniLM on Zen 4

## Verdict

**Recommendation: keep ORT as the shipping CPU floor and park the owned x86 path.** At the production thread policy (8 physical cores), the owned f16 AVX-512 path delivered **12,250.2 tok/s**, 20.6% behind ORT and 46.0% behind CPU-only llama.cpp. Faer delivered **9,453.2 tok/s**, 38.7% behind ORT and 58.3% behind llama.cpp. Neither owned substrate wins the floor.

This is not a recommendation for another GEMM-only tuning wave. The hand path already spends 49.7% of its measured wall time outside timed GEMM calls. Its zero-GEMM Amdahl ceiling is about 24.6k tok/s, only 8.6% above the measured llama.cpp row, before accounting for any unavoidable GEMM work. Closing the remaining gap requires a whole-graph CPU effort: persistent packed layouts, fewer layout transforms and allocations, and fused pointwise/attention work. That is materially larger than this substrate spike.

ORT therefore remains the honest portable floor. The owned code and evidence are retained as a measured prototype, not a shipping provider.

## Default-policy result

All rows used the first 400 records of the canonical MiniLM corpus and the canonical tokenizer path. Throughput is computed from **66,783 real post-truncation tokens**, never an engine-reported or padded count. The owned runtime executed 76,507 padded batch tokens internally and reports that counter separately.

The thread policy is `ceil(available logical CPUs / 2) = 8`, pinned to CPUs `0-7`, which are the eight distinct physical Zen 4 cores. Owned values are pass 3 from two fresh processes; incumbent values are two fresh-process means.

| Path | Storage / compute | Tok/s repeats | Mean tok/s | vs ORT | vs llama.cpp |
|---|---|---:|---:|---:|---:|
| llama.cpp CPU-only | GGUF f16 / ggml CPU | 22,368.8, 22,999.2 | **22,684.0** | +47.0% | baseline |
| ORT CPU | ONNX f32 / ORT MLAS | 15,274.8, 15,588.7 | **15,431.8** | baseline | -32.0% |
| owned hand kernel | f16 RHS / f32 accumulate | 12,357.1, 12,143.3 | **12,250.2** | -20.6% | -46.0% |
| owned faer | f32 / f32 | 9,313.2, 9,593.2 | **9,453.2** | -38.7% | -58.3% |
| owned faer f16 experiment | f16 RHS expanded per call / f32 | 8,722.3 | **8,722.3** | -43.5% | -61.5% |

The f16 faer experiment confirms that faer 0.22.6 has no useful native mixed f16-storage/f32-compute path. Packing weights to f16 and expanding them for each call retained parity but was 7.7% slower than faer f32. It is not a candidate.

## Correctness gates

The ORT vectors staged with the rig are the frozen reference. Both required owned candidates pass cosine `>= 0.9999` and mean top-10 overlap `>= 0.995` on every measured pass at every thread count.

| Path | Mean cosine | Mean top-10 overlap | Gate |
|---|---:|---:|---|
| faer f32 | 0.999999999999 | 1.000000 | pass |
| hand f16 AVX-512 | 0.999999822613 | 0.999250 | pass |
| faer f16 experiment | 0.999999822541 | 0.999250 | pass |
| llama.cpp CPU-only | 0.999986649982 | not reported by incumbent lane | cosine pass |

The hand result uses f16-rounded weights and f32 FMA accumulation. Its small difference from faer f32 is expected and remains well inside both certification gates.

## Thread scaling

Candidate cells are pass 3 from one process; the 8-thread default cells above add a second fresh-process repeat. ORT and llama.cpp values below are two fresh-process means. Affinity used one physical core per thread through 8 threads, then both SMT siblings at 16 threads.

| Threads | Affinity | faer f32 tok/s | hand f16 tok/s | ORT tok/s | llama.cpp tok/s |
|---:|---|---:|---:|---:|---:|
| 1 | `0` | 1,612.1 | 2,232.1 | 3,179.7 | 3,915.0 |
| 2 | `0-1` | 3,073.2 | 4,196.2 | 6,017.0 | 7,453.0 |
| 4 | `0-3` | 5,837.3 | 7,474.6 | 10,426.9 | 13,699.0 |
| 8 | `0-7` | 9,313.2 | 12,357.1 | 15,431.8 | 22,684.0 |
| 16 | `0-15` | 11,640.5 | 13,951.7 | 15,453.6 | 19,577.6 |

SMT helps the owned paths but does not change the decision. At 16 threads, hand reaches 13,951.7 tok/s versus 15,453.6 for ORT and 19,577.6 for llama.cpp. llama.cpp is fastest at eight physical threads; ORT is effectively flat from 8 to 16.

## Phase split and Amdahl limit

Timers surround all GEMM substrate calls. `pointwise` is the remaining measured inference wall time, so it includes layer normalization, GELU, residual/bias work, attention packing and softmax, pooling, allocation, and tokenizer/runner overhead inside the workload window. It is deliberately an end-to-end residual rather than a claim of isolated device timing.

| Owned path, 8 threads | GEMM wall | Pointwise/residual wall | GEMM share | Logical GEMMs | Provider dispatches |
|---|---:|---:|---:|---:|---:|
| faer f32, two-process mean | 2.622 s | 4.444 s | 37.1% | 57,780 | 240 |
| hand f16, two-process mean | 2.742 s | 2.710 s | 50.3% | 57,780 | 240 |

The same repaired graph runs both substrates. The residual differs because the f32 faer path has a larger cache footprint and different Rayon scheduling/cache state; the residual is not treated as a pure pointwise microbenchmark.

For the hand path, removing every measured GEMM would reduce the 5.452 s pass only to 2.710 s, or about 24.6k tok/s. llama.cpp already completes the whole pass in about 2.94 s. A GEMM-only follow-up therefore has almost no credible margin to beat llama.cpp.

## Shared graph repair before measurement

The original Spike C CPU graph invoked both attention GEMMs separately for every document and every head. A preliminary 8-thread run produced **57,780 provider dispatches**, only 4.9k tok/s for hand and 4.4k tok/s for faer, both below 50% of ORT. Per the stop rule, measurement paused and the shared defect was repaired before any published cell counted.

The substrate-neutral repair:

1. packs `[batch, sequence, hidden]` Q/K/V into `[batch * heads, sequence, head_dim]`;
2. submits each QK and probability-times-V stage as one batched provider operation;
3. parallelizes attention packing/unpacking, mask-softmax rows, layer normalization, bias, residual adds, and GELU through the same capped Rayon pool; and
4. preserves 57,780 logical matrix products while reducing provider dispatches to **240** per corpus pass.

Parity was rerun after the repair and passed before timing resumed. This structure also gives a future Apple CPU provider a better graph, but it does not change the shipping decision in this report.

## Candidate implementation

### Faer library path

The library path pins `faer = 0.22.6`. It creates zero-copy row-major matrix views and passes transposed weight views without repacking. Large static projections use faer's capped Rayon parallelism; folded attention batches parallelize over independent head matrices and run each inner faer GEMM sequentially to avoid nested oversubscription.

### Hand-kernel path

The owned C microkernel is selected at runtime:

- AVX-512F + F16C + FMA: 4 rows by 16 output columns;
- AVX2 + F16C + FMA fallback: 4 rows by 8 output columns; and
- scalar compatibility fallback.

Weights are packed to contiguous K-by-N f16 storage. Static RHS packs are cached during warmup; dynamic attention RHS values are packed per batched call. Inputs and accumulators remain f32. The spike's generic tensor loader still retains original f32 weights alongside the f16 cache, so this prototype proves execution behavior but does not yet realize the full resident-memory saving a production loader could obtain.

## Rig and ISA scope

The rig is an AMD Ryzen 7 7700: 8 cores, 16 threads, 32 MiB L3, Ubuntu 24.04, rustc 1.97.0, GCC 13.3.0. `lscpu` reported the following relevant flags:

`fma f16c avx avx2 avx512f avx512dq avx512ifma avx512cd avx512bw avx512vl avx512_bf16 avx512vbmi avx512_vbmi2 avx512_vnni avx512_bitalg avx512_vpopcntdq`

AVX-512F is present, so this is an AVX-512 result rather than an AVX2 fallback result. The CPU does **not** report `avx512_fp16`; the hand path therefore converts f16 weights to f32 with the AVX-512 conversion instruction and performs f32 FMA. It does not claim native AVX-512-FP16 arithmetic.

All four canonical data digests matched `RIG-CPU.txt` immediately before work:

| File | SHA-256 |
|---|---|
| `minilm-corpus-1000.jsonl` | `b7c8424f5b6bc5df61d96146a03642671789c1d41cbe37e82864117330996a10` |
| `ort-minilm-1000-vectors.jsonl` | `7589eea5148562f6141c864d3357bab5dceb6881055afcf93b80efbdcae7d24d` |
| `modernbert-corpus-400.jsonl` | `b4ff00f6d2d9f0652146b7438c2ecd421746bcead466cccf18ec79e45ff79aa8` |
| `modernbert-ort-400-vectors.jsonl` | `d1fb6aaf48c36c8ed7b06b9c69e6244f01393e085d32f49b15194671f7a44000` |

The staged Hugging Face snapshot symlinks and ORT directory were incomplete on arrival. The exact public MiniLM and GGUF blobs were restored and checked against the staged blob hashes (`53aa5117...` and `797b70c4...`); official ONNX Runtime 1.23.2 was restored from its release archive. Corpus and reference data were never replaced.

## Incumbent construction

llama.cpp was cloned and built on the box at commit:

`91c631b21d6e5d09e9c6659efdf6baeef5a44ddb`

CMake configuration:

```text
-DCMAKE_BUILD_TYPE=Release
-DGGML_NATIVE=ON
-DGGML_OPENMP=ON
-DGGML_CUDA=OFF
-DGGML_VULKAN=OFF
-DGGML_METAL=OFF
-DLLAMA_CURL=OFF
```

CMake selected the CPU backend with `-march=native`. There was no CUDA target or CUDA build environment, and the lane additionally passed `--gpu-layers 0`. The measured server is therefore CPU-only.

ORT used the official 1.23.2 shared library through `ORT_DYLIB_PATH`, graph optimization level 3, and an explicit intra-op thread count for every cell.

## Load and pinning discipline

Every cell logged ISO time, `/proc/loadavg`, affinity, and the four highest-CPU processes before launch. The exact one-minute load averages were:

| Threads | faer / hand | ORT r1 / r2 | llama.cpp r1 / r2 |
|---:|---:|---:|---:|
| 1 | 3.24 / 1.28 | 6.89 / 5.28 | 3.66 / 2.91 |
| 2 | 1.12 / 1.67 | 4.06 / 3.60 | 2.48 / 2.41 |
| 4 | 1.86 / 2.81 | 3.36 / 3.41 | 2.43 / 2.55 |
| 8 | 3.05, 0.97 / 4.66, 3.04 | 3.46 / 3.82 | 2.67 / 3.10 |
| 16 | 4.89 / 6.81 | 4.24 / 5.18 | 3.49 / 3.49 |

Load averages were sometimes elevated by the immediately preceding owned cell because the one- and five-minute averages decay slowly. The live process snapshots showed no foreign process consuming even 0.1% CPU before any retained cell, so no cell crossed the 10% foreign-load discard rule. Raw timestamps, all three load averages, run-queue counts, and process snapshots are in `results/cpu-day1/*-scaling-load.log`.

## Commands

Builds:

```sh
cargo build --release -p spike-unified-rt -p lane-ort-embed -p lane-llama
cargo test --release -p spike-unified-rt cpu_backend

cmake -S /work/llama.cpp -B /work/llama.cpp/build-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF
cmake --build /work/llama.cpp/build-cpu --target llama-server -j8
```

Representative owned 8-thread cell (replace `hand` with `faer`):

```sh
taskset -c 0-7 target/release/spike-unified-rt \
  --model /work/model-minilm \
  --tokenizer /work/model-minilm/tokenizer.json \
  --corpus /work/data/minilm-corpus-1000.jsonl \
  --reference /work/data/ort-minilm-1000-vectors.jsonl \
  --limit 400 --dtype f32 --device cpu --cpu-gemm hand --cpu-threads 8 \
  --shapes exact --passes 3 --out /work/cpu-results/final-hand-t8.json
```

Representative ORT cell:

```sh
ORT_DYLIB_PATH=/work/bin/onnxruntime-linux-x64-1.23.2/lib/libonnxruntime.so \
LD_LIBRARY_PATH=/work/bin/onnxruntime-linux-x64-1.23.2/lib \
  taskset -c 0-7 target/release/lane-ort-embed \
  --model /work/model-minilm-onnx/model.onnx \
  --tokenizer /work/model-minilm/tokenizer.json \
  --corpus /work/data/minilm-corpus-1000.jsonl --limit 400 \
  --pooling mean --max-length 512 --attention-units 4000000 \
  --model-label minilm-ort-f32 --intra-threads 8 \
  --out /work/cpu-results/final-ort-t8-r1.json
```

Representative llama.cpp cell:

```sh
taskset -c 0-7 target/release/lane-llama embed \
  --model /work/model-minilm-gguf/all-MiniLM-L6-v2-ggml-model-f16.gguf \
  --tokenizer /work/model-minilm/tokenizer.json \
  --corpus /work/data/minilm-corpus-1000.jsonl --limit 400 \
  --reference /work/data/ort-minilm-1000-vectors.jsonl --min-parity 0.9999 \
  --model-label minilm-f16-llama-cpu --lane-label llama-cpu-embed \
  --server-binary /work/llama.cpp/build-cpu/bin/llama-server \
  --pooling mean --embd-normalize 2 --ctx-size 512 \
  --batch-size 4096 --ubatch-size 1024 --gpu-layers 0 --parallel 1 --threads 8 \
  --out /work/cpu-results/final-llama-t8-r1.json
```

Raw result JSON, load logs, the exact `lscpu` output, toolchain versions, build flags, model digests, and aggregate summary are committed under `results/cpu-day1/`.

## Scope boundary

MiniLM answered the floor question within the day-one bound, so gte-modernbert was not measured. The existing ModernBERT scalar path also bypasses the repaired provider-level batched attention seam and would require another graph refactor before it could produce a fair substrate comparison. Given that neither MiniLM candidate beats either incumbent, that work is not justified as part of this spike.
