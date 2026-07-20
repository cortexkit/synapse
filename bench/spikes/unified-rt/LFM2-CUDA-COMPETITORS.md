# LFM2 CUDA competitor sweep — RTX 4090

**Measured:** 2026-07-20. **Protocol:** one request at a time, raw completion, greedy (`temperature=0`, `top_k=1`, `top_p=1`), 64-token cap, first 12 rows of [`decode-prompts.jsonl`](decode-prompts.jsonl), two repeats. Decode excludes prefill. Raw per-prompt numbers are in [`LFM2-CUDA-COMPETITORS.json`](LFM2-CUDA-COMPETITORS.json).

## Results

`DQ` means the cell was timed but disqualified as a product baseline after the fixed `completion-08` correctness spot-check. `n/r` means the engine did not expose or the run did not capture that value. Owned LFM2 rows are the validated checkpoints requested by the original sweep; the cross-engine rows use the current same-checkpoint LFM2.5 family as directed by the follow-up measurement recipe.

### OWNED: validated LFM2 family

| Model | dtype | decode tok/s (median) | cold load | VRAM peak | W avg / peak | quality |
|---|---:|---:|---:|---:|---:|---|
| LFM2-350M | fp32 | **466.36** | n/r | 1,894 MiB | 74 / 209 | not gated |
| LFM2-350M | Q8_0 | **740.41** | n/r | 2,330 MiB | 71 / 188 | not gated |
| LFM2-700M | fp32 | **249.47** | n/r | 3,490 MiB | 94 / 242 | not gated |
| LFM2-700M | Q8_0 | **510.16** | n/r | 4,346 MiB | 68 / 299 | not gated |
| LFM2-1.2B | fp32 | **177.20** | n/r | 5,008 MiB | 88 / 242 | prior gate 20/20 exact |
| LFM2-1.2B | Q8_0 | **361.63** | n/r | 6,298 MiB | 65 / 344 | prior quant gate; informational |

The LFM2-350M and -700M family loader paths accepted both fp32 and Q8_0. These runs used the shipped CUDA feature and no code changes. The 1.2B numbers reproduce the existing 4090-class result: 178.35 fp32 / 361.80 Q8_0 in `QUANT-DECODE.md` within normal host variation.

### Cross-engine same-checkpoint: LFM2.5

| Model | Engine | dtype | decode tok/s (median) | cold load | VRAM at rest | W avg / peak | quality |
|---|---|---:|---:|---:|---:|---:|---|
| LFM2.5-230M | llama.cpp | F16 GGUF | **1,070.20** | 0.75 s | 978 MiB | 113 / 142 | **DQ** |
| LFM2.5-230M | llama.cpp | Q8_0 GGUF | **1,289.50** | 0.75 s | 776 MiB | 104 / 124 | **DQ** |
| LFM2.5-230M | vLLM | BF16 safetensors | **539.21** | 25.01 s | 18,500 MiB | 88 / 119 | **DQ** |
| LFM2.5-230M | SGLang | BF16 safetensors | **791.64** | 33.03 s | 18,784 MiB | 99 / 138 | **DQ** |
| LFM2.5-350M | llama.cpp | F16 GGUF | **808.44** | 1.01 s | 1,240 MiB | 152 / 232 | **DQ** |
| LFM2.5-350M | llama.cpp | Q8_0 GGUF | **1,056.82** | 1.00 s | 926 MiB | 137 / 191 | **DQ** |
| LFM2.5-350M | vLLM | BF16 safetensors | **592.20** | 38.02 s | 18,376 MiB | 100 / 177 | **DQ** |
| LFM2.5-350M | SGLang | BF16 safetensors | **796.29** | 34.02 s | 18,776 MiB | 129 / 203 | **DQ** |
| LFM2.5-1.2B-Instruct | llama.cpp | F16 GGUF | **352.56** | 1.26 s | 2,846 MiB | 209 / 269 | **DQ** |
| LFM2.5-1.2B-Instruct | llama.cpp | Q8_0 GGUF | **546.90** | 1.00 s | 1,804 MiB | 170 / 253 | **DQ** |
| LFM2.5-1.2B-Instruct | vLLM | BF16 safetensors | **357.64** | 39.02 s | 18,516 MiB | 150 / 260 | **DQ** |
| LFM2.5-1.2B-Instruct | SGLang | BF16 safetensors | **353.15** | 30.02 s | 18,800 MiB | 205 / 275 | **DQ** |

The owned loader also accepted **LFM2.5-1.2B-Instruct** without source changes: fp32 **177.01** tok/s and Q8_0 **367.79** tok/s (VRAM 5,008 / 6,298 MiB; representative power 77 / 246 W and 58 / 343 W). It rejected LFM2.5-230M and -350M before timing with the honest error `parse LFM2 backbone config: missing field rope_theta`; those cells were not hacked or substituted.

## Rig, versions, and telemetry

- Dedicated Vast.ai contract **45369168**, one RTX 4090 24 GiB, reliability **0.997**, driver **590.48.01**, CUDA 12.8 image; the PyTorch wheels reported CUDA 13.0. A 1024x1024 CUDA matmul smoke kernel passed before installing Rust or Python engines.
- Rental rate was **$0.3422/hour**. Vast's final charge was **$0.314** (0.9141564 GPU hours plus disk; no transfer charge). The contract was destroyed after capture and `show instances-v1` showed only the pre-existing campaign contract **45265161**.
- OWNED was `spike-unified-rt`, release, `--features cuda`, fp32 storage/accumulation, uncaptured CUDA launches, and `--weight-quant q8-0` for quant rows. Its aggregate runtime rate is over the generated-token wall time per repeat.
- llama.cpp was built from current master commit `178a6c44937154dc4c4eff0d166f4a044c4fceba`, CUDA, `CMAKE_CUDA_ARCHITECTURES=89`. It served official LiquidAI F16 and Q8_0 GGUFs with one slot and no continuous batching. Cold load is process-to-health; VRAM is `nvidia-smi memory.used` at rest.
- vLLM was `0.25.1` with torch `2.11.0+cu130`; startup used plain `vllm serve`, BF16, max model length 4096, one sequence, 512 max batched tokens, and 0.75 GPU utilization. vLLM logged `Lfm2ForCausalLM` and served all three LFM2.5 checkpoints.
- SGLang was `0.5.15.post1` / `sglang-kernel 0.4.4`; release support loaded all three LFM2.5 checkpoints. The CUDA 13 wheel needed the image's `nvidia/cu13/lib` added to `LD_LIBRARY_PATH` because `libnvrtc.so.13` was not on the default loader path. This was a runtime-library path fix, not an architecture modification. The first release check without that path failed with `Could not load any common_ops library ... libnvrtc.so.13 ... cannot open shared object file`.
- Power was sampled with `nvidia-smi power.draw` every 500 ms during a large timed cell. Load averages were recorded at each window; the shared host varied roughly from 4–11 on the three-minute scale. No foreign GPU compute process was present during the final windows.

## Correctness and honest-notes

The fixed spot-check was `completion-08`, “In a causal transformer, the KV cache stores”. The LFM2.5-230M outputs from llama.cpp and SGLang degenerated into repeated VCC/VK-cache text; the 350M outputs repeated KV/VK-cache text and formula continuations; the 1.2B-Instruct rows continued an unrelated multiple-choice answer under raw completion. Those outputs are coherent enough to preserve as diagnostics but are not a reliable answer to the fixed prompt, so **all LFM2.5 competitor rows are DQ and must not be used as a product-quality baseline**. The sidecar preserves the per-prompt rates; the captured text was used for this spot-check decision and is not repeated in the numeric sidecar.

The engines are not byte-for-byte timing-equivalent. OWNED emits one aggregate rate per repeat; llama.cpp exposes native `timings.predicted_per_second`; vLLM and SGLang use client arrival timestamps between streamed token chunks, excluding the first chunk. EOS-short rows consequently have null/short decode samples. One-token EOS rows in llama.cpp expose a `1,000,000` tok/s sentinel in the raw timing object; it is retained in the sidecar, but it does not affect the reported medians. No prefill number is ranked against decode.

vLLM and SGLang reserve about 18.4–18.8 GiB even for the 230M checkpoint. Their VRAM cells are resting allocator reservations, not model weight sizes. SGLang's default hybrid path used CUDA graphs for decode after its warmup; no mamba/conv-state environment knobs were changed. The optional vLLM optimization-level-3 comparison was not run because the requested plain serving cell already consumed the available wall-time budget.

## Verdict

On the validated original LFM2 family, OWNED is already a useful CUDA baseline: Q8_0 lifts decode from 466/249/177 tok/s to 740/510/362 tok/s for 350M/700M/1.2B, about 1.59–2.96x depending on size, while retaining the known 1.2B correctness evidence. The same-checkpoint LFM2.5 diagnostic ladder shows llama.cpp ahead of the owned 1.2B-class result by about 1.9x at F16 and 1.5x at Q8_0, while vLLM/SGLang add tens of seconds of cold start and reserve almost the whole 24 GiB board. The small-model gaps are consistent with launch/orchestration overhead plus incomplete hybrid-conv specialization rather than pure DRAM bandwidth: Q8 scaling is substantial but below the byte-ratio ceiling, and server allocators dominate VRAM. Fixing the LFM2.5 loader config and certifying native f16 first are the next owned-runtime steps; competitor throughput is a directional diagnostic only until the raw-completion quality failure is resolved.
