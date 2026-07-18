# LFM2-1.2B competitive decode matrix — RTX 4090

**Measurement date:** 2026-07-18
**Model:** [`LiquidAI/LFM2-1.2B`](https://huggingface.co/LiquidAI/LFM2-1.2B) at revision `933cee00d754fb3bfe06c644c0cb95453f2d8bb2`
**Workload:** the first 12 rows of [`decode-prompts.jsonl`](decode-prompts.jsonl), raw completion mode, greedy sampling (`temperature=0`, `top_k=1`, `top_p=1`), maximum 64 new tokens, one request at a time.

## Result matrix

Rates are medians of the 12 per-prompt rates. A prompt may finish at EOS before the 64-token cap; no EOS-forcing was used. `Prefill†` is an API limitation note described below.

| Engine | Precision / artifact | Decode tok/s | Prefill tok/s | Cold load | VRAM at rest | Diverged prompts vs fp32 oracle | Rig |
|---|---|---:|---:|---:|---:|---:|---|
| **Owned** | fp32, resident CUDA path | **178.49** | **1,710.6** | n/r in source | n/r in source | **0/20 (0%)** | owned rig |
| llama.cpp | F32, derived GGUF | 184.8 | 707.8 | 2.01 s | 5,076 MiB | 2/12 (16.7%) | llama rig |
| llama.cpp | F16, official GGUF | 334.7 | 1,097.0 | 1.45 s | 2,844 MiB | 1/12 (8.3%) | llama rig |
| llama.cpp | Q8_0, official GGUF | 521.4 | 1,757.6 | 1.01 s | 1,802 MiB | 6/12 (50.0%) | llama rig |
| llama.cpp | Q4_K_M, official GGUF | **717.8** | **2,068.3** | 1.01 s | 1,310 MiB | 11/12 (91.7%) | llama rig |
| vLLM | F32, safetensors cast | 190.1 | 486† | 33.06 s | 18,478 MiB | 1/12 (8.3%) | serving rig |
| vLLM | F16, safetensors cast | 353.4 | 513† | 32.56 s | 18,504 MiB | 3/12 (25.0%) | serving rig |
| vLLM | AWQ/GPTQ | **unsupported** | — | — | — | — | no official checkpoint |
| SGLang | BF16, native safetensors | 349.7 | 940† | 20.48 s | 18,776 MiB | 6/12 (50.0%) | serving rig |
| SGLang | F16 | **not runnable** | — | — | — | — | release conv-state dtype error |
| SGLang | AWQ/GPTQ | **unsupported** | — | — | — | — | no official checkpoint |

The owned row is copied from [`LFM2-CUDA.md`](LFM2-CUDA.md): its correctness-executable N=12 run reported 1,710.6 prefill tok/s and 178.49 single-stream decode tok/s. Its source document reports token-exact CUDA fp32 decode on 20/20 prompts, but does not record cold-load or resting-VRAM readings, so those cells remain `n/r` rather than being inferred.

`†` vLLM and SGLang's OpenAI-compatible streaming APIs expose prompt-token count and time-to-first-token, but not the llama.cpp-style per-request prompt-evaluation timer. Their displayed prefill value is therefore `prompt tokens / TTFT`, including the first-token scheduling/evaluation path. It is a conservative prefill proxy, not a claim that those cells are directly comparable to llama.cpp's internal prompt-eval timer.

## Rig and provenance

| Label | Vast contract | GPU / driver | CUDA image / host | Use |
|---|---:|---|---|---|
| owned rig | `45249270` | RTX 4090 24 GiB, driver 595.58.03 | CUDA 12.6, 32 effective EPYC cores | copied source result |
| llama rig | `45257396` | RTX 4090 24 GiB, driver 575.57.08 | CUDA 12.8, 16 effective EPYC cores | llama.cpp build and four GGUF cells |
| serving rig | `45259750` | RTX 4090 24 GiB, driver 590.48.01 | CUDA 12.8, 16 effective EPYC cores | vLLM and SGLang cells |

All three are Ada RTX 4090 measurements, but they are not the same physical host. The owned source run used a driver 20.01.05 newer than the llama rig and 5.10.02 newer than the serving rig. The driver was at least 570 on every rig; CUDA 12.8 `nvcc` and the 590.48.01 driver were required for the vLLM 0.25.1 CUDA 13 wheel. GPU resting-VRAM values are `nvidia-smi memory.used` after health and before the timed warmup.

The final benchmark ledger was **$0.522**, below the $15 cap. It includes the final serving rig ($0.270), the llama rig ($0.211), and failed/stopped launch overhead ($0.006 plus the incompatible-driver probe at $0.035). All instances were destroyed; the post-run `show instances-v1` list was empty.

### Model artifacts

- Base safetensors SHA-256: `60fef6ef4481c533ce7427793bed50200b55b3c68d0d00c52bc56f207a9acecd`.
- Official GGUF repository: [`LiquidAI/LFM2-1.2B-GGUF`](https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF), downloaded 2026-07-18.
- Official `LFM2-1.2B-F16.gguf`: SHA-256 `0ddedfb8c5f7f73e77f19678bbc0f6ba2554d0534dd0feea65ea5bca2907d5f2`.
- Official `LFM2-1.2B-Q8_0.gguf`: SHA-256 `0d9ec100a0f33048168d1d5b9fb6403f4836adcbbe9c3f2ab7794c96ffee3c3b`.
- Official `LFM2-1.2B-Q4_K_M.gguf`: SHA-256 `55175400e3f509a9616227afeffd58d87e80b9f628a5d3d54ada884d85221fed`.
- No official F32 GGUF is published. The F32 cell is a transparent, comparable derivative of the pinned BF16 safetensors, produced with the same current llama.cpp converter used to verify the architecture: `convert_hf_to_gguf.py --outtype f32`. Derived F32 SHA-256: `e1c2e62c711e96c511a52f67668d7cad5e0a459435f812ca4d25f384346e960e`.
- No official LFM2-1.2B AWQ or GPTQ safetensors checkpoint was found. vLLM and SGLang quant cells are consequently unsupported; no self-quantized checkpoint was substituted.

## Support and parity findings

### llama.cpp

The current llama.cpp master build was commit `571d0d540df04f25298d0e159e520d9fc62ed121`, built with CUDA and `CMAKE_CUDA_ARCHITECTURES=89`. It loaded the derived F32 file and all three official GGUF files. `llama-server` was run as a raw completion server, not through a chat wrapper, with one slot and `--no-cont-batching`.

Tokenizer parity was checked before timing: llama.cpp and Transformers reported the same prompt token counts for all 12 rows: `6, 20, 8, 7, 12, 12, 14, 11, 9, 16, 13, 9`. The GGUF metadata carried BOS 1 and EOS 7, and no chat-template transformation was applied to the raw completion prompts. The F32 engine still diverged on two prompts after the token-count parity check; this is why the quality column is informational rather than a correctness gate.

The exact llama.cpp serving configuration was:

```text
-ngl all -c 4096 -b 4096 -ub 1024 -np 1
--no-cont-batching --flash-attn on --no-ui --perf
```

### vLLM

The latest PyPI release available during the run was `vllm==0.25.1`. Its startup log resolved the model as `Lfm2ForCausalLM`; both `--dtype float32` and `--dtype float16` served the pinned safetensors checkpoint. The server used:

```text
--dtype <float32|float16> --max-model-len 512
--max-num-seqs 1 --max-num-batched-tokens 512
--gpu-memory-utilization 0.75
```

Requests were sequential and `max-num-seqs=1`; vLLM's scheduler remains enabled because the release does not expose a true no-batching mode. No concurrent requests or batching tricks were used. The quant cells are unsupported because there is no official LFM2 AWQ/GPTQ checkpoint.

### SGLang

The latest PyPI release available during the run was `sglang==0.5.15.post1`. Its package contains an LFM2 model implementation and the server detected the LFM2 chat/tool parser. Native BF16 served successfully with one running request, radix caching disabled, and the default CUDA graph path.

The F16 cell was not forced into a result. With the default CUDA graph path, startup failed with `Expected conv_state.scalar_type() == input_type to be true`; retrying with the release-recommended graph-disabled fallback failed with the same dtype mismatch during the first prefill. This is a release/runtime implementation failure in the F16 LFM2 path, not an assertion that the architecture is absent. The honest cell verdict is therefore **not runnable**, while BF16 is measured above.

The measured BF16 command was:

```text
sglang serve --model-path /workspace/models/LFM2-1.2B
  --dtype bfloat16 --context-length 512 --tp-size 1
  --mem-fraction-static 0.75 --max-running-requests 1
  --disable-radix-cache --host 127.0.0.1 --port 8000
```

## Quality spot-check

The oracle was regenerated from the pinned safetensors with Transformers 5.14.1 and torch 2.11.0, fp32 weights/arithmetic, raw prompts, greedy sampling, and the same 64-token cap. Divergence means the generated token sequence differed at any position for that prompt; it is not a pass/fail gate. EOS-short outputs are compared at their natural length.

The divergence rates show two effects: normal fp32 implementation/tie-breaking differences already appear in the derived llama F32 and vLLM F32 rows, while quantization and lower-precision arithmetic add materially larger changes. In particular, the official Q8 and Q4 files are genuine published artifacts and are not being presented as fp32-equivalent models.

## What the ladder is worth

The owned fp32 decode rate is already close to the same-class llama.cpp and vLLM fp32 cells: 178.5 tok/s versus 184.8 and 190.1. The owned prefill result is much stronger than llama.cpp F32, although the serving-engine prefill values use the TTFT proxy and should not be ranked as exact apples-to-apples timings.

The missing owned F16 path is the clearest near-term opportunity. On this model and architecture, llama.cpp improves from F32 to F16 by 1.81x and vLLM improves by 1.86x. Applying that measured scaling to the owned 178.49 tok/s baseline gives an indicative owned-F16 range of roughly **323–332 tok/s**, with about half the resident weight storage. The actual owned result needs its own exactness gate and kernel measurement; this is a prioritization estimate, not a forecast guarantee.

The published llama.cpp quant ladder reaches 2.92x the owned fp32 rate at Q8_0 and 4.02x at Q4_K_M, but the gains are below the raw weight-byte ratios. The GGUF files are approximately 4.4 GiB (F32-derived), 2.2 GiB (F16), 1.2 GiB (Q8_0), and 0.68 GiB (Q4_K_M): the theoretical storage/bandwidth reductions are about 2.0x, 3.7x, and 6.5x, while measured decode scaling is about 1.8x, 2.9x, and 4.0x. Short-convolution state updates, GQA/cache work, dequantization, and fixed per-token orchestration consume the difference from a pure weight-bandwidth model.

**Priority:** implement and certify owned F16 first; it should nearly double decode while reducing weight traffic and VRAM with a manageable quality target. Follow with Q8_0 as the quality/performance compromise, using the official Q8 output as the comparison target. Treat Q4_K_M as an opt-in throughput tier: its measured 717.8 tok/s is compelling, but 11/12 prompt outputs diverged from the fp32 oracle in this informational spot-check, so it needs an explicit product quality policy rather than being treated as a transparent storage optimization.

## Reproduction notes

- llama.cpp build: current master commit `571d0d540df04f25298d0e159e520d9fc62ed121`, CUDA, SM 8.9.
- llama.cpp timing: one warmup outside the timed set; each prompt posted sequentially to `/completion` with `cache_prompt=false`; prompt and generation timings came from the server's `timings` object.
- vLLM timing: one warmup outside the timed set; sequential streamed `/v1/completions`; cold load is process spawn to `/health`; decode rate uses tokens after the first streamed token.
- SGLang timing: one warmup outside the timed set; sequential streamed `/generate`; cold load is process spawn to `/health`; decode rate uses tokens after the first streamed token.
- The 12 rows stop at the natural EOS where applicable: completion rows 09, 10, and 12 generated fewer than 64 tokens in the fp32 oracle. This preserves greedy behavior instead of padding the workload with synthetic tokens.
