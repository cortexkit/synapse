# Qwen3.5 LoRA scale ladder: 2B → 4B → 9B

Date: 2026-07-16

Labels: `qwen35-4b-lora-v1` and `qwen35-9b-lora-v1`

Verdict: **Scaling this fixed LoRA recipe from 2B to 4B produced a clear gain, but scaling again to 9B did not improve the 40-job point estimate. The 4B rung is the best trained model measured in this curve.**

## Result at a glance

| model | training | params | natural file F1 | natural line Jaccard | contract-valid | natural jobs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen35-2b-lora-v1` | LoRA r32/alpha64 | 1.9B | 0.553 | 0.566 | 77.5% | 35/40 |
| `qwen35-2b-sft-v1-fixed` | full FT reference | 1.9B | 0.561 | 0.542 | 82.5% | 37/40 |
| `qwen35-4b-lora-v1` | LoRA r32/alpha64 | 4.2B | **0.637** | **0.639** | **87.5%** | **37/40** |
| `qwen35-9b-lora-v1` | LoRA r32/alpha64 | 9.0B | 0.615 | 0.571 | **87.5%** | 34/40 |
| `qwen36-27b` | zero-shot reference | 27B | 0.818 | 0.677 | 80.0% | 26/40 |

The 2B, 4B, and 9B LoRA rows use the same curated data, rank/alpha, seven projection targets, optimizer schedule, seed, packing, assistant-only objective, export quantization, disabled-thinking serving mode, and fixed 40-job evaluation. The 2B full-FT row is a method reference. The Qwen3.6-27B zero-shot row is a reference line, not a controlled rung: it is a different generation and its bake-off run emitted an average of 724 thinking tokens per trajectory.

## Controlled setup

Both new runs used one NVIDIA H100 SXM 80 GB and replayed the pinned 2B LoRA chain.

| pin | 4B box | 9B box |
| --- | --- | --- |
| Vast contract / location | `45028752` / India | `45028751` / Czechia |
| Image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` | same |
| Image digest | `sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014` | same |
| GPU driver | 560.35.03 | 555.58.02 |
| Axolotl | `09d325b4fd1288b1473c8a330dd19e3c91b1ac32` (`0.17.0.dev0`) | same |
| Torch / CUDA | 2.9.1+cu128 / Torch CUDA 12.8 / image nvcc 12.4 | same |
| Transformers / Accelerate / TRL / PEFT | 5.9.0 / 1.13.0 / 1.5.1 / 0.19.1 | same |
| FlashAttention / FLA / xFormers | 2.8.3 / 0.4.1 / 0.0.33.post2 | same |
| Base model revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Config SHA-256 | `d387e65b4c2d9f76b3b154219d478bade54f2929960b0cf2d3e07b859f91275a` | `ad04d9dcf35f4586cbe811b0d4a134f604a2947c7b7139d36652d3706fe3ba19` |

Both pre-install CUDA gates loaded the image's Torch 2.5.1/CUDA 12.4 stack and completed a finite BF16 matrix product. The pinned setup then passed `pip check` and a FlashAttention BF16 smoke. The Accelerate full-output upcast patch remained enabled so chunked cross entropy did not recreate a full 32k × 248,320 FP32 logits allocation.

### Model and tokenizer gates

The configs force `Qwen3_5ForCausalLM` with `Qwen3_5TextConfig`, avoiding the repositories' declared multimodal conditional-generation class.

| proof | 4B | 9B |
| --- | ---: | ---: |
| Text-model parameters loaded | 4,205,751,296 | 8,953,803,264 |
| Parameter tensors | 426 | 427 |
| Vision parameters | 0 | 0 |
| Missing / unexpected / mismatched keys | 0 / 0 / 0 | 0 / 0 / 0 |
| Forward logits shape | 1 × 4 × 248,320 | 1 × 4 × 248,320 |

Every downloaded weight shard was SHA-verified. The per-shard hashes are recorded in `qwen35-4b-text-only-load.json`, `qwen35-9b-text-only-load.json`, and the environment records.

The copied curated dataset was read-only on both boxes: 1,864 rows, SHA-256 `005f0cad5fa3da8a21b448fce6e68a3787a516c090f938616556cbae56094859`. Each model's single-tokenizer audit rendered all rows without truncation or overflow. Both produced the same distribution as 2B: p50 14,394, p95 24,949, maximum 32,411, and zero rows over 32,768. The 4B and 9B tokenizer audits were byte-identical to each other on all five render probes. Their official chat-template files have SHA-256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`, which differs from the earlier 2B repository metadata, but the rendered curated probes and token counts are unchanged. Training explicitly used the audited `qwen35-aft.jinja` template at SHA-256 `fef15c2f760736e14982abe133807eae881eaf6c585becaedc190663642e40e8`.

The real Axolotl `ChatTemplateStrategy` loss-mask check passed source rows 155, 340, and 677 for both models. The rows contained 19,046 / 13,371 / 11,041 tokens and trained 2,074 / 1,011 / 858 tokens respectively; user and tool-result context remained masked and full rendered text remained byte-identical.

## Training

### Thirty-step gates

| run | loss first → last | first-five → last-five mean | peak allocation | result |
| --- | --- | --- | ---: | --- |
| 4B LoRA | 0.7343 → 0.6096 | 0.6853 → 0.6042 | 44.31 GiB | pass; finite and decreasing |
| 9B LoRA | 0.6695 → 0.5640 | 0.6287 → 0.5596 | 56.48 GiB | pass; finite and decreasing |

The 9B run fit the original 16-way chunked-cross-entropy setting, so neither the allowed 32-chunk mitigation nor a sequence-length change was needed.

### Three-epoch curves

| run | epoch | steps | mean train loss | first loss | last loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4B LoRA | 1 | 112 | 0.5610 | 0.7343 | 0.5141 |
| 4B LoRA | 2 | 112 | 0.4991 | 1.0422 | 0.4852 |
| 4B LoRA | 3 | 112 | 0.4787 | 0.5483 | 0.4611 |
| 9B LoRA | 1 | 112 | 0.5194 | 0.6695 | 0.4743 |
| 9B LoRA | 2 | 112 | 0.4626 | 0.9628 | 0.4518 |
| 9B LoRA | 3 | 112 | 0.4436 | 0.5115 | 0.4323 |

| run | validation loss: initial → epoch 1 → epoch 2 → epoch 3 |
| --- | --- |
| 4B LoRA | 0.7364 → 0.4998 → 0.4857 → 0.4841 |
| 9B LoRA | 0.6688 → 0.4609 → 0.4473 → 0.4456 |

Both runs processed 88,014,848 packed input tokens and 8,821,337 supervised tokens in 336 steps. The 4B run trained 42,467,328 of 4,248,218,624 parameters (0.9997%) in 8,936 seconds; its first-ten/last-ten losses were 0.7134/0.4801. The 9B run trained 58,195,968 of 9,011,999,232 parameters (0.6458%) in 13,480 seconds; its first-ten/last-ten losses were 0.6507/0.4465. Checkpoints 330 and 336 were retained for each run.

## Export and evaluation

Both adapters were merged with Axolotl's **legacy** merger. The memory-efficient merger was not used because the proven 2B chain showed that it can silently match zero tensors for these checkpoints. Conversion-only config copies changed `mtp_num_hidden_layers` from 1 to 0 before llama.cpp conversion, preventing an extra MTP block from entering the text-only GGUF.

| artifact | bytes | SHA-256 |
| --- | ---: | --- |
| 4B adapter | 169,903,320 | `088e5c7640d46401621147bb0ba90eade776d63f6ce88f675bd2bdf2a5a8613d` |
| 4B merged safetensors | 8,411,558,400 | `5f20d3315b8f4887101a7f53709054b05f8c95bc7e0dc07ea3e1635d65a2fcd7` |
| `qwen35-4b-lora-v1-f16.gguf` | 8,424,393,024 | `6b871e0df380bd8740dce7bacd4398420b726c9179a89d4b6c8038a9ebb25e57` |
| `qwen35-4b-lora-v1-q8_0.gguf` | 4,482,402,624 | `b4bd287f23c797576e6662e35a298c07d8e7b94963f6ac5763ed9d3be81eaac3` |
| 9B adapter | 232,818,064 | `b20b8c1434aee23ba8bf2e6fc8af11ba047affdf05fe306104d11a5f371900cd` |
| 9B merged safetensors | 17,907,663,008 | `a4756fe5b6a2d3a3b8e49081a72bafe863e43396874d9fd934d70bfc5e8b0969` |
| `qwen35-9b-lora-v1-f16.gguf` | 17,920,696,704 | `986a7719f45f06ae240ab24192946cf010c518da55722367e1a2a33c3a5ddde2` |
| `qwen35-9b-lora-v1-q8_0.gguf` | 9,527,501,184 | `dbadd7a4490832a88b7910e8a112ff6885a0c43e2be2c8310e6c79bd13f9e8a4` |

Both Q8_0 files were hash- and size-verified after download to the Mac. They were also copied to the M1 bench machine and SHA-verified there. Each serving topology used llama.cpp build 9580 (`b4e3dc613`), Metal offload, FlashAttention, 131,072 served context, Jinja, and `--chat-template-kwargs '{"enable_thinking":false}'`. The `/apply-template` gate for each model ended with the required `<think>\n\n</think>\n\n` generation suffix.

The 4B evaluation was already 34/40 jobs complete when the operator redirected remaining serving work away from the busy development Mac, so it finished locally. The 9B llama-server ran under the exclusive `[bench-user-home]/bench.lock` on `[bench-host-alias]`; the fixed harness, AFT tool proxy, corpora, scoring, and SSH tunnel remained local. The remote server was stopped and the lock released after scoring.

| model | natural file F1 | natural line Jaccard | contract-valid | API errors | avg tool calls | natural jobs | budget outcomes | wall time/traj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `qwen35-2b-lora-v1` | 0.553 | 0.566 | 77.5% | 0.0% | 14.10 | 35/40 | N 35/F 3/A 0/I 2 | 20.6s |
| `qwen35-4b-lora-v1` | **0.637** | **0.639** | **87.5%** | 0.0% | 12.38 | 37/40 | N 37/F 2/A 0/I 1 | 72.5s |
| `qwen35-9b-lora-v1` | 0.615 | 0.571 | **87.5%** | 0.0% | 11.68 | 34/40 | N 34/F 2/A 0/I 4 | 149.1s |

## Updated size ladder and stock references

| family / model | method | natural file F1 | contract-valid | comparison note |
| --- | --- | ---: | ---: | --- |
| Qwen3.5-2B stock, thinking off | zero-shot | n/a | 22.5% | No natural completions in the fixed stock A/B |
| Qwen3.5-2B | LoRA | 0.553 | 77.5% | Controlled trained rung |
| Qwen3.5-2B | full FT | 0.561 | 82.5% | Method reference |
| Gemma 4 E4B IT | zero-shot | 0.460 | 95.0% | Stock baseline near the 4B scale |
| Qwen3.5-4B | LoRA | **0.637** | 87.5% | Controlled trained rung |
| Ornith-9B | zero-shot | 0.563 | 65.0% | Stock baseline at the 9B scale |
| Qwen3.5-9B | LoRA | 0.615 | 87.5% | Controlled trained rung |
| Qwen3.6-27B | zero-shot | 0.818 | 80.0% | Larger-model reference line |

### Scale-curve verdict

From 2B LoRA to 4B LoRA, natural file F1 rose by 0.084 and contract validity rose by 10 percentage points. From 4B to 9B, F1 fell by 0.022, contract validity stayed at 87.5%, and natural completions fell from 37 to 34. The 9B rung still beats 2B by 0.062 F1 and 10 validity points, but this measured curve is not monotonic with parameter count.

The answer for this fixed data and LoRA recipe is therefore **scale to 4B, not automatically to 9B**. The 9B run optimized training and validation loss more strongly than 4B, yet did not translate that advantage into a better fixed-job score. With only 40 jobs, the 4B/9B ordering should be treated as a point estimate rather than proof that 9B is intrinsically worse, but there is no evidence here to pay the 9B serving and training premium. The 27B zero-shot reference remains 0.181 F1 above the best trained rung and motivates a future controlled 27B LoRA run only if its additional cost is justified.

## Rental cost and teardown

| run | contract | all-in rate | lifetime | calculated cost |
| --- | ---: | ---: | ---: | ---: |
| 4B LoRA | 45028752 | $1.9811/h | 7.932 h | $15.71 |
| 9B LoRA | 45028751 | $2.2867/h | 7.932 h | $18.14 |
| **Total** | | | | **$33.85** |

The total was $11.15 below the $45 hard cap. Account credit moved from $102.59 before rental to $68.75 after teardown, an observed delta of $33.84; the cent-level difference from contract duration × rate is display timing and rounding. Both boxes were destroyed immediately after the pulled Q8_0 files, configs, and run records passed local verification. The final instance query was empty.

Machine-readable evidence is in the two `*-environment.json`, `*-tokenizer-audit.json`, `*-text-only-load.json`, `*-loss-mask-verification.json`, `*-gate-summary.json`, `*-training-summary.json`, and `*-gguf.json` records beside this report.
