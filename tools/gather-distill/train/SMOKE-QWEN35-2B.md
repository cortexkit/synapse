# Qwen3.5-2B SFT smoke run

Date: 2026-07-15

Label: `qwen35-2b-sft-v1`

Verdict: **the full provision → SFT → checkpoint → GGUF → served eval → ladder chain completed. The chain needed four reproducible fixes, while the resulting 2B model scored 0.000 contract-valid and did not close the gatherer contract gap.**

## Result at a glance

| item | result |
| --- | --- |
| Dataset | 1,864 curated rows, SHA-256 `005f0cad5fa3da8a21b448fce6e68a3787a516c090f938616556cbae56094859` |
| Token audit | 28,096,989 input tokens/pass; p50 14,394, p95 24,949, p99 29,766, max 32,411; zero rows over 32,768 |
| Text-only model | `Qwen3_5ForCausalLM`, 1,881,825,088 trainable parameters, zero vision parameters |
| Preprocessed loss tokens | 3,015,230 supervised tokens/pass; 25,081,759 masked context tokens |
| 30-step gate | loss 0.8221 → 0.5971; first-five mean 0.7991 → last-five mean 0.6518; no NaN; checkpoints passed |
| Full run | 3 configured epochs, 336 optimizer steps, 88,014,848 packed input tokens, 8,821,337 supervised tokens |
| Full loss | 0.8573 → 0.3519; first-ten mean 0.7414 → last-ten mean 0.3661 |
| Throughput | 20,173 packed input tokens/s; 2,022 supervised tokens/s; 4,363 s trainer time |
| Peak VRAM | 63.19 GiB PyTorch allocated; 66,301 MiB from `nvidia-smi` |
| Export | 3.52 GiB F16 GGUF and 1.87 GiB Q8_0 GGUF; Q8_0 pulled to `data/students/models/` |
| Eval | 0/40 contract-valid, 0/40 natural, 40/40 invalid-final, 0 API errors, 1.02 tool calls/traj |
| Rental | 3.944 h before stop, $8.63 at the observed all-in $2.1889/h |

Machine 44961461 was stopped after the Q8_0 artifact, score JSON, and run evidence were retrieved.

## Environment pins

- vast.ai contract `44961461`: 1× NVIDIA H100 SXM 80 GB, Malaysia; driver `595.71.05`.
- Image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, digest `sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014`.
- The before-install CUDA gate passed under the image's Torch 2.5.1/CUDA 12.4 stack: `torch.cuda.is_available()` was true and a BF16 CUDA matrix multiplication returned finite output.
- Axolotl commit `09d325b4fd1288b1473c8a330dd19e3c91b1ac32` (`0.17.0.dev0`).
- Installed stack: Torch 2.9.1/CUDA 12.8, Transformers 5.9.0, TRL 1.5.1, PEFT 0.19.1, FlashAttention 2.8.3, flash-linear-attention/fla-core 0.4.1, causal-conv1d 1.6.2.post1, xFormers 0.0.33.post2.
- FlashAttention built with the image's CUDA 12.4 `nvcc`; an H100 BF16 `flash_attn_func` smoke test passed after installation.
- Base model revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; source weight SHA-256 `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`.
- llama.cpp commit `b4e3dc613baa92a3884d4151e3d631395c81934a` / build 9580, compiled with CUDA SM90 support.

The complete machine/package record is in `qwen35-2b-environment.json`.

## Dataset and tokenizer audit

The ignored curated dataset was copied from the parent checkout and independently checked as 1,864 lines with the required SHA-256. The audit rendered every row with the pinned Qwen3.5-2B tokenizer, tools enabled, and truncation disabled:

| percentile | tokens |
| --- | ---: |
| p50 | 14,394 |
| p90 | 21,879 |
| p95 | 24,949 |
| p99 | 29,766 |
| max | 32,411 |

No row exceeded 32,768, so no examples were dropped. The 2B and pinned 9B token vocabularies were byte-identical (`420ab4b96193b4325156f56cd7c3876a8a0d46f515fe4f711a7a6bf5553bf8fa`) and produced identical curated-set distributions. Their Hub revisions have different chat-template bytes, which the audit records explicitly rather than treating template identity as implied by shared vocabulary.

## Text-only load evidence

The Hub config is multimodal and declares `Qwen3_5ForConditionalGeneration`. An unconstrained Axolotl normalization therefore identified the model as multimodal. The final YAML forces the text path with:

```yaml
model_type: Qwen3_5ForCausalLM
cls_model_config: Qwen3_5TextConfig
```

A CUDA load and forward pass established:

- runtime class `Qwen3_5ForCausalLM`;
- 1,881,825,088 parameters across 320 tensors;
- all 1,881,825,088 parameters trainable for full fine-tuning;
- no parameter names containing vision, visual, image, video, or merger components;
- no missing, unexpected, or mismatched keys.

The saved checkpoint remains text-only: `model_type=qwen3_5_text`, architecture `Qwen3_5ForCausalLM`, and no `vision_config`. See `qwen35-2b-text-only-load.json` and `qwen35-2b-training-summary.json`.

## Loss-mask verification

The stock Qwen3.5 template raises `No user query found in messages` when Axolotl renders its system-only prefix probe. `axolotl/templates/qwen35-aft.jinja` changes only that guard when Axolotl supplies `real_last_index`; normal full renders remain byte-identical to the tokenizer template.

Direct `ChatTemplateStrategy` checks on curated rows 155, 340, and 677 passed all invariants:

- full rendered bytes equal the tokenizer's official render;
- prefix token IDs are stable;
- assistant text, tool-call bodies, and assistant EOS tokens are trained;
- system, user, tool-result, and separator tokens are masked;
- the final assistant answer is loss-bearing.

The spot checks covered 43,458 tokens, of which 3,943 were trained. The full preprocess then produced 28,096,989 tokens with 3,015,230 supervised labels and zero overlength examples. See `qwen35-2b-loss-mask-verification.json` and `qwen35-2b-preprocess-summary.json`.

## Training

Configuration: BF16 full fine-tune, FlashAttention 2, 32,768 sequence length, sample packing, micro-batch 1, gradient accumulation 8, 3 epochs, LR `2e-5`, checkpoint every 10 optimizer steps, and two retained checkpoints.

The first 32k forward attempt exhausted memory while materializing the full `32k × 248,320` vocabulary loss tensor; cross entropy requested another 30.31 GiB after the process had reached 53.8 GiB. The sequence length was not reduced. Enabling 16-way Axolotl chunked cross entropy fixed the loss-memory spike.

### 30-step gate

- 7,864,320 packed input tokens and 770,696 supervised tokens.
- Loss 0.8221 → 0.5971; first-five mean 0.7991, last-five mean 0.6518.
- No NaN/Inf and no second OOM.
- Checkpoints 10, 20, and 30 wrote successfully; the two-checkpoint retention policy kept 20 and 30.
- 63.19 GiB peak allocated and 67,623 MiB peak device usage.

The gate projected the real run well inside the remaining nine-hour cap, so the requested 3 epochs were retained.

### Full run

| epoch | mean train loss | first loss | last loss |
| ---: | ---: | ---: | ---: |
| 1 | 0.6174 | 0.8573 | 0.5758 |
| 2 | 0.4408 | 1.0620 | 0.4205 |
| 3 | 0.3654 | 0.4946 | 0.3519 |

Validation loss was 0.8464 at initialization, 0.5609 after epoch 1, 0.5652 after epoch 2, and 0.5868 at the end. The minimum validation loss was therefore at epoch 1 even though the smoke followed the requested 3-epoch endpoint.

The 336-step trainer ran for 4,363 seconds (72.7 minutes), processed 88,014,848 packed input tokens and 8,821,337 supervised tokens, and retained checkpoints 330 and 336. Checkpoint intervals were roughly two minutes at steady state, comfortably below the 15-minute spot-death target. Final safetensors SHA-256: `8e448d11ef6d71cbc3c489df26f5e590b2dab62e239125da18a877d76cf49815`.

## GGUF and serving

Full-FT output is already merged, so the final text-only safetensors directory was converted directly. The first GGUF carried `block_count=25` because the text config retained `mtp_num_hidden_layers=1`, although `Qwen3_5ForCausalLM` contains no MTP tensors. llama.cpp correctly rejected the absent block 24. A conversion-only config copy set `mtp_num_hidden_layers=0`, yielding the correct 24 blocks and 320 tensors without modifying checkpoint weights.

| artifact | bytes | SHA-256 |
| --- | ---: | --- |
| `qwen35-2b-sft-v1-f16.gguf` | 3,775,708,704 | `3c7eb04f8f31660635e8afc73522ef0ab96445aef4e8c7655e61171abd6370de` |
| `qwen35-2b-sft-v1-q8_0.gguf` | 2,012,012,064 | `d63ed1ca210afbd84c5507ae1a312e7520a3b7344042655ccce3562764ef849a` |

llama-server loaded the Q8_0 model at 131,072 context with all layers on the H100, reported 1,881,825,088 parameters, used about 4,253 MiB at idle, and passed `/health` plus a chat-completion smoke. The Q8_0 hash was verified again after pulling it to this Mac.

## Eval versus Opus gold

`eval-student.sh` ran locally in SSH tunnel mode against the remote llama-server, with the local fixed 40-job set, gold rows, corpus repos, and AFT proxy. The produced ladder row is:

| model | natural file F1 | natural line Jaccard | contract-valid rate | API-error rate | avg tool calls | thinking tokens/traj | natural jobs | budget outcomes | served context | wall time/traj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| qwen35-2b-sft-v1 | n/a | n/a | 0.0% | 0.0% | 1.02 | 57 | 0/40 | N 0/F 0/A 0/I 40 | 131072 | 1.3s |

All 40 requests reached the model and none were API errors, but every trajectory ended without parseable final JSON. Nineteen trajectories made at least one tool call; the remainder stopped after short reasoning or prose. The model averaged 172 output tokens and 57 thinking tokens. This is a valid 0.000 quality dot, not an infrastructure error: SFT at 2B did **not** close the contract gap relative to the 0.000 1–1.2B zero-shot tier, and it remains far below Gemma 4 E4B zero-shot at 0.460.

Scores are in `data/students/qwen35-2b-sft-v1-scores.json`; the row is in `data/students/LADDER.md`.

## Chain-link verdicts

| link | verdict | fix or evidence |
| --- | --- | --- |
| Provision and CUDA | pass | Pre-install CUDA/matmul gate passed. |
| Dataset transfer | pass | 1,864 rows and required SHA-256. |
| 2B tokenizer audit | pass | Shared 9B vocab/distribution; zero overflow. |
| Text-only load | fixed, then pass | Explicit causal-LM class plus nested text config prevented vision loading. |
| Loss-mask proof | fixed, then pass | Qwen3.5 Jinja guard made Axolotl prefix probes stable while preserving full renders. |
| Full preprocessing | pass | 28.10M tokens, 3.02M supervised, zero overlength. |
| 30-step gate | fixed, then pass | Chunked cross entropy removed the one full-vocabulary OOM at 32k. |
| Three-epoch train | pass | Decreasing loss, no NaN/OOM, 10-step checkpoints. |
| Merged checkpoint | pass | Text-only 3.76GB safetensors and config. |
| GGUF conversion | fixed, then pass | Conversion-only MTP metadata corrected block count 25 → 24. |
| Quantize and serve | pass | Q8_0 loaded, health/chat checks passed. |
| Tunnel eval and score | pass | 40/40 jobs scored; model-quality result was 0.000. |
| Artifact retention | pass | Q8_0 copied and hash-verified locally before stopping the instance. |

## Time and cost projection

The instance ran 14,197 seconds (3.944 hours). Its observed all-in rate was $2.1889/h including disk, for about **$8.63**, versus the roughly $25 balance and nine-hour stop cap. The 2B trainer itself used 1.212 hours; setup, downloads/builds, preprocessing, gate, export, eval, and artifact transfer used the remaining 2.73 hours.

A simple lower-bound projection holds the 88.0M packed-token workload fixed, scales throughput inversely with active parameter count, and assumes ideal two-GPU scaling for 27B:

| ladder lane | GPUs | projected wall h | projected H100 h | projected rental at $2.1889/H100-h |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-2B full FT (measured) | 1 | 1.21 | 1.21 | $2.65 |
| Gemma 4 E4B LoRA | 1 | 2.58 | 2.58 | $5.64 |
| Qwen3.5-9B LoRA | 1 | 5.80 | 5.80 | $12.69 |
| Qwen3.5-27B LoRA | 2 | 8.69 | 17.39 | $38.06 |
| **training total** |  | **18.28** | **26.97** | **$59.04** |

Sharing the measured one-time 2.73-hour chain overhead puts a lower-bound full-ladder projection near **$65**. Repeating that overhead independently for each lane puts it near **$83** before contingency. A prudent budget is therefore roughly **$78–100** (20% margin), and the 27B lane alone projects beyond a nine-hour rental once setup/eval are included. These are scheduling estimates, not benchmarked larger-model throughput; LoRA memory savings do not eliminate forward/backward compute, and multi-GPU scaling will be below ideal.
