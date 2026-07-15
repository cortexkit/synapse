# Qwen3.5-2B LoRA and data-scaling ablations

Date: 2026-07-15

Labels: `qwen35-2b-lora-v1` and `qwen35-2b-half-v1`

Verdict: **LoRA r=32 is ladder-safe at 2B, and this half-data run gives no evidence that adding more curated rows is the next quality lever.**

## Result at a glance

| model | method | training rows | natural file F1 | natural line Jaccard | contract-valid |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen35-2b-sft-v1-fixed` | full FT | 1,864 | 0.561 | 0.542 | 82.5% |
| `qwen35-2b-lora-v1` | LoRA r32/alpha64 | 1,864 | 0.553 | 0.566 | 77.5% |
| `qwen35-2b-half-v1` | full FT | 932 | 0.600 | 0.535 | 87.5% |

The baseline row is the same Q8_0 checkpoint served with thinking disabled; it is not the original broken thinking-on row.

## Controlled setup

Both runs used one NVIDIA H100 SXM 80 GB and the smoke's pinned image, model, tokenizer, chat template, sequence length, packing, assistant-only masking, three epochs, and seed. The only intended LoRA-lane method change is the adapter block. The only intended half-data change is the selected dataset and output paths.

| pin | value |
| --- | --- |
| Image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (`sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014`) |
| Axolotl | `09d325b4fd1288b1473c8a330dd19e3c91b1ac32` (`0.17.0.dev0`) |
| Torch | `2.9.1+cu128`; Torch CUDA 12.8; image `nvcc` CUDA 12.4 |
| Training libraries | Transformers 5.9.0, Accelerate 1.13.0, TRL 1.5.1, PEFT 0.19.1, xFormers 0.0.33.post2 |
| CUDA extensions | FlashAttention 2.8.3, flash-linear-attention/fla-core 0.4.1, causal-conv1d 1.6.2.post1 |
| Base model | `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Model weights | SHA-256 `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1` |
| llama.cpp | `b4e3dc613baa92a3884d4151e3d631395c81934a` / build 9580, CUDA SM90 on the boxes |
| Training template | `train/axolotl/templates/qwen35-aft.jinja`, SHA-256 `fef15c2f760736e14982abe133807eae881eaf6c585becaedc190663642e40e8` |

The pre-install CUDA gates passed on both rentals under the image's Torch 2.5.1/CUDA 12.4 stack: `torch.cuda.is_available()` was true and a BF16 matrix product was finite. The post-install FlashAttention BF16 smoke and `pip check` also passed. Complete records are in `qwen35-2b-lora-environment.json` and `qwen35-2b-half-environment.json`.

### Reproducible half selection

The 1,864-row curated source was copied rather than symlinked and matched SHA-256 `005f0cad5fa3da8a21b448fce6e68a3787a516c090f938616556cbae56094859`. `select_stratified_half.py` matched each curated row back to the source metadata, formed 120 `tags.request_class` × `tags.language` strata, apportioned exactly half with deterministic SHA-256 ranks at seed 8918, and preserved source order in the output. The ignored 932-row output has SHA-256 `57223038cebabdee720c907bc3e4d901e25e59f33046efa6a6e57f8967fbb9e6`. The committed zero-based index list has SHA-256 `fc69db4481c92c234968ebee7a569a59edb4e1df53ada49829811d3d24b385a9`.

No stratum differs from its exact half by more than one row. `qwen35-2b-half-selection.json` records every full/half stratum count and both source hashes.

### Training objective and memory fix

Both configs use `chat_template: jinja` plus the audited Qwen3.5 template explicitly, `roles_to_train: [assistant]`, `train_on_eos: turn`, 32,768-token sample packing, BF16, FlashAttention 2, and 16-way chunked cross entropy. The full-data loss-mask audit passed rows 155, 340, and 677 (43,458 tokens; 3,943 trained). Curated source row 340 maps to half-dataset row 170 and passed the same byte-equality and mask invariants (13,371 tokens; 1,011 trained).

Accelerate 1.13.0 otherwise upcasts the complete returned BF16 logits before Trainer consumes the already-computed scalar chunked loss. At 32,768 × 248,320 elements this recreates the exact 30.31 GiB allocation that chunking avoids. `setup-trainbox.sh` disables that redundant full-output upcast while leaving Axolotl's per-chunk FP32 loss calculation intact. The first unpatched launch reproduced the expected step-zero OOM on each box; the patched 32k gates peaked at 37.47 GiB for LoRA and 48.03 GiB for half-data full FT.

## Training

### Thirty-step gates

| run | loss first → last | first-five → last-five mean | retained checkpoints | result |
| --- | --- | --- | --- | --- |
| LoRA r32/alpha64 | 0.8573 → 0.7385 | 0.7874 → 0.7284 | 20, 30 | pass; decreasing, finite |
| Half-data full FT | 0.8409 → 0.6547 | 0.7943 → 0.6508 | 20, 30 | pass; decreasing, finite |

LoRA was decreasing at step 30, so the alpha=128 retry was not triggered.

### Three-epoch curves

| run | epoch | steps | mean train loss | first loss | last loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| LoRA r32/alpha64 | 1 | 112 | 0.6808 | 0.8573 | 0.6321 |
| LoRA r32/alpha64 | 2 | 112 | 0.6196 | 1.2995 | 0.6059 |
| LoRA r32/alpha64 | 3 | 112 | 0.5978 | 0.6703 | 0.5813 |
| Half-data full FT | 1 | 57 | 0.6415 | 0.8409 | 0.6622 |
| Half-data full FT | 2 | 57 | 0.4422 | 0.4674 | 0.4428 |
| Half-data full FT | 3 | 57 | 0.3798 | 0.4954 | 0.4176 |

| run | validation loss: initial → epoch 1 → epoch 2 → epoch 3 |
| --- | --- |
| LoRA r32/alpha64 | 0.8464 → 0.6149 → 0.5979 → 0.5954 |
| Half-data full FT | 0.7988 → 0.5475 → 0.5542 → 0.5693 |

The LoRA run trained 21,823,488 of 1,903,648,576 parameters (1.1464%). Its 336 steps processed 88,014,848 packed input tokens and 8,821,337 supervised tokens in 4,071 seconds. Loss moved from 0.8573 to 0.5813; the first-ten/last-ten means were 0.8222/0.6039. Peak PyTorch allocation was 37.47 GiB, and checkpoints 330 and 336 were retained.

The half-data run used 171 steps for 44,433,408 packed input tokens and 4,394,734 supervised tokens in 2,199 seconds. Loss moved from 0.8409 to 0.4176; the first-ten/last-ten means were 0.7281/0.3785. Peak PyTorch allocation was 48.03 GiB, and checkpoints 170 and 171 were retained. Its packed-token total is 50.5% of the full-data smoke total; packing and the per-dataset validation split account for 171 rather than exactly half of 336 steps.

Machine-readable curves and artifact hashes are in `qwen35-2b-lora-training-summary.json` and `qwen35-2b-half-training-summary.json`.

## GGUF export and fixed local evaluation

The LoRA adapter was 87,319,256 bytes with SHA-256 `0daac44d913b1c829176b76d62d8067a86b0c51f0d817028362aa19c06760be6`. Axolotl's memory-efficient merger matched 0/632 tensors, so that output was rejected; the documented legacy merger applied the adapter and produced merged safetensors SHA-256 `4e3b6aed6d48bbb29849f9afe0cd14c63a4316e2de4a507180c4fca46828aef3`. Both conversion-only config copies set `mtp_num_hidden_layers=0`, preserving the smoke's 24-block/320-tensor fix.

| artifact | bytes | SHA-256 |
| --- | ---: | --- |
| `qwen35-2b-lora-v1-f16.gguf` | 3,775,708,608 | `defa8d90bbed19e9c1dafd553cb9eab1fcb2aae42f7f9d26ab3cf9e5a20ff296` |
| `qwen35-2b-lora-v1-q8_0.gguf` | 2,012,011,968 | `bab75e9826fafea79961b6d6901af5d0c893f2f3a93122f7edcc23db9f12ce85` |
| `qwen35-2b-half-v1-f16.gguf` | 3,775,709,344 | `932f0a05d278fa41410edac801feadb5b3a0131d6fca738db9ef95f479c26edd` |
| `qwen35-2b-half-v1-q8_0.gguf` | 2,012,012,704 | `d7b7ed43136552e0536ab8e774d60c8114ca7c290e4b9467b594883737277526` |

The Q8_0 hashes were verified again after both files were pulled to this Mac. Each local llama.cpp build-9580 `/apply-template` gate ended with the required `<think>\n\n</think>\n\n` suffix when thinking was disabled. The same pinned AFT binary, read-only corpora, 40 fixed jobs, concurrency 2, and 131,072 served context were used for both local evaluations.

| model | natural file F1 | natural line Jaccard | contract-valid rate | API-error rate | avg tool calls | thinking tokens/traj | natural jobs | budget outcomes | served context | wall time/traj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `qwen35-2b-sft-v1-fixed` | 0.561 | 0.542 | 82.5% | 0.0% | 13.75 | 0 | 37/40 | N 37/F 3/A 0/I 0 | 131072 | 35.9s |
| `qwen35-2b-lora-v1` | 0.553 | 0.566 | 77.5% | 0.0% | 14.10 | 0 | 35/40 | N 35/F 3/A 0/I 2 | 131072 | 20.6s |
| `qwen35-2b-half-v1` | 0.600 | 0.535 | 87.5% | 0.0% | 12.78 | 0 | 37/40 | N 37/F 1/A 0/I 2 | 131072 | 26.7s |

The serving gate for each model must show the `/apply-template` generation suffix `<think>\n\n</think>\n\n`. Both evaluations use:

```sh
EVAL_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}' \
  scripts/eval-student.sh MODEL.gguf LABEL
```

## Verdicts

1. **Is LoRA r=32 ladder-safe for this task? Yes.** Relative to full FT, natural file F1 changed by -0.008, natural line Jaccard by +0.024, contract validity by -5 percentage points, and natural completions by -2 jobs. That is not a bad method gap on this 40-job eval, and there were no API errors. Keep r=32/alpha64 for the 4B/9B/27B ladder rather than conflating size with a switch to full FT. The two invalid finals and lower contract-valid point estimate are worth monitoring on the larger rungs, but they do not justify full-FT ladders or an alpha=128 retry.
2. **Does data scaling pay? Not at 932 → 1,864 rows in this experiment.** Half-data was +0.039 file F1, -0.007 line Jaccard, +5 percentage points contract-valid, and equal on natural completions versus the full-data baseline. A single 40-job point estimate does not establish that less data is better, but it does rule out the large quality drop that would make more rows the immediate lever. Deprioritize tranche-3 generation and spend the next ladder budget on model scale.

## Rental cost and teardown

| run | vast.ai contract | location | all-in rate | lifetime | observed cost |
| --- | ---: | --- | ---: | ---: | ---: |
| LoRA r32/alpha64 | 44989492 | Malaysia | $2.1889/h | 2.685 h | $5.88 |
| Half-data full FT | 44989494 | India | $2.0381/h | 2.162 h | $4.41 |
| **Total** | | | | | **$10.28** |

The observed total was $6.72 below the $17 hard cap, and each box remained below its approximately $8 lane cap. Credit moved from $17.71 before rental to $7.44 after teardown, with the few-cent difference from the per-contract total attributable to account display rounding. Contract 44989494 was destroyed immediately after the half-data Q8_0 pull/hash check; contract 44989492 was destroyed immediately after the LoRA Q8_0 pull/hash check. The final instance query was empty.
