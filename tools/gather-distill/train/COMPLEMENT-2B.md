# Qwen3.5-2B complement-half control

Date: 2026-07-15

Label: `qwen35-2b-comp-v1`

## Result at a glance

| model | training rows | natural file F1 | natural line Jaccard | contract-valid |
| --- | ---: | ---: | ---: | ---: |
| `qwen35-2b-half-v1` | 932 | 0.600 | 0.535 | 87.5% |
| `qwen35-2b-comp-v1` | 932 | 0.582 | 0.553 | 82.5% |
| `qwen35-2b-sft-v1-fixed` | 1,864 | 0.561 | 0.542 | 82.5% |

The full-data row is the fixed, thinking-disabled evaluation of the same full fine-tune used by the half-data comparison.

## Controlled replay

`train/axolotl/qwen35-2b-comp-full.yaml` is byte-for-byte identical to the half-data full-fine-tune config except for its dataset, prepared-dataset, and output paths. It retains seed 8918, the explicit `qwen35-aft.jinja` template, assistant-only labels, `train_on_eos: turn`, 32,768-token packing, BF16, FlashAttention 2, 16-way chunked cross entropy, and three epochs.

| pin | value |
| --- | --- |
| Image | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (`sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014`) |
| GPU | 1× NVIDIA H100 80 GB, driver 560.35.05 |
| Axolotl | `09d325b4fd1288b1473c8a330dd19e3c91b1ac32` (`0.17.0.dev0`) |
| Training stack | Torch 2.9.1+cu128, Transformers 5.9.0, Accelerate 1.13.0, TRL 1.5.1, PEFT 0.19.1, FlashAttention 2.8.3 |
| Base model | `Qwen/Qwen3.5-2B` at `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| llama.cpp | `b4e3dc613baa92a3884d4151e3d631395c81934a`, build 9580 |

The pre-install CUDA gate passed: the image's Torch 2.5.1/CUDA 12.4 stack reported CUDA available and produced a finite BF16 matrix product. The pinned-install FlashAttention BF16 smoke and `pip check` also passed. `qwen35-2b-comp-environment.json` records the full pin and hash set.

The on-box loss-mask check passed on complement rows 155, 340, and 677 with the pinned template. Each example had byte-identical full rendering, trainable assistant tool calls and final answers, and masked user and tool-result spans.

## Complement selection

The ignored curated source has 1,864 rows and SHA-256 `005f0cad5fa3da8a21b448fce6e68a3787a516c090f938616556cbae56094859`. The committed half index list has 932 sorted zero-based entries and SHA-256 `fc69db4481c92c234968ebee7a569a59edb4e1df53ada49829811d3d24b385a9`.

`select_stratified_half.py --complement` reads that committed list instead of rewriting it, independently recomputes the seed-8918 stratified draw, and rejects a list that does not match. It writes every source-order row whose index is absent from the list. The replay verified 932 complement rows, an index-list intersection of zero, and a union of 1,864 rows. The output SHA-256 is `05d66a7f7b05e7e5cb58570eeaac8b68d027b719f834605a6fb5a4b611ef45e7`. `qwen35-2b-comp-selection.json` records all 120 request-class × language strata; each complement count is the full stratum count minus the committed-half count.

## Training

### Thirty-step gate

| loss first → last | first-five → last-five mean | retained checkpoints | result |
| --- | --- | --- | --- |
| 0.8580 → 0.6555 | 0.7966 → 0.6182 | 20, 30 | pass; decreasing and finite |

The gate used the same 32k configuration and peaked at 48.03 GiB PyTorch allocated. No retry was needed.

### Three-epoch curve

| epoch | steps | mean train loss | first loss | last loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 55 | 0.6410 | 0.8580 | 0.6672 |
| 2 | 55 | 0.4557 | 1.0740 | 0.3908 |
| 3 | 55 | 0.3863 | 0.8653 | 0.3751 |

| validation point | initial | epoch 1 | epoch 2 | epoch 3 |
| --- | ---: | ---: | ---: | ---: |
| loss | 0.8679 | 0.5944 | 0.5997 | 0.6154 |

The full run completed 165 steps in 2,187 seconds. Its loss moved from 0.8580 to 0.3751, the first-ten/last-ten means were 0.7503/0.3849, and no NaN or Inf occurred. The maximum PyTorch allocation was 48.03 GiB; `nvidia-smi` observed 53,506 MiB and 703.47 W at peak. Checkpoints 160 and 165 were retained. `qwen35-2b-comp-training-summary.json` contains the machine-readable curve.

## Export and local evaluation

The conversion-only copy set `mtp_num_hidden_layers=0`, preserving the text-only 24-block/320-tensor GGUF fix. The Q8_0 artifact was pulled to the local ignored models directory and its SHA-256 was checked again before the rental was destroyed.

| artifact | bytes | SHA-256 |
| --- | ---: | --- |
| `qwen35-2b-comp-v1-f16.gguf` | 3,775,709,312 | `cb775ae2623f37bc3dc55352d8d026cc31fe817f362b9e0cdd7ce47e4216e9ca` |
| `qwen35-2b-comp-v1-q8_0.gguf` | 2,012,012,672 | `b91b3dc052c033590b156c4b129286160376ffc58697e96c44b343a1d6873b54` |

The remote and local build-9580 `/apply-template` gates both ended in the required `<think>\n\n</think>\n\n` suffix with `enable_thinking:false`. The local 40-job evaluation used the same read-only corpus, fixed jobs, gold rows, AFT binary, concurrency 2, 131,072 served context, and command shape:

```sh
EVAL_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}' \
  scripts/eval-student.sh MODEL.gguf qwen35-2b-comp-v1
```

| model | natural file F1 | natural line Jaccard | contract-valid rate | API-error rate | avg tool calls | thinking tokens/traj | natural jobs | budget outcomes | served context | wall time/traj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `qwen35-2b-comp-v1` | 0.582 | 0.553 | 82.5% | 0.0% | 12.68 | 0 | 37/40 | N 37/F 0/A 0/I 3 | 131072 | 23.4s |

The row is appended to `data/students/LADDER.md`; raw rows, scores, datasets, checkpoints, and GGUFs remain ignored.

## Verdict

The data-scaling conclusion holds across both independent halves: the original half scored 0.600 and the disjoint complement scored 0.582, both above the 1,864-row full-data score of 0.561. The complement trails the first half by 0.018, so the initial draw contributed ordinary sample variation, but it does not look like a lucky draw that reversed the conclusion: neither 932-row half showed the large quality drop that would make adding rows the immediate lever. With only 40 evaluation jobs these are point estimates rather than evidence that fewer rows are intrinsically better; prioritize model-scale work over tranche expansion while retaining the full data for future training.

## Rental and teardown

The India H100 rental ran at $2.0381/h and cost $4.63, below the $6 hard cap. It was destroyed, not stopped, immediately after the local Q8_0 hash matched the remote hash; the active-instance query was empty after teardown.
