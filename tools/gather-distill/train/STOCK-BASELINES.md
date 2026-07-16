# Stock Qwen3.5 4B + 9B no-thinking baselines

Date: 2026-07-16

Labels: `qwen35-4b-stock-nothink` and `qwen35-9b-stock-nothink`

## Artifact provenance and family controls

| size | Qwen training-base revision | Unsloth GGUF revision | Q8_0 file | bytes | SHA-256 |
| --- | --- | --- | --- | ---: | --- |
| 4B | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `unsloth/Qwen3.5-4B-GGUF@e87f176479d0855a907a41277aca2f8ee7a09523` | `Qwen3.5-4B-Q8_0.gguf` | 4,482,403,488 | `10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1` |
| 9B | `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` | `unsloth/Qwen3.5-9B-GGUF@3885219b6810b007914f3a7950a8d1b469d598a5` | `Qwen3.5-9B-Q8_0.gguf` | 9,527,502,048 | `809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4` |

The Unsloth card metadata declares `base_model:Qwen/Qwen3.5-4B` plus `base_model:quantized:Qwen/Qwen3.5-4B` for 4B, and the equivalent two tags for `Qwen/Qwen3.5-9B` for 9B. At evaluation lookup, the Qwen repository heads matched the revisions used to train the corresponding LoRA rungs. As with the 2B control, the publisher records the base repository but not a conversion-input commit, so this establishes the requested revision family rather than bit-level conversion provenance.

The ModelScope mirrors were used only as download transport after the direct Hugging Face card/API requests returned CloudFront 504 responses. The official Hugging Face LFS pointers were still retrieved and supplied the canonical object IDs, file sizes, and repository revisions above; each local mirror download matched that official SHA-256 exactly. The mirror revisions were `167b4afc359863325cb4164418c715421b4e9118` (4B) and `ae90f0d1c1be2b9250b0ef68265615f6fe3c777b` (9B).

## Fixed replay

Both controls replayed the stock-2B no-thinking arm against the fixed 40 jobs, gold rows, read-only corpus checkouts, and pinned AFT binary at concurrency 2. The M1 server used llama.cpp build 9580 (`b4e3dc613baa92a3884d4151e3d631395c81934a`), Q8_0, `-ngl 99 --jinja -fa on -c 131072`, and `--chat-template-kwargs '{"enable_thinking":false}'`. The local harness connected through `EVAL_REMOTE_ENDPOINT=[bench-host-alias]`.

| fixed input | SHA-256 |
| --- | --- |
| 40 eval jobs | `ca25a1fc77b001fc1b582ab0ff9112eb59938139a9e66037341000a6d09ecf9c` |
| gold rows | `c469e507ed900913e553c1aa63ad59d216729903ac71501b863ad89273600483` |
| AFT binary | `25cafa202e726a6b2d363fef4efac6e60ee6128105e7dbc42da7119e82b9a294` |

The 4B and 9B servers ran sequentially under `[bench-user-home]/bench.lock`; `/tmp/aft-measure.lock` was absent and no `Runner.Worker` was active before either lock was acquired. The copied GGUF was SHA-verified on the M1 before each server launch. For both sizes, `/apply-template` ended with the required disabled-thinking generation suffix:

```text
<think>

</think>

```

No thinking-enabled arm was run.

## Results

### Completed controls table

| size | stock nothink (F1 / valid / naturals) | LoRA trained (F1 / valid / naturals) | full-FT (2B only) (F1 / valid / naturals) |
| --- | --- | --- | --- |
| 2B | n/a / 22.5% / 0/40 | 0.553 / 77.5% / 35/40 | 0.561 / 82.5% / 37/40 |
| 4B | 0.448 / 25.0% / 5/40 | 0.637 / 87.5% / 37/40 | — |
| 9B | 0.665 / 72.5% / 16/40 | 0.615 / 87.5% / 34/40 | — |

Natural file F1 is computed only over natural completions. It is therefore undefined for the stock 2B row rather than zero.

### Measured SFT lift

| size | trained comparison | change from stock no-thinking |
| --- | --- | --- |
| 2B | LoRA / full FT | LoRA: +55.0 validity points and +35 natural jobs. Full FT: +60.0 points (22.5% → 82.5%) and +37 natural jobs (0 → 37); natural-only F1 becomes 0.561. |
| 4B | LoRA | +62.5 validity points (25.0% → 87.5%), +32 natural jobs (5 → 37), and +0.189 natural file F1 (0.448 → 0.637). |
| 9B | LoRA | +15.0 validity points (72.5% → 87.5%), +18 natural jobs (16 → 34), and -0.050 natural file F1 (0.665 → 0.615). |

### Appended ladder rows

| model | natural file F1 | natural line Jaccard | contract-valid | API errors | avg tool calls | thinking tokens/traj | natural jobs | budget outcomes | served context | wall time/traj |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `qwen35-4b-stock-nothink` | 0.448 | 0.295 | 25.0% | 0.0% | 12.12 | 0 | 5/40 | N 5/F 19/A 0/I 16 | 131072 | 145.4s |
| `qwen35-9b-stock-nothink` | 0.665 | 0.372 | 72.5% | 5.0% | 13.78 | 0 | 16/40 | N 16/F 16/A 2/I 6 | 131072 | 203.8s |

The same rows are appended by `scripts/eval-student.sh` to `data/students/LADDER.md`; raw rows, ledgers, score JSON, server logs, and GGUFs remain local and ignored.

## Verdict

**SFT lift is large through 4B, then much smaller at 9B.** The 2B full-FT reference raised contract validity by **+60.0 points** and natural jobs from **0 to 37** (the 2B LoRA row is +55.0 points and 0 to 35). The controlled 4B LoRA rung is similarly large at **+62.5 points** and **5 to 37** natural jobs, with natural file F1 rising by **0.189**. At 9B, the same LoRA recipe raises validity by only **+15.0 points** and natural jobs from **16 to 34**; its natural file F1 falls by **0.050**. More 9B trajectories reached natural completion, but their natural-only citation overlap was lower.

**Stock capability growth does not explain the trained 4B > 9B F1 inversion.** Stock 9B is substantially better than stock 4B on this fixed eval: natural file F1 is **0.665 vs 0.448** (+0.217), contract validity is **72.5% vs 25.0%** (+47.5 points), and natural jobs are **16 vs 5**. The trained inversion instead appears after applying this fixed LoRA recipe: 4B LoRA reaches 0.637 F1 while 9B LoRA reaches 0.615, despite equal 87.5% contract validity. With 40 jobs, these are point estimates rather than proof that the 9B base model or the recipe is intrinsically worse, but they rule out a weak stock 9B as the explanation for the inversion.
