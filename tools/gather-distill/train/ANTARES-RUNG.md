# Antares gather-SFT rung

Date: 2026-07-22
Question: can search-specialized pretraining replace model scale at the bridge tier?

## Verdict

**No.** Antares-1B transferred enough repository-search behavior to produce a non-zero
natural row (0.260 file F1), but it reached only 20% contract validity and was far
below the Qwen3.5-2B SFT rows (0.553-0.600). Antares-350M produced no valid natural
package. The search-pretraining prior is not a substitute for the 2B student class
under the production contract, so neither size is a useful draft/verify partner.

The stock Antares-1B control was also run: it produced no valid package. This anchors
the useful behavior to the format retrain rather than to the base model alone.

## Protocol and format gate

- Dataset: the frozen 1,864-row `train/sft-dataset-curated.jsonl`; SHA-256 is recorded
  in `antares-artifacts.json`.
- The target was the production gather contract: Granite role delimiters, our
  `<tool_call><function=...><parameter=...>` calls, and the final JSON package. The
  Antares native terminal `<think>/<tool_call>` surface was **not** used as the SFT
  target. This deliberate format retrain tests transfer of its search behavior, not
  surface-format compatibility.
- `granite-aft.jinja` was verified with `tokenizer.apply_chat_template` for both
  checkpoints before training. Both rendered assistant turns with
  `<|end_of_text|>\n`, had vocab size 100,352 and EOS id 100257, and had zero rows
  over the 32,768-token training limit (maximum 32,298 tokens). The disabled-thinking
  generation knob was `enable_thinking=false`, rendering the empty
  `<think>\n\n</think>\n\n` suffix.
- Axolotl accepted the Granite-4 hybrid architecture as
  `granitemoehybrid` and applied its existing Granite hybrid packing patch; no
  architecture mapping was hand-added. The fallback TRL path was not needed.
- Both runs used LoRA r=32/alpha=64, the seven Qwen ladder projection targets,
  assistant-only loss, EOS-per-turn training, 32,768 packing, BF16, gradient
  checkpointing, and three epochs. The learning rate was 2e-5 and remained stable.

## Environment and training

One H100 SXM 80GB was used sequentially for training. The immutable image, Axolotl
commit, package pins, GPU, and Accelerate upcast patch are in
`antares-1b-environment.json`.

| rung | base revision | train time | final train loss | final eval loss | peak device memory | train tokens |
|---|---|---:|---:|---:|---:|---:|
| Antares-1B LoRA | `10417eb35641b32e7141157db19c76eb545193b6` | 2:55:56 | 0.774 | 0.8957 | 22.29 GiB | 88.64M |
| Antares-350M LoRA | `cdf6d054fa5f491553ccb1704269cbd1954c6c6e` | 1:06:35 | 1.103 | 1.823 | 15.65 GiB | 88.64M |

The final 1B and 350M adapters were merged with the legacy PEFT
`merge_and_unload` path. This explicitly avoids the known memory-efficient merger
failure mode that can silently match zero tensors. Conversion and Q8 quantization
used llama.cpp commit `b4e3dc613baa92a3884d4151e3d631395c81934a`; the artifact
hashes are in `antares-artifacts.json`.

Both Q8 GGUFs passed a 16-token `llama-cli` smoke decode on the H100. The 1B smoke
produced a concise repository sentence; the 350M smoke also produced a concise
sentence. Granite conversion reported `GraniteMoeHybridForCausalLM` and emitted the
llama.cpp `granite` architecture successfully.

## Fixed 40-question evaluation

The frozen jobs/gold and mechanical scorer were unchanged. Evaluation ran locally on
the M1 through `/opt/zerobrew/bin/llama-server`; the Q8 GGUFs and adjacent config
records are referenced by the artifact manifest. “Natural” F1/Jaccard are the
standard natural-only ladder columns; `n/a` means no natural completion existed.

| model | natural file F1 | natural line Jaccard | contract valid | API errors | avg tool calls | thinking tok/traj | natural jobs | budget outcomes | served ctx |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Antares-1B stock | n/a | n/a | 0.0% | 0.0% | 12.53 | 1 | 0/40 | N0/F12/A0/I28 | 131072 |
| Antares-1B LoRA | 0.260 | 0.198 | 20.0% | 12.5% | 7.53 | 0 | 14/40 | N14/F2/A5/I19 | 131072 |
| Antares-350M LoRA | 0.000 | 0.000 | 0.0% | 0.0% | 0.00 | 0 | 17/40 | N17/F0/A0/I23 | 32768 |
| Qwen3.5-2B stock (nothink) | n/a | n/a | 22.5% | 0.0% | 14.97 | 0 | 0/40 | N0/F23/A0/I17 | 131072 |
| Qwen3.5-2B SFT full | 0.561 | 0.542 | 82.5% | 0.0% | 13.75 | 0 | 37/40 | N37/F3/A0/I0 | 131072 |
| Qwen3.5-2B SFT half | 0.600 | 0.535 | 87.5% | 0.0% | 12.78 | 0 | 37/40 | N37/F1/A0/I2 | 131072 |
| Qwen3.5-4B LoRA | 0.637 | 0.639 | 87.5% | 0.0% | 12.38 | 0 | 37/40 | N37/F2/A0/I1 | 131072 |
| Qwen3.5-9B LoRA | 0.615 | 0.571 | 87.5% | 0.0% | 11.68 | 0 | 34/40 | N34/F2/A0/I4 | 131072 |

### Serving diagnosis

The required diagnosis pass was performed before treating the 1B result as a
capability result. The 1B raw rows contained eight mechanically valid packages and
several natural JSON packages with incorrect paths or schema fields; the GGUF smoke
also produced normal text with thinking disabled. This is not the Qwen “0/40 because
the serving template never closed thinking” failure mode. The remaining errors are
therefore reported as contract/search quality failures. The 350M raw output showed
repeated malformed or incomplete JSON and no valid package; its zero is a genuine
capacity result for this retrain.

## Cost ledger and cleanup

All three instance charges were closed and the trainbox was destroyed immediately
after artifact pulls and H100 smoke tests.

| Vast contract | use | charge |
|---:|---|---:|
| 45544868 | failed SSH bootstrap probe | $0.490 |
| 45545605 | failed SSH recovery probe | $0.112 |
| 45545842 | H100 setup, both trainings, merges, conversions, smoke tests | $13.974 |
| **total** | | **$14.576** |

The total is below the $15 cap. The HF token was copied to the box only as a runtime
file, was never committed or printed, and the box was destroyed.

## Artifacts

Tracked experiment metadata and hashes: `train/antares-artifacts.json`,
`train/antares-tokenizer-audit.json`, `train/antares-1b-environment.json`, the two
Axolotl configs, `train/axolotl/templates/granite-aft.jinja`, and
`train/verify_antares_template.py`. GGUFs, raw rows, ledgers, server logs, and scorer
JSON remain local/ignored; their paths and SHA-256 values are recorded in the
manifest. Historical Qwen controls are documented in `SCALE-LADDER.md` and
`STOCK-2B-THINKING-AB.md`.
