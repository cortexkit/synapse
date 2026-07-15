# Gatherer SFT ladder preparation

This directory converts the Anthropic-shaped gather trajectories, audits the four real student tokenizers, and defines Axolotl launch configs. The generated `sft-dataset.jsonl` is local and ignored by Git. No training or GPU rental is part of this step.

## Reproduce the local artifacts

Run from `tools/gather-distill`:

```bash
# The ignored source must already exist at data/dataset-v1.jsonl.
bun run train/convert.ts

uv venv train/.venv --python 3.11
uv pip sync --python train/.venv/bin/python train/requirements.txt
HF_HUB_DISABLE_XET=1 train/.venv/bin/python train/audit_tokenizers.py
```

The converter always rebuilds `train/sft-dataset.jsonl` from the source and writes `train/conversion-report.json`. The audit renders every row without truncation and writes `train/tokenizer-audit.json`. Model repository revisions are pinned in the audit script and recorded in `PREP-REPORT.md`.

### Required overflow gate

Do **not** launch a config against the unreviewed JSONL. All configs have `sequence_len: 32768`, while the audit found a small, explicit overflow set. Review every `overflow_records` entry in `tokenizer-audit.json`, approve its proposed `longer-context variant`, transaction-boundary `split`, or `drop` disposition, and point the config at the resulting curated JSONL. Truncation is not an allowed disposition.

Qwen3.5 and Qwen3.6 have one important adapter boundary: their official Hugging Face templates expect `function.arguments` mappings, while OpenAI Chat Completions correctly stores arguments as JSON strings. Axolotl 0.17.0's chat-template strategy parses those strings before rendering. The standalone audit performs the same in-memory parse and never changes the committed target shape.

## Loss-mask check

A full Axolotl 0.17.0 environment was installed locally. On macOS its CLI import stops at the unavailable `bitsandbytes` package, so the verification invokes the installed `ChatTemplateStrategy` directly—the same strategy used by `type: chat_template` datasets:

```bash
uv venv train/.venv-axolotl --python 3.11
uv pip install --python train/.venv-axolotl/bin/python axolotl
HF_HUB_DISABLE_XET=1 \
  train/.venv-axolotl/bin/python train/verify_loss_mask.py
```

The script runs three real examples and writes token/label runs to `train/loss-mask-verification.json`. Qwen3's stock template changes its last-assistant rendering during Axolotl's prefix probes, which can shift labels into the next tool-result header. The 1.7B config therefore uses `templates/qwen3-aft.jinja`: the pinned tokenizer template plus a `real_last_index` override for prefix probes. The verifier proves that full-conversation output remains byte-identical, then independently rejects masked assistant payload tokens or trainable user/tool-result tokens.

## Axolotl launch matrix

Run commands from `tools/gather-distill` only after the overflow gate and after changing each dataset path to the approved curated JSONL.

| Config | Method | Expected hardware | Launch |
|---|---|---|---|
| `train/axolotl/qwen3-1.7b-full.yaml` | Full fine-tune | 1x H100 80 GB | `axolotl train train/axolotl/qwen3-1.7b-full.yaml` |
| `train/axolotl/gemma4-e4b-lora.yaml` | LoRA r32/alpha64 | 1x H100 80 GB | `axolotl train train/axolotl/gemma4-e4b-lora.yaml` |
| `train/axolotl/qwen35-9b-lora.yaml` | LoRA r32/alpha64 | 1x H100 80 GB | `axolotl train train/axolotl/qwen35-9b-lora.yaml` |
| `train/axolotl/qwen36-27b-lora-fsdp2.yaml` | LoRA r32/alpha64 + FSDP2 | 2x H100 80 GB with NVLink | `CUDA_VISIBLE_DEVICES=0,1 axolotl train train/axolotl/qwen36-27b-lora-fsdp2.yaml --launcher torchrun -- --nproc_per_node=2 --nnodes=1` |

All four preserve the tokenizer's full-conversation rendering, per-record `tools`, assistant-only roles, per-assistant-turn EOS training, 32,768-token packed sequences, BF16, FlashAttention 2, gradient checkpointing, and safetensors. The 1.7B Jinja file is a mask-boundary patch whose full render is byte-identical to Qwen3's pinned tokenizer template; the other configs use `tokenizer_default` directly. The 27B wrap class is `Qwen3_5DecoderLayer`: the Qwen3.6 checkpoint declares `model_type: qwen3_5` and Transformers 5.13.1 implements it with the Qwen3.5 family classes.

Qwen3.5/Qwen3.6 also contain Gated DeltaNet projections. The requested seven-target LoRA survey block covers `q/k/v/o/gate/up/down_proj`; it does not target the separate linear-attention projections. Keep that scope explicit when comparing the ladder runs.

## Launch-time pin record

Never update Axolotl, Transformers, TRL, PEFT, PyTorch, FlashAttention/FLA, CUDA, or the container independently. At the start of each actual run, save all of the following beside the run output:

```bash
# For a source checkout, this is the authoritative Axolotl pin.
git -C /path/to/axolotl rev-parse HEAD
python - <<'PY'
from importlib.metadata import version
for package in ["axolotl", "transformers", "trl", "peft", "torch", "flash-attn", "flash-linear-attention"]:
    try:
        print(package, version(package))
    except Exception as error:
        print(package, "NOT INSTALLED", error)
PY

docker image inspect --format '{{json .RepoDigests}}' "$TRAINING_IMAGE"
```

Record the immutable image digest, Axolotl commit (not only a release label), package versions, CUDA/driver versions, model repository revision, config SHA-256, curated dataset SHA-256, and tokenizer-audit JSON. Qwen3.5/3.6 packed training currently requires the launch image's compatible `flash-linear-attention` build; pin it with the rest of the stack.
