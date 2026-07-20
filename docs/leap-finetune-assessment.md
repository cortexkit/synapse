# leap-finetune smoke assessment

**Assessment date:** 2026-07-20  
**Reference checkout:** `~/Work/OSS/leap-finetune`, `4dbf17b11542f6c167ac032d4b2dd0a3f8961542`  
**Decision status:** environment, dense SFT/resume, independent dense GRPO/vLLM, and LoRA adapter-save smokes passed. The GGUF exporter remains blocked by a first-party tensor-mapping bug.

## Executive result

The physical runs confirm that the environment, dense SFT/resume path, dense
GRPO with colocated vLLM, and LoRA adapter saving work on a Linux CUDA host. The
train-to-serve path remains blocked: the bundled GGUF converter fails on a
standard LFM2-350M checkpoint before `llama-cli` can run. GGUF is independent
of GRPO and LoRA, so those two decision-critical paths were run after recording
the exporter failure.

The gatherer-tier **LFM2-8B-A1B cannot use GRPO in this checkout**. Although a
`MOE_GRPO` defaults dictionary exists, the actual text GRPO runner rejects MoE
models before constructing the trainer. The shipped MoE examples and runners
cover SFT and DPO, not a working MoE GRPO path.

The custom-reward seam is suitable for the gatherer's package-vs-gold scorer:
extra dataset columns survive the GRPO normalization/filter path and are
forwarded as keyword arguments to every reward callable. A reward recipe can
compose multiple weighted callables, but its `required_columns` attribute is
documentation only; the reward function itself must accept the exact column
names and `**kwargs`.

## Execution boundary and spend

A qualifying RTX 4090 was rented and the physical run started. The admission gate
passed: driver `580.159.03`, CUDA capability 13.0, reliability `99.8246%`, and
an RTX 4090 with 24,564 MiB. The final box used
`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`; the leap environment itself
resolved the pinned CUDA 13 stack.

The first rental run stopped at the GGUF exporter, while two earlier rentals on
one host failed to leave the SSH proxy in a usable state and were destroyed. A
second qualifying rental then ran the independent GRPO and LoRA paths. Final
labeled-instance destruction was verified with `vastai show instances-v1
--label leap-finetune-smoke`, which returned zero instances.

| Step | Requested proof | Physical result | Wall time / peak VRAM |
|---|---|---|---|
| 0 | Driver gate and `uv sync` | **Passed**; FA2 usable | `uv sync` 50.73 s; driver gate passed |
| 1 | Dense 350M SFT, checkpoint, falling loss, resume | **Passed**; checkpoint, optimizer/scheduler state, falling loss, and resume to step 30 | initial SFT 92.71 s / 3,964 MiB; resume 76.60 s / VRAM sample not retained |
| 2 | F16/Q8_0 GGUF and `llama-cli` prompt | **Hard blocker**; bundled converter cannot map `model.layers.0.feed_forward.w1.weight` | exporter did not complete; `llama-cli` not reached |
| 3 | 350M GRPO, custom reward, vLLM, finite update | **Passed**; reward flowed, finite metrics, nonzero policy gradients on steps 1–3 | 135.58 s / 7,723 MiB |
| 4 | 350M LoRA SFT and adapter save | **Passed**; adapter config and safetensors written | 96.00 s / 1,803 MiB |

Campaign spend is **$0.200** across the five rentals used by this task: $0.017
for the two failed startup attempts, $0.073 for the first successful SFT/GGUF
host, and $0.110 for the second GRPO/LoRA host (GPU, storage, and download
charges). This remains below the $6 cap. The account invoice also contains
unrelated instances that were not touched.

The commands below remain the exact smoke recipe and include the deviations and
measured outcomes from this run.

## Working recipe for the CUDA box

### 1. Admit the host before installing anything

Use a single Linux x86_64 box with Python 3.12 and an RTX 4090-class GPU. Reject
the offer if the NVIDIA driver is below 580; the default lock selects the
CUDA 13 / Torch 2.11 FlashAttention wheel, and the driver gate must happen
before any package or model download.

```bash
set -euo pipefail

export LEAP_ROOT=/lambdafs/leap-finetune
export HF_HOME=/lambdafs/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export UV_CACHE_DIR=/lambdafs/uv-cache
export LLAMA_CPP_DIR=/lambdafs/llama.cpp

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
import subprocess

raw = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    text=True,
).splitlines()[0].strip()
major = int(raw.split(".", 1)[0])
if major < 580:
    raise SystemExit(f"reject host: CUDA 13 smoke requires preferred driver >= 580; got {raw}")
print(f"driver gate passed: {raw}")
PY

git clone https://github.com/Liquid4All/leap-finetune.git "$LEAP_ROOT"
cd "$LEAP_ROOT"
uv python install 3.12
/usr/bin/time -f 'uv_sync_wall_seconds=%e' uv sync
uv run leap-finetune env fa2-status --require
```

The reference declares `.python-version` as `3.12`. The root CUDA lock/profile
pins or constrains the relevant stack as follows:

| Component | Pin/resolution in the reference |
|---|---|
| Python | 3.12, Linux x86_64 wheel target |
| PyTorch | Linux lock resolves 2.11.0; project constraint `>=2.10,<2.12` |
| Transformers | `>=5.3.0,<5.4.0` (the lock override keeps this range) |
| TRL | `1.2.0` |
| vLLM | `0.22.0` in the default CUDA group |
| Ray | `2.51.1` |
| FlashAttention | `2.8.3`, pinned CUDA 13 / Torch 2.11 / CPython 3.12 x86_64 wheel |
| PEFT | `>=0.15.2` |

Record the actual resolution and the box identity in the run directory:

```bash
uv run python - <<'PY' | tee smoke/environment.txt
from importlib.metadata import version
import platform
import torch

for package in ("torch", "transformers", "trl", "vllm", "ray", "flash-attn", "peft"):
    try:
        print(f"{package}={version(package)}")
    except Exception as exc:
        print(f"{package}=NOT_INSTALLED ({exc})")
print(f"python={platform.python_version()}")
print(f"platform={platform.platform()}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
```

The first successful rental resolved 284 packages and completed `uv sync` in
50.73 s. The repeat rental, with an empty package cache, completed it in 66.07 s;
there was no resolution or wheel-build failure. `uv sync` may spend most of its
time resolving or downloading the pinned accelerator wheels, especially
FlashAttention. If the FlashAttention wheel cannot be resolved, the README
documents a deliberate fallback:

```bash
uv sync --no-group flash-attn
uv run leap-finetune env install-fa2
```

That fallback is acceptable for SFT only if `fa2-status` explains the SDPA
fallback. It is **not** an equivalent GRPO/vLLM validation: the GRPO smoke must
use the complete CUDA/vLLM profile and should stop rather than silently claiming
the pinned rollout stack was tested.

### 2. Make a 200-row local dataset

This avoids a second Hugging Face dataset download. The same file intentionally
contains an SFT-style `messages` column and a `target_keyword` column. For GRPO,
leap-finetune removes the last assistant turn into `solution`, creates `prompt`
from the preceding turns, and preserves `target_keyword` for the custom reward.

```bash
cd "$LEAP_ROOT"
mkdir -p smoke
python - <<'PY'
import json
from pathlib import Path

rows = []
for i in range(200):
    rows.append(
        {
            "messages": [
                {"role": "system", "content": "Return the requested marker."},
                {"role": "user", "content": f"Record {i}: reply with LIQUID."},
                {"role": "assistant", "content": "LIQUID"},
            ],
            "target_keyword": "LIQUID",
        }
    )
path = Path("smoke/chat-200.jsonl")
with path.open("w", encoding="utf-8") as stream:
    for row in rows:
        stream.write(json.dumps(row) + "\n")
print(f"wrote {len(rows)} rows to {path}")
PY
```

### 3. Dense SFT smoke and resume

Create `smoke/sft-350m.yaml`:

```yaml
project_name: "lfm2_350m_sft_smoke"
model_name: "LFM2-350M"
training_type: "sft"

ray:
  num_workers: 1
  resources_per_worker:
    GPU: 1

dataset:
  path: "./chat-200.jsonl"
  type: "sft"
  limit: 200
  test_size: 0.1

training_config:
  extends: "DEFAULT_SFT"
  max_steps: 20
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 5e-5
  max_length: 256
  logging_steps: 1
  save_strategy: "steps"
  save_steps: 10
  save_total_limit: 2
  eval_strategy: "steps"
  eval_steps: 10
  save_only_model: false
  bf16: true
  gradient_checkpointing: true

peft_config:
  use_peft: false
```

Run it and retain the full log; start the documented `nvidia-smi` sampler in parallel when collecting peak VRAM:

```bash
cd "$LEAP_ROOT"
set -o pipefail
(
  /usr/bin/time -f 'sft_wall_seconds=%e' \
    -o smoke/sft.time \
    uv run leap-finetune smoke/sft-350m.yaml 2>&1
) | tee smoke/sft.log
```

The run should create a project directory under
`outputs/lfm2_350m_sft_smoke/`. `LeapCheckpointCallback` renames standard HF
checkpoint directories, writes a `latest` pointer, and records loss history.
Verify the checkpoint and the optimizer/scheduler state rather than checking
only for model weights:

```bash
cd "$LEAP_ROOT"
RUN_DIR=$(dirname "$(find outputs/lfm2_350m_sft_smoke -type l -name latest -print -quit)")
CHECKPOINT=$(realpath "$RUN_DIR/latest")
test -f "$CHECKPOINT/config.json"
test -f "$CHECKPOINT/trainer_state.json"
test -f "$CHECKPOINT/optimizer.pt" || test -f "$CHECKPOINT/optimizer.bin"
test -f "$CHECKPOINT/scheduler.pt" || test -f "$CHECKPOINT/scheduler.bin"

python - "$CHECKPOINT/trainer_state.json" <<'PY'
import json
import math
import sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
losses = [
    item["loss"]
    for item in state.get("log_history", [])
    if isinstance(item.get("loss"), (int, float))
]
assert len(losses) >= 2, losses
assert all(math.isfinite(value) for value in losses), losses
assert losses[-1] < losses[0], (losses[0], losses[-1])
print({"loss_first": losses[0], "loss_last": losses[-1], "loss_decreased": True})
PY
```

Prove resume, not just parser acceptance. Make a second config with a larger
`max_steps` and `resume_from_checkpoint: latest`:

```bash
cd "$LEAP_ROOT"
cp smoke/sft-350m.yaml smoke/sft-350m-resume.yaml
python - <<'PY'
from pathlib import Path
import yaml

path = Path("smoke/sft-350m-resume.yaml")
config = yaml.safe_load(path.read_text())
config["training_config"]["max_steps"] = 30
config["training_config"]["resume_from_checkpoint"] = "latest"
path.write_text(yaml.safe_dump(config, sort_keys=False))
PY
set -o pipefail
uv run leap-finetune smoke/sft-350m-resume.yaml 2>&1 | tee smoke/sft-resume.log
RUN_DIR=$(dirname "$(find outputs/lfm2_350m_sft_smoke -type l -name latest -print -quit)")
CHECKPOINT=$(realpath "$RUN_DIR/latest")
python - "$CHECKPOINT/trainer_state.json" <<'PY
import json
import sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
assert state["global_step"] >= 30, state["global_step"]
print({"resumed_global_step": state["global_step"]})
PY
```

A failure to find `latest`, or a resumed run that starts at step 0, is a hard
blocker for the BRIDGE program. `save_only_model: false` is required because
optimizer state, scheduler state, RNG state, and the training step are part of
the resume contract.

### 4. Export GGUF and close the train-to-serve loop

The bundled exporter directly supports `F16` and `Q8_0`; it does not need a
separately installed Python converter. Export both from the resumed dense
checkpoint:

```bash
cd "$LEAP_ROOT"
RUN_DIR=$(dirname "$(find outputs/lfm2_350m_sft_smoke -type l -name latest -print -quit)")
CHECKPOINT=$(realpath "$RUN_DIR/latest")
uv run leap-export-gguf "$CHECKPOINT" \
  --quant F16 \
  --quant Q8_0 \
  --output-dir smoke/gguf-sft
```

Build `llama-cli` once for load verification. The exporter only needs a
`llama.cpp` checkout for K-quants, but the load test needs the CLI binary:

```bash
if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON
cmake --build "$LLAMA_CPP_DIR/build" \
  --target llama-cli \
  --config Release \
  --parallel "$(nproc)"
git -C "$LLAMA_CPP_DIR" rev-parse HEAD | tee smoke/llama-cpp-revision.txt

for quant in F16 Q8_0; do
  "$LLAMA_CPP_DIR/build/bin/llama-cli" \
    -m "smoke/gguf-sft/LFM2-350M-${quant}.gguf" \
    -p "Reply with exactly LIQUID." \
    -n 16 \
    -ngl 99 \
    --temp 0 \
    2>&1 | tee "smoke/llama-cli-${quant}.log"
done
```

On the physical run, this command stopped during the first F16 conversion:

```text
ValueError: Can not map tensor 'model.layers.0.feed_forward.w1.weight'
```

The bundled converter's LFM2 mapping does not recognize the model's standard
`feed_forward.w1` tensor, so neither F16 nor Q8_0 was produced and `llama-cli`
was not reached. This is a first-party exporter blocker, not a CUDA or
llama.cpp load failure. Do not claim the train-to-serve loop is closed until
Liquid fixes the mapping (or documents an unmodified supported checkpoint
shape).

### 5. Dense GRPO with a custom callable

Create `smoke/keyword_reward.py`:

```python
def _completion_text(completion):
    if isinstance(completion, list):
        if not completion:
            return ""
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    return str(completion)


def keyword_reward(completions, target_keyword=None, **kwargs):
    """Score one when the completion contains its row's target keyword."""
    if isinstance(target_keyword, list):
        keywords = target_keyword or [None]
    else:
        keywords = [target_keyword]

    scores = []
    for index, completion in enumerate(completions):
        keyword = keywords[index % len(keywords)]
        text = _completion_text(completion).lower()
        target = "" if keyword is None else str(keyword).lower()
        scores.append(1.0 if target and target in text else 0.0)
    return scores
```

Create `smoke/grpo-350m.yaml`:

```yaml
project_name: "lfm2_350m_grpo_smoke"
model_name: "LFM2-350M"
training_type: "grpo"

ray:
  num_workers: 1
  resources_per_worker:
    GPU: 1

dataset:
  path: "./chat-200.jsonl"
  type: "grpo"
  limit: 200
  test_size: 0.1

rewards:
  funcs:
    - "./keyword_reward.py::keyword_reward"
  weights: [1.0]

training_config:
  extends: "DEFAULT_GRPO"
  max_steps: 4
  num_train_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 1
  num_generations: 2
  max_completion_length: 64
  learning_rate: 1e-6
  logging_steps: 1
  save_strategy: "steps"
  save_steps: 2
  save_total_limit: 2
  eval_strategy: "no"
  remove_unused_columns: false
  bf16: true
  gradient_checkpointing: true
  use_vllm: true
  vllm_mode: "colocate"
  vllm_gpu_memory_utilization: 0.2
  vllm_enable_sleep_mode: true
```

Launch on the same one-GPU box:

```bash
cd "$LEAP_ROOT"
set -o pipefail
uv run leap-finetune smoke/grpo-350m.yaml 2>&1 | tee smoke/grpo.log
```

`DEFAULT_GRPO` already selects TRL v1 `GRPOTrainer`, `use_vllm: true`, and
colocated vLLM. The short config reduces generations and completion length so
one 4090 can exercise the plumbing without a second GPU. Verify that the
checkpoint has policy updates and finite reward metrics:

```bash
cd "$LEAP_ROOT"
GRPO_CHECKPOINT=$(realpath outputs/lfm2_350m_grpo_smoke/latest)
python - "$GRPO_CHECKPOINT/trainer_state.json" <<'PY'
import json
import math
import sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
assert state.get("global_step", 0) >= 4, state.get("global_step")
reward_values = []
loss_values = []
gradient_values = []
for item in state.get("log_history", []):
    for key, value in item.items():
        if not isinstance(value, (int, float)):
            continue
        if "reward" in key.lower():
            reward_values.append(value)
        if key == "loss":
            loss_values.append(value)
        if "grad_norm" in key:
            gradient_values.append(value)
assert reward_values, "no reward metrics found"
assert all(math.isfinite(value) for value in reward_values), reward_values
assert all(math.isfinite(value) for value in loss_values), loss_values
assert all(math.isfinite(value) for value in gradient_values), gradient_values
print({
    "global_step": state["global_step"],
    "reward_samples": reward_values[:8],
    "rewards_finite": True,
    "loss_finite": True,
    "gradient_finite": True,
})
PY
```

The reward file is intentionally a plain Python callable, not an OpenEnv
adapter. The observed smoke must show a reward metric named for
`keyword_reward`, at least one completed optimizer step, and no `NaN` in reward,
loss, or gradient metrics. A vLLM startup or version mismatch is a GRPO hard
blocker even if the callable unit test passes.

### 6. LoRA rerun and adapter GGUF

Copy the SFT config and enable the shipped LFM2 LoRA defaults (`r=8`,
`alpha=16`, dropout `0.1`, with LFM attention, GLU, and convolution targets):

```bash
cd "$LEAP_ROOT"
cp smoke/sft-350m.yaml smoke/sft-350m-lora.yaml
python - <<'PY'
from pathlib import Path
import yaml

path = Path("smoke/sft-350m-lora.yaml")
config = yaml.safe_load(path.read_text())
config["project_name"] = "lfm2_350m_sft_lora_smoke"
config["peft_config"] = {"extends": "DEFAULT_LORA", "use_peft": True}
path.write_text(yaml.safe_dump(config, sort_keys=False))
PY
set -o pipefail
uv run leap-finetune smoke/sft-350m-lora.yaml 2>&1 | tee smoke/sft-lora.log
```

The ordinary checkpoint directory contains `adapter_config.json` and adapter
weights. The end-of-run callback also writes a merged `-lora_m-` model; do not
mistake the merged model for the PEFT adapter when testing adapter export:

```bash
cd "$LEAP_ROOT"
ADAPTER_CONFIG=$(find outputs/lfm2_350m_sft_lora_smoke \
  -type f -name adapter_config.json -print -quit)
test -n "$ADAPTER_CONFIG"
ADAPTER=$(dirname "$ADAPTER_CONFIG")

uv run leap-export-gguf "$ADAPTER" \
  --base-model-path "LiquidAI/LFM2-350M" \
  --quant F16 \
  --quant Q8_0 \
  --output-dir smoke/gguf-lora

test -s smoke/gguf-lora/*-lora-F16.gguf
test -s smoke/gguf-lora/*-lora-Q8_0.gguf
```

Adapter exports support F16/BF16/F32/Q8_0 directly. K-quants require merging
the adapter into the base model first, then exporting the merged checkpoint.

## Capacity and timing expectations

The following are the measurements retained from the two successful rentals.
Peak VRAM was sampled every 500 ms with `nvidia-smi`; the resume sampler file
was not retrieved before destroying the first host, so that cell is explicitly
not reported.

| Operation | Configuration | Wall time | Peak VRAM |
|---|---|---:|---:|
| Environment | Python 3.12.13, Torch 2.11.0+cu130, FA2 2.8.3 | 50.73 s | N/A |
| Dense SFT | LFM2-350M, 200 rows, 20 steps, BF16, full weights | 92.71 s | 3,964 MiB |
| Resume | Same run, `latest`, continued to step 30 | 76.60 s | Not retained |
| GGUF export | Resumed checkpoint, F16 first | Blocked | Not applicable |
| GRPO / vLLM | LFM2-350M base, 4 steps, 2 generations, colocated | 135.58 s | 7,723 MiB |
| LoRA / adapter save | LFM2-350M base, 20 steps, `DEFAULT_LORA` | 96.00 s | 1,803 MiB |

The SFT sampler reported a 1 MiB idle baseline and a 3,964 MiB maximum. The
training log reported loss falling from `6.8755846` to `0.2050134`; the resumed
checkpoint contained `optimizer.pt`, `scheduler.pt`, `rng_state.pth`, and
`trainer_state.json` at global step 30. Ray emitted repeated metrics-exporter
connection warnings, but they did not affect training or checkpointing.

GRPO reached global step 4 and started the colocated vLLM path successfully at
TRL `1.2.0` / vLLM `0.22.0`. The custom metric was
`rewards/keyword_reward/mean`; steps 1–3 had reward mean `0.5`, reward std
`0.7071`, and gradient norms `45.25`, `69.5`, and `50.25`, proving policy
updates. Step 4 had all-positive completions and therefore zero reward
variance, zero gradient, and zero loss; it was finite but not an additional
policy update. The recorded GRPO loss history was
`[-0.70595, 0.06616, 0.21946, 0.0]`, with no NaN or Inf.

The LoRA run reached global step 20, wrote `adapter_config.json` and
`adapter_model.safetensors` plus optimizer/scheduler state, and saved the
merged `-lora_m-` model as a separate artifact. Its loss fell from `6.87558` to
`4.95767`. Adapter GGUF export was intentionally skipped because the same
first-party converter mapping blocker applies to the adapter train-to-serve
path.

For scale planning only, the unmeasured model-state floor remains approximately
2 bytes per parameter for BF16 weights and 12 bytes per parameter for BF16
weights + BF16 gradients + two FP32 Adam moments, before activations and
workspaces. That gives approximately 0.7/2.4/5.2 GB of weights and
4.2/14.4/31.2 GB of full-update state at 350M/1.2B/2.6B respectively. These
are floors, not measured VRAM; the requested run did not execute at 1.2B+.
The measured colocated GRPO peak was 7,723 MiB with
`vllm_gpu_memory_utilization: 0.2`, `num_generations: 2`, and
`max_completion_length: 64`. Its 1.2B/2.6B extrapolation remains unmeasured;
GGUF failure does not invalidate this independent GRPO result.

## Custom-reward seam verdict

### Dataset contract

The relevant path is concrete, not just README prose:

1. `normalize_columns("grpo")` converts an SFT-style `messages` list into
   `prompt` containing all non-assistant turns and `solution` containing the
   last assistant text. A pre-existing `solution` is preserved.
2. The GRPO validator requires only a non-empty `prompt` string or message list.
   It leaves `solution`, `target_keyword`, `package`, `gold`, and other columns
   intact.
3. `grpo_run` passes the resulting dataset to TRL `GRPOTrainer` and resolves
   custom functions from `path.py::function_name`.
4. The reward signature is `reward_fn(completions, **kwargs) -> list[float | None]`.
   TRL supplies each extra dataset column by keyword, as well as runtime fields
   such as `prompts`, `completion_ids`, and `trainer_state`.

**Verdict:** the gatherer scorer can be a normal function such as
`score_package(completions, gold_package, package_id, **kwargs)`. Keep column
names Python-safe, include `**kwargs`, and return exactly one float or `None` per
completion. If the scorer needs a whole package object, serialize it in a
column that the local Ray/Arrow path can carry, or derive it from a stable ID
inside the worker; do not rely on an unpicklable closure or process-local state.

The `Recipe.required_columns` tuple is useful documentation but is not
validated by the loader. Missing columns therefore fail when the callable is
invoked, not at config parse time. Add an explicit dataset preflight for the
real gatherer schema before renting a box.

### Multi-component rewards

Yes, multi-component rewards are supported:

```yaml
rewards:
  funcs:
    - "./rewards/format.py::format_reward"
    - "./rewards/overlap.py::gold_overlap_reward"
  weights: [0.1, 1.0]
```

The loader resolves recipe rewards first, then individual functions, then an
optional judge reward. It validates that an explicit `weights` list has exactly
one numeric entry per resolved callable and assigns it to TRL's
`reward_weights`. A recipe can return `(callable, weight)` pairs with defaults.

This is enough for a mechanical gatherer objective such as
`0.1 * valid_package_format + 1.0 * gold_overlap`. It is not a separate
component API: all components see the same completion batch and row kwargs,
and their outputs are combined by TRL. Keep each component deterministic,
side-effect-free, and batch-aligned. Returning `None` marks a sample as not
applicable and excludes it from advantage aggregation; use that deliberately
for missing gold, not as a generic error path.

## MoE story: definitive answer

The shipped configuration surface is misleading at first glance:

- `job_configs/moe_sft_example.yaml` selects `training_type: moe_sft` and
  `extends: MOE_SFT`.
- `job_configs/moe_dpo_example.yaml` selects `training_type: moe_dpo` and
  `extends: MOE_DPO`.
- `job_configs/moe_ep_sft_example.yaml` is the expert-parallel MoE SFT shape.
- There is no shipped `moe_grpo_example.yaml`.
- `training/default_configs/grpo_configs.py` does define `MOE_GRPO`, but
  `training/grpo.py::grpo_run` checks `is_moe_model_from_name(model_name)` and
  raises `ValueError("GRPO for MoE models is not supported in this EP branch")`.

The parser's parallelism validation also restricts expert parallel size to the
MoE SFT/DPO types. Therefore the definitive answer for reference revision
`4dbf17b...` is:

> **MoE configs cover SFT and DPO only. MoE GRPO is not supported, despite the
> unused `MOE_GRPO` defaults dictionary. LFM2-8B-A1B is not a gatherer GRPO
> candidate on this stack.**

A future MoE GRPO implementation would need an actual trainer/runtime path,
not only a YAML alias: expert routing/sharding, reward rollout placement,
checkpointing, and finite-update tests would all need to be added.

## Tool-calling data contract

For LFM2, definitions belong in the system message and the assistant tool call
is pre-baked in LFM bracket notation. This is the canonical JSON shape to use
for LFM2 training data:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "List of tools: <|tool_list_start|>[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}]<|tool_list_end|>"
    },
    {"role": "user", "content": "What's the weather in Boston?"},
    {
      "role": "assistant",
      "content": "<|tool_call_start|>[get_weather(location=\"Boston\")]<|tool_call_end|>"
    },
    {
      "role": "tool",
      "content": "{\"temperature\": 72, \"condition\": \"sunny\"}"
    },
    {"role": "assistant", "content": "It's 72 F and sunny in Boston."}
  ]
}
```

The exact markers and ordering are significant:

- LFM2 expects `<|tool_list_start|>` and `<|tool_list_end|>` around system tool
  definitions, and `<|tool_call_start|>[func(args)]<|tool_call_end|>` in the
  assistant `content`.
- A legacy LFM2 assistant call must be tool-call-first; prose before the call
  is rejected by validation. LFM2.5 has a different allowance and template.
- A `role: "tool"` message contains only the response payload. Do **not** put
  `<|tool_response_start|>` or `<|tool_response_end|>` in that content; the
  LFM2 chat template adds them during tokenization.
- Structured OpenAI `tool_calls` fields are auto-converted to the bracket form.
  Foreign `<tool_call>` XML, Mistral `[TOOL_CALLS]`, and similar markers are
  rejected with a conversion hint.

The SFT tokenizer calls `apply_chat_template(..., tools=row.get("tools"))`, and
`load_model` applies the pinned template override where the model family needs
it. This makes tool-call formatting part of the data validation/tokenization
path rather than an Axolotl-only convention.

## Comparison with the existing Axolotl pipeline

| Concern | leap-finetune for LFM2 | Existing Axolotl pipeline |
|---|---|---|
| Chat template | Liquid-specific loader, pinned LFM2/LFM2.5 templates, tool-format validation, and assistant masking in the training path. | Our configs explicitly provide custom Jinja templates and a loss-mask verifier; each new model/template requires the same audit and prefix-probe fixes. |
| GGUF | `leap-export-gguf` is shipped with the trainer, supports full checkpoints and PEFT adapters, and directly emits F16/Q8_0. | Current Axolotl configs produce safetensors/PEFT outputs; GGUF requires a separate llama.cpp conversion/build step and an adapter/base-model decision. |
| GRPO | TRL v1 `GRPOTrainer`, plain Python rewards, weighted reward composition, and colocated/server-mode vLLM are part of the pinned default stack. | The existing Axolotl launch matrix is SFT/full/LoRA/FSDP2; it does not provide this LFM2 GRPO+vLLM recipe or reward seam. |
| Model scope | First-party LFM2 dense SFT/DPO/GRPO and MoE SFT/DPO paths. | Broader general-purpose training surface and an already audited Qwen3/Qwen3.5 ladder. |
| Operational tradeoff | Narrower stack and heavier CUDA/vLLM dependency resolution; the CUDA smoke must be run on Linux. | Existing local preparation/audit evidence is useful, but it is not evidence that LFM2's template, exporter, or GRPO path works. |

The practical decision is to use leap-finetune for the LFM2-350M smoke and, if
that smoke passes, the 1.2B-class voice BRIDGE SFT. Use the custom reward only
for a dense gatherer GRPO experiment until the project ships a real MoE GRPO
runner. Keep Axolotl for the existing Qwen student ladder rather than treating
its successful template audit as transferable LFM2 evidence.

## Evidence reviewed

The assessment is based on the reference README/AGENTS.md and these source
contracts at the revision recorded above:

- `pyproject.toml`, `uv.lock`, `.python-version`, and the example SFT, LoRA,
  GRPO, and MoE YAML files;
- `src/leap_finetune/checkpointing/model_loading.py` and
  `job_configs/chat_templates/lfm2_chat_template.jinja`;
- `src/leap_finetune/data_loading/validate_dataset_format.py`,
  `validate_tool_format.py`, and `tokenize_data.py`;
- `src/leap_finetune/training/grpo.py`, `default_configs/grpo_configs.py`,
  `moe_sft.py`, and `moe_dpo.py`;
- `src/leap_finetune/rl/rewards/loader.py`, `rewards/README.md`, and the RL
  contract tests;
- the existing repository's `tools/gather-distill/train/README.md` and
  Axolotl configs/templates.

Claims requiring a GPU are marked passed only where the retained run evidence
supports them. GGUF load remains explicitly unmeasured because the first-party
exporter failed before producing a file; GRPO/vLLM and LoRA adapter-save evidence
is recorded above.
