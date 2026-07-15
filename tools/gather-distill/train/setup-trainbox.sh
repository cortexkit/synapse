#!/usr/bin/env bash
# Reproduce the CUDA training and GGUF toolchain used by the Qwen3.5-2B smoke run.
set -euo pipefail
IFS=$'\n\t'

AXOLOTL_REVISION="09d325b4fd1288b1473c8a330dd19e3c91b1ac32"
LLAMA_CPP_REVISION="b4e3dc613baa92a3884d4151e3d631395c81934a"
BASE_IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
BASE_IMAGE_DIGEST="sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
AXOLOTL_DIR="${AXOLOTL_DIR:-$WORKSPACE_ROOT/axolotl}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$WORKSPACE_ROOT/llama.cpp}"
ENVIRONMENT_OUTPUT="${ENVIRONMENT_OUTPUT:-$WORKSPACE_ROOT/trainbox-environment.json}"
MAX_JOBS="${MAX_JOBS:-8}"

export DEBIAN_FRONTEND=noninteractive
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

# Run this before apt or pip so a host with a broken container runtime costs only
# the few seconds needed for this probe.
python - <<'PY'
import json
import torch

if not torch.cuda.is_available():
    raise SystemExit("pre-install CUDA gate failed: torch.cuda.is_available() is false")
a = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
b = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
c = a @ b
if not torch.isfinite(c).all().item():
    raise SystemExit("pre-install CUDA gate failed: BF16 matrix product is not finite")
print(json.dumps({
    "gate": "pre-install-cuda",
    "passed": True,
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
}))
PY

apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates cmake git ninja-build python3-dev
rm -rf /var/lib/apt/lists/*

python -m pip install --upgrade \
  pip "setuptools>=64" "setuptools_scm>=8" wheel packaging ninja psutil

# Install the cu128 wheels before native extensions so their build metadata sees
# the same Torch ABI as the smoke environment.
python -m pip install --no-cache-dir \
  "torch==2.9.1" "torchvision==0.24.1" \
  --index-url https://download.pytorch.org/whl/cu128
# The base image's optional torchaudio build pins its original Torch 2.5.1 ABI.
python -m pip uninstall -y torchaudio

python -m pip install --no-cache-dir \
  "transformers==5.9.0" \
  "accelerate==1.13.0" \
  "trl==1.5.1" \
  "peft==0.19.1" \
  "xformers==0.0.33.post2" \
  "flash-linear-attention==0.4.1" \
  "fla-core==0.4.1"

python -m pip install --no-cache-dir --no-build-isolation \
  "causal-conv1d==1.6.2.post1"

if [[ ! -d "$AXOLOTL_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/axolotl-ai-cloud/axolotl.git "$AXOLOTL_DIR"
fi
git -C "$AXOLOTL_DIR" fetch --depth 1 origin "$AXOLOTL_REVISION"
git -C "$AXOLOTL_DIR" checkout --detach "$AXOLOTL_REVISION"

# Let Axolotl resolve its pinned Python dependencies without starting a second
# FlashAttention build. The extension is built once, explicitly for H100 below.
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE \
  python -m pip install --no-cache-dir --no-build-isolation \
  -e "$AXOLOTL_DIR[flash-attn]"

# Axolotl's chunked loss already upcasts one logits chunk at a time. Accelerate
# otherwise duplicates the complete 32k x vocabulary BF16 output in FP32 before
# Trainer reads the scalar loss, recreating the 30.31 GiB allocation this config
# is meant to avoid. Trainer does not consume the returned training logits.
python - <<'PY'
from pathlib import Path
import inspect
import py_compile
import accelerate.accelerator

path = Path(inspect.getsourcefile(accelerate.accelerator))
text = path.read_text()
replacements = {
    "model.forward = convert_outputs_to_fp32(autocast_context(model_forward_func))": (
        "model.forward = autocast_context(model_forward_func)"
    ),
    "model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)": (
        "model.forward = MethodType(model.forward.__func__, model)"
    ),
}
for original, replacement in replacements.items():
    if text.count(replacement) == 1 and original not in text:
        continue
    if text.count(original) != 1:
        raise SystemExit(f"unexpected Accelerate output-upcast patch target in {path}: {original}")
    text = text.replace(original, replacement)
path.write_text(text)
py_compile.compile(str(path), doraise=True)
print(f"patched training-output upcast in {path}")
PY

unset FLASH_ATTENTION_SKIP_CUDA_BUILD
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
FLASH_ATTN_CUDA_ARCHS=9.0 \
MAX_JOBS="$MAX_JOBS" \
  python -m pip install --no-cache-dir --no-build-isolation --no-deps \
  --force-reinstall --no-binary=flash-attn "flash-attn==2.8.3"

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
fi
git -C "$LLAMA_CPP_DIR" fetch --depth 1 origin "$LLAMA_CPP_REVISION"
git -C "$LLAMA_CPP_DIR" checkout --detach "$LLAMA_CPP_REVISION"
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON
cmake --build "$LLAMA_CPP_DIR/build" \
  --target llama-quantize llama-server \
  --config Release --parallel "$MAX_JOBS"

python -m pip check
python - <<'PY'
import json
from importlib.metadata import version

import torch
from flash_attn import flash_attn_func

q = torch.randn((1, 64, 4, 64), device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q)
if not torch.isfinite(out).all().item():
    raise SystemExit("post-install FlashAttention smoke produced non-finite output")
expected = {
    "axolotl": "0.17.0.dev0",
    "torch": "2.9.1+cu128",
    "torchvision": "0.24.1+cu128",
    "transformers": "5.9.0",
    "accelerate": "1.13.0",
    "trl": "1.5.1",
    "peft": "0.19.1",
    "flash-attn": "2.8.3",
    "flash-linear-attention": "0.4.1",
    "fla-core": "0.4.1",
    "causal-conv1d": "1.6.2.post1",
    "xformers": "0.0.33.post2",
}
actual = {package: version(package) for package in expected}
if actual != expected:
    raise SystemExit(f"package pin mismatch: expected {expected}, got {actual}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"expected Torch CUDA 12.8, got {torch.version.cuda}")
print(json.dumps({
    "gate": "post-install-flash-attention",
    "passed": True,
    "packages": actual,
    "torch_cuda": torch.version.cuda,
}))
PY

mkdir -p "$(dirname "$ENVIRONMENT_OUTPUT")"
BASE_IMAGE="$BASE_IMAGE" \
BASE_IMAGE_DIGEST="$BASE_IMAGE_DIGEST" \
AXOLOTL_DIR="$AXOLOTL_DIR" \
AXOLOTL_REVISION="$AXOLOTL_REVISION" \
LLAMA_CPP_DIR="$LLAMA_CPP_DIR" \
LLAMA_CPP_REVISION="$LLAMA_CPP_REVISION" \
ENVIRONMENT_OUTPUT="$ENVIRONMENT_OUTPUT" \
python - <<'PY'
import json
import os
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch

packages = [
    "axolotl", "torch", "torchvision", "transformers", "accelerate", "trl", "peft",
    "flash-attn", "flash-linear-attention", "fla-core", "causal-conv1d", "xformers",
]
report = {
    "recorded_at": datetime.now(timezone.utc).isoformat(),
    "base_image": os.environ["BASE_IMAGE"],
    "base_image_digest": os.environ["BASE_IMAGE_DIGEST"],
    "axolotl_commit": subprocess.check_output(
        ["git", "-C", os.environ["AXOLOTL_DIR"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "llama_cpp_commit": subprocess.check_output(
        ["git", "-C", os.environ["LLAMA_CPP_DIR"], "rev-parse", "HEAD"], text=True
    ).strip(),
    "packages": {package: version(package) for package in packages},
    "torch_cuda": torch.version.cuda,
    "nvidia_smi": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True
    ).strip(),
    "system_nvcc": subprocess.check_output(["nvcc", "--version"], text=True).splitlines()[-1],
    "accelerate_full_output_upcast_disabled": True,
}
if report["axolotl_commit"] != os.environ["AXOLOTL_REVISION"]:
    raise SystemExit("Axolotl checkout changed while recording the environment")
if report["llama_cpp_commit"] != os.environ["LLAMA_CPP_REVISION"]:
    raise SystemExit("llama.cpp checkout changed while recording the environment")
Path(os.environ["ENVIRONMENT_OUTPUT"]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
