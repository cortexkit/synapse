#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! "$1" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Usage: %s <N> <out>\n' "$0" >&2
    exit 2
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
case "$2" in
    /*) output="$2" ;;
    *) output="$PWD/$2" ;;
esac
export XDG_CACHE_HOME="$output/cache/runtime" HF_HOME="$output/cache/huggingface"
export TORCH_HOME="$output/cache/torch" NUMBA_CACHE_DIR="$output/cache/numba"
export MPLCONFIGDIR="$output/cache/matplotlib" JOBLIB_TEMP_FOLDER="$output/cache/joblib"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$script_dir/run_qwen3_asr_full.py" --workers "$1" --output "$output"
