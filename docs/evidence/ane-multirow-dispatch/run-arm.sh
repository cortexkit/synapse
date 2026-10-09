#!/bin/sh
# Usage: run-arm.sh <layout:rows:width> <output.json> [extra env assignments...]
set -eu
arm="$1"; out="$2"; shift 2
root="$HOME/mason-ane-multirow"
bin=$(ls -t "$root"/target/release/deps/ck_synapse_worker_ane_direct-* | grep -v "\.d$" | head -1)
snapshot="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
cd "$root/synapse/crates/synapse-worker-ane-direct"
exec env -u TMPDIR \
  ANE_MULTIROW_ARM="$arm" \
  ANE_TEST_PACKAGES="$root/packages" \
  ANE_MULTIROW_INPUT="$HOME/.local/share/cortexkit/synapse/aft-headtohead/engram.jsonl" \
  ANE_MULTIROW_TOKENIZER="$snapshot/tokenizer.json" \
  ANE_MULTIROW_OUT="$out" \
  "$@" \
  "$bin" --ignored --exact multirow::hardware::multirow_experiment --nocapture --test-threads=1
