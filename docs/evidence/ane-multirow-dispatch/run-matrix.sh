#!/bin/sh
# Runs every arm in its own process, one after another; each writes <dir>/<arm>.json.
set -u
dir="$1"; mkdir -p "$dir"
for spec in 2:128 4:128 8:128 2:256 4:256; do
  for layout in batch-axis width-segmented width-folded width-block-mask; do
    arm="$layout:$spec"
    name=$(echo "$arm" | tr ":" "-")
    start=$(date +%s)
    "$HOME/mason-ane-multirow/run-arm.sh" "$arm" "$dir/$name.json" > "$dir/$name.log" 2>&1
    echo "$arm exit=$? seconds=$(( $(date +%s) - start ))"
  done
done
