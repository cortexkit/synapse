#!/bin/sh
# Final evidence run: every arm in its own process, then the replays.
set -u
root="$HOME/mason-ane-multirow"; d="$root/runs/final"; mkdir -p "$d/arms" "$d/replay"
arm() { n=$(echo "$1" | tr : -); s=$(date +%s); shift_arm="$1"; shift; "$root/run-arm.sh" "$shift_arm" "$d/$n.json" "$@" > "$d/$n.log" 2>&1; echo "$shift_arm exit=$? seconds=$(( $(date +%s) - s ))"; }
cd "$d/arms" 2>/dev/null
for spec in 2:128 3:128 4:128 8:128 2:256 4:256; do
  for layout in batch-axis width-segmented width-folded width-block-mask; do d2=$d; d=$d2/arms; arm "$layout:$spec"; d=$d2; done
done
for spec in 2:64 4:64 8:64 2:32 4:32 8:32; do
  for layout in width-segmented width-folded; do d2=$d; d=$d2/arms; arm "$layout:$spec"; d=$d2; done
done
d2=$d; d=$d2/arms; arm batch-axis:2:64; d=$d2
d2=$d; d=$d2/replay
arm width-folded:2:128 ANE_MULTIROW_REPLAY=1 ANE_MULTIROW_REPLAY_ALL=1
arm width-folded:4:64 ANE_MULTIROW_REPLAY=1
arm width-folded:8:32 ANE_MULTIROW_REPLAY=1 ANE_MULTIROW_REPLAY_EXTRA=width-folded:4:64
d=$d2
echo DONE
