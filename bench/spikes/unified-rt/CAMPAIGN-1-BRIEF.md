# Campaign 1 brief — Qwen3-Embedding-0.6B f16 Metal throughput (Athena v3 validation workload)

Prepared for the v3 central-executor (ALF Slice B acceptance fixture). Everything here is frozen against committed artifacts; nothing requires interpretation at campaign time.

## Objective

Maximize steady-state embedding throughput of the owned runtime's Qwen3-Embedding-0.6B f16 Metal path on the locked M1 rig, above the frozen baseline, without violating the correctness floor. The original motivating gap (0.88x vs MLX) was closed by the graduation probe — current standing is 1.117x MLX — so the campaign objective is absolute: **beat the frozen owned baseline**.

## Frozen baseline (graduation probe, 2026-07-12, GRADUATION-PROBE.md)

- Owned runtime Qwen3 f16 Metal, exact shapes, package-HIT, in-process pass 3: **7,000 tok/s** (result JSON: results/graduation-probe/owned-qwen3-steady.json).
- Comparators (context, not gates): llama-server Metal 4,153; MLX Python bf16 6,266.
- Baseline binary: source c0df8540, sha256 3b92806c…e2b3. Campaign candidates build from current master (the baseline number is the bar, not the binary).

## Accept gate (dual-threshold, frozen at classify — the campaign's round-termination predicate)

1. **Minimum win**: a candidate variant must beat the current best (starting at the frozen baseline) by **>= 3%** steady tok/s, measured by the pinned rig, or it is pruned. Prune verdicts are banked to the leaderboard graveyard with full profiles.
2. **Correctness floor (disqualifies regardless of speed)**: mean cosine **>= 0.9999** AND mean top-10 rank overlap **>= 0.995** vs the frozen ORT fp32 reference, every measured run. Reference: qwen3 400-row corpus sha256 5a9bfdc8…630c, reference vectors sha256 cacee1f…cf46 (staged on the rig at $SYNAPSE_BENCH_ROOT/bench-tools/unified-rt-serving/data/).
3. **Reconciliation**: rig-canonical vs candidate-reported token divergence > 1% = invalid run (rig enforces).

## Measurement protocol (rig-owned, candidate-blind)

- Rig: `synapse-rig` (bench/rig, merged 2f8330d) — hash-pinned at campaign freeze; executor asserts rig_metadata.sha256 + git_revision per result.
- Candidate entry: `spike-unified-rt --serve-stdio` (protocol v1 frames per bench/rig/RIG.md).
- Rig invocation per candidate: `--device metal --dtype f16 --shapes exact --passes 3 --max-length 512 --attention-units 4000000`; steady = pass 3. Two fresh-process repeats per candidate; the WORSE steady of the two is the scored number (anti-noise, anti-lucky-run).
- Paired control: current-best runs beside every round's candidates (same protocol, same session); if the control drifts > 3% from its banked value, the round re-baselines instead of scoring (thermal/OS drift guard).

## Rig profile (M1 box)

- Host: <bench-host> (Apple M1 Max 64GB, macOS 26.5.2, Xcode 26.6, ssh alias $SYNAPSE_BENCH_HOST). Model snapshot + corpus + reference pre-staged under $SYNAPSE_BENCH_ROOT/bench-tools/unified-rt-serving/.
- Lock discipline: `mkdir $SYNAPSE_BENCH_ROOT/bench.lock` before any timed run (trap-released rmdir), `pgrep -f Runner.Worker` must be empty (the box doubles as the macos-metal CI runner; CI jobs finish in 10-20 min — wait, never stop the service).
- Power/rig metadata: macmon 0.7.2 via the rig's metadata block.
- AVAILABILITY CAVEAT: this box ships out within days; its successor (rented Scaleway M4) is a DIFFERENT machine profile — the campaign must re-freeze the baseline on the successor before any cross-machine score comparison. Machine identity is part of every banked number.

## Search-space guidance for members (advisory, not binding)

Productive directions from the campaign record: MPSGraph op-fusion opportunities in the GQA attention block (broadcast repeat vs strided views), per-shape compilation descriptor options beyond O0 (EXECUTABLE.md measured O0-vs-O1 for MiniLM only), bucket-policy interactions (policy v1 known-regressive at 8 rows — a row-ladder variant is a legitimate candidate class), activation residency between blocks (fp32 islands audit: which casts are load-bearing for parity vs habit). Dead branches (do not re-propose without new mechanism): see f16-evidence/EVIDENCE.md + COUNCIL-VERDICT archives — O1 lazy compile, multi-shape append packages.

## Deliverable per round (executor-owned)

Leaderboard row per candidate: steady tok/s (both repeats), gate verdicts, parity numbers, rig metadata (P-state class, GPU W), patch identity; graveyard rows for pruned variants with the same completeness.
