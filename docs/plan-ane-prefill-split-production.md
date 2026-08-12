# ANE-prefill + GPU-decode split: productionization discussion page

Status: DISCUSSION PREP (2026-08-12) — no epic fired; decision points at the end.
Spike evidence: bench/spikes/ane-prefill-split/ANE-PREFILL-SPLIT.md (merged 9175e17, locked M1).

## What the spike proved
- Stateless CoreML W128 prefill graph with explicit per-layer KV outputs sidesteps the
  stateful-ANE wall entirely (99.905% ANE placement, coremltools 8.3 / torch 2.5.1).
- KV transfer ANE->Metal 6.02ms (under the pre-registered 20ms kill rule).
- TTFT 39.4ms vs 688ms pure-GPU = 17.45x, request energy 1.46x better.
- Correctness: 20/20 token-exact across 64 decode tokens; ANE fp16 prefill drift did
  not flip a single token on the battery.
- Dead branch (do not revisit): W32x4 windowed prefill — windowed KV structurally invalid (9/20).

## What production requires beyond the spike
1. **Artifact story**: a per-machine compiled .mlmodelc prefill package per (family, bucket).
   Follows the ANE embed lane precedent (bucketed packages, compile-scale LOAD budget,
   mlmodelc manifest support already in the loader since gen-12).
2. **Identity**: the ANE prefill graph is part of the decode processing identity — a new
   processing_fingerprint input (prefill_engine=ane-w128 vs metal), NOT a new vector-space
   fingerprint (decode output remains token-exact; the certification battery is the proof).
   Certification: extend the decode probe with a split-prefill arm reusing the spike's
   20x64 battery + boundary rows. The records epic makes this cheap: a new engine identity
   rotates the profile, probe re-certifies, done — no record rebinds.
3. **Routing**: per-request TTFT-sensitivity is unknowable; the win is machine-level.
   Proposal: knob-level policy — quiet/balanced knobs prefer split-prefill when certified,
   performance knob keeps pure-GPU (max throughput under sustained decode).
4. **Worker plumbing**: ck-synapse-worker-decode grows an optional ANE prefill stage
   (Swift CoreML shim like the ANE embed worker, or in-process objc bridge). The KV
   handoff format is the spike's f16 layout convert (6ms measured).
5. **Failure semantics**: ANE prefill failure = transparent fallback to GPU prefill
   (same request, no error surface) + health counter; repeated failures quarantine the
   split arm per lane, never the lane itself.

## Sizing (honest)
- ~1 spec-pipeline epic: worker stage + package build/ingest + probe arm + routing knob
  + failure semantics + fleet enable. 5-6 slices, most mechanical; the Swift shim and
  the probe arm are the two real ones.
- Hardware note: M1 numbers are the authority; M5 expected better (bigger ANE). No
  non-Mac surface (CUDA/Vulkan unaffected).

## Why now / why not now
- FOR: biggest UX lever we have for cold-prompt latency (voice bridge, interactive
  oneshots); GPU freed during prefill (embed batches keep running); energy story.
- AGAINST: current consumers (dreamer classify class) are latency-tolerant; the win
  is user-facing interactivity that arrives with the CK-app voice/persona work — 
  shipping it early means certifying and carrying a second execution shape with no
  consumer feeling it yet.

## Decision points for Ufuk
1. Fire the epic now, or park until the first latency-sensitive consumer (WERNI persona
   plane / voice bridge) has a date?
2. If fired: knob-level routing (proposal above) or config-only enable like chain-K?
3. Scope guard: Qwen3-0.6B only first, or both families (LFM2 conv-cache adds a
   wrinkle — conv state must also transfer, unmeasured)?
