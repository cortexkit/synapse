# Study: speculative-serving link sweep (2026-08-24)

Four sources examined for the owned-LLM-loop program (draft:
`.cortexkit/alfonso/drafts/2026-08-24-owned-llm-loop-agentic-scale-local-serving-27b-class.md`).
Verdicts first; numbers are quoted from the sources, not re-measured.

## 1. dFlash 2 (inco.ai/blog/dflash2, checkpoints on HF)

**What it is:** lossless speculative decoding with a one-pass block-diffusion
drafter (a separate ~2B model, NOT an MTP head). dFlash 2 adds two learned
components over dFlash 1: a candidate path selector (keeps top-16 per draft
position, scores adjacent pairs with 256-dim token embeddings; +2M params,
+0.6% latency, +0.34 acceptance) and a two-tap dynamic depthwise convolution
around every sublayer (+3% params, +0.7% latency) that restores local
inter-token dependency without making drafting autoregressive.

**Numbers that matter to us (Qwen3.8-27B, block size 8 = 7 draft tokens):**
- Mean acceptance length: dFlash 2 **4.80** vs native MTP **4.28** vs
  DSpark 3.62 (GSM8K 5.46, HumanEval 4.39, MT-Bench 4.10).
- SGLang batch-1 speedups 2.67-3.43x; at concurrency 32 they collapse to
  1.01-1.45x — single-stream is where block drafting pays, which is exactly
  our serving discipline.
- llama.cpp **Metal, M5 Pro 64GB**, target Q4_K_M: 1.77-1.85x over serial
  (18.4-19.3 tok/s vs 10.4), acceptance ~5.0; draft quantization barely
  matters (BF16/Q8_0/Q4_K_M within 5%).
- MLX quantized caveat from the dFlash repo: `block_size <= 5` because their
  quantized matmul kernel degrades at larger verification widths.

**Catch:** the drafter is trained target-specific against the frozen target.
A published Qwen3.8-27B dFlash 2 checkpoint exists (2B BF16), so adoption
means implementing the drafter forward (block diffusion + selector + conv),
not training. The verifier stays our existing exact-match machinery.

**Also settled here:** Qwen3.8-27B is **dense** (official card + our staged
challenge weights: no expert keys, 64 layers x 5120). The MoE evidence in the
dFlash line is dFlash 1 on Qwen3-Coder-30B-A3B (3.5x/2.6x/3.2x batch-1 on
HumanEval/LCB/MBPP) — MoE targets work, but dFlash 2 is unvalidated on MoE.

## 2. FreeToken (FlashML-org)

CUDA-only Python/PyTorch serving runtime whose one idea is serving large MoE
models with experts host-resident: GPU expert-slot LRU cache with Triton-side
eviction, hybrid CPU/GPU expert execution decided by a measured
CPU-bandwidth-vs-PCIe calibration (hybrid when CPU MoE bandwidth > 2x PCIe
gather), prefill streamed through double-buffered full expert layers, radix
prefix KV cache plus recurrent-state snapshots for hybrid models. No
speculative decoding machinery. Nothing runs on Metal.

**Transplantable for our wave 2 (MoE):** the expert-slot LRU data model and
hybrid fetch policy, the MoE-vs-KV memory budgeting split, the FTW aligned
sharded weight format (streaming loader, 8GiB shards, GiB/s-reporting
loader), and the radix/recurrent-snapshot cache shapes (which independently
converge with TokenSpeed's aligned-grain lesson). The PCIe cost model does
NOT transplant: Apple unified memory has no host/device split, so expert
"offload" becomes a residency/admission policy over shared MTLBuffers, not a
streaming problem.

## 3. magnitude (magnitudedev)

Local-first coding-agent product. Relevance is strategic, not mechanical:
its inference layer is a **Rust daemon (ICN) managing pinned llama.cpp GGUF
models**, with a model catalog that already ships speculative-decoding
configurations (Qwen 3.5/3.6/3.8, Gemma 4, LFM2.5, DeepSeek V4 Flash, GLM
5.2...) — independent convergence on "own the local serving daemon, pin the
engine, catalog the models." Agent-loop patterns worth noting: tools as
typed schema contracts with streaming structural tool-call parsing
(field-path chunks, incremental JSON validation), and tool-call limits
enforced twice (provider grammar + mechanical harness abort) — the same
belt-and-suspenders we use for constrained decoding. It has NO native UI
grounding (browser is a user panel, not an agent surface; vision is
describe-only) — relayed to CEREB as competitive intel.

## 4. MTPLX v2.7.1 -> v2.9.1 delta (405 commits)

We banked the v2.7.1 depth-controller design already
(`docs/reference-mtp-depth-controller.swift`). The delta has four things:

**(a) Telemetry stack worth copying.** Three layers: per-round decode events
(depth requested/selected, per-draft acceptance flag/probability, correction
token, verify strategy, per-stage timing); ~1Hz JSONL trace buckets
(acceptance rate by depth, draft/verify-forward/verify-eval/accept/repair/
commit/bonus time split, memory and cache sizes); and a rotating
flight-recorder JSONL per request (begin/prefill/1Hz/end/postcommit events,
`end` written even on cancel/disconnect). Report panels render per-depth
acceptance bars and a normalized draft/verify/accept/other time split. This
is the concrete shape for our draft's "acceptance-rate and depth telemetry
in the result envelope" acceptance criterion.

**(b) Depth-tuner warmup trap — we hit this exact class.** Their v2.9.0 fix:
the tuner now warms every candidate depth before timed rows because
model-load/compile cost was biasing it toward shallow depths. Same disease
as our certification battery's cold-slope TTFT sampling (fixed with
converge-then-measure warmup). Validates steady-state-only timing as a rule
for any depth policy we ship.

**(c) Postcommit-starvation spiral — transplantable scheduling lesson.**
Their background KV-commit waited for foreground-idle before even attempting
the lock; agent fan-out keeps foreground continuously busy, so saves hit
deadlines without ever trying, and preemption discarded the job terminally —
a fast agent chain could starve the same save forever (committed stream
froze at 15k tokens while generation passed 76k). Fix shape: the resource
lock attempt is the gate (not an idleness predicate), bounded yield/grace
under pressure, preempted jobs re-arm on an idle queue (capped retries),
every job version-fenced against session state, strict priority lanes
(foreground > idle-commit > persistence). Rule for our loop: preemption must
never mean discard, and background work must contend on the actual resource,
not on a proxy signal.

**(d) Draft-head quantization pairing.** v2.9.0 requantized MTP heads to
match trunks (INT4 head for INT4 trunk, INT8 for INT8) with a draft-only
requantization path that never touches target weights. Their shipped
MoE+MTP contract (Qwen3.6-35B-A3B) pins exact per-module quantization
(q8/g64 router + q4/g64 experts on target; dense BF16 router + q4/g32
routed on MTP) with install-time numerical self-checks and fail-closed
verification — the same artifact-identity discipline as our certification
records, applied to a draft/target pair. Their K1 is shipped; K2 is opt-in
with a three-row verify — matching our finding that depth pays only when
head-forward cost is well under a backbone step.

## Consequences for the owned-LLM-loop draft

- The speed-layer open question is now three-way with published evidence:
  native MTP head (4.28) / dFlash2 drafter (4.80, +2B resident, new forward
  path) / head with paired quantization (MTPLX pattern). Draft updated.
- Telemetry criterion gets a concrete reference shape (MTPLX flight
  recorder); steady-state warmup is a hard rule for depth tuning AND any
  timing gate.
- Wave-2 MoE inherits FreeToken's budget/LRU data model reshaped for
  unified memory, and MTPLX's exact per-module quant-contract discipline.
- KV persistence/prefix-reuse design must carry the postcommit lesson:
  lock-contention as the gate, bounded yield, re-arm not discard, version
  fences.
