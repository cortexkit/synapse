# Study: Inco Splash (2026-09-19)

Source: https://inco.ai/blog/splash/ and github.com/incoai/splash at f58d36d ("Splash 1.0", Apache-2.0), cloned to `~/Work/OSS/splash`. Read against the owned-LLM-loop epic (27B-class agentic serving on Metal, `.cortexkit/alfonso/drafts/2026-08-24-owned-llm-loop-*.md`) and the shipped owned decode engine. Every claim below is cited to a file; the blog's numbers are theirs, unreproduced here.

## What it is

A model-specialized local engine for Apple silicon (M3+, macOS 26.4+, 36 GB minimum): C++/Objective-C runtime (`runtime/`, ~30 k lines), hand-fused Metal kernels per model shape (`runtime/metal/kernels/`), a Python HTTP front (`server/`) speaking OpenAI Chat/Responses and Anthropic Messages, and a `dev/tuning` rig that produces precompiled kernel presets per GPU family. Two supported models at 1.0: Qwen3.6-35B-A3B and Qwen3.8-27B (`runtime/model/Qwen3_6Moe.*`, `Qwen3_8.*`). Everything not shared is rebuilt per model: fused shape-specific kernels, a trained DFlash 2 draft, a memory plan, pinned baselines. No generic runtime, no fallback path, no user tuning beyond `--max-memory` and `--max-context`.

The thesis is the one we reached from the other direction in July: a runtime that knows the model's exact shapes beats a general one, and the way to afford that is to support few models and automate the kernel work. Their kernel authoring is agentic ("our in-house kernel agents"), which is our campaign program with a different harness.

## The four mechanisms, from source

### 1. Scheduler: alternate prefill and decode at command boundaries, cap prefill when contended

`runtime/engine/Scheduler.cpp:161-185` (`next`): one batch plan at a time (`active_` guard). If both a decode plan and a prefill plan exist and share a priority, the scheduler **alternates** by whichever kind was committed last — "prefill and fixed-eight decode use different Metal graphs and cannot be packed into one command", so fairness is at command granularity, not token granularity.

Prefill selection (`nextPrefill`, 187-234) orders by priority, then an **overtake counter** (`kMaximumOvertakes = maximumBatchWidth - 1 = 3`: a lane yields to shorter arrivals at most three prefill commands in a row before it leads), then shortest remaining prompt, then arrival order. Prefill rows in one command are bounded by `prefillBudget` (236-263): the model's `prefillTokenBudget` (2048 for both models, `Model.hpp:251`) unless the lane is **contended** — a same-or-higher-priority peer is decoding, or a peer's prefill could finish within this slice — in which case rows halve until `rows * measured_ms_per_token <= 500 ms` (`kContendedPrefillMilliseconds`), floored at 64 rows. So a 32 K cold read is chunked to ~500 ms slices only when someone else is waiting; alone it runs 2048-token slices.

Decode selection (`nextDecode`, 265-303) batches up to `maximumBatchWidth = 4` requests that share priority, **cohort**, and **decode stage**; requests waiting on a grammar mask (`ApplyInitialMask`) are batched one at a time because applying a mask can terminate a request or start drafting.

Relation to ours: our DECODE scheduler is quantum-bounded (N = 16 tokens, prefill chunked to K = 16 mat-mat with yield points), which is the same idea with a fixed quantum instead of a measured-time budget. Their `prefillMillisecondsPerToken_` is measured at runtime and the budget adapts; ours is a constant. Their overtake counter is a cleaner anti-starvation rule than our oldest-anchor aging for the equal-priority case.

### 2. Memory plan: budget from the working set, everything else derived

`runtime/engine/MemoryPlan.cpp:288-303` with `MemoryPlan.hpp:50-71`: `headroom = max(2 % of recommendedMaxWorkingSetSize, 1 GiB)`; `hardBudget = min(recommended - headroom, --max-memory)`; `dynamicBudget = hardBudget - fixedRuntimeBytes` where fixed = target weights + draft weights + vision weights + pipeline reserve + runtime overhead, all known per model (`MemoryPlan.cpp:75-97` refuses a model spec that omits any of them). KV pages are sized from the model's KV layout, so the number of concurrent contexts that fit is arithmetic at startup, which is how they admit 16 concurrent 32 K requests on 48 GB where "a general-purpose memory policy accepted nine" (blog, Concurrency). `MemoryGovernor.cpp` reclaims cached state under pressure (`Cache::reclaimCache`, `Cache.hpp:110`) with a `ReuseBacking | ReleaseBacking` mode.

Relation to ours: our owned-LLM-loop spec has a hard 32 K reservation with eviction protection (S08). Theirs is the same shape with one addition worth taking: **the fixed-bytes ledger is a required part of the model descriptor and the plan refuses a model that does not declare it**, so "does it fit" is never a runtime discovery.

### 3. Paged KV as a chain keyed by (parent page, token span); recurrent state snapshotted at planned boundaries

`runtime/engine/Cache.cpp` (`lookup`): the prompt is walked in `pageTokens = 32`-token blocks (`SPLASH_TARGET_KV_BLOCK_TOKENS`, `ExecutionGeometry.h:16`); each block is looked up as `(parent block id, 32 token ids, image identity)` — a **chain**, not a flat prefix hash, so a hit at block N implies blocks 1..N-1 hit and no hash collision can splice two prompts. One token is always left un-cached ("regenerate request-specific anchor logits"). The deepest cached block wins and is touched (LRU).

For the hybrid model, the KV chain is not enough — Gated DeltaNet layers carry recurrent state that a KV page does not capture. `StateCache::acquireDeepest` (`StateCache.cpp:46-52`) walks the matched chain backwards for the deepest block that has a **composite state snapshot**. Snapshots are not taken at every page: `Engine::configureDraftStatePlan` (`Engine.cpp:471-514`) plans boundaries before prefill at (a) every `prefillCheckpointTokens = 2 * draftContextTokens = 4096` tokens, (b) the junction boundary, and (c) the replay boundary, because "arbitrary chunk ends do not carry a complete draft state". So a prefix hit lands on the nearest planned boundary at or below the KV match and replays from there.

Relation to ours: our KV reuse proof in the spec requires byte-identical continuation from a retained state bound to its producing fingerprint. Their chain keying gives that property structurally (a page is only reachable through its exact parent). The GDN snapshot planning is the TokenSpeed "aligned-grain guard" for recurrent caches (`docs/study-tokenspeed.md`) implemented the other way round: instead of refusing misaligned chunks, they choose the chunk ends up front so every end is aligned. LFM2's conv-cache would want the same planning if we ever cache LFM2 prefixes.

### 4. DFlash 2 draft as the decode path, per request, with grammar masks for the whole block

`runtime/model/Model.hpp:249-260` (`ExecutionLimits`): `draftProposalTokens = 7`, `targetVerifyRows = 8` (= proposals + the anchor), `draftQueryRows = 8`, `draftContextTokens = 2048` (the draft's sliding window; draft memory is bounded regardless of context, blog §"The draft stays small"). `DFlashDraft.hpp:100-116` keeps persistent draft K/V per batch lane, so each of the four batched requests carries its own draft state that advances with its accepted tokens — speculation inside the batch, not around it.

The part that matters most for us is **constrained decoding under speculation**, which our sidecar program parked as unsolved. `server/constraints.py:25-60`: the engine sends the draft's proposed tokens (`simulationTokens`, `Engine.cpp:164-166`, up to `MAX_ROWS - 1 = 8`) to the Python side; llguidance's `fill_next_token_bitmask_par_with_draft_tokens` validates the proposal against the grammar (`validate_tokens` returns how many of the K draft tokens are grammar-legal) and fills **one bitmask row per position, K+1 rows in one call**; the engine applies row i to verify position i. A draft token that leaves the grammar simply fails verification at that row. Our per-token host mask (`TokenMask` applied at the pre-commit tap, chain-K forced to 1 under grammar) is exactly this with K = 0. The blog's "when output is not constrained by a schema, drafting, verification, acceptance and state updates are one submission" is the unmasked fast path; masked decode takes a `WaitingMask` round-trip per step (`Scheduler.hpp:20`, `Engine.cpp:129-134`).

KV cache is **8-bit** (`runtime/ops/Q8PageStorage.*`, decode attention kernels read it directly: `runtime/metal/kernels/decode/attention_q8.metal`, `common/q8_attention_tile.h`) with a q8 store pass after each step (`attention_q8_store.metal`). Weights are 4-bit (`linear_q4.metal`, `q4_mpp_tiles.h`).

### Two smaller facts

- Python-to-engine transport is `FdTransport` (`runtime/engine/FdTransport.cpp`): a pipe for completion wakes and a framed protocol (`Protocol.cpp`, 1.5 k lines); per-step payloads are small (token ids, mask rows), logits never cross the boundary. Our worker protocol is the same shape.
- Kernel presets are built by `dev/tuning/*` and keyed by identity from `dev/tools/build_identity.py` (GPU family, core count, workload); the model package ships them precompiled. Our equivalent is per-machine certification with the harness measuring; theirs precomputes and pins.

## What to take, and what not to

Take:
1. **Measured-time prefill budget under contention** (`prefillBudget`): halve rows until a slice costs <= 500 ms when someone is decoding, otherwise run the full budget. Cheaper and fairer than a fixed quantum; it is one measured scalar plus the rule.
2. **Overtake counter** for equal-priority prefill fairness (`kMaximumOvertakes = width - 1`).
3. **Grammar masks for the whole draft block in one call** via llguidance's draft-aware bitmask fill. This is the missing piece that made our SO-BANK sidecar a negative (its structural hints were paid per token on the host). With K+1 rows filled once per step, speculation and JSON schema compose, and chain-K need not drop to 1 under grammar.
4. **Planned recurrent-state boundaries**: choose chunk ends before prefill so every snapshot is a complete state, rather than snapshotting wherever a chunk happens to end.
5. **Required fixed-bytes ledger in the model descriptor**, refused if absent.

Do not take:
- The Python front. Ours is a SubC module; the wire is already typed.
- The per-model precompiled preset model. Our lane is certified per machine by measurement, which is what lets an uncertified machine fail closed instead of running a preset built for a different chip.
- DFlash 2 itself as a dependency. The blog says the trained drafts are published; the engineering is the block-verify loop we already have (Leviathan acceptance, `DraftSource` seam in S05). Whether a DFlash 2 draft beats Qwen 3.8's native MTP head on our stack is the measurement the epic's wave-1/wave-2 split already schedules.

## Open question this raises

Their scheduler only batches decode across requests that share a **cohort**, and prefill and decode never share a command. Our owned-LLM-loop spec fixed a four-queue vocabulary (S09) but did not say whether a prefill quantum and a decode quantum may share a command buffer on Metal. Splash says no, for kernel-shape reasons. Worth stating explicitly in the spec before the first multi-request measurement, since it decides whether "prefill while streaming" costs one extra command boundary per alternation or none.
