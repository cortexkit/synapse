# Study: TokenSpeed (lightseekorg/tokenspeed)

Cloned to ~/Work/OSS/tokenspeed (studied @ d34dcf1, 2026-08-13). A speed-of-light
LLM inference engine for agentic workloads — TensorRT-LLM-class performance with
vLLM-class usability; PyTorch Ecosystem member; day-0 enablement for Qwen3.8-2.4T,
Kimi K3, TML Inkling. Four components: local-SPMD modeling layer with a static
communication compiler, C++ scheduler control plane + Python execution plane,
layered kernel registry, SMG-integrated entrypoint.

Frame for reading this study: they serve frontier MoE models on B200 clusters at
high concurrency; we serve small models on end-user devices. The transferable
material is ARCHITECTURE (contracts, guards, registries), not kernels or
parallelism. Sections ordered by usefulness to us.

## 1. Typestate request lifecycle — the FSM is the type system

Their request lifecycle is not a state enum + switch; it is
`std::variant<Bootstrapping, Submitted, Prefilling, PrefillDone, Decoding,
Retracted, Finished>` (fsm/states.h:32) where every event is a functor with one
`operator()` overload PER LEGAL SOURCE STATE, consuming the state BY MOVE and
returning the destination state (fsm/forward_events.h). An illegal
(event, state) pair falls through to `InvalidTransitionHandler`, which throws
with both type names (fsm/base_event.h:31-41). `Request::Apply` is four lines of
`std::visit` (scheduler/request.h:46-51).

The part worth stealing: **states own their resources, and transitions move
them**. Block tables (KV ownership) live inside the state object; `FinishEvent`
and `AbortEvent` can only produce `Finished` by first `TakeBlockTables()` +
`FreeRequest` (forward_events.cpp:117-122). A code path that forgets to release
KV does not compile, because the state that held it was consumed. ForwardState
is move-only (forward_states.h:86-89).

Synapse mapping: our decode worker supervision (owned-decode-worker
supervisor.rs) and durable-job lifecycle are runtime-checked state machines.
Rust does this pattern natively (enum variants owning resources, `match` on
move). When multi-stream decode serving happens (scheduler already quantum-
sequences), the request-owns-its-KV-by-construction shape is the one to build —
not a `state: String` column consulted by discipline.

## 2. Kernel registry priority bands — our lane selection, generalized

tokenspeed-kernel's registry (registry.py:75-132) solves "many implementations
of one op, per hardware/dtype/shape" — the exact problem our engine lanes +
certification machinery solve — with three ideas we don't have:

- **Priority BANDS with reserved plugin headroom**: REFERENCE=0, PORTABLE=4..7,
  PERFORMANT=8..11, SPECIALIZED=12..15, PLUGIN=16..19. In-tree kernels may not
  use the plugin band, so out-of-tree backends always have room to win without
  auditing every registration. Bands are a portability/performance CONTRACT,
  not a number fight.
- **Reference always registered, never auto-selected** (priority 0): the
  PyTorch ground truth sits in the same registry as the fast paths and is
  selectable by name in tests. Our ort-fp32 oracle lanes follow the same
  philosophy but live outside the selection machinery; theirs is uniform.
- **Selection = capability gate → trait filter → priority tiebreak → optional
  per-family SelectionOracle → cached** (selection.py), with `override=` and
  config-file overrides as the dev escape hatch. Format signatures
  (dtype/layout tuples, signature.py) are first-class selection keys — the
  kernel-dispatch analog of our fingerprint discipline.

Synapse mapping: today our lane choice is certification-gated but mostly
hand-routed (family dispatch + engine preference). At 5+ engines x quants x
platforms, a declarative registry with bands and capability gates is the shape
that stops the `if cfg!(...)` sprawl. Bank for when lane count next grows.

## 3. The aligned-grain guard — turning silent degradation into a hard error

`aligned_max_scheduled_tokens` (engine/scheduler_utils.py:139-187): recurrent-
state cache groups (Mamba-class — family=State) register their state snapshot
ONLY when a prefill chunk ends exactly on a cache-block boundary. A chunk size
not a multiple of that grain never registers a snapshot, and prefix-cache reuse
silently degrades to ZERO for the whole model. Their fix is structural: floor
the chunk size to the LCM of all state grains, and RAISE if the budget is
smaller than one block ("raising is safer than increasing a limit that may
already have sized executor buffers").

This is directly load-bearing for us: **LFM2's conv-cache is exactly a
recurrent-state group**. When we build KV/state reuse for hybrid models (note
#661 territory) or the LFM2 split-prefill follow-up, snapshot-at-aligned-
boundaries-only is the trap, and the guard belongs at config time, not in a
comment. The general pattern is one we already preach (kill the problem class
structurally) applied to a performance cliff rather than a correctness bug.

## 4. Prefix-cache logits contract — prefix_replay_tokens

SchedulerConfig.prefix_replay_tokens (scheduler/types.h:65-69): the minimum
prompt TAIL that must be recomputed after a prefix-cache hit, "zero preserves
the default logits contract, which already recomputes at least the final prompt
token" — and their DSpark speculative path RAISES this because the drafter
needs runtime state rebuilt from a longer tail, advertised through the draft
model's config as a CAPABILITY, fail-closed without it
(resolve_dspark_prefix_replay_tokens, scheduler_utils.py:314-330).

Synapse mapping: when we do KV reuse for repeated oneshot prefixes (dreamer
system prompts are the obvious win), the contract questions are exactly these:
how many tail tokens must re-run to keep logits/tap semantics intact, and which
downstream mechanisms (grammar mask state, sidecar pickup, chain-K) need state
rebuilt from the replayed tail. Their answer — a single integer contract,
capability-advertised, fail-closed — is the right shape.

## 5. Scheduler as pure planner; retraction and recovery discipline

The C++ scheduler emits an `ExecutionPlan` (forward batch + cache transfer ops
+ `pages_to_zero`) and mutates nothing at execution time; Python executes and
feeds results back as typed events via `Advance` (scheduler.cpp:415-457). Our
module/worker split has the same grain (module plans, worker executes over
IPC), but two of their details are worth keeping:

- **pages_to_zero** (execution_plan.h:43-46): newly assigned cache child pages
  are enumerated in the plan and zeroed by the runtime before use, with group
  identity because an LCM parent can hold live siblings — leftover-bytes
  hygiene made explicit in the contract instead of assumed.
- **Retraction + recovery head-of-line** (forward.cpp:527-554): under capacity
  pressure a Decoding request is RETRACTED (device pages released immediately,
  tokens requeued as prefill). Recovery runs strictly head-of-line — priority 0
  with an explicit comment: starting another recovery earlier "can make the two
  requests repeatedly evict each other". Livelock prevented by ordering
  discipline, not by backoff. Our scheduler yields at quantum boundaries and
  never retracts; if we ever add capacity-pressure preemption for long decodes,
  this is the reference design.
- **overlap_schedule_depth ∈ {0,1}** (types.h:53-56): CPU scheduling of step
  N+1 overlaps GPU execution of step N by AT MOST one step, enforced as a
  validated contract, not a tunable. Bounded overlap = bounded speculation
  about uncommitted decode lengths.

## 6. Agentic-workload specializations (what the label actually means)

- **Tokenizer prefix caches default-ON for gateway launches**
  (cli/serve_smg.py:202-225): L1 prefix-caching at special-token boundaries
  cuts TTFT ~30% for shared-system-prompt traffic; they inject the flags and
  intercept `--no-` forms for opt-out. The lesson is the default direction:
  agentic traffic has massive shared prefixes, so prefix machinery should be
  on unless refused.
- **Honest cache metrics**: `prefix_cache_hits_total` documented as a ratio
  against `prompt_tokens_total` (metrics/collector.py:359-367) — tokens served
  from cache, not "requests that hit".
- **Draft-block decode trick** (mla.py:125-131): DFLASH/DSpark drafters
  propose a whole block in one forward by expanding each request into
  spec_num_tokens single-query rows sharing the block-end seq_len — non-causal
  within the block WITHOUT a mask shape change. Clever and portable; relevant
  if our batched-verify substrate ever grows a block-drafter.
- TTFT is a first-class histogram; first-token time recorded at the output
  processor with a "pure" generation-time split (output_processor.py:334-340).

## 7. Placement compiler (SPMD) — noted, not actionable

Modules carry DTensor-inspired placement annotations (Replicate/Shard/Partial
per parallel group — placement.py:36-50); a compile pass walks each decoder
layer's execution plan, tracks (hidden, residual) placement state, and inserts
the minimal collectives at module boundaries, fusing the reduce into a
following norm when groups allow (compiler.py:178-472). Users never hand-write
parallelism. Elegant; irrelevant to single-device serving. If owned-CUDA ever
goes multi-GPU, start here rather than inventing annotation vocabulary.

## Verdict

Nothing here says our architecture is wrong; several things say where it goes
next as serving concurrency grows. Concretely banked for synapse:

1. Typestate + resource-owning states for any future multi-stream decode
   scheduler (Rust-native pattern; KV freed by construction).
2. Kernel/lane registry with priority bands + always-present reference when
   engine-lane count next grows.
3. The aligned-grain structural guard, specifically for LFM2 conv-cache state
   reuse and any recurrent-state snapshot machinery.
4. prefix_replay_tokens as the contract shape for oneshot KV-prefix reuse
   (dreamer shared system prompts), capability-advertised and fail-closed.
5. Retraction head-of-line discipline if capacity preemption ever lands.

Non-goals confirmed by contrast: their whole parallelism/MoE/disaggregated-PD
surface exists because of multi-GPU frontier serving — none of it earns its
complexity at our model sizes and request rates.
