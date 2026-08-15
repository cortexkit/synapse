# MTPLX study (v2.7.1 @ 963b923, 2026-08-15)

Repo: github.com/youssofal/MTPLX, cloned at ~/Work/OSS/MTPLX. Formerly a watch-shelf item; now directly load-bearing for two live programs: the Qwen 3.8 27B native-MTP challenge entry and the owned-LLM-loop epic.

## Verdict

MTPLX has become the most relevant external reference for both programs. It is a Python/MLX serving stack for Apple Silicon whose whole thesis is ours: use the model's OWN MTP head to draft, verify exactly against the target, never change the emitted stream. It already serves **Qwen 3.8 27B with its native multi-step MTP head** (the `qwen3_next` backend with a Qwen 3.8 contract overlay), with drop-day A/B receipts dated 2026-08-14 — one day after the model dropped. Its highest-value transplantable asset is a measured, cost-model-based **adaptive draft-depth controller**; its receipts also carry sampler-vs-acceptance findings specific to this exact model family.

Differences that keep it a reference rather than a dependency: exact speculative SAMPLING against the target distribution (temperature-legal acceptance, Leviathan-style) where the challenge and our microllm serving are GREEDY serial-exact; Python host loop (our loop is Rust/Metal); MLX kernels (ours are hand-written Metal).

## The depth controller (mtplx/adaptive.py) — the transplant target

Three policies behind one `observe(attempted_depth, accepted_depths)` interface:

1. `AdaptiveDepthPolicy` — streak heuristic: +1 depth after `increase_after=4` consecutive full accepts; -1 after early rejections (reject position <= depth/2) `decrease_after=1` times. Simple, stateless-ish, the baseline.
2. `ExpectedValueDepthPolicy` — draft-then-decide: EWMA per-depth acceptance (priors 0.92/0.64/0.32, alpha 0.12), continue drafting depth d+1 only if `prefix_accept x next_accept x confidence >= extra_cost x baseline_tok_s x (1+margin)`. The confidence factor reads the DRAFT head's top-2 margin and top-1 probability (tanh-scaled, weight 0.35, clamped 0.25-1.75) — per-round adaptivity from signals we also have at the tap.
3. `CostModelDepthPolicy` — the strongest: ported from omlx's `_DepthController` with measured design decisions documented inline:
   - `score(d) = (1 + p1 + p1p2 + ...) / t_est(d)` — maximize expected committed tokens per wall-clock cycle, not acceptance.
   - Acceptance EMA in TOKEN domain (property of model/content); cost EMA in WALL-CLOCK horizon (tracks context growth, thermals, external GPU load; tau 400ms) with one-off-spike damping (ratio 2.0 -> alpha x0.25).
   - Marginal cost of an extra verify row = measured slope between cheapest/priciest observed depths, not a constant (fallback 7ms).
   - Bidirectional, staleness-directed probes, duty-bounded to 15% of cycles; re-measuring a SHALLOWER rival breaks the depth-2 lock (stale-high t[1] hides depth 1 forever) — a failure mode they measured, not theorized.
   - Hysteresis 1.03 on switching; cost EMA deliberately NOT staleness-age-weighted (measured worse: probe-burst noise entered the decision).
   - Cycle wall-time cap 5000ms so agent-serving queue waits don't poison the cost model.

Challenge fit: the ranked track's editable `draftPolicy` (constant 2 as shipped, adaptive 0-8 allowed) is exactly this controller's slot. Their design already handles what the challenge measures (median paired speedup across 8 prompts under thermal gating — wall-clock-aware depth choice is precisely the right objective). Port shape: Swift port of `CostModelDepthPolicy` with the challenge's depth ceiling 8, cycle time from the harness's own loop.

Owned-loop fit: the same controller drops into our Metal step engine's chain/batched-verify machinery (`verify_tokens` + rollback already merged); `accepts_cycle_ms` matches our instrumented per-cycle timings.

## Qwen 3.8-specific receipts (mtplx/backends/descriptors.py)

- Qwen 3.8 27B shares the Qwen 3.6 trunk geometry (`qwen3_next` backend) but ships its own contract: official thinking sampler (T=1.0, top_p 0.95, top_k 20), reasoning_effort xhigh/medium/low, and a **multi-step-trained MTP head** (deeper-than-1 drafting is trained, not improvised — consistent with the challenge shipping depth-2 default and allowing 8).
- Draft-sampler A/B (2026-08-14, thermally controlled): keeping the official target sampler ON THE DRAFT (T=1.0) beat a cooled draft (T=0.6): 46.05 vs 42.79 tok/s with higher D2/D3 acceptance. For the greedy challenge this exact knob is moot (greedy both sides), but the meta-finding — the head is trained under the target's distribution and drafting under a different one hurts acceptance — cautions against "clever" draft-side distribution tweaks.
- reasoning_effort default: medium vs xhigh completed the same task in 51.5s vs 314.9s — product-level, not challenge-relevant, but a real number for the owned-loop UX tier.
- Paired A/B receipts (docs/perf/qwen27b-gdn/): K3 vs K2 chained timing confirmed +0.87% mean (6/6 wins, spread 0.51%) — their margins at the K2/K3 boundary are thin, which says the depth controller (not a bigger static K) is where the win lives. Their harness discipline (paired arms, census decode windows, receipts as JSON) mirrors our rig conventions.

## What we did NOT extract (bounded)

Acceptance-rate tables per depth and the historical int4 acceptance-collapse data were not located in this pass (the v0.1-era research doc is a pointer shell; benchmarks live as runners + receipts, not tables). The quant-fragility question that matters for the challenge — bf16 head + 4-bit backbone acceptance — is better answered empirically by our own baseline run than by archaeology: the challenge pins exactly that pairing, and `scripts/step_acceptance.py` + `mtp_depth_sweep.py` exist to regenerate curves if we want MTPLX's own numbers on our hardware.

## Actions

1. Challenge entry: port `CostModelDepthPolicy` to the challenge's Swift `draftPolicy` seam (constants as shipped; re-tune PROBE/HYSTERESIS only against ranked-runner-shaped local runs). This plus kernel work on the verify path is the entry's spine.
2. Owned-loop epic: the controller + the "score = expected committed tokens / cycle time" objective goes into the epic's decode-loop design as the depth policy, feeding on our existing per-cycle instrumentation.
3. Watch: MTPLX's sustained no-fan long-context throughput is their own named weak spot — our quiet-tier/energy-knob story remains differentiated.
