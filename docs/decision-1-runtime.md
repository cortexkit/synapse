# Decision #1: Synapse's local inference engine strategy

Status: DRAFT — measured numbers pending (idle-gated matrix on AFT's real corpus).
Decision owner: Ufuk. Reviewers: AFT-Alfonso (driving), SUBC-Alfonso (offered).

## Question

Does Synapse serve local models through in-house engine lanes (thin Rust module
orchestrating proven inference runtimes), wrap an existing server app (Ollama /
LMStudio), or adopt one unifying framework (burn)?

## Method

Empirical, on the machine class we actually target (Apple Silicon first). Two
workloads from real consumer shapes:

- **A: embed-corpus-v1** — 15,271 chunks, byte-exact AFT embed_text (their real
  chunker output; regenerable via aft's export_chunks example).
- **B: microllm-oneshot-v1** — 100 one-shot classification prompts (the AFT/MC
  intent-call shape), greedy, 16 max tokens.

Metrics per lane: cold-load s (first-class: it decides always-on vs load-per-call),
infer wall s, input tok/s, peak RSS, avg/peak CPU+GPU+ANE watts, energy J (macmon
sampling), and **output parity** (mean cosine vs fp32 reference on identical inputs
— speed claims cannot hide wrong outputs) or label validity (workload B).

Measurement protocol: serial lanes, hard idle gate (machine must be <=15% CPU /
<=5% GPU for 6s before any measured run), same corpus/tokenizer accounting across
lanes. Contaminated runs are registered and excluded.

## Candidates and how each was handled

| Candidate | Handling | Outcome |
|-----------|----------|---------|
| raw ort (CPU, AFT policies) | measured (reference lane) | pending matrix |
| mlx-rs (Metal, bf16) | measured | pending matrix |
| llama.cpp (llama-server child, Metal) | measured | pending matrix |
| burn (wgpu/Metal) | measured (embed only) | pending matrix |
| LMStudio (wrap) | measured (workload A) | pending matrix |
| Ollama (wrap) | not installed; LMStudio represents the wrap class; install+measure only if wrap survives on other grounds | dispositioned unless revived |
| vllm | dispositioned by inspection, then CORRECTED and re-checked | core vllm: no in-tree Metal (macOS CPU-only), confirmed. Ecosystem: official out-of-tree vLLM-Metal plugin exists (alpha, MLX-backed, dev-wheel installs, Python 3.12 + source-built vllm core) — real but not shippable to end users; its own Rust frontend still spawns the Python engine. Remains relevant as a REMOTE endpoint users point us at |
| vllm-mlx (waybarrios) | researched | independent MLX server, alpha, not the upstream path; self-reported strong numbers; Python-heavy — disposition for shipping |
| sglang | researched | Python-first, datacenter-oriented, no Windows story; official Apple lane exists and is MLX underneath — disposition as primary, same remote-endpoint relevance as vllm |
| oMLX | researched | strongest wrap-class option found (DMG/Homebrew service, OpenAI+Anthropic APIs, healthy project); still a ~750MB Python/MLX app — candidate optional EXTERNAL backend via the remote-endpoint lane, not a subprocess engine |
| unsloth | dispositioned by inspection | training-first; its own serving delegates to llama-server + MLX — independent confirmation of the hybrid we're evaluating |
| LFM2.5-230M (model, not runtime) | added to workload B | runs on llama-server b9580 out of the box; NOT runnable on the mlx-rs lane without hand-implementing its hybrid architecture — which is the mlx-rs finding restated |
| ANE (CoreML-direct / Core AI) | deferred spike with written rationale | on-Mac LLM decode loses joules/token to GPU despite lower watts; wins are memory + always-on niches; revisit when STT/always-on lands. ORT-CoreML-EP dead end stands. |

## Integration findings (independent of measured speed)

These are architecture facts learned by building the lanes; they hold regardless of
what the matrix measures:

1. **burn**: burn-onnx REFUSED the stock Qwen3-Embedding ONNX export ("Nodes are not
   topologically sorted"); only its pre-validated model list imported cleanly
   (MiniLM). Import is compile-time codegen — the binary is pinned to one ONNX
   snapshot, so serving a model the user downloads at runtime is architecturally
   impossible without shipping a compiler. f16-only on Metal (no bf16). 105s Metal
   shader cold start in our smoke. Disqualifying for a model-SERVING module
   regardless of throughput; remains interesting for baked-in fixed models.
2. **mlx-rs**: full parity achieved (0.9958 vs fp32) but every model family must be
   hand-implemented in Rust (we wrote Qwen3's forward pass; a new architecture =
   new code). 13-minute mlx-sys release build; needs cmake + full Xcode Metal
   toolchain in CI. Viable as a hand-tuned fast lane for CHOSEN embedding models,
   not as the general engine.
3. **llama.cpp (child process)**: runtime model loading (GGUF ecosystem), one
   engine for embed + LLM, correct last-token pooling via --pooling last (parity
   0.9999994), per-request server timings, clean child lifecycle under our
   supervision — matches SUBC's module-spawns-runtime-children model exactly.
4. **wrap (LMStudio)**: smoke parity 0.9996 through /v1/embeddings — plus the
   standing evidence: GUI app lifecycle outside subc supervision, and the live
   incident (6 concurrent AFT processes buried it; no machine-wide admission) is
   the class of failure wrapping cannot fix from outside.
5. **ort CPU**: the proven universal floor; AFT's shipped policies reproduced
   exactly (Level3, ceil(cores/2) threads, 4M attention-unit batching).
6. **vllm family (measured empirically after the by-inspection disposition)**:
   vllm-CPU 0.24.0 serves MiniLM on macOS but needed memory-reservation flags,
   single-process mode, and a truncated corpus (MiniLM 256-token limit surfaced
   as opaque 400s/timeouts); smoke 7.8k tok/s — 3.7x SLOWER than our bounded ort
   on the same model, with a 1.3 GB venv and 13s cold start. vllm-metal 0.3.0
   installs (with workspace + Xcode workarounds) but CANNOT serve MiniLM at all:
   "Model type bert not supported" (its pooling support today is
   Qwen3-Embedding/Reranker only). vllm-mlx serves MiniLM only as an auxiliary
   model hitched to a supported primary LLM (BERT rejected as primary; also
   needed a transformers version pin to even start). Its raw smoke number
   (54.8k tok/s, contended machine, unvalidated quality) shows MLX kernel
   potential, but three-of-three modes required version pins or config surgery
   to serve a 5-year-old, 22M-parameter industry-standard embedder — the
   packaging-fragility disposition is now empirical fact, not inspection.

## The convergence finding

Every 2026 Apple-Silicon serving stack examined (vLLM-Metal, vllm-mlx, sglang's
Apple lane, oMLX, unsloth Studio, LMStudio's MLX engine) delegates compute to MLX.
The engine layer on Apple hardware is decided — MLX or llama.cpp-Metal; everything
above it is packaging. Decision #1 therefore reduces to: whose packaging — theirs
(Python stacks we cannot ship to end users) or ours (Rust module with
mlx-rs/llama-server lanes, both parity-proven in this bench).

## Results

Final set measured 2026-07-05 (single sequential run, bench/results/night-20260705/)
on M5 Max (18 cores, 128 GB), macOS 26.5.1, idle-gated (CPU <= 15%, GPU <= 5%
preflight; mid-run foreign-CPU attribution, ALL 16 runs `contaminated=false`,
foreign CPU 1-9%). Corpus: AFT's real chunk export, 15,271 chunks / 1,569,409
tokens Qwen3-tokenized, 2,349,040 MiniLM-tokenized (byte-exact embed_text).
Parity = mean cosine vs ort-cpu fp32 reference on identical inputs. Performance
(tok/s, cold load) is the primary axis; energy is secondary (macmon watts are
machine totals under the idle gate — comparatively valid, not per-process
attribution). Sub-2s runs get no meaningful power sample (250ms sampler).

### Workload A: batch embedding, Qwen3-Embedding-0.6B (quality-upgrade model)

| lane | precision | tok/s | parity | cold load | avg W | energy |
|---|---|---|---|---|---|---|
| ort-cpu (9 threads) | fp32 | 1,222 | reference | 0.9s | 30.3 | 38.9 kJ |
| mlx-rs Metal | bf16 | 8,521 | 0.99600 | 0.2s | 38.5 | 7.2 kJ |
| mlx-embeddings Metal (Python) | 4-bit DWQ | 23,421 | 0.9671* | 1.5s | 36.6 | 2.8 kJ |
| llama-server Metal | f16 gguf | 7,685 | 1.00000 | 0.8s | 32.6 | 6.7 kJ |
| wrap: LMStudio | (server, gguf) | 4,343 | 0.99973 | n/a | 38.9 | 14.1 kJ |

*DWQ parity is quality-disqualifying despite the throughput — see the
rank-stability finding below.

- The Metal lanes are 6.3-7x the CPU floor at equal quality (bf16/f16), and the
  DWQ quant reproduces AFT's June spike number (23.4k vs their 22.8k tok/s).
- llama-server f16 is numerically indistinguishable from the fp32 reference
  (1.00000 over all 15,271 chunks); mlx bf16's 0.9960 is a REAL vector-space
  difference that must surface as a distinct model fingerprint (MC contract).
- Wrapping LMStudio costs 1.8x throughput vs supervising llama-server directly
  ON THE SAME ENGINE AND QUANT CLASS (4.3k vs 7.7k): HTTP + app overhead, no
  control, and the machine-wide-admission failure class on top.

### Quantization is not free: the DWQ rank-stability finding

mlx-community's 4-bit DWQ quant of the same model doubles throughput
(26.3k tok/s smoke vs 9.1k bf16) — and quietly rewrites retrieval results.
Against the fp32 reference on 400 real code chunks (k=10 neighbor overlap):

| metric | value |
|---|---|
| mean cosine | 0.9664 |
| mean top-10 overlap | 0.836 |
| p50 / p10 overlap | 0.90 / 0.70 |
| worst-decile mean | 0.63 |
| worst query | 0.40 |

Full-corpus confirmation (15,271 chunks, 306 rank queries, k=10): mean cosine
0.9671, mean overlap 0.829, p50 0.90, p10 0.70, worst-decile mean 0.58, min
0.40. The median query loses 1 of its 10 nearest neighbors; the worst decile
loses ~4. A 0.967 mean cosine "looks fine" while a visible minority of searches
degrades — the canonical proof that cosine-only parity gates are insufficient.
Consequence: DWQ is an opt-in speed tier with its own fingerprint (full reembed
to adopt), never a silent default. bf16/f16 are the quality-safe Apple lanes.

### Workload A floor: all-MiniLM-L6-v2 across every engine (today's production model)

| lane | mode | tok/s | parity | cold load | avg W | energy |
|---|---|---|---|---|---|---|
| llama-server Metal | f16 gguf | 92,777 | 1.00000 | 0.28s | 22.0 | 0.57 kJ |
| burn wgpu/Metal | f32 (compiled-in) | 64,865 | 1.00000 | 7.6s | 41.9 | 1.85 kJ |
| MLX Metal (mlx-embeddings) | bf16 | 38,039 | n/a | 1.1s | 14.7 | 0.88 kJ |
| ort Rust CPU (9 threads) | fp32 | 29,643 | reference | 0.08s | 26.9 | 2.14 kJ |
| ts onnxruntime-node CPU | fp32 | 28,909 | n/a | 0.16s | 31.4 | 2.39 kJ |
| ts transformers.js CPU | q8 (MC ships this) | 26,129 | n/a | 2.1s | 29.6 | 2.55 kJ |
| ts transformers.js CPU | fp32 | 24,365 | n/a | 3.0s | 32.1 | 2.98 kJ |
| llama-server CPU (-ngl 0) | f16 gguf | 11,836 | 1.00000 | 0.26s | 38.9 | 7.76 kJ |
| vllm CPU (smoke, wrap) | fp32 | 7,793 | n/a | 13s server | n/a | n/a |

Findings:

- **Correction of our own contended-machine smoke**: with the machine quiet, GPU
  wins MiniLM too. The earlier "tiny encoders lose on GPU" readout (11.6k) was
  contention noise; clean MLX GPU is 38k and llama-Metal 92.8k. The quadrant
  rule that survives: GPU wins BIG on 600M-class; on 150M-class GPU still wins
  but CPU is within 3x, so CPU-only machines remain first-class for MiniLM.
- **llama-server Metal at 92.8k tok/s with parity 1.00000 and 22W** is the
  single best MiniLM result on every axis at once: 3.1x the CPU floor, 3.8x less
  energy, 0.28s cold load. The whole 15,271-chunk corpus embedded in 26 seconds.
- **Today's TS production path is not slow** — ort-node hits 97.5% of Rust ort;
  transformers.js costs ~12-18% over raw ort-node. The upgrade Synapse offers
  MiniLM-class users is llama-Metal's 3.1x + CPU freed, not "Rust instead of JS".
- **burn's 64.9k** (2nd place, parity 1.0) comes with the architectural
  disqualifiers unchanged: model compiled into the binary at build time (no
  runtime loading), 7.6s Metal cold start every process start.
- **q8 beat fp32 in transformers.js on the full corpus** (26.1k vs 24.4k) —
  the smoke had it reversed; quantized wins once the corpus amortizes warmup.

### Saturation audit (2026-07-06): three table rows above are undersaturated

Post-run GPU-utilization auditing (utilization capture now lands in every
measure.json) showed the night-20260705 configs left GPU on the table in three
lanes. Root cause everywhere: mixed-length padded batches — batches pad to
their longest member, so short chunks burn GPU on padding tokens. Fix: sort by
tokenized length (output is keyed by id; order is irrelevant) + bigger batch
budgets. Contended-machine PROBE numbers (relative evidence only, not table
grade):

| lane | night config | GPU util | fixed config | probe result |
|---|---|---|---|---|
| burn MiniLM | 4M attn units, unsorted | 68% | 16M units, sorted | 82k → 137.5k tok/s (parity 1.0 unchanged) |
| mlx-python MiniLM | 8k budget, unsorted | 77% | 32k/256, sorted | 178k → 355k tok/s probe* |
| mlx-rs Qwen3 | unsorted | ~85% | sorted | 8.5k → 13.4k tok/s, GPU 98% |
| ort CPU MiniLM | 9 threads | 43% CPU | (by design) | 18 threads: +20% — AFT policy row stays the default, saturated column reported separately |

*probe subset skews long (158 vs 142 avg tokens); treat direction, not magnitude.

Consequences pending the rev-2 clean rerun (script updated, run pending):
- The "llama-server wins GPU embedding on every axis" conclusion does NOT
  survive saturation fixes; MLX leads GPU embedding in both model classes in
  the probes. The Recommendation section's engine assignment for embed will be
  re-decided on rev-2 numbers.
- mlx-rs bf16 parity moved 0.99600 → 0.99626 under sorted batching: batch
  shape perturbs bf16 numerics. One more reason fingerprints must capture
  runtime config, not just model+quant.
- llama.cpp caveat (petejm/apple-silicon-embed-bench, community-corroborated):
  current llama.cpp macOS 26.5 builds report `has tensor = false` — Apple M5
  tensor accelerators are DISABLED in llama.cpp while MLX uses them. Part of
  the MLX-vs-llama gap is API state, closable by future llama.cpp releases:
  engine assignments must stay pluggable, not locked to today's winner.

### Model matrix (provisional — smoke-grade, full-corpus rev-2 run pending)

Model axis added 2026-07-06 (D-009: best options per hardware class + a
speed-vs-energy knob; bench spread supports the knob — same workload spans
14.7W to 62W by engine/batch choice). Model classes: static 16M (Model2Vec
potion-code-16M) / 22M MiniLM / 150M ModernBERT-class / 600M Qwen3. Cross-MODEL
quality comes from public data (MTEB/CoIR) + our own retrieval eval
(bench/eval-coir, cosqa first, semble's 1,250-query code-search dataset as
second column); intra-model QUANT quality from our parity+rank-overlap tooling.

Qwen3-Embedding-0.6B quant ladder (400-chunk smoke vs fp32 reference, k=10;
first public quant-quality data for these embedders — none published anywhere):

| quant | mean cosine | rank overlap | worst decile |
|---|---|---|---|
| Q8_0 (GGUF) | pending full run | | |
| Q6_K (GGUF) | 0.9963 | 0.965 | 0.88 |
| Q4_K_M (GGUF) | 0.9750 | 0.869 | 0.67 |
| 4bit DWQ (MLX) | 0.9664 | 0.836 | 0.63 |
| 8bit (MLX) | 0.8899 | 0.837 | 0.10 (min 0.0) |

- Q6_K is the quant sweet spot so far: 0.965 rank stability at ~60% of f16 size.
- The MLX 8bit row is ANOMALOUS (8-bit scoring below 4-bit is not how
  quantization error works): suspected broken mlx-community upload or an
  mlx-embeddings handling bug for that format. Under investigation — do not
  cite as a real measurement.
- ModernBERT-class candidates verified runnable cross-engine (200-chunk smokes,
  ort fp32 vs llama GGUF): gte-modernbert-base 0.99997 (CLS pooling),
  nomic-modernbert-embed 0.99931 (mean + search prefixes), jina-v5-nano
  0.99999 (EuroBERT, last pooling; NOTE: produces 768-dim vectors on both
  engines, not the 512 the repo docs suggest — consistent cross-engine, needs
  pooling-config verification before quality claims).
- Static class (Model2Vec): semble's own ablations put raw potion-code-16M at
  NDCG 0.650 vs 0.765 for a 137M transformer — viable ONLY behind hybrid
  retrieval (BM25 carries symbol queries), i.e. AFT-shaped consumers on very
  constrained machines; disqualified for MC's pure vector search. Lane built;
  speed/footprint numbers pending.

### Workload C (rerank) and retrieval-quality eval: in flight

Qwen3-Reranker-0.6B via llama-server /v1/rerank (lane in build). Latency per
20-doc rerank request decides interactive-path vs background-only for AFT.
Quality eval harness (bench/eval-coir) scores THE ARTIFACTS WE SHIP (lane
vectors, per model x quant x engine) on public retrieval tasks — quant damage
gets priced in nDCG, not just neighbor churn. Pipeline-vs-pipeline vs semble
agreed with AFT (four columns: raw dense / naive hybrid / AFT full stack /
semble published — if AFT signals don't clear naive hybrid on their
distribution, that's the finding).

### Workload B: micro-LLM one-shot classification, 100 prompts

| lane | model | combined tok/s | decode tok/s | valid labels | cold load |
|---|---|---|---|---|---|
| llama-server Metal | Qwen3-0.6B q8_0 | 12,110 | 558 | 97/100 | 0.55s |
| mlx-rs Metal | Qwen3-0.6B bf16 | 7,265 | 45* | 96/100 | 0.32s |
| llama-server Metal | LFM2.5-230M q8_0 | 30,278 | 1,171 | 81/100 | 0.27s |

*mlx decode rate is unbatched greedy decoding in our hand-rolled lane — an
implementation artifact (no speculative/batched decode), not an MLX ceiling.

- LFM2.5-230M is 2.3x faster and materially less accurate on this task (81%
  valid labels vs 97%): attractive tok/s, unusable accuracy for one-shot
  classification without prompt work. Model choice stays a per-task decision.
- Qwen3-0.6B q8_0 on llama-server is the current quality/speed sweet spot.

## Recommendation

Direction set by Ufuk (2026-07-04): **borrow the kernels, own the stack.** Native
engine layers under a fully-owned Rust serving stack; no adopted Python packaging.
Rationale: every capable Apple-Silicon stack already concedes compute to native
engines (see convergence finding); the packaging layer is the part Synapse must own
anyway (subc supervision, machine-wide admission, credential integration, model
lifecycle), and Python packaging is the part that fails our end-user constraints
(install burden, interpreter footprint, cold start, energy).

> **REVISION PENDING (2026-07-06):** the saturation audit above invalidates
> the embed-workload engine assignments below — MLX leads GPU embedding in
> post-fix probes (both model classes). The assignments stand as written for
> micro-LLM and the CPU floor; the embed rows will be re-decided from the
> rev-2 clean rerun before D-005 lock. Kept unedited meanwhile so the review
> trail stays honest.

The measured matrix (above) answers the remaining question — which engine
carries which workload. Engine assignments from the clean full-corpus run:

**llama.cpp (supervised llama-server child) is the primary local engine on
Apple Silicon — for both workloads and both model classes.**

- MiniLM-class embed: 92.8k tok/s, parity 1.00000, 0.28s cold, 22W — best on
  every axis simultaneously.
- Qwen3-class embed: 7.7k tok/s at parity 1.00000. mlx-rs bf16 is 11% faster
  (8.5k) but at 0.996 parity (distinct fingerprint) and it costs a hand-written
  Rust forward pass per model family; llama.cpp gets new architectures free
  from the GGUF ecosystem (LFM2.5's hybrid arch ran day-one unmodified).
- Micro-LLM: 12.1k tok/s combined, 97% label validity, full server timings.

**ort (in-process Rust) is the universal CPU floor** — every platform, no GPU
required: 29.6k MiniLM / 1.2k Qwen3. Proven AFT policies reproduce exactly.
On CPU-only machines it beats llama-cpu by 2.5x on MiniLM.

**mlx-rs**: not an engine for v1. The 11% embed edge on Qwen3 doesn't pay for
per-architecture Rust implementations plus the Metal-toolchain CI burden; its
unbatched decode is 12x slower than llama-server's. Revisit only if a workload
appears where MLX kernels are unbeatable AND the model family is stable enough
to hand-implement once (the mlx-embeddings DWQ lane shows 23.4k is reachable —
but that's a quant-quality tradeoff, not an mlx-rs advantage).

**burn**: dispositioned (compile-time model binding, 7.6s cold start), evidence
recorded at 64.9k tok/s / parity 1.0 for the baked-in-model niche.

**Remote/wrap endpoints** (user-pointed LMStudio/Ollama/vllm/sglang/oMLX):
supported as explicit remote backends behind the same surface, never the
default local engine — wrapping LMStudio measured 1.8x slower than supervising
the same engine class directly, with the machine-wide-admission failure class
unfixable from outside.

**Model defaults per workload** (fingerprint-stable choices):
- embed floor: MiniLM fp32 (ort, CPU) — today's spaces remain valid.
- embed quality tier: Qwen3-Embedding-0.6B f16 GGUF (llama-Metal) on Apple;
  ort fp32 elsewhere. DWQ quant: opt-in only (rank-stability finding).
- micro-LLM: Qwen3-0.6B q8_0 (llama-Metal). LFM2.5-230M rejected at 81% label
  validity despite 2.5x throughput.

The lock decision for D-005 is therefore: Rust module owning admission, model
lifecycle, and the subc surface; llama-server as supervised child engine; ort
in-process as floor; remote endpoints as a peer backend lane. No Python in the
shipped path.
