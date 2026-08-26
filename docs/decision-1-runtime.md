# Decision #1: Synapse's local inference engine strategy

Status: **RATIFIED (D-005, 2026-07-08)** — the documented engine strategy was
validated by the implemented Lane 1 v1 surface. Later amendments added a
two-lane engine program (adopt plus owned runtime graduating by measurement),
best-per-hardware support including MLX/ANE on Macs, and the implemented
fingerprint contract (`docs/wire-contract-v1.md`).
Decision ownership and review records are retained outside the public tree.

Evidence base: 60+ idle-gated full-corpus lane runs across 4 hardware classes
(M5 Max, M1 Max, RTX 3060, Ryzen Z1 Extreme), retrieval-quality evals on CoIR
(cosqa + CSN-python), reranker cross-validation against a Python reference
implementation, and source-grounded engine research. Raw artifacts under
bench/results/ (local, gitignored): night-20260705, m1-night-20260707,
ally-20260708, cuda3060-20260708.

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
| raw ort (CPU, AFT policies) | measured (reference lane) | Apple CPU floor + ONNX/DirectML lane (loses to llama-CPU ~2x on x86) |
| mlx-rs (Metal, bf16) | measured | Apple embed fast lane confirmed (M1: 2.6x llama-Metal on MiniLM); shipping form (mlx-rs vs sidecar) = the one open sub-decision |
| llama.cpp (llama-server child) | measured on Metal, CUDA, Vulkan, CPU | general workhorse on every platform: micro-LLM, non-Apple GPU embed, rerank, x86 CPU floor |
| burn (wgpu/Metal) | measured (embed only) | dispositioned: compile-time model binding, 7.6s Metal cold start (64.9k tok/s recorded for baked-in niche) |
| LMStudio (wrap) | measured (workload A) | wrap class dispositioned: 1.8x slower than supervising the same engine directly; admission unfixable from outside |
| Ollama (wrap) | not installed; LMStudio represents the wrap class; install+measure only if wrap survives on other grounds | dispositioned unless revived |
| vllm | dispositioned by inspection, then CORRECTED and re-checked | core vllm: no in-tree Metal (macOS CPU-only), confirmed. Ecosystem: official out-of-tree vLLM-Metal plugin exists (alpha, MLX-backed, dev-wheel installs, Python 3.12 + source-built vllm core) — real but not shippable to end users; its own Rust frontend still spawns the Python engine. Remains relevant as a REMOTE endpoint users point us at |
| vllm-mlx (waybarrios) | researched | independent MLX server, alpha, not the upstream path; self-reported strong numbers; Python-heavy — disposition for shipping |
| sglang | researched | Python-first, datacenter-oriented, no Windows story; official Apple lane exists and is MLX underneath — disposition as primary, same remote-endpoint relevance as vllm |
| oMLX | researched | strongest wrap-class option found (DMG/Homebrew service, OpenAI+Anthropic APIs, healthy project); still a ~750MB Python/MLX app — candidate optional EXTERNAL backend via the remote-endpoint lane, not a subprocess engine |
| unsloth | dispositioned by inspection | training-first; its own serving delegates to llama-server + MLX — independent confirmation of the hybrid we're evaluating |
| LFM2.5-230M (model, not runtime) | added to workload B | runs on llama-server b9580 out of the box; NOT runnable on the mlx-rs lane without hand-implementing its hybrid architecture — which is the mlx-rs finding restated |
| ANE (CoreML-direct / Core AI) | surveyed 2026-07-08 (see ANE section) | REVERSED for encoders: fixed-bucket CoreML conversion runs BERT→Qwen3-0.6B-class encoders ~99.8% on ANE at ~2W. The old dead-end verdict was the ORT-CoreML-EP path only. Spike specified; quiet-tier engine candidate. |

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

Consequences (M5 rev-2 clean rerun demoted to opportunistic — the M1 clean
run and cross-platform rows already settle every decision; M5 embed rows keep
a probe-grade caveat until an idle window allows the rerun):
- The "llama-server wins GPU embedding on every axis" conclusion does NOT
  survive saturation fixes; MLX leads GPU embedding in both model classes
  (confirmed CLEAN on M1: 132.6k vs 51.8k MiniLM). The engine-assignment
  section reflects this.
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

Qwen3-Embedding-0.6B quant ladder — quality (rank metrics from 400-chunk smoke
vs fp32 reference, k=10; parity from the M1 full-corpus run, 15,271 chunks;
first public quant-quality data for these embedders — none published anywhere):

| quant | full-corpus cosine | rank overlap | worst decile | cosqa NDCG@10 |
|---|---|---|---|---|
| f16 (GGUF) | 1.00000 | — | — | 0.3487 |
| Q8_0 (GGUF) | 0.99973 | — | — | — |
| Q6_K (GGUF) | 0.99684 | 0.965 | 0.88 | 0.3486 |
| Q4_K_M (GGUF) | 0.97761 | 0.869 | 0.67 | 0.339 |
| 4bit DWQ (MLX) | 0.9671 | 0.829 | 0.58 | — |
| 8bit (MLX) | 0.8899* | 0.837 | 0.10 (min 0.0) | — |

The retrieval eval (cosqa) prices the ladder in NDCG: Q6_K is quality-FREE
(0.3486 vs f16's 0.3487) at ~60% of f16 size; Q4_K_M costs ~3% NDCG. The
rank-overlap and NDCG orderings agree — our cheap parity tooling predicts
expensive retrieval-quality outcomes.

- Q6_K is the quant sweet spot: 0.965 rank stability, NDCG-identical to f16.
- *The MLX 8bit row is ANOMALOUS (8-bit scoring below 4-bit is not how
  quantization error works): suspected broken mlx-community upload or an
  mlx-embeddings handling bug for that format. Reproduced on the M1 full
  corpus — still under investigation, do not cite as a real measurement.
- ModernBERT-class verified cross-engine on the M1 FULL corpus (ort fp32 vs
  llama GGUF): gte-modernbert-base 1.00000 (CLS pooling),
  nomic-modernbert-embed 0.99938 q8 (mean + search prefixes), jina-v5-nano
  0.99999 (EuroBERT, last pooling, 768-dim on both engines).

### Retrieval-quality screen (cosqa, NDCG@10; bench/eval-coir)

| model | params | NDCG@10 | R@10 |
|---|---|---|---|
| gte-modernbert-base f16 | 149M | **0.360** | 0.630 |
| Qwen3-Embedding-0.6B f16 | 595M | 0.3487 | 0.602 |
| Qwen3-Embedding-0.6B Q6_K | 595M | 0.3486 | 0.606 |
| Qwen3-Embedding-0.6B Q4_K_M | 595M | 0.339 | — |
| all-MiniLM-L6-v2 fp32 | 22M | 0.284 | 0.498 |
| potion-code-16M (static) | 16M | 0.185 | — |
| jina-v5-nano | 199M | 0.142* | — |

*jina number is a broken-config artifact (task-adapter/pooling suspected), not
a model verdict — under investigation.

- **gte-modernbert-base beats Qwen3-0.6B at a quarter of the size** — the
  150M ModernBERT class is the quality/footprint sweet spot and the strongest
  default-model candidate (also CPU-viable: see cross-platform table).
- cosqa is a SMOKE screen only (single-qrel NL queries; see rerank section for
  how it misled) — absolute values are noisy, ordering is directional. The
  semble 1,250-query code-search dataset is the planned second column.
- Static class (Model2Vec potion-code-16M): measured NDCG 0.185 vs MiniLM's
  0.284 — 35% quality drop confirms semble's own ablations. Quality bar ruling
  (Ufuk): standalone vector quality is the bar; hybrid-fusion rescue does not
  qualify a default. Potion = explicit constrained-machine opt-in only. Speed
  measured: ~2.7M tok/s CPU, 0.4s cold load, ~30MB — the extreme-frugality
  point if ever needed.

### Workload C: reranking (verified end-to-end)

Measured via llama-server b9580 /v1/rerank, cross-validated against a Python
transformers reference implementation, and quality-scored on CoIR (cosqa +
CSN-python 2,000 queries / 280k-doc corpus).

| reranker | params | cosqa delta | CSN-py delta (MiniLM / gte front-end) | p50 latency (50 docs) |
|---|---|---|---|---|
| gte-reranker-modernbert-base f16 | 149M | +0.051 | **+0.112 / +0.063** | 510-666ms (real code); 283ms (cosqa snippets) |
| bge-reranker-v2-m3 | 568M | -0.028 | — | 567ms |
| Qwen3-Reranker-0.6B | 595M | -0.224 | — | 304ms |

Findings, in verification order:

1. **The engine is faithful for ModernBERT-class rerankers**: llama.cpp scores
   match the Python reference at Pearson 0.999992 over 2,500 pairs (tie-aware
   top-1 agreement 1.0). Engine exonerated.
2. **The Qwen3-Reranker path in llama.cpp b9580 is broken** (template/token
   handling: known-relevant docs score ~e-11 vs distractors ~e-06). Parked;
   watch llama.cpp PR #20009. Not a model verdict.
3. **cosqa initially produced a false conclusion** ("reranking hurts strong
   dense retrievers") that the reference implementation *reproduced* — it is a
   dataset property (single-qrel NL queries penalize rerankers for surfacing
   other relevant docs), not an engine or model property. On code-shaped CSN-python
   the reranker helps BOTH front-ends monotonically: MiniLM 0.781→0.893,
   gte 0.917→0.980. cosqa is hereby demoted to smoke-only.
4. **gte-reranker-modernbert-base (149M, Apache-2.0, official ONNX) is the
   working default reranker.** Caveat carried: gte's absolute CSN numbers are
   likely training-contaminated — trust the deltas, not the absolutes.
5. Latency scales with doc length (real code ~2x cosqa snippets); top-20
   requests land ~250-350ms — rerank-on-by-default is justified for search
   paths tolerating ~0.5s, background-only otherwise.

Pending: ms-marco ORT floor (llama.cpp blocked on token_type_ids, PR #21729),
Qwen3-Reranker-4B quality ceiling, semble dataset column, joint 4-column
pipeline run with AFT (raw dense / naive hybrid / AFT stack / semble published).

### Cross-platform campaign (2026-07-07/08): four hardware classes, one engine stack

Same corpus, same llama.cpp build (b9580), same lane code everywhere.
M1 Max = clean idle-gated macOS row (25 lanes, all contaminated=false).
RTX 3060 (rented, $0.49 total) = consumer-CUDA row with per-GPU watts via
nvidia-smi. ROG Ally X (Z1 Extreme, Zen4 + RDNA3 iGPU) = low-end Windows row.
CPU rows on the 3060 box carry a shared-host caveat; its GPU rows are exclusive.

**MiniLM-class embed (tok/s):**

| path | M5 Max | M1 Max | RTX 3060 | Ally Z1E |
|---|---|---|---|---|
| best GPU | 355k probe* (MLX) | 132.6k (MLX) | 57.4k (CUDA, 76W) | 35.9k (Vulkan) |
| llama GPU | 92.8k | 51.8k | 57.4k | 35.9k |
| best CPU | 29.6k (ort) | 11.5k (ort) | 7.2k (llama) | 17.2k (llama) |

**gte-modernbert-base (150M class):** M1 11.5k Metal / 1.1k ort-CPU; 3060
18.7k CUDA (94W) / 0.9k CPU; Ally 6.7k Vulkan / 2.1k CPU. GPU-accelerated
everywhere including the handheld iGPU; CPU-viable for indexing smaller corpora.

**Qwen3-Embedding-0.6B:** M5 13.4k (MLX sorted) → M1 7.7k (MLX) → 3060 4.9k
(CUDA, 98W) → Ally 1.6k (Vulkan q6k) → ~250-275 pure CPU (any platform).

**Micro-LLM (Qwen3-0.6B q8, 100 one-shots):** label validity 97-98/100 on every
platform. Combined tok/s: M5 12.1k → M1 5.2k → 3060 4.0k → Ally-Vulkan 2.0k →
Ally-CPU 509 → 3060-CPU 232.

Campaign findings that reshape the recommendation:

1. **The 600M embed tier requires an accelerator.** Pure-CPU Qwen3-0.6B is
   ~250 tok/s everywhere (a 15k-chunk corpus = ~100 minutes). CPU-only machines
   cap at the 150M class — which the quality screen independently crowned.
2. **"ort is the universal CPU floor" is Apple-only.** On Zen4/x86, llama-CPU
   beats ort by ~2x (Ally: 17.2k vs 8.6k; 3060 box: 7.2k vs 4.0k). The floor
   engine is per-platform: ort on Apple, llama.cpp on x86.
3. **Vulkan on RDNA3 iGPUs is a real acceleration tier**: 2-4x over CPU on
   every workload at parity 1.0 on a handheld gaming PC. llama.cpp's single
   GGUF + multi-backend story (Metal/CUDA/Vulkan/CPU) covered every platform
   we touched with zero lane-code changes.
4. **Quant speed behavior is architecture-dependent**: q6k fastest on the
   bandwidth-bound iGPU (beats f16 by 19%), all quants within 5% on the
   compute-bound 3060, DWQ pays only on M5 (2.5x) and is 6x SLOWER than bf16
   on M1. No global quant default is correct.
5. **MLX-vs-llama.cpp flips by Apple Si generation** (M1: MLX 2.6x llama on
   MiniLM; M5 probes: ~3.8x with tensor-accelerator asymmetry) — but MLX's
   Qwen3 lead over llama-Metal on M1 is modest (7.7k vs 3.4k... 2.3x). Engine
   choice per hardware requires measurement, not tables.
6. **llama.cpp CUDA builds offload batch matmuls even at -ngl 0** (67W GPU
   draw with zero layers "offloaded") — pure-CPU measurement on CUDA builds
   requires CUDA_VISIBLE_DEVICES="". Recorded as a telemetry-hygiene trap.

### ANE survey (2026-07-08): the quiet tier exists

Source-grounded survey (ane-book, CoreML-LLM PR #169,
smpanaro/ModernBERT-AppleNeuralEngine, Apple docs) REVERSES the inherited
"CoreML dead end for embedders" verdict for the direct-CoreML path (the dead
end was ORT-CoreML-EP specifically):

- Encoders convert and run on ANE today: fixed-shape token buckets
  (128/256/512) fp16, Linear→Conv2d(1x1) 4D layout, CPU_AND_NE; verified via
  MLComputePlan placement + powermetrics ANE counters.
- Proof points: Qwen3-0.6B-class encoder at ~99.8% ANE residency (100.6ms/doc
  @512, M4 Max); ModernBERT-on-ANE at ~2.1W vs our GPU lanes' 14-62W.
- Traps mapped: naive int8 collapses on these encoders (rotation/outlier
  mitigation required); 8192-token buckets pass static placement but fail at
  runtime (chunk-and-pool for long context); dynamic shapes fall off ANE
  (bucket + GPU catch-all).
- macOS 27 "Golden Gate" Core AI = compute-unit preference + tooling, no
  direct ANE API — do not wait for it.
- Quiet-tier ladder: MiniLM → ModernBERT-class → Qwen3-0.6B, all ~2W-class,
  GPU catch-all for shapes that fall off ANE. Integration: Swift sidecar
  serving .mlmodelc first (ane-book runtime pattern), objc2-core-ml native
  path later. Spike specified (MiniLM 256/512 buckets; gates: cosine parity,
  MLComputePlan NE placement, GPU-idle powermetrics; measure docs/s + J/doc).

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

### Engine assignments (final, campaign-informed)

**llama.cpp (supervised llama-server child) is the general workhorse on every
platform.** One GGUF artifact, four backends measured (Metal/CUDA/Vulkan/CPU),
runtime model loading, new architectures free from the GGUF ecosystem, parity
1.00000 at f16, and the only engine that covered ALL our hardware rows with
zero code changes. Carries: micro-LLM everywhere; embedding on every
non-Apple GPU (CUDA/Vulkan); reranking (/v1/rerank, reference-faithful for
ModernBERT-class); CPU floor on x86 (beats ort ~2x on Zen4).

**MLX is the Apple-Silicon embedding fast lane** (132.6k vs 51.8k MiniLM on
M1; probes say the gap widens on M5 where llama.cpp's tensor-accelerator
support lags). HOW we ship MLX remains the one open engine question:
mlx-rs (hand-written forward passes, bf16 = distinct fingerprint at 0.996
parity) vs a slim mlx-embeddings sidecar. The M5-vs-M1 flip and llama.cpp's
closable tensor-accelerator gap both argue: keep the assignment PLUGGABLE and
let the onboarding probe decide per machine.

**ort (in-process Rust) is the Apple CPU floor and the ONNX-ecosystem lane**
(29.6k MiniLM M5 / 11.5k M1 — 1.6-2x llama-CPU on Apple; loses the same ratio
on x86). Also the DirectML door on Windows (lane already ported, load-dynamic).

**ANE (CoreML sidecar) is the quiet-tier engine candidate** — ~2W encoder
serving with the ladder MiniLM → ModernBERT → Qwen3-0.6B. Spike next; not a
v1 lock item.

**burn**: dispositioned (compile-time model binding, 7.6s Metal cold start);
evidence recorded for the baked-in-model niche.

**Remote/wrap endpoints** (user-pointed LMStudio/Ollama/vllm/sglang/oMLX):
supported as explicit remote backends behind the same surface, never the
default local engine — wrapping LMStudio measured 1.8x slower than supervising
the same engine class directly, with the machine-wide-admission failure class
unfixable from outside.

### Model defaults per hardware class (D-009 input)

Quality screen + cross-platform speed jointly produce the ladder. All choices
fingerprint-stable; quant policy: Q8_0/Q6_K quality-safe, 4-bit opt-in only.

| hardware class | embed default | embed quality tier | micro-LLM | rerank |
|---|---|---|---|---|
| Apple Silicon (any) | MiniLM (compat floor) | gte-modernbert f16 → Qwen3-0.6B f16 | Qwen3-0.6B q8 Metal | gte-reranker-modernbert |
| Windows/Linux + dGPU | MiniLM | gte-modernbert f16 → Qwen3-0.6B q6k/q8 (CUDA/Vulkan) | Qwen3-0.6B q8 | gte-reranker-modernbert |
| iGPU-class (handhelds, thin laptops) | MiniLM Vulkan | gte-modernbert f16 Vulkan; Qwen3 q6k for patient indexing | Qwen3-0.6B q8 Vulkan | gte-reranker (background) |
| CPU-only | MiniLM (ort on Apple, llama on x86) | gte-modernbert (indexing-speed caveat) | Qwen3-0.6B q8 (slow: ~250 tok/s) | background-only |
| quiet tier (ANE, post-spike) | MiniLM ANE | ModernBERT-class ANE → Qwen3-0.6B ANE | GPU or declined | MiniLM cross-encoder ANE |

Key model verdicts feeding the table: gte-modernbert-base (149M) beats
Qwen3-0.6B on code retrieval at a quarter of the size and stays GPU-viable on
a handheld; the 600M tier needs an accelerator (pure-CPU ~250 tok/s);
LFM2.5-230M rejected for one-shots (81% label validity); potion static =
explicit constrained-machine opt-in only (quality bar ruling); DWQ = opt-in
speed tier on M5-class only (rank-stability + it's slower on M1).

### The onboarding bench (end-game requirement, Ufuk 2026-07-08)

The campaign's strongest meta-finding: optimal config is NOT predictable from
specs. MLX-vs-llama flips by Si generation; DWQ is 2.5x faster on M5 and 6x
slower on M1; ort-vs-llama CPU flips by ISA; quant speed inverts between
bandwidth-bound and compute-bound GPUs; ANE residency depends on op placement
only a live probe can verify; Core AI showed the same export recipe regressing
2.2x across an OS update. Therefore Synapse ships a first-run probe (~1-2 min):
detect hardware → micro-bench available engines on a built-in corpus → verify
quality (parity vs shipped reference vectors; MLComputePlan for ANE) → present
what the machine supports with measured numbers → map the speed-vs-energy knob
(performance|balanced|quiet) to MEASURED per-machine configs. Re-probe on
OS/driver updates. Lab campaign data = the probe's priors + shipped reference
vectors. (Independent validation: MTPLX's auto-tune arrived at the same
design for the same reason — measure real configs on the user's machine
against a kept baseline, save only verified wins, honest verdict UI.)

### Fingerprint and equivalence contract (pre-agreed by all three consumers, 2026-07-08)

Converged through documented cross-consumer review:

**Strict identity, declared equivalence, probe-enforced.** Fingerprints are
strict per (model, quant, engine-lane, runtime-config). Interchangeability is
an explicit, revocable ALIAS TABLE layered on top — never baked into the
identity string. Engine is recorded as provenance metadata beside every
response.

- **The gate**: equivalence is certified against CANONICAL SHIPPED REFERENCE
  VECTORS (model-canonical), never pairwise engine-vs-engine (transitivity
  drift with mixed-provenance indexes). The bar is parity AND worst-decile
  rank-overlap — tail-sensitive by design (the DWQ finding: 0.967 mean cosine
  still fails; means hide tail rank damage). Certified per (machine, engine,
  model, quant, runtime-config) by the onboarding probe; re-checked on
  re-probe.
- **Explicit re-probe triggers** (MC pin): engine version bump, runtime-config
  change, model file hash change. Periodic checks may be added; they are never
  the sole trigger.
- **Revocation is never retroactive** (MC pin): vectors written under a
  certified fingerprint remain valid forever; a lane that falls out of the
  class on re-probe is demoted going forward only (alias row retracted, no
  identity churn).
- **Threshold revisions re-certify lanes, never churn identity** (MC pin): the
  gate definition is Synapse-owned and versioned; a threshold change triggers
  re-certification, and identity moves only when actual membership of the
  writing lane changes.
- **Table epoch** (AFT rider): the alias table carries a version bumped on any
  row change; every embed/rerank response carries (fingerprint, table_epoch).
  Consumers cache interchangeability verdicts per (index_fingerprint,
  table_epoch) and revalidate only on epoch change.
- **Mixed-provenance rule** (AFT rider): while A≡B holds, an A-keyed index may
  legitimately accumulate B-written vectors. On revocation, a pure-A index
  remains servable; an index whose written-provenance set spans a retracted
  pair is invalidated (internally inconsistent in a way neither fingerprint
  names). Consumers record the provenance set per index; the embed surface
  supplies provenance on every response to make that possible.
- **Migration contract**: promoting a non-faithful engine (e.g. MLX bf16) or
  demoting a lane triggers background re-embed with the old index served
  until swap — never a cold hole.
- Day-1 declared pair: llama-server f16 GGUF ≡ ort fp32 (measured 1.00000
  mean cosine, full 15,271-chunk corpus). Day-1 fleet is one class; the first
  alias event exercises the machinery.

Consumer API consequences (lock inputs): fingerprint is a first-class
queryable field on the embed surface (with equivalent_to list); rerank
returns per-candidate raw scores + fingerprint; admission never hides queue
latency inside per-call latency (fast-fail/degraded signal for interactive
budgets); error responses classify transient-vs-permanent at the source.

### The lock decision

D-005 as proposed for lock: a Rust module owning admission (machine-wide),
model lifecycle (shared content-addressed cache), fingerprints (model + quant
+ engine + runtime-config), and the subc surface; llama-server as the
supervised child workhorse across all platforms/backends; ort in-process as
the Apple CPU floor and ONNX/DirectML lane; MLX as the Apple embed fast lane
(shipping form TBD — the one open sub-decision); ANE spike scheduled for the
quiet tier; remote endpoints as a peer backend lane; engine-per-workload
assignments made per-machine by the onboarding probe, never hardcoded. No
Python in the shipped path.

Deferred / watch items: llama.cpp M5 tensor-accelerator support (closes the
MLX gap?), llama.cpp Qwen3-reranker template fix (PR #20009), ms-marco
token_type_ids (PR #21729), Core AI as engine (macOS 27+), MTP/speculative
decode for the future agentic-LLM lane, MLX 8bit anomaly, jina-v5 config.
