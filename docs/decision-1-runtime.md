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
4. **wrap (LMStudio)**: [pending smoke] — plus the standing evidence: GUI app
   lifecycle outside subc supervision, and today's live incident (6 concurrent AFT
   processes buried it; no machine-wide admission) is the class of failure wrapping
   cannot fix from outside.
5. **ort CPU**: the proven universal floor; AFT's shipped policies reproduced
   exactly (Level3, ceil(cores/2) threads, 4M attention-unit batching).

## The convergence finding

Every 2026 Apple-Silicon serving stack examined (vLLM-Metal, vllm-mlx, sglang's
Apple lane, oMLX, unsloth Studio, LMStudio's MLX engine) delegates compute to MLX.
The engine layer on Apple hardware is decided — MLX or llama.cpp-Metal; everything
above it is packaging. Decision #1 therefore reduces to: whose packaging — theirs
(Python stacks we cannot ship to end users) or ours (Rust module with
mlx-rs/llama-server lanes, both parity-proven in this bench).

## Results

Measured 2026-07-04/05 on M5 Max (18 cores, 128 GB), macOS 26.5.1, idle-gated
(CPU <= 15%, GPU <= 5% preflight; mid-run foreign-CPU attribution, all runs
`contaminated=false`). Corpus: AFT's real chunk export, 15,271 chunks /
1,569,409 tokens (byte-exact embed_text). Parity = mean cosine vs ort-cpu fp32
reference on identical inputs. Energy = combined avg watts x wall seconds.

### Workload A: batch embedding, Qwen3-Embedding-0.6B

| lane | precision | tok/s | parity | cold load | peak RSS | avg W | energy |
|---|---|---|---|---|---|---|---|
| ort-cpu (9 threads) | fp32 | 1,211 | reference | 1.7s | (lane) | 34.3 | 44.5 kJ |
| mlx-rs Metal | bf16 | 9,078 | 0.99600 | 0.4s | n/a | 61.7 | 11.1 kJ |
| llama-server Metal | f16 gguf | 7,783 | 0.9999994 | 0.8s | 1.55 GB | 38.8 | 7.9 kJ |
| wrap: LMStudio | (server) | [pending] | 0.99958 (smoke) | n/a | external | [pending] | [pending] |

- llama-server wins energy (7.9 vs 11.1 kJ) despite lower tok/s: it draws 39W
  where mlx-bf16 draws 62W. Joules per corpus is the end-user metric.
- Parity is a fingerprint story: llama f16 is numerically indistinguishable from
  the fp32 reference; mlx bf16's 0.9960 is a REAL vector-space difference that
  must surface as a distinct model fingerprint (MC contract).
- ort-cpu is 6.4-7.5x slower and 4-5.6x more energy-hungry than the Metal lanes
  on the same model: the CPU floor exists for compatibility, not for daily use.

### Workload A floor pair: all-MiniLM-L6-v2 (burn cannot import Qwen3)

| lane | precision | tok/s | parity | cold load | energy |
|---|---|---|---|---|---|
| ort-cpu (9 threads) | fp32 | 28,915 | reference (same model) | 0.08s | 2.4 kJ |
| burn wgpu/Metal | f32 | [pending] | [pending] | [pending] | [pending] |

### Workload B: micro-LLM one-shot classification, 100 prompts

| lane | model | combined tok/s | decode tok/s | valid labels | cold load |
|---|---|---|---|---|---|
| llama-server Metal | Qwen3-0.6B q8_0 | 12,352 | 573 | 97/100 | 0.55s |
| mlx-rs Metal | Qwen3-0.6B bf16 | 7,896 | 49* | 96/100 | 0.35s |
| llama-server Metal | LFM2.5-230M q8_0 | 28,776 | 1,140 | 81/100 | 0.33s |

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

The measured matrix decides the remaining question — which engine carries which
workload per platform: ort CPU floor everywhere; MLX lane (mlx-rs) vs llama.cpp
child process on Apple Silicon per workload; remote endpoints (user-pointed
vllm/sglang/Ollama/LMStudio/oMLX) as another backend behind the same surface.

[MEASURED TABLE PENDING — idle-gated matrix on AFT's corpus.]
