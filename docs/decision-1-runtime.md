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

[TABLE PENDING — generated from bench/results/*.json after idle-gated runs]

## Recommendation

[PENDING measured numbers. Preliminary shape, to be confirmed or falsified by the
matrix: hybrid in-house module — ort CPU floor everywhere, MLX lane where it wins
on Apple Silicon, llama.cpp child process as the general local-LLM/GGUF engine,
remote endpoints (including user-pointed vllm/Ollama/LMStudio) as just another
backend behind the same surface. Wrap-as-primary looks dead on architecture grounds
(supervision, admission control, reliability ceiling) before speed even enters.]
