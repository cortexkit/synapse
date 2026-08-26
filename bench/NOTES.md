# Decision #1 evaluation — working notes

Evidence accumulator for the tradeoff doc. Raw run JSONs land in bench/results/.

## Bench environment

- Apple M5 Max, 18 cores, 128 GB RAM, macOS 26.5.1 (build 25F80)
- rustc 1.96.1; Xcode 26.6 (via DEVELOPER_DIR override, xcode-select points at CLT)
- llama-server build 9580 (b4e3dc613), zerobrew
- Power sampling: macmon 0.7.2 (sudo-free; cpu/gpu/ane watts, JSON pipe)
- Corpus v1 (provisional): 4,000 independently licensed test chunks, 1,416,127
  tokens, avg 354 tok/chunk, Qwen3-Embedding tokenizer, line-based chunker.
  Replace this fixture with a reviewed public corpus before publishing final numbers.

## Measurement protocol

- ALL published numbers come from `synapse-bench power` runs, which hard-refuse to
  start unless the machine is idle (6s macmon preflight: avg CPU <= 15%, GPU <= 5%).
  `--skip-idle-check` exists for integration smoke only, never for published numbers.
- Lanes run SERIALLY, one at a time, idle-gated between lanes (bench/run-matrix.sh).
- Deterministic outputs (parity reference vectors) may be generated on a busy machine;
  timing/power from those runs is discarded.
- Contaminated-run register: bench/results/smoke-ort.json timings are
  integration proof only and are not published measurements.

## Dispositions by inspection

### vllm — dispositioned (not benched)

Python server stack (setuptools/CMake, fastapi/prometheus/otel footprint), CUDA-first
(CMake default target cuda; GPU build errors without CUDA/HIP). macOS is forcibly CPU-only
(setup.py flips darwin to cpu; no metal/mps platform in vllm/platforms). Native Windows
unsupported (setup.py: Linux/macOS only). Counter-evidence honestly noted: real embeddings
API (/v1/embeddings) and a substantive ARM-NEON CPU backend — but CPU-only-on-mac Python
serving is strictly dominated by our existing lanes for a desktop daemon.
Residual relevance: users may point Synapse at a remote vllm endpoint; that is the
remote-backend lane, not a local runtime candidate.

### unsloth — dispositioned (not benched)

Training/fine-tuning-first (Triton kernels, trl.SFTTrainer core). Its "fast_inference"
literally hands off to vLLM (models/llama.py); its Studio serving app delegates GGUF to a
llama-server subprocess and Apple Silicon to MLX (studio/backend/core/inference/*). GGUF
export shells through llama.cpp (save.py clones/builds ggerganov/llama.cpp). Confirms the
founding doc's suspicion. Not a runtime; relevant later only for fine-tune workflows.
Notable: unsloth Studio's own architecture (llama-server child process + MLX on Apple
Silicon) independently converges on the same hybrid we're evaluating.

## Candidate lanes to measure

| Lane | Runtime | Model (embed) | Model (micro-LLM) | Status |
|------|---------|---------------|-------------------|--------|
| ort-cpu | raw ort, threads=ncpu/2 | Qwen3-Embedding-0.6B or MiniLM | n/a (floor is embed-only) | pending |
| mlx | mlx-rs, Metal | Qwen3-Embedding-0.6B bf16 | Qwen3-0.6B-class | pending |
| llama-metal | llama-server child, Metal | Qwen3-Embedding-0.6B GGUF | Qwen3-0.6B GGUF | pending |
| burn-wgpu | burn, wgpu/Metal | Qwen3-Embedding-0.6B via burn-onnx (f16; Metal lacks bf16) | dropped — see below | pending |
| wrap-lmstudio | LMStudio HTTP | qwen3-embedding-0.6b | n/a | pending (workload A only) |
| wrap-ollama | Ollama HTTP | same | not installed — decide whether to install | pending |

Parity: reference vectors from the ort-cpu lane (deterministic); embed lanes report mean
cosine vs reference. Generative workload B judged on output validity, not parity.

External technical review: use a reviewed public corpus for final numbers;
rerank remains deferred to the API design round; COLD-LOAD is a first-class
metric for the micro-LLM workload (always-on vs load-per-call architecture
fork); wrap lanes report predicted and measured results side by side.
Internal queue design later: scheduler-owns-readiness (workers only receive runnable
jobs), never workers-block-on-locks. API contract must state timeout tiers explicitly in
manifest/op descriptions (25s-default-vs-60s-cold-load scar).

## Burn inspection

- ONNX import moved OUT of the main repo to external burn-onnx (v0.21.0, 2026-05-12,
  first dedicated release): 160 ops, model checks include ModernBERT, all-MiniLM-L6-v2,
  and a Qwen check; SDPA-pattern coalescing into native Attention. Unsupported ops =
  refuse/skip-codegen, no fallback. Dynamic shapes: common transformer patterns OK,
  general case still edge-casey (open issues on runtime axes/split sizes).
- Metal story: burn-wgpu -> CubeCL/wgpu -> MSL. f16 supported, bf16 NOT supported on
  Metal. No published Metal benches in-repo (benchmarks.toml is CUDA-only); external
  models repo reports M3 Max MiniLM wgpu at 18ms/sentence.
- LLM: primitives exist (MhaCache, RoPE, RMSNorm) but NO GGUF loader, no sampling stack,
  quantization "active development"; llama-burn (models repo) documents tch/cuda/vulkan,
  not Metal, with an open "llama-burn + wgpu broken?" issue.
- Safetensors direct load: solid (burn-store SafetensorsStore + PyTorchToBurnAdapter).
- Verdict for bench: burn enters the EMBED lane only (burn-onnx import of
  Qwen3-Embedding-0.6B at f16, wgpu/Metal). LLM lane for burn is dispositioned by
  inspection: no GGUF, no Metal-documented llama path, quantization immature — would be
  a weeks-scale risky integration for a bench candidate that already trails on evidence.
- Strategic note: burn as "one Rust API over heterogeneous GPUs" fails the founding
  question for the LLM half today; it can only challenge the N-backends conclusion for
  embeddings.

## ANE re-survey (sources dated, retrieved 2026-07-04)

Old conclusion (ORT-CoreML-EP dead end) holds for THAT path only; direct paths moved:

- CoreML DIRECT is viable today: stateful models/MLState (KV cache) since macOS 15;
  coremltools StateType; public proof of ANE-first transformers (john-rocky/CoreML-LLM:
  Qwen3.5-0.8B ~48 tok/s @99.9% ANE on iPhone 17 Pro; EmbeddingGemma-300M @99.8% ANE).
  Caveat: heavy per-model ANE surgery (chunking, Conv2d rewrites, state workarounds).
- Core AI (WWDC26, macOS 27 beta era): Apple's forward runtime for custom on-device
  models (Swift runtime + Python export; official Qwen3-0.6B/4B/8B, Mistral, Gemma
  recipes in apple/coreai-models). Promising, pre-GA — spike-only for now.
- Foundation Models framework: NO public embeddings API (absence-of-evidence as of
  2026-07-04); app-side completions only, constrained model choice. Not for Synapse.
- Orion: proves private-API ANE execution (ANEClient direct, 170+ tok/s GPT-2 on M4
  Max) — research-only, version-fragile, not shippable.
- Rust: objc2-core-ml / cidre make classic CoreML callable from Rust; Core AI and
  Foundation Models realistically need a Swift shim. objc2 marks FoundationModels
  Swift-only.
- Economics on MAC (third-party bench, M4 Max, Gemma 4 E2B): CoreML/ANE 12.7W but
  0.48 J/token vs MLX/GPU 24.7W at 0.24 J/token — ANE draws less power but LOSES
  joules/token on decode because it's slower. ANE's real win: memory footprint (241MB
  vs 1279MB on a 2B model) and thermal retention — i.e. always-on niches, iPhone more
  than Mac.
- Verdict for decision #1: ANE is NOT a runtime candidate for round 1 (Mac throughput
  workloads favor GPU decisively). Recorded as a deferred spike: CoreML-direct
  embedder + small-LLM measurement behind a Swift shim, relevant when always-on
  micro-LLM/STT workloads land. This defers with evidence, not by omission.

## Lane integration results (masons, smoke = correctness only, timings contaminated)

| Lane | Parity vs reference | Cold load (smoke) | Integration findings |
|------|--------------------:|------------------:|----------------------|
| ort-cpu (Qwen3-Emb fp32) | reference | 1.4s | KV-cache inputs in onnx-community export fed empty; reference policies reproduced exactly |
| llama-metal (f16 GGUF, llama-server child) | 0.9999994 | 13.1s (incl model load) | --pooling last + --embd-normalize 2 verified; server returns per-request timings; chat_template_kwargs.enable_thinking=false works on build 9580; clean child lifecycle incl error paths |
| mlx (bf16, mlx-rs 0.25.3) | 0.9958 | 4.4s | Full Qwen3 forward pass hand-written (RMSNorm/RoPE/GQA/q&k-norm); plain completion gave 0.000 label validity → Qwen chat template + thinking-disabled gives 1.000; mlx-sys release build 13m08s, needs cmake + DEVELOPER_DIR (real CI cost) |
| burn-wgpu (MiniLM fallback, f32) | 1.0000 (vs ort same model) | 105.5s (Metal shader setup) | Qwen3 ONNX REFUSED by burn-onnx: "Nodes are not topologically sorted (ONNX spec violation)" — the validated-models path works, arbitrary exports don't; compile-time codegen pins the binary to one ONNX snapshot (no runtime model swap); 4m31s release build |
| wrap-lmstudio | pending | n/a | blocked on another workload |

Workload B smoke (10 prompts, greedy, 16 max tokens):
- llama-metal: 10/10 valid labels, server decode ~281 tok/s (contaminated)
- mlx: 10/10 valid labels with chat template, decode ~21.9 tok/s (contaminated, and
  hand-rolled decode loop — production would batch/optimize; treat as floor not ceiling)

Architecture signal already visible (pre-measurement): burn's compile-time-codegen
model binding is structurally wrong for a model-SERVING module (models arrive at
runtime); mlx-rs requires hand-implementing every architecture (worked for Qwen3, but
each new model family = new Rust code); llama.cpp child process gives runtime model
loading + per-request timings + one binary for embed+LLM.

## New candidates round 2 (user request, 2026-07-04 evening)

- LFM2.5-230M: added to workload B on the llama lane (Q8_0, verified working on
  llama-server b9580 — valid label answer in manual probe). NOTE: it cannot join the
  mlx lane without hand-implementing LFM2.5's hybrid conv/attention architecture in
  mlx-rs — which is itself the mlx-rs finding restated: every model family = new
  Rust code; llama.cpp got it for free from the GGUF ecosystem.
- sglang: Python-first serving stack, quickstart still Linux+CUDA
  sm80+, no Windows story — BUT it now has an official Apple Silicon path via a
  separate MLX/Metal guide (docs.sglang.io hardware-platforms/apple_metal, v0.5.14
  2026-06-26 shows active MLX work). Embeddings first-class (/v1/embeddings,
  Qwen3-Embedding-0.6B is their own bench default). Verdict: dispositioned as
  primary desktop runtime (Python stack, datacenter-oriented, no Windows); its
  Apple lane is MLX underneath — see the convergence note below.
- vllm Metal claim: CORRECTION to our disposition. vLLM-Metal is
  REAL: an official-org out-of-tree plugin (vllm-project/vllm-metal, created
  2025-12-12, alpha, ~1.4k stars), MLX-backed (deps: mlx, mlx-lm, mlx-vlm),
  registered via vllm.platform_plugins. Core vllm still has no in-tree Metal
  backend, so the repo-inspection finding stands for core; the disposition wording
  was stale about the ecosystem. Limits: installer pulls dev wheels + builds vllm
  0.24.0 core from source (Python 3.12 arm64 + uv + Xcode CLT burden), pooling
  experimental/LAST-only, narrow GGUF. Even its experimental vllm-rs Rust frontend
  spawns the Python engine. Verdict unchanged for shipping: not a consumer
  child-process candidate; noted honestly in the doc.
- waybarrios/vllm-mlx (same worker): independent Apple MLX server with vLLM-like
  batching/paged-KV (delegates to mlx-lm/mlx-vlm/mlx-embeddings), alpha, not the
  upstream path (official docs point at vllm-metal instead). Self-reported 417.9
  tok/s Qwen3-0.6B-8bit on M4 Max. Python-heavy; disposition for shipping.
- oMLX (omlx.ai, jundot/omlx): the strongest wrap-class candidate found — DMG +
  Homebrew service + CLI serve, OpenAI+Anthropic APIs, embeddings/rerank, SSD KV
  cache; 17.5k stars, 100+ contributors, v0.4.4 2026-06-16. Still a ~750MB
  Python/MLX app stack, macOS 15+. Verdict: not a subprocess-class engine; a
  candidate optional EXTERNAL backend behind our remote-endpoint lane (same class
  as LMStudio, materially healthier project).

CONVERGENCE NOTE (feeds the doc): every 2026 Apple-Silicon serving stack examined
(vllm-metal, vllm-mlx, sglang-apple, oMLX, unsloth Studio, LMStudio's MLX engine)
delegates compute to MLX. The ecosystem has already voted: on Apple hardware the
engine layer is MLX or llama.cpp-Metal; everything above it is packaging. This
narrows decision #1 to: whose packaging — theirs (Python stacks we can't ship to
end users) or ours (Rust module + mlx-rs/llama-server lanes, both parity-proven
in our bench).

## Rerank smoke dataset

- `bench/data/rerank-smoke-v1.jsonl` is a deterministic smoke/latency set: 50 requests,
  20 documents per request, 10 planted-relevance topics repeated 5 times with rotated
  distractors.
- The first document in each request is the known relevant chunk; the remaining 19 are
  distractors. The cheap smoke sanity check uses that planted first-document convention.
- This checkout does not contain `bench/data/corpus-smoke.jsonl`, so the rerank smoke set
  was hand-built from committed bench source snippets instead of a missing corpus export.
  Each document string keeps its source file and line range in the text for traceability.
