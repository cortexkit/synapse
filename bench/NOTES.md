# Decision #1 evaluation — working notes

Evidence accumulator for the tradeoff doc. Raw run JSONs land in bench/results/.

## Bench environment

- Apple M5 Max, 18 cores, 128 GB RAM, macOS 26.5.1 (build 25F80)
- rustc 1.96.1; Xcode 26.6 (via DEVELOPER_DIR override, xcode-select points at CLT)
- llama-server build 9580 (b4e3dc613), zerobrew
- Power sampling: macmon 0.7.2 (sudo-free; cpu/gpu/ane watts, JSON pipe)
- Corpus v1 (provisional): 4000 chunks from cortexkit/aft, 1,416,127 tokens,
  avg 354 tok/chunk, Qwen3-Embedding tokenizer, line-based chunker.
  To be superseded by AFT's real chunk export for final numbers.

## Dispositions by inspection (worker [task-id], evidence cited in report)

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

AFT steers folded in (pm_61591b08): corpus swap to AFT's real SemanticChunk export when
it lands; rerank deferred to the API design round (with AFT before freeze); COLD-LOAD is
a first-class metric column for the micro-LLM workload (always-on vs load-per-call
architecture fork); wrap lanes measured with predicted-vs-measured stated side by side.
Internal queue design later: scheduler-owns-readiness (workers only receive runnable
jobs), never workers-block-on-locks. API contract must state timeout tiers explicitly in
manifest/op descriptions (25s-default-vs-60s-cold-load scar).

## Burn inspection (worker [task-id], evidence cited in report)

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

## ANE re-survey (worker [task-id], sources dated, retrieved 2026-07-04)

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
