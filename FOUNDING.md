# Synapse (CortexKit AI module) — founding handoff

Written by AFT-Alfonso for the new project session. AFT is the first consumer, Magic Context the second. This document carries everything the AFT session learned that the new module inherits. Treat it as input, not gospel: re-derive the architecture yourself, but do not re-learn these measurements the hard way.

## Mission

One subc-supervised module that is the AI house for the whole CortexKit family: local inference (embeddings, rerankers, micro-LLMs; later image gen, STT, TTS) AND the gateway for remote AI endpoints (OpenAI-compatible, Ollama, LMStudio, provider APIs), so consumers talk to ONE surface regardless of where the model runs. LMStudio-like capability, but daemon-native (subc module, no GUI), multi-consumer, credential-integrated.

- Consumers: AFT (semantic search — extraction of its embedding pipeline is already-decided architecture), Magic Context (embeddings; they hit the same class of problems), Alfonso/llm-runner (micro-LLM calls, TBD scope split), future products.
- Remote auth: through the cortexkit-credentials module (~/Work/Projects/CortexKit/cortexkit-credentials) — the module never holds raw user API keys itself.
- subc supervises the module process; consumers reach it over subc routing like any module.

## SUBC's founding answers (verbatim-sourced, 2026-07-04, pm_b4bff216)

1. **Prior art / name**: "embedding-engine" in early subc sketches was a placeholder — nothing designed or reserved; SUPERSEDED by this module. The founding session owns the name (avoid llm-runner-adjacent names). Manifest: register as a **ManagementSurface** (operations embed / rerank / infer / models.list, kind Query), NOT a ToolProvider — these are module-to-module capabilities, not agent-facing tools. execution_mode only exists on Tool entries; moot unless agent-facing tools are exposed later (then: stateless inference = pure, artifact-writing = mutating).

2. **Credentials contract** (implemented, production-proven via llm-runner's `AuthMode::FromVault`): vault = subc-supervised cortexkit-credentials module (reserved, spawn-attested). Flow: operator imports credentials offline via CLI and mints capability handles (`ckh_…`); the consuming module reads its handles from `~/.config/cortexkit/<module>/vault-handles.json` (0600), opens an ordinary consumer connection, route.opens to the credentials module, requests payload by handle. Handles are the only scoping primitive in v1. Implemented already: read path, crash-safe OAuth refresh (epoch-CAS, kill-9-proven), HMAC audit chain, enumeration limiter, keychain master-key custody, and a `report_auth_failure` wire op (`{handle, provider_status}` on 401/403 so the vault can mark/refresh). Credential-id convention: `<method>:<provider>` (apikey:openai, oauth:anthropic). For remote AI endpoints: copy the llm-runner pattern exactly.

3. **Day-one constraints**: subc-protocol >= 0.6.x + subc-transport >= 0.3.x (principal era; SDK auto-attaches consumer_identity). Two-crate split per module template: `<x>-core` (pure logic) + `<x>-module` (wire binary); **ai-provider-quota is the canonical reference** (quota-module/src/main.rs). Storage: cortexkit-store descriptor delivered via HELLO_ACK, database-per-module (`~/.local/share/cortexkit/<module>/store.db`), never self-invented paths. Config: standard cortexkit locations, module is a read-only consumer. **Spawn model for GPU runtimes: subc supervises the MODULE PROCESS ONLY; heavy model runtimes (llama.cpp server, MLX process) are CHILDREN the module itself spawns and supervises** — keeps subc thin, lets the module swap runtimes without losing registration/routes, matches the subc→module→workers hierarchy. Module must degrade (not crash) on child death. CI: Blacksmith runners + cortexkit-ci App secrets for the cross-repo subconscious checkout (llm-runner's ci.yml is the template). Peer-owns-repo from day one.

4. **Scope vs llm-runner — separate modules, crisp line**: llm-runner = durable agentic SESSION engine (multi-turn loops, WAL/replay, chat wire families, tool dispatch). This module = model SERVING — stateless capability inference (embed batch, rerank candidates, transcribe audio, one-shot micro-LLM completion as a raw primitive). Rule: conversation/session/tool semantics → llm-runner; stateless capability inference → this module. Deferred convergence (do not design now): llm-runner may later consume this module's local models as a provider backend — lands as a ProviderSpec on llm-runner's side, zero changes here if the serving surface is clean.

SUBC offered to review the founding doc when drafted; the new session should ping them directly once registered.

## Decision #1 to make: in-house engine vs existing solutions

The user explicitly wants this evaluated honestly, including these candidates (all pre-cloned under ~/Work/OSS/ for study):

- **burn** (~/Work/OSS/burn, tracel-ai): Rust-native deep learning framework, many backends (wgpu, Metal, CUDA, ndarray, candle interop). Attractive as ONE Rust API over heterogeneous GPUs — evaluate whether its inference maturity and model-import story (ONNX import) is production-grade for our model set, vs being a training-first framework.
- **vllm** (~/Work/OSS/vllm): the serving-throughput king, but Python + CUDA-first; evaluate honestly whether we'd ever ship it to end-user machines (likely: no for local desktop, maybe as a remote-endpoint backend users point us at).
- **unsloth** (~/Work/OSS/unsloth): fine-tuning-focused (user suspects llama.cpp under the hood — verify; it's primarily Triton-kernel training with GGUF export THROUGH llama.cpp). Probably relevant later for fine-tune workflows, not for the serving core.
- **llama.cpp** (not cloned yet): the obvious local-LLM serving engine; GGUF ecosystem, Metal/CUDA/CPU. Evaluate llama-cpp-rs bindings vs supervising a llama-server child process (SUBC's spawn model favors child processes anyway).
- **ANE revisit** (~/Work/OSS/Orion, mechramc): our ORT-CoreML-EP→ANE conclusion (dead end, 4 measurements) was specific to THAT path and is months old; new macOS-version developments and projects like Orion warrant a fresh look. Owner should re-survey: CoreML direct (not via ORT), Apple's new APIs in the upcoming macOS, and whether ANE-class efficiency matters for our workload shapes (embeddings are throughput-bound where GPU wins; ANE's win is efficiency at small batch — may matter for always-on micro-LLM/STT).

Frame from AFT's experience:

- **Wrapping Ollama/LMStudio**: both exist, both serve OpenAI-compatible HTTP, both manage model downloads. But: user-facing apps with their own lifecycle (LMStudio is GUI-first, both are extra install burden), no subc supervision, no credential integration, quirky under load (see remote-endpoint scar tissue below — LMStudio 400s under concurrent embed load, cold-load stalls). Wrapping them means our reliability ceiling is their bug tracker.
- **In-house engines**: we have PROVEN spikes for the two hard paths: raw `ort` (ONNX) for cross-platform CPU embedding (shipped in AFT v0.35+, thread-capped, memory-bounded) and `mlx-rs` for Apple Silicon GPU (parity-verified, see numbers). llama.cpp (via llama-cpp-rs or direct FFI) is the obvious third engine for local LLM/GGUF. "In-house" really means "thin Rust module orchestrating proven inference runtimes" — we are not writing kernels.
- Likely answer is a hybrid: in-house module + engine backends (ort, MLX, llama.cpp), with remote endpoints as just another backend behind the same API. But run the design pass properly — including whether serving Ollama's API shape as OUR surface buys ecosystem compatibility for free.

## AFT's requirements as consumer #1 (extraction contract)

What moves OUT of AFT into this module: embedding generation, vector storage/ANN, model-based reranking, intent micro-LLM calls. What stays in AFT: the retrieval plane — chunking, lexical/trigram search, RRF fusion, heuristic rerank, callgraph enrichment, query shaping.

Semantic-store requirement (user: "+999 for semantic-store in subc"): the module owns the vector store and returns RICH per-candidate data — chunk_id, raw cosine, rank, enough for the consumer to fuse/rerank/threshold on its side. Not just "top-k texts".

Concrete API needs from AFT day one:
- embed_batch(texts, model_identity) -> vectors; deterministic model fingerprint in the response (AFT caches keyed on backend identity; see flip-flop below).
- Incremental upsert/delete by (project_key, chunk_id); query by vector with k + min_cosine.
- Status/warmup surface (cold model load can be 60s+ for big local models; consumers need building/ready/degraded states, never silent hangs — AFT's honest-reporting convention).
- Backpressure semantics: AFT retries with backoff on transient; the module should queue, not 500, under concurrent consumer load.

MC's parallel need (coordinate with them): they also embed; TODAY AFT+MC on one project each build separate indexes and can flip-flop a shared cache when backend identities mismatch (memory 6101). One module owning embeddings dissolves that class: single model identity per machine, both consumers share vectors or at least share the embedder.

## Hard-won engineering facts (do not re-learn)

Local embedding (AFT shipped these in v0.35):
- fastembed hardcodes all-cores ONNX threading: 1.7x slower and 3.5x more CPU than capping intra-op threads at num_cpus/2. Use raw `ort` with explicit thread caps.
- Attention memory is quadratic-ish in batch token count: token-budget batching capped AFT's worst-case peak RSS 4.92 GB -> 1.34 GB with zero vector drift. Cap the per-batch token budget, not the item count.
- Chromium-scale (176k files) linear accumulation hits 13.5 GB RSS: stream (collect -> embed -> flush-to-disk -> drop per shard). Never materialize the full corpus of vectors in RAM during a build.
- Mean-pooling + L2-normalize for MiniLM-class parity; tokenizer truncation forced to 512 (Qdrant tokenizer.json embeds 128, fastembed built at 512 — trust the model card, not the tokenizer file).

Apple Silicon GPU (spiked, not yet shipped):
- ORT CoreML EP is a DEAD END for both ANE and GPU on transformer embedders — four independent measurements, ANE fixed at 0. Do not retry this path.
- MLX is the real path: Python mlx Qwen3-Embedding-0.6B hit 22.8K tok/s on code (7.4x faster than ONNX CPU) at ~60W GPU / ~11W CPU (CPU essentially freed). mlx-rs from Rust reproduces it (~62W GPU) with cosine parity >= 0.9994 vs Python on real bf16 weights (last-token pool + L2-normalize).
- mlx-rs build needs full Xcode + Metal toolchain component — CI implication (macOS runners need the Metal component or prebuilt artifacts).
- Spike artifacts: the CoreML/ANE spike source survives at aft:.alfonso/spikes/coreml_spike.rs (the dead-end evidence). The MLX spikes (Python throughput, Rust GPU offload, Rust bf16 parity vs mlx-embeddings with last-token pool + L2-normalize) were ephemeral — code cleaned up after proof — but the method is fully described in AFT session history and reproducible in ~1 day: reference vectors from Python mlx-embeddings, token-ID parity harness, vendored mlx-rs path dep. Ask AFT-Alfonso for the transcript if needed.
- Platform GPU landscape (from the same investigation): no framework unifies GPUs. Apple Silicon = MLX (proven). NVIDIA = ORT CUDA EP (mature, cheap to add — unlike its CoreML EP). AMD/Intel = fragmented, low ROI. Universal floor = bounded-CPU ort. Burn is the one candidate that could challenge the N-backends conclusion — evaluate it against this table.

Model landscape (as of our research; re-survey):
- gte-modernbert-base was our chosen upgrade target for code retrieval (CoIR ~79) — one model servable on both the MLX lane and the bounded-CPU ort lane.
- model2vec (Potion Code 16M) from a contributor PR: static embeddings, no attention, no OOM class at all — interesting as a low-end/CPU-floor tier. Quantized vector storage + RRF fusion ideas from the same PR (cortexkit/aft#87) worth mining.
- Qwen3-Embedding-0.6B proven on MLX lane.

Remote endpoints (scar tissue from AFT's semantic backends):
- OpenAI-compatible: reqwest .json() sets Content-Type; adding it manually creates duplicate headers that break OpenAI's parser (real bug we shipped). 
- LMStudio: transient 400s under concurrent embed load (needs self-heal retry), 8B model cold-load blows 25s timeouts (AFT ships 60s semantic timeout + retry-with-backoff + circuit breaker that preserves the cached index on transient failure instead of rebuilding — the "GPU storm" fix). Ollama on loopback needs an SSRF-guard exemption.
- SSRF guard: user-tier LAN base URLs allowed, project-tier trust-dropped (AFT's config posture; the module will own this class of policy for remote endpoints).
- Every remote call needs: timeout tiers (connect vs read), transient classification, bounded retry with backoff, and a circuit breaker. Port AFT's semantics rather than reinventing.

## subc integration notes (from AFT's module experience)

- Transport: loopback TCP + HMAC handshake via shared subc-transport/subc-protocol crates. Pin them in LOCKSTEP (diamond-dependency E0308 if you bump one without the other; cargo tree -i subc-protocol must show ONE version).
- Frames: dedicated reader task, never read_frame inside select! (cancellation-safety desync — AFT hit stream desync in production tests).
- Long-running calls: inference is stateless but cold model loads run 60s+. AFT built a deferred-response pattern for its long bash calls; as a ManagementSurface this module should still avoid blocking its wire loop on model loads — async load + building/ready states, or ask SUBC what the ManagementSurface equivalent of deferred responses is.
- Principal trust: direct/reserved/absent; the module should decide day one what untrusted binds may do (probably: nothing without credentials scoping).
- Storage convention: config ~/.config/cortexkit/<module>.jsonc, data ~/.local/share/cortexkit/<module>/. Models are BIG — a shared model cache dir (~/.local/share/cortexkit/models/?) shared across module restarts, with checksummed downloads (AFT's downloader has the TOFU/sha256 patterns).
- GPU residency: one module process holding Metal/GPU memory for hot models; decide model eviction policy (LRU by last-use with TTL?) early — this is the LMStudio-parity feature users feel.

## Suggested first moves for the new session

1. Read ai-provider-quota (canonical module template) and llm-runner's FromVault + ci.yml before writing any code.
2. Talk to MC-Alfonso: their embedding usage, model identity, and what API they'd consume (second consumer keeps the API honest).
3. Decision #1 design pass (in-house hybrid vs wrap): write the tradeoff doc, Oracle it, get Ufuk's call.
4. Spike order that de-risked best for AFT: serve ONE model over subc with the ort CPU lane first (smallest end-to-end slice), then MLX lane, then remote-endpoint backend, then the store.
