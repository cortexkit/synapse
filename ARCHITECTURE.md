# Architecture

## Pattern Overview

**Overall:** Serial, Idle-Gated Multi-Lane Benchmarking Harness AND the Production Synapse Engine runtime.

**Key Characteristics:**
- **Local Inference Service:** The primary production system (`synapse-module`) acts as a persistent SubC node that receives embedding, generation, and reranking requests. It routes work via a 3-class fair-share aging scheduler to underlying local hardware engine lanes, or to external provider pools via the remote gateway.
- **Hardware-Specific Workers:** Local model inference runs outside the host process via supervised binary children (`ck-synapse-worker-mlx`, `ck-synapse-worker-ane`, `ck-synapse-worker-llama`). The host speaks to them over UNIX domain sockets or Windows named pipes using a fast binary framing protocol.
- **Content-Addressed Cache & Durable Jobs:** Persistent SQLite storage manages model downloading (with concurrent shared-lease readers and a two-phase GC), machine capability probing, alias translation, and restartable generation requests (tracking execution/retention TTLs and checkpointed pages).
- **Serial Execution under Idle-Gate Constraints (Bench Harness):** Prevent measurement contamination by ensuring the host machine is idle (average CPU <= 15%, GPU <= 5% for 6 seconds) before starting any evaluation run.
- **Self-Contained Execution Lanes (Bench Harness):** Separate binaries or runtime environments for each target evaluate hardware backends before promoting them to production workers.
- **Numerical Parity Auditing (Bench Harness):** Quantify accuracy drift across acceleration targets by calculating the mean cosine similarity of generated embeddings against a CPU-based `ort` (ONNX Runtime) reference lane.
- **Retrieval Quality and Reranking Parity Auditing:** Assess retrieval quality using offline evaluation datasets (COSQA, CodeSearchNet-Python) from the CoIR suite. Reranking workloads compare candidate scores against reference Alibaba-NLP/gte-reranker-modernbert-base scores to evaluate score drift and rank stability.
- **Native Constrained Decoding (Spike):** Restrict causal generation sequences to a schema/grammar (e.g. JSON or JSON Schema) using a vocabulary-specific bitset mask on logits, ensuring token-by-token structural compliance before token commitment.

## Layers

**Synapse SubC Module (`synapse-module`):**
- Purpose: The main service listening on the SubC bus. Handles route binding, job admission, the model cache, remote provider dispatch, worker lifecycle supervision, and in-process execution via the owned engine.
- Location: `crates/synapse-module`
- Contains: A 3-class aging scheduler, SQLite durable job and cache lease state, machine probe certification logic, socket/pipe-based worker host, the remote gateway client, and direct bindings to `synapse-engine-owned`.
- Depends on: `synapse-core`, `synapse-engine-owned`, `subc-client-rs`, `rusqlite`, `tokio`.

**Remote Gateway (`synapse-module/src/remote`):**
- Purpose: Executes remote provider dispatch through interactive-first turnover pools, circuit breakers, and loopback-verified clients.
- Location: `crates/synapse-module/src/remote`
- Contains: `ProviderRuntime`, client dispatch, vault credential management via `cortexkit-credentials` SubC route, HTTP validators, mock provider e2e, and checkpoint-driven continuity logic.
- Depends on: `synapse-core`, `subc-client-rs`, `reqwest`.

**Synapse Owned Engine (`synapse-engine-owned`):**
- Purpose: Primary in-process execution engine for Apple Silicon (macOS), providing exact-match Metal MPSGraph inference for ModernBERT, Qwen3, and MiniLM models. 
- Location: `crates/synapse-engine-owned`
- Contains: Rust-to-Objective-C bindings, Metal shader graphs, and tensor operations for embedding and reranking. The module stays the sole tokenizer owner; this engine strictly consumes canonical token IDs and executes tensor logic.
- Depends on: `synapse-core`, `safetensors`, `half`, Apple's `Metal` and `MPSGraph` frameworks.
- Used by: `synapse-module` as the primary local engine.

**Synapse Worker Lanes (`synapse-worker-*`):**
- Purpose: Execute in-memory tokenization, tensor forward passes, and pooling for specific hardware classes (Apple Silicon MLX, Apple Neural Engine, Llama GGUF).
- Location: `crates/synapse-worker-mlx`, `crates/synapse-worker-ane`, `crates/synapse-worker-llama`
- Contains: Metal-accelerated customized MLX models, CoreML graphs (including the `gte-modernbert` embedder and reranker for the ANE quiet-tier), and `llama.cpp` inference processes.
- Depends on: `synapse-core`, `mlx-rs`, `coreml` (via Swift), `reqwest`.
- Used by: The `synapse-module` host spawning them dynamically based on user requests and capability tiers.

**Synapse Core Abstractions (`synapse-core`):**
- Purpose: Core vocabulary structs, engine traits, machine capability profiles, and error contracts shared between the host and its workers.
- Location: `crates/synapse-core`
- Contains: `WorkerHello` handshake and binary framing logic, `EngineError` contract, `MachineProfile` with `ane_subtype` chip-identity mapping, `RuntimeConfig`, `TokenBatch`, and scheduling traits.

**Benchmark Harness Core:**
- Purpose: Provides CLI commands for corpus generation, power-monitored process wrapping, result schema definition, and numerical parity functions.
- Location: `bench/harness`
- Contains: CLI entry parsing, idle-gating checks, telemetry collection wrapping, JSONL dataset loading, and cosine similarity calculations.
- Depends on: `clap`, `serde`, `serde_json`, `tokenizers`, `reqwest`.
- Used by: All inference lanes (compiled as the `synapse-bench` library dependency).

**Benchmark Measurement Rig (`synapse-rig`):**
- Purpose: A hash-pinned external measurement harness split out of the candidate tree. Drives candidate inference as a subprocess to guarantee strict execution walls, exact tokenizer application, canonical token accounting, and un-tampered semantic parity metrics.
- Location: `bench/rig`
- Contains: Length-prefixed JSON stdio framing protocol (`rig_protocol.rs`), exact-shape tokenizer constraints, canonical throughput calculation, and result schema enforcement.
- Depends on: `synapse-core`, `tokenizers`, `serde_json`.
- Used by: All modern lane runners evaluating throughput, correctness, or parity on candidate backends.

**Native Engine Inference Lanes:**
- Purpose: Execute in-memory tokenization, tensor forward passes, and pooling over target platforms.
- Location: `bench/lanes/ort-embed`, `bench/lanes/mlx`, `bench/lanes/burn`, `bench/lanes/mlx-minilm`, `bench/lanes/ts-embed`, `bench/lanes/potion`, `bench/spikes/unified-rt`
- Contains: Bounded-thread ONNX Runtime embedding logic, Metal-accelerated MLX custom model implementations, unified-rt candidate implementations (Vulkan cooperative-matrix/plain shaders on RDNA3 with device-local memory staging and budget validation, CUDA cuBLASLt fused graphs and fused QK norm RoPE single-launch kernels on NVIDIA, Metal graph execution optimization levels O0/O1, package caching, and custom direct Metal decode step kernels in `bench/spikes/unified-rt/src/qwen3_decode_metal_step.rs`), LFM2 hybrid causal backbone, LFM2-Audio ASR speech encoder (FastConformer and Slaney mel filterbank frontend), Qwen3-0.6B f16 Metal decode throughput optimizations, WGPU-based Burn ONNX imports, python-based MLX community/source loading, Model2Vec static embedding (`potion-code-16M`), and TypeScript setups.
- Depends on: `bench/harness` or `bench/rig`, target runtime libraries (`ort`, `mlx-rs`, `vulkano`, `cudarc`), and `tokenizers`.
- Used by: The benchmark suite runners `bench/run-matrix.sh` and `bench/run-night.sh`.

**Supervised Child Server Lane:**
- Purpose: Spawns, monitors, and terminates a child server process (`llama-server`) and routes inference requests over standard HTTP endpoints.
- Location: `bench/lanes/llama`
- Contains: Supervised subprocess spawning, TCP port binding probes, `/health` API polling, and batched OpenAI-compatible API request orchestration. Also handles `/v1/rerank` request routing for reranker workloads.
- Depends on: `bench/harness`, `reqwest`, `serde`, `serde_json`.
- Used by: The benchmark suite runner `bench/run-matrix.sh` and `bench/run-night.sh`.

**CoIR Retrieval Evaluation Harness:**
- Purpose: Score retrieval quality and rerank similarity of Synapse embeddings offline.
- Location: `bench/eval-coir`
- Contains: Data preparation scripts for download and conversion, numpy brute-force cosine search and pytrec_eval scoring, and candidate-vs-reference reranker validation.
- Depends on: `uv`, `numpy`, `pytrec_eval`, and `transformers` (for reference reranking).
- Used by: Developers running evaluation checks during model selection and quality screening.

**External Service Wrapper Lane:**
- Purpose: Integrates and profiles pre-existing external inference servers (LMStudio, Ollama) that run out-of-process.
- Location: `bench/lanes/wrap-embed`
- Contains: HTTP request client, input pre-truncation, rate-limit backpressure handling, and external process name RSS sampling.
- Depends on: `bench/harness`, `reqwest`, `tokenizers`.
- Used by: The benchmark suite runner `bench/run-matrix.sh`.

**External Gather-Distillation Harness:**
- Purpose: Standalone Bun/TypeScript data generation for the production gatherer contract, supporting both Anthropic API (with multi-account OAuth rotation) and local OpenAI-compatible endpoints.
- Location: `tools/gather-distill`
- Contains: Trajectory generation, work queue handling, `AftClientPool` process wrapping, validation, gold-overlap scoring, and zero-shot gatherer bake-off evaluation leaderboards (`tools/gather-distill/BAKEOFF-ZEROSHOT.md`).
- Depends on: Bun, pinned `aft-v0.46.0` binary, and `@cortexkit/anthropic-auth-core`.
- Used by: Developers running qgen, gather, validate, or score campaigns.

**Synapse Operator CLI (`synapse-opctl`):**
- Purpose: Drive Synapse operations (models catalog, probe runs, scheduling, batch embedding, and jobs paging) through the fleet subc daemon connection.
- Location: `crates/synapse-opctl`
- Contains: CLI command parsing and formatting logic for operator management.
- Depends on: `subc-client-rs`, `clap`, `serde_json`, `tokio`.
- Used by: Operators and deployment scripts monitoring or triggering runtime actions.

**Management Call Utility (`subc-call`):**
- Purpose: Send raw method calls and JSON params to any module over the fleet daemon.
- Location: `crates/synapse-module/src/bin/subc_call.rs`
- Contains: Direct IPC client call wrapping and formatted envelope printing.
- Depends on: `subc-client-rs`, `tokio`, `serde_json`.
- Used by: Developers and scripts executing low-level management surface functions.

**Decode Campaign Harness:**
- Purpose: Execute and coordinate the sandboxed Athena V3 single-stream decode campaign.
- Location: `bench/campaign`
- Contains: Integrity validation of model snapshots, fixtures, and target runners; deterministic verification of intervention hooks; candidate-owned temporary workspace staging and build output/target directories; toolchain environment forwarding (`RUSTUP_HOME`, `CARGO_HOME`); split-stream append-mode logging (`.log` and `.log.stderr`); and failure scene preservation.
- Depends on: `spike-unified-rt` runner.
- Used by: Automated evaluation gates to confirm decode performance and correctness.


## Data Flow

**Production Inference Flow:**

1. Initialize layered configuration from `SYNAPSE_CONFIG_PATH` (`synapse.jsonc`), rejecting unknown fields and applying `microllm` ceilings — `crates/synapse-module/src/remote/config.rs`
2. Route request received via SubC — `crates/synapse-module/src/lib.rs`
3. Validate alias surfaces, apply machine capability profiles with `ane_subtype` chip identity (Perf/Quiet tiers), verify microLLM certifications (refusing execution on uncertified fingerprints), or map user-tier `remote_providers` profiles — `crates/synapse-module/src/store.rs`
4. Admit job to the DB (checking active attempt ID CAS, request-digest idempotency, and page counts of existing results to resume from checkpoints) — `crates/synapse-module/src/store.rs`
5. Dispatch based on route:
   - **Local:** Download/Verify models through content-addressed cache with shared leases and 24-hour age-floored temporary blob cleanup, admit to 3-class Aging Scheduler, spawn/handshake Worker lane (UNIX sockets / Windows pipes), submit binary frames.
   - **Remote:** Forward through `ProviderRuntime` pools, passing circuit breakers and p90 estimators, fetching credentials via vault trait, executing strict loopback-validated HTTP calls, and serializing `recommended_batch` policies in model listings — `crates/synapse-module/src/remote/runtime.rs`
6. Commit checkpointed pages sequentially according to byte size limits (`result_page_bytes`) as the job runs (allowing page-while-running for snapshots and continuity hooks), mark job complete (applying execution/retention TTL split), and return envelope. If the client queries a job, they can follow pages via `page` parameters — `crates/synapse-module/src/store.rs`

**Constrained Decoding Flow:**

1. Extract logit values for the next token from the causal decode execution.
2. Query the constraint state machine (such as the JSON schema `JsonParser`) to determine valid byte sequences.
3. Compute the vocabulary-wide bitset `TokenMask` by matching allowed byte sequences against the token vocabulary trie.
4. Apply the `TokenMask` to the logits (forcing unallowed token logits to negative infinity).
5. Select the next token from the masked logits, notify the pre-commit tap hooks, advance the constraint parser state, and commit the token.


**Corpus Generation Flow (Bench):**

1. Scan files recursively from a source code tree — `bench/harness/src/main.rs`
2. Parse text and split contents into chunks constrained by token budgets — `bench/harness/src/corpus.rs`
3. Export structured chunks to a JSONL dataset containing IDs and chunk texts — `bench/harness/src/corpus.rs`

**Idle-Gated Power Telemetry Run:**

1. Monitor system activity using `macmon` and block execution until utilization satisfies idle thresholds — `bench/harness/src/metrics.rs`
2. Spawn the target command as a child process and initialize CPU, GPU, ANE watt telemetry, machine utilization (CPU/GPU avg/peak usage percentages), and RSS profiling — `bench/harness/src/metrics.rs`
3. Sample metrics periodically, tracking energy consumption in Joules — `bench/harness/src/metrics.rs`
4. Serialize telemetry data to an output metrics JSON file — `bench/harness/src/metrics.rs`

**Inference and Numerical Parity Check:**

1. Launch `synapse-rig` which spawns the target inference candidate as a subprocess over stdio, or run a legacy native lane script directly — `bench/rig/src/main.rs` or `bench/lanes/*/src/main.rs`
2. The rig or legacy lane parser prepares inputs. For rig runs, the rig sends batched length-prefixed JSON frames (`prepare_shapes`, `embed`) to the candidate.
3. The candidate returns vector arrays. The rig ensures exact canonical token accounting independent of the candidate padding tricks.
4. Calculate mean cosine similarity of produced output vectors against the `ort-cpu` baseline reference, and top-k neighbor overlap metrics to check for rank stability — `bench/harness/src/parity.rs`
5. Write results structured in the `LaneResult` schema to the output results JSON — `bench/harness/src/results.rs`

**CoIR Retrieval Evaluation Flow:**

1. Prepare evaluation task files (COSQA and CodeSearchNet-Python) into uniform JSONL shape (queries and corpus) — `bench/eval-coir/prepare.py`
2. Run target inference lanes with document/query prefixes to generate vector outputs — `bench/lanes/*/src/main.rs`, `bench/lanes/potion/main.py`, etc.
3. Execute brute-force cosine retrieval and calculate metrics (MRR@10, NDCG@10, Recall@10) — `bench/eval-coir/score.py`

**Reranking Quality Check Flow:**

1. Start `synapse-rig` targeting a candidate backend (such as `unified-rt` or `llama-server`) with `--rerank-requests` — `bench/rig/src/main.rs`
2. The rig submits length-prefixed JSON batches of query-document pairs to the candidate, accumulating strict canonical real-token counts — `bench/rig/src/main.rs`
3. Generate reference scores using Hugging Face reference implementation (`Alibaba-NLP/gte-reranker-modernbert-base`) — `bench/eval-coir/reference_rerank.py`
4. The rig automatically calculates Pearson correlation and tie-aware top-1 overlap against reference scores, rejecting the candidate if it drifts below the `.999` and `.98` thresholds.

**Gather-Distillation and Evaluation Flow:**

1. Generate question candidates grounded on corpus manifests and entry files using `qgen` — `tools/gather-distill/src/qgen.ts`
2. Process questions round-robin via the gather queue, proxying all tool calls (`search`, `outline`, `zoom`, etc.) to a background `aft` v0.46.0 subprocess over an NDJSON stream — `tools/gather-distill/src/gather.ts`
3. Force the final turn with the toolset intact (`tool_choice: "none"`) to retrieve structured evidence JSON and snippet citations — `tools/gather-distill/src/gather.ts`
4. Validate trajectory JSON rows, confirming commit SHAs, file bounds, and citation content against the pinned repo — `tools/gather-distill/src/validate.ts`
5. Perform offline gold-overlap scoring to evaluate candidate trajectory quality (tracking line-range Jaccard, file F1, and token usage) — `tools/gather-distill/src/scorer.ts`

**LFM2 Causal Decode and LFM2-Audio ASR Flow (unified-rt):**

1. Detect model family from config `model_type` (`lfm2` or `lfm2-audio`).
2. Initialize LFM2 hybrid backbone containing 10 short-convolution layers and 6 full-attention layers with tied embeddings and GQA KV cache.
3. If processing audio (`lfm2-audio`):
   - Read mono 16 kHz WAV file, apply pre-emphasis, centered STFT with Hann window, power spectrum, and 128-bin Slaney-normalized mel filterbank.
   - Normalize log-mel features and process through a noncausal FastConformer encoder and audio projector to map speech inputs to 2048-wide vectors.
   - Splice projected audio embeddings into the LFM2 backbone text token space.
4. Execute greedy causal decode using `DecodeModel` cache and token taps, keeping execution token-exact against Python references.

## Key Abstractions

**Worker Framing Protocol:**
- Purpose: A byte-exact IPC mechanism sending dynamic float and integer arrays between the Rust host and worker children over sockets.
- Location: `crates/synapse-core/src/worker_protocol.rs`
- Pattern: Binary Serialization (e.g., `decode_f32_frame`, `encode_i32_frame`).

**Fair-Share Scheduler:**
- Purpose: Manage execution time slices across queued, active, and completed jobs, guaranteeing that background operations don't starve foreground priority work.
- Location: `crates/synapse-core/src/scheduler.rs`
- Pattern: Trait-based State Machine Interface.

**Content-Addressed Lease Cache:**
- Purpose: Safe and crash-resilient model storage using atomic file renaming and reference-counted shared leases preventing active models from two-phase garbage collection.
- Location: `crates/synapse-module/src/store.rs`
- Pattern: Persistent DB Leasing.

**ProviderRuntime & Turnover Pools:**
- Purpose: Manages interactive-first workload routing to remote gateways using turnover queues, latency-sized sub-batches, and bucketed p90 estimators.
- Location: `crates/synapse-module/src/remote/runtime.rs`
- Pattern: Concurrency Pool with Circuit Breakers (half-open lease, censor floor).

**Gateway HTTP Substrate:**
- Purpose: Hardened, loopback-verified HTTP bindings ensuring security boundary enforcement (no-redirect, exact preflight loopback matching, max-body bounds).
- Location: `crates/synapse-module/src/remote/client.rs`
- Pattern: Network Facade with Strict Validation.

**LaneResult:**
- Purpose: Unified schema for reporting execution performance, throughput metrics, telemetry outputs, and parity calculations.
- Location: `bench/harness/src/results.rs`
- Pattern: Serializable Data Struct.

**Chunk / Prompt:**
- Purpose: Uniform representations of inputs for embedding (Chunk) and classification (Prompt) workloads.
- Location: `bench/harness/src/parity.rs`
- Pattern: Deserializable Data Structures.

**Rerank Query Row:**
- Purpose: Represents a query and its candidate documents to be reranked.
- Location: `bench/lanes/llama/src/main.rs`
- Pattern: Deserializable Data Structure.

**System Idle Gate:**
- Purpose: Preflight check to guard executions and prevent run telemetry contamination by background system tasks.
- Location: `bench/harness/src/metrics.rs`
- Pattern: Guard clause loop checking CPU and GPU utilization thresholds.

**AftClientPool:**
- Purpose: A bounded pool managing background `aft` subprocesses, configuring repositories with trigram-indexing preflight checks and routing commands over NDJSON.
- Location: `tools/gather-distill/src/tools.ts`
- Pattern: LRU process cache with timeout recovery.

**AccountPool:**
- Purpose: Managed pool for token rotation, concurrency limits, and cooldowns across multiple credentials.
- Location: `tools/gather-distill/src/auth.ts`
- Pattern: Rotating Credentials Pool with in-flight caps.

**LFM2 Causal Mixer:**
- Purpose: Alternates 10 short-convolution layers and 6 full-attention layers with tied embeddings and GQA KV cache, supporting modern `layer_types` configurations.
- Location: `bench/spikes/unified-rt/src/lfm2.rs`
- Pattern: Causal hybrid model architecture.

**FastConformer Audio Encoder:**
- Purpose: Translates Mel-spectrogram DSP features into projection-aligned backbone embeddings for ASR splicing.
- Location: `bench/spikes/unified-rt/src/lfm2_audio.rs`
- Pattern: Feature extraction and modality alignment.

**Vulkan Device-Local Weight Stager:**
- Purpose: Performs staging buffer transfers (`vkCmdCopyBuffer`) to isolated device-local Vulkan memory and tracks heap budgets via `VK_EXT_memory_budget`.
- Location: `bench/spikes/unified-rt/src/vulkan_backend.rs`
- Pattern: Isolated memory placement.

**DecodeConstraint / JsonParser:**
- Purpose: Enforce grammars and JSON Schema specifications during causal decoding.
- Location: `bench/spikes/unified-rt/src/json_constraint.rs`
- Pattern: Incremental state-based byte recognizer returning cached token bitsets (`TokenMask`).

**Admission Semaphore:**
- Purpose: Guard inline execution pools, tracking waiters and recording percentile wait statistics.
- Location: `crates/synapse-module/src/lib.rs`
- Pattern: Concurrency Semaphore with observable stats wrapper.

**Certification Status and Demotion:**
- Purpose: Track local hardware engine capability status, storing whether a measured fingerprint is `certified` or `uncertified`.
- Location: `crates/synapse-module/src/store.rs`
- Pattern: SQLite-backed schema with automatic demotion upon failed re-certification.

**Machine Profile:**
- Purpose: Capture machine hardware and engine runtime identities (OS build, arch, chip model, RAM class, `ane_subtype` chip mapping, sorted engine identities) into a stable hash for fingerprinting and certification.
- Location: `crates/synapse-core/src/machine_profile.rs`
- Pattern: Serializable Identity Profile with SHA-256 fingerprinting.

**Metal Custom Step Engine:**
- Purpose: Execute single-token Qwen3 decode steps bypassing MPSGraph via direct Metal compute kernels (`qwen3_decode_metal_step.rs`, `qwen3_decode_metal_step.m`, `qwen3_decode_metal_step.metal`), leveraging SIMDgroup RMSNorm, position-parallel attention, and Q8 GEMV routines.
- Location: `bench/spikes/unified-rt/src/qwen3_decode_metal_step.rs`
- Pattern: Direct Metal Compute Kernel Pipeline.


## Entry Points

**Synapse Module Main (`ck-synapse`):**
- Location: `crates/synapse-module/src/main.rs`
- Triggers: Starts the primary SubC worker process.
- Responsibilities: DB initializations, environment bootstrapping, SubC binding registrations, and polling the scheduler.

**Worker Binaries (`ck-synapse-worker-*`):**
- Location: `crates/synapse-worker-*/src/main.rs`
- Triggers: Spawned directly by `synapse-module/src/worker_host/mod.rs`.
- Responsibilities: Initializing accelerator graphs/sessions (MLX, ANE, Llama), pipe/socket handshaking, loop listening for compute requests, returning tensors.

**Bench Harness CLI (`synapse-bench`):**
- Location: `bench/harness/src/main.rs`
- Triggers: Execution of the `synapse-bench` binary.
- Responsibilities: Routes commands to either chunk source files into a corpus, execute telemetry-monitored child commands, or calculate top-k neighbor rank-overlap parity.

**Benchmark Measurement Rig (`synapse-rig`):**
- Location: `bench/rig/src/main.rs`
- Triggers: Direct invocation by orchestrators during candidate evaluation and performance testing.
- Responsibilities: External measurement harness, bounding execution timings, hashing candidates for validation, enforcing gate thresholds for parity and padding waste, and generating `LaneResult` json.

**Inference Lane Runners:**
- Location: `bench/lanes/ort-embed/src/main.rs`, `bench/lanes/wrap-embed/src/main.rs`, `bench/lanes/llama/src/main.rs`, `bench/lanes/mlx/src/main.rs`, `bench/lanes/burn/src/main.rs` (Rust crates); `bench/spikes/unified-rt/src/main.rs` (spike unified-rt runner); `bench/lanes/mlx-minilm/main.py` (Python script); `bench/lanes/ts-embed/main.mjs` (TypeScript script); `bench/lanes/potion/main.py` (Python script)
- Triggers: Invocation by the power wrapper or direct script executions.
- Responsibilities: Model initialization, cold-load timing tracking, batched inference execution (including causal decoding `--generate-prompts` and ASR transcribing `--asr-audio` for LFM2), and vector/result output generation.

**CoIR Evaluation Entry Points:**
- Location: `bench/eval-coir/prepare.py`, `bench/eval-coir/score.py`, `bench/eval-coir/reference_rerank.py`
- Triggers: Invoked by developers during retrieval and reranking quality audits.
- Responsibilities: Retrieval task data prep, vector scoring, and reference rerank scoring.

**Matrix Orchestration Script:**
- Location: `bench/run-matrix.sh`
- Triggers: Triggered by a developer to run the complete benchmarking suite.
- Responsibilities: Autodetect cached HuggingFace snapshots, sequence idle-gated executions of Qwen3, MiniLM, and LFM candidate lanes, check dependencies, and log to `bench/results/matrix.log`.

**Nightly Suite Orchestrator:**
- Location: `bench/run-night.sh`
- Triggers: Scheduled execution or manual trigger by developer.
- Responsibilities: Precondition validation, sequential idle-gated run of all 16 target lanes on the full AFT corpus, full-corpus parity and rank-overlap calculations, and archiving outputs under `bench/results/night-YYYYMMDD/`.

**Vulkan Capability Prober:**
- Location: `bench/spikes/unified-rt/src/bin/vulkan_probe.rs`
- Triggers: Execution of the `vulkan_probe` binary.
- Responsibilities: Queries and logs Vulkan physical heaps, memory types, and memory budget metrics.

**Gather-Distillation CLI (`gather-distill`):**
- Location: `tools/gather-distill/src/cli.ts`
- Triggers: Invocation of Bun running the CLI commands.
- Responsibilities: Routes commands to qgen (question generation with optional `--avoid-from` to avoid duplicating question lists), gather (trajectory collection), validate (trajectory inspection), and score (gold-overlap performance comparison and zero-shot bake-off evaluation).

**Synapse Operator CLI (`synapse-opctl`):**
- Location: `crates/synapse-opctl/src/main.rs`
- Triggers: Execution of the `ck-synapse-opctl` binary.
- Responsibilities: Routes commands to list models, view scheduler stats, start probes, run batches, and fetch paged job results.

**Management Surface SubC Caller (`subc-call`):**
- Location: `crates/synapse-module/src/bin/subc_call.rs`
- Triggers: Execution of the `subc_call` binary.
- Responsibilities: Connects to fleet daemon and sends management calls directly to target modules.

**Inline Embedding Throughput Client (`inline_embed_throughput`):**
- Location: `crates/synapse-module/src/bin/inline_embed_throughput.rs`
- Triggers: Execution of `inline_embed_throughput` binary.
- Responsibilities: Evaluates local batch throughput and query concurrency/latencies under load.

**Athena V3 Decode Campaign Harness:**
- Location: `bench/campaign/decode-harness.sh`
- Triggers: Invocation of `decode-harness.sh` by an evaluation runner.
- Responsibilities: Manages snapshot validation, locked candidate execution sandbox, candidate-owned workspace staging, toolchain environment forwarding, correctness verification, split-stream append-mode logging, failure scene preservation, and performance evaluation.


## Error Handling

**Strategy:** Fail-fast utilizing `anyhow::Result` and typed subsystem errors (`SubcModuleError`, `EngineError`) with contextual layers.
- **Worker Crash Domain:** If a worker binary crashes, deadlocks, or hangs, the `synapse-module` supervisor reclaims the job. Workers isolate dirty driver states, preventing host process termination.
- **Gateway Continuity:** The remote gateway tracks `ContinuityCheck` hooks for checkpointed streams, catching upstream disconnects or token censorship, while maintaining stable HTTP error unions.
- **Durable Job Resiliency:** Jobs track their generation cycles. Crash-interrupted requests can be recovered via idempotent request-digest keys if the host restarts.
- **SubC Communication:** Submodule failures strictly return properly formatted error envelopes detailing the specific layer failure (e.g., CacheMiss, EngineOOM).

**Bench Harness Strategy:** Fail-fast utilizing `anyhow::Result` error propagation with contextual layers (`.context()`).
- **Child Supervision:** Spawned subprocesses (`llama-server`) are tracked via PID. If a child dies or fails to bind to its designated port within `HEALTH_TIMEOUT` (120s), the lane runner fails immediately rather than silently hanging. Platform-specific process control signals (such as SIGTERM on Unix) are gated appropriately so subprocess lifecycles function seamlessly on both Windows and Unix platforms.
- **HTTP Resiliency:** Requests to external wrapping endpoints (`wrap-embed`) implement read timeouts, connect timeouts, and bounded retry loops with backoff to recover from transient rate limits or cold-load stalls.
- **Campaign Failure Preservation:** The decode campaign harness separates standard output and standard error streams into append-only logs (`.log` and `.log.stderr`) to avoid truncating diagnostics. If a candidate build, verification, or run fails, the harness dumps logs and staging details to the results directory as a preserved failure scene before cleaning up.

## Cross-Cutting Concerns

**Logging:** Console outputs are printed directly. Matrix status tracking, parameters, and outputs write directly to `bench/results/matrix.log`.
**Caching:** Model files are located from HuggingFace cache snapshots. Content-addressed downloads will follow atomic tmp+rename patterns in `~/.local/share/cortexkit/models/`.
**Storage:** Structured outputs are written under `bench/results/` as telemetry metrics (`.measure.json`), parity vectors (`-vectors.jsonl`), and results summary (`.json`) files.
