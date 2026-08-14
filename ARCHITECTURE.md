# Architecture

## Pattern Overview

**Overall:** Serial, Idle-Gated Multi-Lane Benchmarking Harness AND the Production Synapse Engine runtime.

**Key Characteristics:**
- **Local Inference Service:** The primary production system (`synapse-module`) acts as a persistent SubC node that receives embedding, generation, and reranking requests. It routes work via a 3-class fair-share aging scheduler to underlying local hardware engine lanes, or to external provider pools via the remote gateway.
- **Hardware-Specific Workers:** Local model inference runs outside the host process via supervised binary children (`ck-synapse-worker-mlx`, `ck-synapse-worker-ane`, `ck-synapse-worker-llama`, `ck-synapse-worker-cuda`, `ck-synapse-worker-decode`, and Swift sidecar `ane-prefill-sidecar`). The host speaks to them over UNIX domain sockets or Windows named pipes using a fast binary framing protocol.
- **Content-Addressed Cache & Durable Jobs:** Persistent SQLite storage manages model downloading (with concurrent shared-lease readers and a two-phase GC), machine capability probing, alias translation, and restartable generation requests (tracking execution/retention TTLs and checkpointed pages).
- **Serial Execution under Idle-Gate Constraints (Bench Harness):** Prevent measurement contamination by ensuring the host machine is idle (average CPU <= 15%, GPU <= 5% for 6 seconds) before starting any evaluation run.
- **Self-Contained Execution Lanes (Bench Harness):** Separate binaries or runtime environments for each target evaluate hardware backends before promoting them to production workers.
- **Numerical Parity Auditing (Bench Harness):** Quantify accuracy drift across acceleration targets by calculating the mean cosine similarity of generated embeddings against a CPU-based `ort` (ONNX Runtime) reference lane.
- **Retrieval Quality and Reranking Parity Auditing:** Assess retrieval quality using offline evaluation datasets (COSQA, CodeSearchNet-Python) from the CoIR suite. Reranking workloads compare candidate scores against reference Alibaba-NLP/gte-reranker-modernbert-base scores to evaluate score drift and rank stability.
- **Native Constrained Decoding (Spike):** Restrict causal generation sequences to a schema/grammar (e.g. JSON or JSON Schema) using a vocabulary-specific bitset mask on logits, ensuring token-by-token structural compliance before token commitment.

## Layers

**Synapse SubC Module (`synapse-module`):**
- Purpose: The main service listening on the SubC bus. Handles route binding, job admission, the model cache, remote provider dispatch, worker lifecycle supervision (offloading worker engine drops to dedicated threads), approval storage and identity-based rollback (`rollback.rs`), runtime admission probe health, storage epochs and rotation ledgers, owned CUDA evidence and declared identities, and in-process execution via the owned engine.
- Location: `crates/synapse-module`
- Contains: A 3-class aging scheduler, SQLite durable job and cache lease state, machine probe certification logic, socket/pipe-based worker host, the remote gateway client, module-side routing (`owned-decode-routing` including ANE split prefill routing `owned-decode-routing/ane_prefill.rs`), grammar compilation and DECODE scheduler (`owned-decode-grammar-scheduler`), certification gates and probes (`owned-decode-certification`), approval rollback (`rollback.rs`), contract manifests (`owned-decode-manifests`), request-scoped semantic-sidecar hint bank normalization and per-field slotting (`owned-decode-sidecar`), and direct bindings to `synapse-engine-owned` and `synapse-engine-cuda`.
- Depends on: `synapse-core`, `synapse-engine-owned`, `synapse-engine-cuda`, `subc-client-rs`, `rusqlite`, `tokio`.

**Remote Gateway (`crates/synapse-module/src/remote`):**
- Purpose: Executes remote provider dispatch through interactive-first turnover pools, circuit breakers, and loopback-verified clients.
- Location: `crates/synapse-module/src/remote`
- Contains: `ProviderRuntime`, client dispatch, vault credential management via the `claustrum` SubC route, HTTP validators, mock provider e2e, and checkpoint-driven continuity logic.
- Depends on: `synapse-core`, `subc-client-rs`, `reqwest`.

**Synapse Owned Engine (`synapse-engine-owned`):**
- Purpose: Primary in-process execution engine for Apple Silicon (macOS), providing exact-match Metal MPSGraph inference for ModernBERT, Qwen3, and MiniLM models, direct Metal step decode engines for Qwen3 and LFM2, supervised decode worker state management, and ModernBERT pair reranking (`rerank_pairs`).
- Location: `crates/synapse-engine-owned`
- Contains: Rust-to-Objective-C bindings, Metal shader graphs (including macOS 15+ `@available`-guarded fused scaled-dot-product attention for ModernBERT with `GRAPH_REVISION` package cache invalidation), direct Metal step decode kernels and models (`owned-decode-engine`), supervised decode worker protocol, boundary, crash budget, sidecar hint bank installation protocol, and supervision state machine (`owned-decode-worker`), and tensor operations for embedding and reranking (`crates/synapse-engine-owned/src/modernbert.rs`). The module stays the sole tokenizer owner; this engine strictly consumes canonical token IDs and executes tensor logic.
- Depends on: `synapse-core`, `safetensors`, `half`, Apple's `Metal` and `MPSGraph` frameworks.
- Used by: `synapse-module` as the primary local engine.

**Synapse CUDA Engine (`synapse-engine-cuda`):**
- Purpose: Primary in-process CUDA execution engine (`owned-cuda-v1`), providing PTX virtual arch `compute_75` (Compute Capability 7.5+ floor, CUDA Driver API 12.040+) inference for MiniLM, GTE-ModernBERT, and Qwen3 models in f16 storage dtype.
- Location: `crates/synapse-engine-cuda`
- Contains: C++/CUDA PTX kernel ports (byte-identical to `unified-rt`), CUDA graphs support, precision-aware embedding execution (`OwnedCudaEmbedEngine`), model family detection (`config.json`), and hardware capability floor verification (`device_meets_floor`).
- Depends on: `synapse-core`, `safetensors`, `half`, `sha2`, CUDA toolkit/driver libraries.
- Used by: `synapse-module` and `synapse-worker-cuda`.

**Synapse Worker Lanes (`synapse-worker-*`):**
- Purpose: Execute in-memory tokenization, tensor forward passes, and token generation for specific hardware classes (Apple Silicon MLX, Apple Neural Engine, Llama GGUF, NVIDIA CUDA, and supervised Metal decode).
- Location: `crates/synapse-worker-mlx`, `crates/synapse-worker-ane`, `crates/synapse-worker-llama`, `crates/synapse-worker-cuda`, `crates/synapse-worker-decode`, `workers/ane-prefill-sidecar`
- Contains: Metal-accelerated customized MLX models, CoreML graphs (including the `gte-modernbert` embedder and reranker for the ANE quiet-tier via `ane-coreml-worker`), `llama.cpp` inference processes, supervised owned CUDA runner (`ck-synapse-worker-cuda`) executing MiniLM, ModernBERT, and Qwen3 embedding batches over IPC, supervised owned Metal decode runner (`ck-synapse-worker-decode`) executing Qwen3 and LFM2 token generation under progress/continuation framing and sidecar hint bank installation, and supervised Swift ANE prefill sidecar (`ane-prefill-sidecar`) executing fixed-window CoreML prefill passes.
- Depends on: `synapse-core`, `owned-decode-worker`, `synapse-engine-owned`, `mlx-rs`, `coreml` (via Swift), `reqwest`.
- Used by: The `synapse-module` host spawning them dynamically based on user requests and capability tiers.

**Synapse Core Abstractions (`synapse-core`):**
- Purpose: Core vocabulary structs, engine traits, machine capability profiles, and error contracts shared between the host and its workers.
- Location: `crates/synapse-core`
- Contains: `WorkerHello` handshake with strict catalog engine identity validation, shared canonical HELLO engine identities (`worker_engine_names.rs`), binary framing logic, `EngineError` contract, `MachineProfile` with `ane_subtype` chip-identity mapping, `RuntimeConfig`, `TokenBatch`, per-request decode chain policy, request-scoped sidecar specification contracts (`sidecar_spec.rs`), and scheduling traits.

**Benchmark Harness Core:**
- Purpose: Provides CLI commands for corpus generation, power-monitored process wrapping, result schema definition, and numerical parity functions.
- Location: `bench/harness`
- Contains: CLI entry parsing, idle-gating checks, telemetry collection wrapping, JSONL dataset loading, and cosine similarity calculations.
- Depends on: `clap`, `serde`, `serde_json`, `tokenizers`, `reqwest`.
- Used by: All inference lanes (compiled as the `synapse-bench` library dependency).

**Benchmark Measurement Rig (`synapse-rig`):**
- Purpose: A hash-pinned external measurement harness split out of the candidate tree. Drives candidate inference as a subprocess to guarantee strict execution walls, exact tokenizer application, canonical token accounting, and un-tampered semantic parity metrics.
- Location: `bench/rig`
- Contains: Length-prefixed JSON stdio framing protocol (`bench/harness/src/rig_protocol.rs`), exact-shape tokenizer constraints, canonical throughput calculation, and result schema enforcement.
- Depends on: `synapse-core`, `tokenizers`, `serde_json`.
- Used by: All modern lane runners evaluating throughput, correctness, or parity on candidate backends.

**Native Engine Inference Lanes:**
- Purpose: Execute in-memory tokenization, tensor forward passes, and pooling over target platforms.
- Location: `bench/lanes/ort-embed`, `bench/lanes/mlx`, `bench/lanes/burn`, `bench/lanes/mlx-minilm`, `bench/lanes/ts-embed`, `bench/lanes/potion`, `bench/spikes/unified-rt`, `bench/spikes/ane-prefill-split`
- Contains: Bounded-thread ONNX Runtime embedding logic, Metal-accelerated MLX custom model implementations, unified-rt candidate implementations (Vulkan cooperative-matrix/plain shaders on RDNA3 with device-local memory staging, budget validation, subgroup-parallel RMSNorm, vectorized loads, Q8 block-address hoisting, f16/Q8 pack-four subgroup rows, and batched mat-mat compute shaders in `bench/spikes/unified-rt/src/qwen3_decode_vulkan.rs`, CUDA cuBLASLt fused graphs and fused QK norm RoPE single-launch kernels on NVIDIA, Metal graph execution optimization levels O0/O1, package caching, true batched speculative verification on `bench/spikes/unified-rt/src/qwen3_decode_metal_step.rs`, and custom direct Metal step kernels for Qwen3 and LFM2 with device-resident conv-cache and Q8_0 hybrid engine in `bench/spikes/unified-rt/src/lfm2_decode_metal_step.rs`), ANE prefill and Metal decode split measurement (`bench/spikes/ane-prefill-split`), LFM2 hybrid causal backbone, LFM2-Audio ASR speech encoder (FastConformer and Slaney mel filterbank frontend), Qwen3-0.6B f16 Metal decode throughput optimizations, WGPU-based Burn ONNX imports, python-based MLX community/source loading, Model2Vec static embedding (`potion-code-16M`), and TypeScript setups.
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
- Purpose: Standalone Bun/TypeScript data generation and SFT training pipeline for the production gatherer contract, supporting both Anthropic API (with multi-account OAuth rotation) and local OpenAI-compatible endpoints.
- Location: `tools/gather-distill`
- Contains: Trajectory generation, work queue handling, `AftClientPool` process wrapping, validation, gold-overlap scoring, zero-shot gatherer bake-off evaluation leaderboards (`tools/gather-distill/BAKEOFF-ZEROSHOT.md`), Axolotl SFT training configs/rungs (`tools/gather-distill/train/`), and student evaluation scale ladder metrics (`tools/gather-distill/train/SCALE-LADDER.md`).
- Depends on: Bun, pinned `aft-v0.46.0` binary, and `@cortexkit/anthropic-auth-core`.
- Used by: Developers running qgen, gather, validate, score, or model distillation and SFT evaluation campaigns.

**Athena Classify Distillation Harness:**
- Purpose: Standalone Bun/TypeScript dataset generation and classification runner for the local `Athena-classify` student model.
- Location: `tools/classify-distill`
- Contains: Vendored ALF rust/ts contracts (sha-pinned provenance), real-export importer (`tools/classify-distill/src/importer.ts`), histogram-driven synthetic qgen (`tools/classify-distill/src/qgen.ts`), mechanical validator port (`tools/classify-distill/src/validator.ts`), multi-account OAuth Anthropic runner with dry-run/mock gates (`tools/classify-distill/src/runner.ts`), and contract parity verification (`tools/classify-distill/src/parity.ts`).
- Depends on: Bun, `claude-sonnet-5` (qgen), `claude-opus-4-8` (run default), and `@cortexkit/anthropic-auth-core`.
- Used by: Developers running Athena classify dataset generation, real attempt importing, parity auditing, or distillation campaigns.

**Synapse Operator CLI (`synapse-opctl`):**
- Purpose: Drive Synapse operations (models catalog, probe runs, scheduling admission stats, approval enablement and rollbacks, batch embedding, and jobs paging) through the fleet subc daemon connection.
- Location: `crates/synapse-opctl`
- Contains: CLI command parsing and formatting logic for operator management, including model status, probe execution, scheduler admission, approval migration, explicit enablement, disablement, emergency rollback, batch submission, and paged results.
- Depends on: `subc-client-rs`, `clap`, `serde_json`, `tokio`.
- Used by: Operators and deployment scripts monitoring or triggering runtime actions.

**Management Call Utility (`subc-call`):**
- Purpose: Send raw method calls and JSON params to any module over the fleet daemon.
- Location: `crates/synapse-module/src/bin/subc_call.rs`
- Contains: Direct IPC client call wrapping, `--identity` override flag support for stamping chair-verb credentials on consumer binds, and formatted envelope printing.
- Depends on: `subc-client-rs`, `tokio`, `serde_json`.
- Used by: Developers and scripts executing low-level management surface functions.

**Campaign Harnesses:**
- Purpose: Execute and coordinate sandboxed evaluation campaigns (single-stream decode, Metal direct step, CUDA quantization, LFM2 CUDA Q8, and Metal embedding).
- Location: `bench/campaign`
- Contains: Integrity validation of model snapshots, fixtures, and target runners; deterministic verification of intervention hooks; candidate-owned temporary workspace staging and build output/target directories; toolchain environment forwarding (`RUSTUP_HOME`, `CARGO_HOME`); split-stream append-mode logging (`.log` and `.log.stderr`); and failure scene preservation.
- Depends on: `spike-unified-rt` runner.
- Used by: Automated evaluation gates to confirm decode and embedding performance and correctness.


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

1. Module parses and validates input JSON schemas (`synapse-json-schema-v1`), enforces grammar limits, compiles a byte-level JSON automaton, and converts it into a `TokenIdJsonConstraintV1` structure shipped directly to the worker (`crates/synapse-module/owned-decode-grammar-scheduler/grammar_compile.rs`).
2. Decode requests enter the dedicated `QueueClass::Decode` scheduler using N-token quantum sequencing (batched 16-token chunks with yield-on-contention release) — `crates/synapse-module/owned-decode-grammar-scheduler/scheduler.rs`. Decode requests forward optional per-request chain policy (`chain_k`) through execution envelopes.
3. Worker extracts logit values for the next token from causal decode execution (Qwen3 or LFM2 Metal step engines).
4. Query constraint state machine (`JsonParser` / automaton) to match allowed byte sequences against the token vocabulary trie and compute the vocabulary-wide bitset `TokenMask`.
5. Apply the `TokenMask` to the logits (forcing unallowed token logits to negative infinity).
6. Select the next token from masked logits, advance constraint parser state, and yield progress or final frame.

**ANE Split Prefill and Decode Flow:**

1. Decode requests with `DecodePrefill::AneSplit` enter `AnePrefillRouter` (`crates/synapse-module/owned-decode-routing/ane_prefill.rs`), which evaluates global gates (platform support, Qwen3 family, greedy top-1 sampling, identity pins) and selects the smallest fitting fixed-window prefill bucket (`W128`, `W256`, `W512`).
2. The router verifies split-arm health (`SplitArmHealth`), checks deadline feasibility against calibrated p95 budgets (`SplitTimingBudgets`), acquires an ANE execution guard, and issues an `EXECUTE` command to `ane-prefill-sidecar` (`workers/ane-prefill-sidecar/`).
3. `ane-prefill-sidecar` executes CoreML prediction on `CPU_AND_NE`, emitting f32 logits sampled at `active_tokens - 1` and f16 KV cache frames.
4. `ck-synapse-worker-decode` ingests the KV cache frame and hands off execution to Metal step decode kernels for token generation. If pre-attempt or execution failures occur, the router tags response provenance with closed bypass (`PrefillBypassReason`) or fallback (`PrefillFallbackReason`) categories.


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

**Athena Classify Distillation Flow:**

1. Import real ALF export attempts into candidate gold/reject rows based on attempt state and `accepted_plan_json` labels — `tools/classify-distill/src/importer.ts`
2. Generate synthetic request prose with class priors from consult histograms via Sonnet-5 — `tools/classify-distill/src/qgen.ts`
3. Execute dry-run or Opus classification runs using static prompt caching (~500 tokens prefix), multi-account OAuth rotation, and mechanical validation against vendored ALF contracts — `tools/classify-distill/src/runner.ts`, `tools/classify-distill/src/validator.ts`
4. Split valid responses to gold JSONL and invalid or failed classifications to reject JSONL with validation errors for evaluation — `tools/classify-distill/src/runner.ts`

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
- Pattern: Binary Serialization (e.g., `decode_f32_frame`, `encode_i32_frame`) with host-side catalog engine identity validation during `WorkerHello` handshake.

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

**Classify Validator & Contract Port:**
- Purpose: Enforce strict output JSON compliance against vendored ALF Rust/TS contract defaults, aliases, optional fields, unknown-field tolerance, route-specific checks, and council class resolver.
- Location: `tools/classify-distill/src/validator.ts`
- Pattern: Structural schema validator and intent resolver.

**Grammar Compiler & Automaton:**
- Purpose: Exclusively compiles `synapse-json-schema-v1` JSON schemas into byte-level JSON automata (`TokenIdJsonConstraintV1`) and enforces checked-in structural limits without exposing raw schema structures across the worker boundary.
- Location: `crates/synapse-module/owned-decode-grammar-scheduler/mod.rs`
- Pattern: Byte-level automaton compilation and vocabulary bitset indexing.

**Decode Scheduler & Quantum Sequencer:**
- Purpose: Dedicated DECODE queue scheduler with weighted boundary arbitration, oldest-anchor aging, execution permits with yield-on-contention release, and N-token (N=16) quantum sequencing for owned generation workloads.
- Location: `crates/synapse-module/owned-decode-grammar-scheduler/scheduler.rs`
- Pattern: Quantum-bounded state machine scheduler.

**Owned Decode Supervisor & Protocol:**
- Purpose: Pure-Rust state machine supervising `ck-synapse-worker-decode` over `owned-metal-decode-worker-v1` IPC, managing sequence/session validation, terminal-control boundary precedence, crash-budget persistence/quarantine, and single-crash token-zero restart.
- Location: `crates/synapse-engine-owned/owned-decode-worker/src/supervisor.rs`
- Pattern: Worker lifecycle supervisor with crash budget and quarantine state.

**Owned Decode Sidecar & Hint Bank:**
- Purpose: Pure data handling module and shared contracts normalizing request-scoped semantic-sidecar results, building target-tokenizer hint banks (`SidecarHintBank`) for non-blocking suffix-match pickup during target decoding, managing per-field layout plan slotting (`PerFieldPlan`), and classifying sidecar outcome precedence (`SidecarOutcome`).
- Location: `crates/synapse-module/owned-decode-sidecar/mod.rs`, `crates/synapse-core/src/sidecar_spec.rs`
- Pattern: Data normalization, layout rendering policy, and tokenizer hint bank indexing.

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
- Purpose: Execute single-token, batched speculative verification (`verify_batch`), or GPU-chained multi-token Qwen3 decode steps bypassing MPSGraph via direct Metal compute kernels (`bench/spikes/unified-rt/src/qwen3_decode_metal_step.rs`, `bench/spikes/unified-rt/src/qwen3_decode_metal_step.m`, `bench/spikes/unified-rt/src/qwen3_decode_metal_step.metal`), leveraging SIMDgroup RMSNorm, position-parallel attention, GPU-side token argmax gathering, and Q8 GEMV routines. Supports opt-in multi-token command buffer chaining via `SYNAPSE_METAL_STEP_CHAIN_K` (default 1) and batched verification via `SYNAPSE_METAL_STEP_BATCHED_VERIFY=1`.
- Location: `bench/spikes/unified-rt/src/qwen3_decode_metal_step.rs`
- Pattern: Direct Metal Compute Kernel Pipeline.

**LFM2 Metal Step Engine:**
- Purpose: Execute LFM2-1.2B hybrid decode steps bypassing MPSGraph via direct Metal compute kernels (`bench/spikes/unified-rt/src/lfm2_decode_metal_step.rs`, `bench/spikes/unified-rt/src/lfm2_decode_metal_step.m`, `bench/spikes/unified-rt/src/lfm2_decode_metal_step.metal`), combining a device-resident short-convolution rolling conv-cache kernel (`lfm2_conv_step`) with reused Qwen3 attention/matvec/RMSNorm kernels and Q8_0 GEMV routines. Gated via two-tier M1 authority signature and structural band invariants.
- Location: `bench/spikes/unified-rt/src/lfm2_decode_metal_step.rs`
- Pattern: Direct Metal Compute Kernel Pipeline with Rolling Conv-Cache.

**OwnedCudaEmbedEngine & Worker:**
- Purpose: Execute CUDA PTX embedding inference for MiniLM, ModernBERT, and Qwen3 in f16 storage dtype across in-process and supervised out-of-process worker configurations.
- Location: `crates/synapse-engine-cuda/src/lib.rs`, `crates/synapse-worker-cuda/src/main.rs`
- Pattern: PTX Kernel Dispatch with CUDA Graph Execution and Hardware Capability Floor (`device_meets_floor`).

**Approval & Emergency Rollback:**
- Purpose: Manages storage approvals, rotation ledgers, explicit `(model_id, decode_fingerprint)` enablement and disablement, and atomic single-transaction emergency rollbacks to instantly revoke serving approvals across all lanes.
- Location: `crates/synapse-module/src/rollback.rs`, `crates/synapse-module/src/store.rs`
- Pattern: Identity-Based Approval Ledger with Atomic Rollback Transaction.

**Worker HELLO Engine Names:**
- Purpose: Centralized canonical worker identity constants preventing identity drift during worker handshakes.
- Location: `crates/synapse-core/src/worker_engine_names.rs`
- Pattern: Shared Identity Constants (`LLAMA_WORKER_ENGINE`, `DECODE_WORKER_ENGINE`, `CUDA_WORKER_ENGINE`, etc.).

**ANE Prefill Router & Split Arm Health:**
- Purpose: Pure routing boundary selecting certified fixed-window CoreML prefill arms (`W128`, `W256`, `W512`), deriving attempt budgets from p95 calibration, managing consecutive-strike quarantine health (`SplitArmHealth`), and mapping closed bypass (`PrefillBypassReason`) and fallback (`PrefillFallbackReason`) provenance.
- Location: `crates/synapse-module/owned-decode-routing/ane_prefill.rs`
- Pattern: Pure routing boundary with p95 attempt budgets, consecutive-strike quarantine, and closed provenance tracking.

**ANE Prefill Sidecar:**
- Purpose: Separately supervised Swift/CoreML process managing fixed-window Qwen3 prefill execution over UNIX domain sockets.
- Location: `workers/ane-prefill-sidecar/`
- Pattern: Out-of-process CoreML stage supporting SHA-verified model loading (`INSTALL`), fixed-window token execution on `CPU_AND_NE` (`EXECUTE`), f32 logits and f16 KV cache streaming, and in-flight prediction aborts (`ABORT`).


## Entry Points

**Synapse Module Main (`ck-synapse`):**
- Location: `crates/synapse-module/src/main.rs`
- Triggers: Starts the primary SubC worker process.
- Responsibilities: DB initializations, environment bootstrapping, SubC binding registrations, and polling the scheduler.

**Worker Binaries (`ck-synapse-worker-*`):**
- Location: `crates/synapse-worker-*/src/main.rs`
- Triggers: Spawned directly by `crates/synapse-module/src/worker_host/mod.rs`.
- Responsibilities: Initializing accelerator graphs/sessions (MLX, ANE, Llama, supervised Metal decode), pipe/socket handshaking, loop listening for compute requests, returning tensors or generated token frames.

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

**Athena Classify Distillation CLI (`classify-distill`):**
- Location: `tools/classify-distill/src/cli.ts`
- Triggers: Invocation of Bun running `tools/classify-distill/src/cli.ts` commands (`import`, `qgen`, `run`, `parity`, `--dry-run`).
- Responsibilities: Routes commands to import real attempt exports, generate synthetic request prose, verify contract parity against ALF, and execute Opus/Haiku classification runs.

**Synapse Operator CLI (`synapse-opctl`):**
- Location: `crates/synapse-opctl/src/main.rs`
- Triggers: Execution of the `ck-synapse-opctl` binary.
- Responsibilities: Routes commands to list models, view scheduler stats, start probes, manage approval migrations/enablements/disablements, run batches, and fetch paged job results.

**Management Surface SubC Caller (`subc-call`):**
- Location: `crates/synapse-module/src/bin/subc_call.rs`
- Triggers: Execution of the `subc_call` binary.
- Responsibilities: Connects to fleet daemon and sends management calls directly to target modules (supporting `--identity` override for chair-only wire ops).

**Inline Embedding Throughput Client (`inline_embed_throughput`):**
- Location: `crates/synapse-module/src/bin/inline_embed_throughput.rs`
- Triggers: Execution of `inline_embed_throughput` binary.
- Responsibilities: Evaluates local batch throughput and query concurrency/latencies under load.

**Constraint Compiler (`compile_constraint`):**
- Location: `crates/synapse-worker-decode/src/bin/compile_constraint.rs`
- Triggers: Execution of `compile_constraint` binary.
- Responsibilities: Compiles JSON Schema grammars into wire-serializable `TokenIdJsonConstraint` structures for worker distribution.

**ANE Prefill Sidecar Binary (`ane-prefill-sidecar`):**
- Location: `workers/ane-prefill-sidecar/Sources/AnePrefillSidecarExecutable/main.swift`
- Triggers: Spawned by host worker supervision during ANE prefill execution.
- Responsibilities: Loads compiled CoreML prefill packages (`INSTALL`), executes fixed-window prediction on `CPU_AND_NE` (`EXECUTE`), streams f32 logits and f16 KV cache frames, and acknowledges in-flight cancels (`ABORT`).

**Campaign Harnesses:**
- Location: `bench/campaign/decode-harness.sh`, `bench/campaign/metal-step-harness.sh`, `bench/campaign/lfm2-cuda-harness.sh`, `bench/campaign/metal-embed-harness.sh`
- Triggers: Invocation of campaign harness scripts by evaluation runners.
- Responsibilities: Manage snapshot validation, locked candidate execution sandbox, candidate-owned workspace staging, toolchain environment forwarding, correctness verification, split-stream append-mode logging, failure scene preservation, and performance evaluation.


## Error Handling

**Strategy:** Fail-fast utilizing `anyhow::Result` and typed subsystem errors (`SubcModuleError`, `EngineError`) with contextual layers.
- **Worker Crash Domain:** If a worker binary crashes, deadlocks, or hangs, the `synapse-module` supervisor reclaims the job. Workers isolate dirty driver states, preventing host process termination. Host worker engine teardown runs on a dedicated thread off the runtime-driving threads to prevent async runtime panics on drop.
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
