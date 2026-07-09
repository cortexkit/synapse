# Architecture

## Pattern Overview

**Overall:** Serial, Idle-Gated Multi-Lane Benchmarking Harness AND the Production Synapse Engine runtime.

**Key Characteristics:**
- **Local Inference Service:** The primary production system (`synapse-module`) acts as a persistent SubC node that receives embedding, generation, and reranking requests. It routes work via a 3-class fair-share aging scheduler to underlying engine lanes.
- **Hardware-Specific Workers:** Model inference runs outside the host process via supervised binary children (`synapse-worker-mlx`, `synapse-worker-ane`, `synapse-worker-llama`). The host speaks to them over UNIX domain sockets using a fast binary framing protocol.
- **Content-Addressed Cache & Durable Jobs:** Persistent SQLite storage manages model downloading (with concurrent shared-lease readers and a two-phase GC), machine capability probing, alias translation, and restartable generation requests.
- **Serial Execution under Idle-Gate Constraints (Bench Harness):** Prevent measurement contamination by ensuring the host machine is idle (average CPU <= 15%, GPU <= 5% for 6 seconds) before starting any evaluation run.
- **Self-Contained Execution Lanes (Bench Harness):** Separate binaries or runtime environments for each target evaluate hardware backends before promoting them to production workers.
- **Numerical Parity Auditing (Bench Harness):** Quantify accuracy drift across acceleration targets by calculating the mean cosine similarity of generated embeddings against a CPU-based `ort` (ONNX Runtime) reference lane.
- **Retrieval Quality and Reranking Parity Auditing:** Assess retrieval quality using offline evaluation datasets (COSQA, CodeSearchNet-Python) from the CoIR suite. Reranking workloads compare candidate scores against reference Alibaba-NLP/gte-reranker-modernbert-base scores to evaluate score drift and rank stability.

## Layers

**Synapse SubC Module (`synapse-module`):**
- Purpose: The main service listening on the SubC bus. Handles route binding, job admission, the model cache, and worker lifecycle supervision.
- Location: `crates/synapse-module`
- Contains: A 3-class aging scheduler, SQLite durable job and cache lease state, machine probe certification logic, and socket-based worker host.
- Depends on: `synapse-core`, `subc-client-rs`, `rusqlite`, `tokio`.

**Synapse Worker Lanes (`synapse-worker-*`):**
- Purpose: Execute in-memory tokenization, tensor forward passes, and pooling for specific hardware classes (Apple Silicon MLX, Apple Neural Engine, Llama GGUF).
- Location: `crates/synapse-worker-mlx`, `crates/synapse-worker-ane`, `crates/synapse-worker-llama`
- Contains: Metal-accelerated customized MLX models, CoreML graphs, and `llama.cpp` inference processes.
- Depends on: `synapse-core`, `mlx-rs`, `coreml` (via Swift), `reqwest`.
- Used by: The `synapse-module` host spawning them dynamically based on user requests and capability tiers.

**Synapse Core Abstractions (`synapse-core`):**
- Purpose: Core vocabulary structs, engine traits, and error contracts shared between the host and its workers.
- Location: `crates/synapse-core`
- Contains: `WorkerHello` handshake and binary framing logic, `EngineError` contract, `RuntimeConfig`, `TokenBatch`, and scheduling traits.

**Benchmark Harness Core:**
- Purpose: Provides CLI commands for corpus generation, power-monitored process wrapping, result schema definition, and numerical parity functions.
- Location: `bench/harness`
- Contains: CLI entry parsing, idle-gating checks, telemetry collection wrapping, JSONL dataset loading, and cosine similarity calculations.
- Depends on: `clap`, `serde`, `serde_json`, `tokenizers`, `reqwest`.
- Used by: All inference lanes (compiled as the `synapse-bench` library dependency).

**Native Engine Inference Lanes:**
- Purpose: Execute in-memory tokenization, tensor forward passes, and pooling over target platforms.
- Location: `bench/lanes/ort-embed`, `bench/lanes/mlx`, `bench/lanes/burn`, `bench/lanes/mlx-minilm`, `bench/lanes/ts-embed`, `bench/lanes/potion`
- Contains: Bounded-thread ONNX Runtime embedding logic, Metal-accelerated MLX custom model implementations (including Qwen3's attention layers and length-uniform batch sorting), WGPU-based Burn ONNX imports, python-based MLX community/source loading, Model2Vec static embedding (`potion-code-16M`), and TypeScript (Transformers.js or native `onnxruntime-node`) setups.
- Depends on: `bench/harness` (for Rust crates), target runtime libraries (`ort`, `mlx-rs`, `burn`, `@huggingface/transformers`, `mlx-embeddings`, `model2vec`), and `tokenizers`.
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

## Data Flow

**Production Inference Flow:**

1. Route request received via SubC — `crates/synapse-module/src/lib.rs`
2. Validate alias surfaces, apply machine capability capability profiles (Perf/Quiet tiers), and admit into job table — `crates/synapse-module/src/store.rs`
3. Download/Verify models through content-addressed cache with shared leases — `crates/synapse-module/src/store.rs`
4. Admit task to the Fair-Share Aging Scheduler (3-class: Embed, Generation, System) — `crates/synapse-core/src/scheduler.rs`
5. Handshake and spawn the appropriate Worker lane — `crates/synapse-module/src/worker_host.rs`
6. Submit frames over UNIX domain socket, accumulate partial outputs or vector results — `crates/synapse-core/src/worker_protocol.rs`
7. Mark durable job completed and return response envelope to SubC — `crates/synapse-module/src/lib.rs`

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

1. Parse arguments, load weights/servers, configure tokenizers, and execute a warmup batch — `bench/lanes/*/src/main.rs` (Rust), `bench/lanes/mlx-minilm/main.py` (Python), `bench/lanes/ts-embed/main.mjs` (JS), `bench/lanes/potion/main.py` (Python)
2. Divide inputs and run model forward execution under token-budget or attention-unit batch limits (optionally sorting inputs by token count to prevent padding waste) — `bench/lanes/*/src/main.rs` (Rust), `bench/lanes/mlx-minilm/main.py` (Python), `bench/lanes/ts-embed/main.mjs` (JS), `bench/lanes/potion/main.py` (Python)
3. Calculate mean cosine similarity of produced output vectors against the `ort-cpu` baseline reference — `bench/harness/src/parity.rs`
4. Calculate top-k neighbor overlap metrics to check for rank stability against the reference lane — `bench/harness/src/parity.rs`
5. Write results structured in the `LaneResult` schema to the output results JSON — `bench/harness/src/results.rs`

**CoIR Retrieval Evaluation Flow:**

1. Prepare evaluation task files (COSQA and CodeSearchNet-Python) into uniform JSONL shape (queries and corpus) — `bench/eval-coir/prepare.py`
2. Run target inference lanes with document/query prefixes to generate vector outputs — `bench/lanes/*/src/main.rs`, `bench/lanes/potion/main.py`, etc.
3. Execute brute-force cosine retrieval and calculate metrics (MRR@10, NDCG@10, Recall@10) — `bench/eval-coir/score.py`

**Reranking Quality Check Flow:**

1. Spawn `llama-server` with `--rerank` argument — `bench/lanes/llama/src/main.rs`
2. Submit batch query and document pairs to the `/v1/rerank` endpoint, accumulating server-reported prompt token counts — `bench/lanes/llama/src/main.rs`
3. Generate reference scores using Hugging Face reference implementation (`Alibaba-NLP/gte-reranker-modernbert-base`) — `bench/eval-coir/reference_rerank.py`
4. Compare candidate rerank scores against the reference scores to calculate delta/drift — `bench/eval-coir/compare_rerank_scores.py`

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

## Entry Points

**Synapse Module Main:**
- Location: `crates/synapse-module/src/main.rs`
- Triggers: Starts the primary SubC worker process.
- Responsibilities: DB initializations, environment bootstrapping, SubC binding registrations, and polling the scheduler.

**Worker Binaries:**
- Location: `crates/synapse-worker-*/src/main.rs`
- Triggers: Spawned directly by `synapse-module/src/worker_host.rs`.
- Responsibilities: Initializing accelerator graphs/sessions (MLX, ANE, Llama), socket handshaking, loop listening for compute requests, returning tensors.

**Bench Harness CLI (`synapse-bench`):**
- Location: `bench/harness/src/main.rs`
- Triggers: Execution of the `synapse-bench` binary.
- Responsibilities: Routes commands to either chunk source files into a corpus, execute telemetry-monitored child commands, or calculate top-k neighbor rank-overlap parity.

**Inference Lane Runners:**
- Location: `bench/lanes/ort-embed/src/main.rs`, `bench/lanes/wrap-embed/src/main.rs`, `bench/lanes/llama/src/main.rs`, `bench/lanes/mlx/src/main.rs`, `bench/lanes/burn/src/main.rs` (Rust crates); `bench/lanes/mlx-minilm/main.py` (Python script); `bench/lanes/ts-embed/main.mjs` (TypeScript script); `bench/lanes/potion/main.py` (Python script)
- Triggers: Invocation by the power wrapper or direct script executions.
- Responsibilities: Model initialization, cold-load timing tracking, batched inference execution, and vector/result output generation.

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

## Error Handling

**Strategy:** Fail-fast utilizing `anyhow::Result` and typed subsystem errors (`SubcModuleError`, `EngineError`) with contextual layers.
- **Worker Crash Domain:** If a worker binary crashes, deadlocks, or hangs, the `synapse-module` supervisor reclaims the job. Workers isolate dirty driver states, preventing host process termination.
- **Durable Job Resiliency:** Jobs track their generation cycles. Crash-interrupted requests can be recovered via idempotent request keys if the host restarts.
- **SubC Communication:** Submodule failures strictly return properly formatted error envelopes detailing the specific layer failure (e.g., CacheMiss, EngineOOM).

**Bench Harness Strategy:** Fail-fast utilizing `anyhow::Result` error propagation with contextual layers (`.context()`).
- **Child Supervision:** Spawned subprocesses (`llama-server`) are tracked via PID. If a child dies or fails to bind to its designated port within `HEALTH_TIMEOUT` (120s), the lane runner fails immediately rather than silently hanging. Platform-specific process control signals (such as SIGTERM on Unix) are gated appropriately so subprocess lifecycles function seamlessly on both Windows and Unix platforms.
- **HTTP Resiliency:** Requests to external wrapping endpoints (`wrap-embed`) implement read timeouts, connect timeouts, and bounded retry loops with backoff to recover from transient rate limits or cold-load stalls.

## Cross-Cutting Concerns

**Logging:** Console outputs are printed directly. Matrix status tracking, parameters, and outputs write directly to `bench/results/matrix.log`.
**Caching:** Model files are located from HuggingFace cache snapshots. Content-addressed downloads will follow atomic tmp+rename patterns in `~/.local/share/cortexkit/models/`.
**Storage:** Structured outputs are written under `bench/results/` as telemetry metrics (`.measure.json`), parity vectors (`-vectors.jsonl`), and results summary (`.json`) files.
