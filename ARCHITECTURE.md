# Architecture

## Pattern Overview

**Overall:** Serial, Idle-Gated Multi-Lane Benchmarking Harness.

**Key Characteristics:**
- **Serial Execution under Idle-Gate Constraints:** Prevent measurement contamination by ensuring the host machine is idle (average CPU <= 15%, GPU <= 5% for 6 seconds) before starting any run.
- **Self-Contained Execution Lanes:** Separate binaries or runtime environments for each target (`lane-ort-embed`, `lane-mlx`, `lane-llama`, `lane-burn`, `lane-wrap-embed`, `mlx-minilm`, `ts-embed`) compile/execute independently to avoid compile-time dependency leakage or driver pollution.
- **Numerical Parity Auditing:** Quantify accuracy drift across acceleration targets by calculating the mean cosine similarity of generated embeddings against a CPU-based `ort` (ONNX Runtime) reference lane. Perform rank-stability auditing (top-k neighbor overlap) to detect quantization-induced reordering.

## Layers

**Benchmark Harness Core:**
- Purpose: Provides CLI commands for corpus generation, power-monitored process wrapping, result schema definition, and numerical parity functions.
- Location: `bench/harness`
- Contains: CLI entry parsing, idle-gating checks, telemetry collection wrapping, JSONL dataset loading, and cosine similarity calculations.
- Depends on: `clap`, `serde`, `serde_json`, `tokenizers`, `reqwest`.
- Used by: All inference lanes (compiled as the `synapse-bench` library dependency).

**Native Engine Inference Lanes:**
- Purpose: Execute in-memory tokenization, tensor forward passes, and pooling over target platforms.
- Location: `bench/lanes/ort-embed`, `bench/lanes/mlx`, `bench/lanes/burn`, `bench/lanes/mlx-minilm`, `bench/lanes/ts-embed`
- Contains: Bounded-thread ONNX Runtime embedding logic, Metal-accelerated MLX custom model implementations (including Qwen3's attention layers and length-uniform batch sorting), WGPU-based Burn ONNX imports, python-based MLX community/source loading, and TypeScript (Transformers.js or native `onnxruntime-node`) setups.
- Depends on: `bench/harness` (for Rust crates), target runtime libraries (`ort`, `mlx-rs`, `burn`, `@huggingface/transformers`, `mlx-embeddings`), and `tokenizers`.
- Used by: The benchmark suite runners `bench/run-matrix.sh` and `bench/run-night.sh`.

**Supervised Child Server Lane:**
- Purpose: Spawns, monitors, and terminates a child server process (`llama-server`) and routes inference requests over standard HTTP endpoints.
- Location: `bench/lanes/llama`
- Contains: Supervised subprocess spawning, TCP port binding probes, `/health` API polling, and batched OpenAI-compatible API request orchestration.
- Depends on: `bench/harness`, `reqwest`, `serde`, `serde_json`.
- Used by: The benchmark suite runner `bench/run-matrix.sh`.

**External Service Wrapper Lane:**
- Purpose: Integrates and profiles pre-existing external inference servers (LMStudio, Ollama) that run out-of-process.
- Location: `bench/lanes/wrap-embed`
- Contains: HTTP request client, input pre-truncation, rate-limit backpressure handling, and external process name RSS sampling.
- Depends on: `bench/harness`, `reqwest`, `tokenizers`.
- Used by: The benchmark suite runner `bench/run-matrix.sh`.

## Data Flow

**Corpus Generation Flow:**

1. Scan files recursively from a source code tree — `bench/harness/src/main.rs`
2. Parse text and split contents into chunks constrained by token budgets — `bench/harness/src/corpus.rs`
3. Export structured chunks to a JSONL dataset containing IDs and chunk texts — `bench/harness/src/corpus.rs`

**Idle-Gated Power Telemetry Run:**

1. Monitor system activity using `macmon` and block execution until utilization satisfies idle thresholds — `bench/harness/src/metrics.rs`
2. Spawn the target command as a child process and initialize CPU, GPU, ANE watt telemetry, machine utilization (CPU/GPU avg/peak usage percentages), and RSS profiling — `bench/harness/src/metrics.rs`
3. Sample metrics periodically, tracking energy consumption in Joules — `bench/harness/src/metrics.rs`
4. Serialize telemetry data to an output metrics JSON file — `bench/harness/src/metrics.rs`

**Inference and Numerical Parity Check:**

1. Parse arguments, load weights/servers, configure tokenizers, and execute a warmup batch — `bench/lanes/*/src/main.rs` (Rust), `bench/lanes/mlx-minilm/main.py` (Python), `bench/lanes/ts-embed/main.mjs` (JS)
2. Divide inputs and run model forward execution under token-budget or attention-unit batch limits (optionally sorting inputs by token count to prevent padding waste) — `bench/lanes/*/src/main.rs` (Rust), `bench/lanes/mlx-minilm/main.py` (Python), `bench/lanes/ts-embed/main.mjs` (JS)
3. Calculate mean cosine similarity of produced output vectors against the `ort-cpu` baseline reference — `bench/harness/src/parity.rs`
4. Calculate top-k neighbor overlap metrics to check for rank stability against the reference lane — `bench/harness/src/parity.rs`
5. Write results structured in the `LaneResult` schema to the output results JSON — `bench/harness/src/results.rs`

## Key Abstractions

**LaneResult:**
- Purpose: Unified schema for reporting execution performance, throughput metrics, telemetry outputs, and parity calculations.
- Location: `bench/harness/src/results.rs`
- Pattern: Serializable Data Struct.

**Chunk / Prompt:**
- Purpose: Uniform representations of inputs for embedding (Chunk) and classification (Prompt) workloads.
- Location: `bench/harness/src/parity.rs`
- Pattern: Deserializable Data Structures.

**System Idle Gate:**
- Purpose: Preflight check to guard executions and prevent run telemetry contamination by background system tasks.
- Location: `bench/harness/src/metrics.rs`
- Pattern: Guard clause loop checking CPU and GPU utilization thresholds.

## Entry Points

**Bench Harness CLI (`synapse-bench`):**
- Location: `bench/harness/src/main.rs`
- Triggers: Execution of the `synapse-bench` binary.
- Responsibilities: Routes commands to either chunk source files into a corpus, execute telemetry-monitored child commands, or calculate top-k neighbor rank-overlap parity.

**Inference Lane Runners:**
- Location: `bench/lanes/ort-embed/src/main.rs`, `bench/lanes/wrap-embed/src/main.rs`, `bench/lanes/llama/src/main.rs`, `bench/lanes/mlx/src/main.rs`, `bench/lanes/burn/src/main.rs` (Rust crates); `bench/lanes/mlx-minilm/main.py` (Python script); `bench/lanes/ts-embed/main.mjs` (TypeScript script)
- Triggers: Invocation by the power wrapper or direct script executions.
- Responsibilities: Model initialization, cold-load timing tracking, batched inference execution, and vector/result output generation.

**Matrix Orchestration Script:**
- Location: `bench/run-matrix.sh`
- Triggers: Triggered by a developer to run the complete benchmarking suite.
- Responsibilities: Autodetect cached HuggingFace snapshots, sequence idle-gated executions of Qwen3, MiniLM, and LFM candidate lanes, check dependencies, and log to `bench/results/matrix.log`.

**Nightly Suite Orchestrator:**
- Location: `bench/run-night.sh`
- Triggers: Scheduled execution or manual trigger by developer.
- Responsibilities: Precondition validation, sequential idle-gated run of all 16 target lanes on the full AFT corpus, full-corpus parity and rank-overlap calculations, and archiving outputs under `bench/results/night-YYYYMMDD/`.

## Error Handling

**Strategy:** Fail-fast utilizing `anyhow::Result` error propagation with contextual layers (`.context()`).
- **Child Supervision:** Spawned subprocesses (`llama-server`) are tracked via PID. If a child dies or fails to bind to its designated port within `HEALTH_TIMEOUT` (120s), the lane runner fails immediately rather than silently hanging.
- **HTTP Resiliency:** Requests to external wrapping endpoints (`wrap-embed`) implement read timeouts, connect timeouts, and bounded retry loops with backoff to recover from transient rate limits or cold-load stalls.

## Cross-Cutting Concerns

**Logging:** Console outputs are printed directly. Matrix status tracking, parameters, and outputs write directly to `bench/results/matrix.log`.
**Caching:** Model files are located from HuggingFace cache snapshots. Content-addressed downloads will follow atomic tmp+rename patterns in `~/.local/share/cortexkit/models/`.
**Storage:** Structured outputs are written under `bench/results/` as telemetry metrics (`.measure.json`), parity vectors (`-vectors.jsonl`), and results summary (`.json`) files.
