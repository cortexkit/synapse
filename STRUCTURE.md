# Codebase Structure

## Directory Layout

```
[project-root]/
├── .alfonso/               # Alfonso AI spikes and planning artifacts
├── .cortexkit/             # Prompts, configurations, and historian logs
├── bench/                  # Main benchmarking workspace containing harness and lanes
│   ├── campaign/           # Verification harness and campaign fixtures for decode models
│   ├── data/               # Evaluation prompt datasets and corpus targets
│   ├── eval-coir/          # CoIR retrieval evaluation harness and tools
│   ├── harness/            # Core benchmark metrics and telemetry runner library
│   ├── lanes/              # Crate workspace members and runtime scripts
│   │   ├── burn/           # WGPU/Metal shader engine using Burn
│   │   ├── llama/          # Supervised llama-server child process executor
│   │   ├── mlx-minilm/     # Python-based MLX GPU executor for MiniLM
│   │   ├── mlx/            # Metal-accelerated MLX GPU executor
│   │   ├── ort-embed/      # Bounded-CPU ONNX runtime embedding runner
│   │   ├── potion/         # Model2Vec static embedding lane (potion-code-16M)
│   │   ├── ts-embed/       # TypeScript Bun/Node runner (Transformers.js or ORT Node)
│   │   └── wrap-embed/     # External API wrapper (Ollama/LMStudio)
│   ├── rig/                # External measurement harness and candidate supervisor
│   ├── spikes/             # Benchmarking experimental spikes (unified-rt, ane-minilm)
│   └── results/            # Saved telemetry metrics, results, and vectors
├── corpus/                 # Code chunk files used for evaluations
├── crates/                 # Core Synapse application and worker binaries
│   ├── synapse-core/       # Shared types, protocol, and scheduler traits
│   ├── synapse-engine-ort/ # In-process ONNX Runtime inference engine
│   ├── synapse-engine-owned/ # Primary owned Metal MPSGraph engine (macOS)
│   ├── synapse-module/     # Main SubC module host, job queue, and worker manager
│   ├── synapse-opctl/      # CLI operator control surface driving SubC commands
│   ├── synapse-worker-ane/ # Apple Neural Engine supervised worker (Swift/CoreML)
│   ├── synapse-worker-llama/ # llama.cpp supervised worker (GGUF)
│   └── synapse-worker-mlx/ # Apple Silicon MLX supervised worker
├── docs/                   # Design logs and decisions documentation
├── tools/                  # Shared system tools and distillation harnesses
│   └── gather-distill/     # External gather-distillation data generation harness
├── Cargo.toml              # Cargo workspace definition
├── DECISIONS.md            # Log of architecture design decisions
└── FOUNDING.md             # Foundational constraints and handoff requirements
```

## Directory Purposes

**.alfonso/:**
- Purpose: Host experiment spikes, prototypes, and workspace development plans.
- Contains: Rust source spikes.
- Key files: `.alfonso/spikes/coreml_spike.rs`

**.cortexkit/:**
- Purpose: Houses agent prompts, configuration setups, and historian logs.
- Contains: Markdown prompts, ignores, and sub-directories.
- Key files: `.cortexkit/alfonso/prompts/lane-mlx.md`, `.cortexkit/alfonso/prompts/lane-llama.md`

**crates/:**
- Purpose: Contains the production Synapse runtime, module host, and inference workers.
- Contains: The main SubC module and supervised worker binaries.
- Key files: `crates/synapse-module/src/main.rs`, `crates/synapse-core/src/lib.rs`

**crates/synapse-core/:**
- Purpose: Defines shared abstractions for engines, worker protocol, caching, and scheduling.
- Contains: Envelopes, machine profile structs, engine traits, and error contracts.
- Key files: `crates/synapse-core/src/worker_protocol.rs`, `crates/synapse-core/src/scheduler.rs`

**crates/synapse-engine-owned/:**
- Purpose: The primary in-process execution engine for Apple Silicon (macOS).
- Contains: Metal MPSGraph inference layers for ModernBERT, Qwen3, and MiniLM models.
- Key files: `crates/synapse-engine-owned/src/lib.rs`, `crates/synapse-engine-owned/src/metal_mpsgraph.m`

**crates/synapse-module/:**
- Purpose: The primary SubC service module. Handles the content-addressed model cache, durable jobs, worker hosting, remote provider dispatch, and route binding.
- Contains: SQLite store initialization, SubC `ModuleHandler` implementation, UNIX socket / Windows pipe worker spawning, and the remote gateway client.
- Key files: `crates/synapse-module/src/lib.rs`, `crates/synapse-module/src/worker_host/mod.rs`, `crates/synapse-module/src/remote/gateway.rs`

**crates/synapse-opctl/:**
- Purpose: Command-line operator control surface driving SubC commands.
- Contains: Commands to query model statuses, run/inspect certification probes, monitor scheduler stats, submit batches, and fetch job pages.
- Key files: `crates/synapse-opctl/src/main.rs`


**crates/synapse-worker-*/:**
- Purpose: Specialized out-of-process inference engines built for specific hardware (ANE, MLX, llama.cpp).
- Contains: Binaries that speak the `worker_protocol` over a local socket.
- Key files: `crates/synapse-worker-mlx/src/main.rs`, `crates/synapse-worker-ane/src/main.rs`
- Build note: `synapse-worker-mlx` requires full Xcode with the Metal toolchain on macOS. If `xcrun` resolves to Command Line Tools only, build with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`; the workspace does not auto-set it.

**bench/:**
- Purpose: Contains all performance evaluation execution infrastructure.
- Contains: Shell scripts, configuration folders, and cargo workspace members.
- Key files: `bench/run-matrix.sh`, `bench/NOTES.md`

**bench/campaign/:**
- Purpose: Verification harness and campaign fixtures for decode models.
- Contains: Locked sandboxed campaign controller script, prompt/reference fixtures, and signature validation checks.
- Key files: `bench/campaign/decode-harness.sh`, `bench/campaign/README.md`


**bench/harness/:**
- Purpose: Acts as the core telemetry library and dataset builder.
- Contains: Cargo manifest, Rust code for metrics, corpus chunking, parity, and results formatting.
- Key files: `bench/harness/src/metrics.rs`, `bench/harness/src/parity.rs`, `bench/harness/src/results.rs`, `bench/harness/src/rig_protocol.rs`

**bench/rig/:**
- Purpose: External measurement harness for strict bounding of benchmark candidate timing, token accounting, and semantics parity.
- Contains: Cargo manifest, subprocess supervisor logic, and reference checking.
- Key files: `bench/rig/src/main.rs`, `bench/rig/RIG.md`

**bench/spikes/:**
- Purpose: Holds discrete architecture experimentation paths and new backend developments.
- Contains: `unified-rt` (CUDA/Vulkan/M1 exact-match execution) and `ane-minilm` (Apple Neural Engine CoreML conversion) spikes.
- Key files: `bench/spikes/unified-rt/src/main.rs`, `bench/spikes/unified-rt/src/vulkan_backend.rs`, `bench/spikes/unified-rt/src/cuda_backend.rs`, `bench/spikes/unified-rt/src/lfm2.rs`, `bench/spikes/unified-rt/src/lfm2_audio.rs`, `bench/spikes/unified-rt/src/lfm2_decode.rs`, `bench/spikes/unified-rt/src/qwen3_decode.rs`

**bench/eval-coir/:**
- Purpose: Hosts the CoIR retrieval and rerank quality evaluation harness.
- Contains: Dataset preparation scripts, numpy metrics scoring, and reference rerank cross-check tools.
- Key files: `bench/eval-coir/prepare.py`, `bench/eval-coir/score.py`, `bench/eval-coir/coir_eval.py`

**bench/lanes/:**
- Purpose: Groups individual workspace crates and runtime scripts that run candidate models.
- Contains: Sub-directories for each runtime backend (Rust crates, Python venvs, Bun packages).
- Key files: `bench/lanes/ort-embed/src/main.rs`, `bench/lanes/mlx/src/main.rs`, `bench/lanes/llama/src/main.rs`, `bench/lanes/mlx-minilm/main.py`, `bench/lanes/ts-embed/main.mjs`, `bench/lanes/potion/main.py`

**bench/results/:**
- Purpose: Storage directory for the output log files, parity vectors, and metrics.
- Contains: JSON, JSONL, and log files.
- Key files: `bench/results/matrix.log`, `bench/results/ort-cpu-embed.json`

**docs/:**
- Purpose: Stores contextual architectural studies and decision analyses.
- Contains: Markdown documents.
- Key files: `docs/decision-1-runtime.md`

**tools/:**
- Purpose: Houses shared development tools, utilities, and datasets generation/distillation harnesses.
- Contains: The `gather-distill` TypeScript project workspace.

**tools/gather-distill/:**
- Purpose: Standalone external harness for generating QA datasets and collecting model tool-use trajectories.
- Contains: Bun workspaces, Anthropic/OpenAI API adapters (supporting OpenAI OAuth transports), AFT child process pools, validation scripts, scoring modules, and utility judge matrix evaluation engines.
- Key files: `tools/gather-distill/src/cli.ts`, `tools/gather-distill/README.md`, `tools/gather-distill/BAKEOFF-ZEROSHOT.md`

## Key File Locations

**Entry Points:**
- `crates/synapse-module/src/main.rs`: The main production SubC module entry point.
- `crates/synapse-worker-*/src/main.rs`: Executables for hardware-specific supervised workers.
- `crates/synapse-opctl/src/main.rs`: Operator command line control surface (`ck-synapse-opctl`).
- `crates/synapse-module/src/bin/subc_call.rs`: Management surface call utility.
- `crates/synapse-module/src/bin/inline_embed_throughput.rs`: Batch throughput execution client.
- `bench/harness/src/main.rs`: CLI runner for corpus generation, power wrapper execution, and parity check.
- `bench/lanes/*/src/main.rs` (Rust), `bench/lanes/mlx-minilm/main.py` (Python), `bench/lanes/ts-embed/main.mjs` (JS), `bench/lanes/potion/main.py` (Python): Main executables for each specific runtime lane.
- `bench/campaign/decode-harness.sh`: Campaign controller script.
- `bench/eval-coir/prepare.py`: Downloads and structures datasets for retrieval tasks.
- `bench/eval-coir/score.py`: Computes retrieval quality metrics on generated vectors.
- `bench/run-matrix.sh`: Global benchmark suite runner.
- `bench/run-night.sh`: Nightly full-corpus multi-lane orchestrator.
- `tools/gather-distill/src/cli.ts`: Entry point for the gather-distillation harness commands (`qgen`, `gather`, `validate`, `score`).
- `bench/spikes/unified-rt/src/bin/vulkan_probe.rs`: Vulkan memory type and budget capability prober.

**Configuration:**
- `Cargo.toml`: Cargo workspace manifest listing all members.
- `bench/lanes/burn/build.rs`: Burn compilation setup for model building.

**Core Logic:**
- `crates/synapse-module/src/remote/runtime.rs`: Provider pool routing, circuit breaker enforcement, and telemetry collection for external model execution.
- `crates/synapse-module/src/worker_host/mod.rs`: Spawns and manages worker lifecycles over Unix domain sockets or Windows named pipes using a binary framing protocol.
- `crates/synapse-module/src/store.rs`: SQLite-backed state for content-addressed model cache, durable jobs, active attempts, and performance tier capabilities.
- `crates/synapse-core/src/scheduler.rs`: 3-class fair-share aging scheduler for managing concurrent inference requests.
- `bench/harness/src/metrics.rs`: Macmon power metrics execution, parsing, and system idle gating.
- `bench/harness/src/parity.rs`: Numerical calculation of cosine similarity, rank stability/overlap checks, and file parsing functions.
- `bench/lanes/mlx/src/main.rs`: Qwen3 model architecture implementation and custom forward passes in MLX.
- `bench/lanes/mlx-minilm/main.py`: Length-sorted batched MLX GPU execution for MiniLM.
- `bench/lanes/ts-embed/main.mjs`: Transformers.js (q8/fp32) and native `onnxruntime-node` embedding logic.
- `bench/lanes/potion/main.py`: Model2Vec static embedding lane utilizing `model2vec` (StaticModel) with `potion-code-16M`.
- `bench/eval-coir/coir_eval.py`: Brute-force numpy cosine retrieval and pytrec_eval scoring logic.
- `bench/eval-coir/reference_rerank.py`: Reference Alibaba-NLP/gte-reranker-modernbert-base rerank calculation.
- `tools/gather-distill/src/auth.ts`: Multi-account credential storage, verification, and rotation pool.
- `tools/gather-distill/src/tools.ts`: Verbatim v0.46.0 tool definitions schema and `AftClientPool` process allocation.
- `tools/gather-distill/src/gather.ts`: Work queue execution loop driving model tool interactions.
- `tools/gather-distill/src/validate.ts`: Citation verification, SHA-checking, and path bounds checker.
- `tools/gather-distill/src/scorer.ts`: Offline gold-standard Jaccard and file F1 overlap quality scorer.
- `tools/gather-distill/src/judge.ts`: OpenAI OAuth validation and judge scoring loop.
- `bench/spikes/unified-rt/src/json_constraint.rs`: Constrained JSON schema grammar state machine and token mask generator.
- `bench/spikes/unified-rt/src/lfm2.rs`: LFM2 model family, short-convolution, and full-attention mixer logic.
- `bench/spikes/unified-rt/src/lfm2_audio.rs`: Mel-spectrogram DSP frontend, FastConformer speech encoder, and audio projector.
- `bench/spikes/unified-rt/src/lfm2_decode.rs`: Causal decoding logic for LFM2 hybrid backbone models.
- `bench/spikes/unified-rt/src/qwen3_decode.rs`: Fast Metal decode optimizations for Qwen3-0.6B f16.

**Tests:**
- Standalone nextest-compatible test suites are managed via library configurations and workspace flags.

## Naming Conventions

**Files:** Snake case for Rust files (`main.rs`, `metrics.rs`) and scripts (`run-matrix.sh`).
**Directories:** Kebab case for lane folder structures (`ort-embed`, `wrap-embed`).
**Binaries:** The fleet convention prefixes runtime executables with `ck-` (e.g., `ck-synapse`, `ck-synapse-worker-mlx`) to group them in Activity Monitor, while preserving the un-prefixed module ID and crate names.

## Where to Add New Code

**New benchmark lane:** For Rust-based lanes, create a new workspace crate under `bench/lanes/[lane-name]/` and register the crate path in the root `Cargo.toml` `members` list. For Python or JavaScript/TypeScript-based lanes, create a new sub-directory under `bench/lanes/[lane-name]/` with the corresponding package or dependency manifest (`requirements.txt`, `package.json`). Follow standard batching structures, output a valid `LaneResult` json structure, then add the runner invocation inside `bench/run-matrix.sh` and `bench/run-night.sh`.
**New worker backend:** Create a new workspace crate `crates/synapse-worker-[name]`, implement the binary frame protocol defined in `crates/synapse-core/src/worker_protocol.rs`, and integrate its lifecycle into `crates/synapse-module/src/worker_host/mod.rs`.
**New benchmark workload:** Add a subcommand and its schema parsing in `bench/harness/src/main.rs`, support loading and typing under `bench/harness/src/parity.rs`, and implement the evaluation logic in the corresponding lane executables.
**Shared utilities:** Place shared functions or data representations within `bench/harness/src/parity.rs` or `bench/harness/src/results.rs`.
**Tests:** Co-locate unit tests within the source files as nested `#[cfg(test)]` modules, and integration tests inside `tests/` directories at the crate roots.
