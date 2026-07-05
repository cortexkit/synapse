# Codebase Structure

## Directory Layout

```
[project-root]/
├── .alfonso/               # Alfonso AI spikes and planning artifacts
├── .cortexkit/             # Prompts, configurations, and historian logs
├── bench/                  # Main benchmarking workspace containing harness and lanes
│   ├── data/               # Evaluation prompt datasets and corpus targets
│   ├── harness/            # Core benchmark metrics and telemetry runner library
│   ├── lanes/              # Crate workspace members for model runtimes
│   │   ├── burn/           # WGPU/Metal shader engine using Burn
│   │   ├── llama/          # Supervised llama-server child process executor
│   │   ├── mlx/            # Metal-accelerated MLX GPU executor
│   │   ├── ort-embed/      # Bounded-CPU ONNX runtime embedding runner
│   │   └── wrap-embed/     # External API wrapper (Ollama/LMStudio)
│   └── results/            # Saved telemetry metrics, results, and vectors
├── corpus/                 # Code chunk files used for evaluations
├── docs/                   # Design logs and decisions documentation
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

**bench/:**
- Purpose: Contains all performance evaluation execution infrastructure.
- Contains: Shell scripts, configuration folders, and cargo workspace members.
- Key files: `bench/run-matrix.sh`, `bench/NOTES.md`

**bench/harness/:**
- Purpose: Acts as the core telemetry library and dataset builder.
- Contains: Cargo manifest, Rust code for metrics, corpus chunking, parity, and results formatting.
- Key files: `bench/harness/src/metrics.rs`, `bench/harness/src/parity.rs`, `bench/harness/src/results.rs`

**bench/lanes/:**
- Purpose: Groups individual workspace crates that run candidate models.
- Contains: Sub-directories for each runtime backend.
- Key files: `bench/lanes/ort-embed/src/main.rs`, `bench/lanes/mlx/src/main.rs`, `bench/lanes/llama/src/main.rs`

**bench/results/:**
- Purpose: Storage directory for the output log files, parity vectors, and metrics.
- Contains: JSON, JSONL, and log files.
- Key files: `bench/results/matrix.log`, `bench/results/ort-cpu-embed.json`

**docs/:**
- Purpose: Stores contextual architectural studies and decision analyses.
- Contains: Markdown documents.
- Key files: `docs/decision-1-runtime.md`

## Key File Locations

**Entry Points:**
- `bench/harness/src/main.rs`: CLI runner for corpus generation and power wrapper execution.
- `bench/lanes/*/src/main.rs`: Main executables for each specific runtime lane.
- `bench/run-matrix.sh`: Global benchmark suite runner.

**Configuration:**
- `Cargo.toml`: Cargo workspace manifest listing all members.
- `bench/lanes/burn/build.rs`: Burn compilation setup for model building.

**Core Logic:**
- `bench/harness/src/metrics.rs`: Macmon power metrics execution, parsing, and system idle gating.
- `bench/harness/src/parity.rs`: Numerical calculation of cosine similarity and file parsing functions.
- `bench/lanes/mlx/src/main.rs`: Qwen3 model architecture implementation and custom forward passes in MLX.

**Tests:**
- Standalone nextest-compatible test suites are managed via library configurations and workspace flags.

## Naming Conventions

**Files:** Snake case for Rust files (`main.rs`, `metrics.rs`) and scripts (`run-matrix.sh`).
**Directories:** Kebap case for lane folder structures (`ort-embed`, `wrap-embed`).

## Where to Add New Code

**New benchmark lane:** Create a new workspace crate under `bench/lanes/[lane-name]/`. Register the crate path in the root `Cargo.toml` `members` list. Follow the standard lane structure (arguments, warmup step, token-based batching, and `LaneResult` generation), then add the command runner invocation inside `bench/run-matrix.sh`.
**New benchmark workload:** Add a subcommand and its schema parsing in `bench/harness/src/main.rs`, support loading and typing under `bench/harness/src/parity.rs`, and implement the evaluation logic in the corresponding lane executables.
**Shared utilities:** Place shared functions or data representations within `bench/harness/src/parity.rs` or `bench/harness/src/results.rs`.
**Tests:** Co-locate unit tests within the source files as nested `#[cfg(test)]` modules, and integration tests inside `tests/` directories at the crate roots.
