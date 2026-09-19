# Synapse workspace

This workspace contains the Synapse SubC module, supervised inference workers, and benchmark lanes described in `STRUCTURE.md` and `ARCHITECTURE.md`.

## Synapse crates

Production crates live under `crates/`:

- `synapse-core`: shared protocol, error, cache, fingerprint, scheduler, and tokenizer types.
- `synapse-engine-owned`: primary in-process Metal/MPSGraph engine for Apple Silicon (embedding, reranking, direct Metal step decode), with the supervised decode worker state machine in its `synapse-engine-owned/owned-decode-worker` subcrate.
- `synapse-engine-cuda`: primary in-process CUDA engine (`owned-cuda-v1`, PTX kernel ports, f16 storage).
- `synapse-engine-ort`: in-process ONNX Runtime embedding engine (universal CPU floor).
- `synapse-module`: SubC management surface, model cache, durable jobs, and worker host. Builds the `ck-synapse` binary (fleet `ck-*` naming convention for Activity Monitor grouping; `module_id` stays `synapse`).
- `synapse-opctl`: operator CLI (`ck-synapse-opctl`) for catalog, probes, admission stats, approvals, and paged results over the fleet daemon.
- `synapse-worker-llama`, `synapse-worker-ane`, `synapse-worker-cuda`, `synapse-worker-decode`: supervised worker binaries (`ck-synapse-worker-*`) that speak the Synapse worker protocol over Unix sockets (named pipes on Windows).

### Metal build requirement

`synapse-engine-owned` compiles Metal shaders and Objective-C MPSGraph drivers, which need the full Xcode Metal toolchain on macOS. Command Line Tools alone can make `xcrun` fail to find `metal` or `metallib`, producing confusing build failures.

Use an explicit developer directory when building the Metal crates or the full workspace on affected hosts:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo build -p synapse-engine-owned
```

Do not auto-set `DEVELOPER_DIR` in scripts; configure the host or invocation so Cargo uses the intended Xcode installation.

### Module config (`synapse.jsonc`)

Example user-tier `~/.config/cortexkit/synapse.jsonc` (project configs must omit the user-only chain setting):

```jsonc
{
  // performance | balanced | quiet
  "knob": "balanced",
  "microllm_max_tokens": 512,
  "grammar_enabled": false,
  // Free-text chain span (1..=16); grammar stays at K=1. Values above 1
  // change free-text execution shape only after certification covers the span.
  "decode_chain_k": 1,
  "cache_max_bytes": 34359738368,
  "alias_admin_enabled": false,
  "worker": {
    "load_timeout_ms": 180000
  },
  "preload_models": [],
  "inline": {
    "max_items": 64,
    "max_tokens": 8192,
    "byte_budget": 67108864,
    "max_queue_ms": 5000,
    "deadline_ms": 30000,
    "estimated_execution_ms": 25,
    "max_concurrent_workers": 2
  },
  "jobs": {
    "ttl_ms": 86400000,
    "result_page_bytes": 524288,
    "bulk_quantum_tokens": 3072
  },
  "probe": {
    "mean_cosine_threshold": 0.999,
    "worst_decile_rank_overlap_threshold": 0.9,
    "ane_placement_threshold": 0.9
  }
}
```

Tests can point at a file with `SYNAPSE_CONFIG_PATH`. Only one synapse module
per machine (singleton lease); a second instance refuses to start.

### Owned-CUDA hardware floor

`ck-synapse-worker-cuda` implements `--probe-floor` (hidden, like the
`--test-abort*` surfaces). It prints one JSON object and exits 0:

```json
{"driver_api": 13030, "compute_capability": {"major": 8, "minor": 9}}
```

The module probes the configured `worker_bin`, the engine's worker-binary
environment override, or the sibling `ck-synapse-worker-cuda`, in that order.
It caches one result per process unless both environment readings parse
successfully. The child wait is bounded to 10 seconds; stdout is capped at
4096 bytes and each pipe completion wait is bounded to another 100 ms.
A missing binary, non-zero exit, timeout, or invalid output produces
`HardwareUnavailable`. Refusal and model evidence carry diagnostic context,
including the last 4096 bytes of stderr when available, under `observed`.
Failed probes do not fabricate numeric hardware readings.

The environment overrides the probe only as a complete, parseable pair.
Otherwise both readings come from the probe; partial overrides are not merged:

- `SYNAPSE_CUDA_DRIVER_API` (alias `CUDA_DRIVER_API`) — the raw CUDA **driver
  API** integer from `cuDriverGetVersion()`, not the marketing driver version.
  For example, a measured driver API value is `13030`. `610.88` is not a valid
  API integer; without a parseable alias, it causes fallback to the probe.
- `SYNAPSE_CUDA_COMPUTE_CAPABILITY` (alias `CUDA_COMPUTE_CAPABILITY`) — device
  0's compute capability as `major.minor`, for example `8.9`.
- `SYNAPSE_CUDA_PACKAGING_DRIVER` — optional; the driver string a packaging
  build was tested against, carried into the refusal for diagnostics.

### Windows owned-CUDA package

The manual Windows CUDA gate packages the worker with runtime DLLs derived
from the same pinned `cuda_cudart` and `libcublas` redistribution archives
used for compilation. `scripts/package-owned-cuda.ps1` places the executable
and DLLs at the ZIP root, includes component licenses, and records source
components and SHA-256 hashes in `manifest.json`. The NVIDIA driver is not
bundled and must already be installed.

Windows worker builds require CUDA 13. The build script rejects other toolkit
major versions before linking, because this package resolves CUDA 13 DLL names.

`scripts/test-owned-cuda-package.ps1 -Archive <zip> -RequireGpu` extracts a
fresh copy, verifies hashes, and checks no-sidecar `--version`, actionable
missing-library refusal, and a real hardware-floor probe with adjacent DLLs
and CUDA removed from PATH. Without `-RequireGpu`, a runner without NVIDIA
hardware may report an explicit driver/device refusal; this is not a GPU
execution pass. Neither mode loads model weights or certifies embeddings.

The worker delays its cuBLASLt import and checks runtime library loading
before CUDA calls. No global PATH changes or extra DLL search directories
are needed. Release-matrix publication remains separate from this manual
gate artifact.

This delay-loading behavior is Windows-only. The Linux ELF worker retains a
`DT_NEEDED` dependency on `libcublasLt.so.12`; without that runtime on the
library search path, even `--probe-floor` exits 127 before reaching the
driver-only probe.
