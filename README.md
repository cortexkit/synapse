# Synapse workspace

This workspace contains the Synapse SubC module, supervised inference workers, and benchmark lanes described in `STRUCTURE.md` and `ARCHITECTURE.md`.

## Synapse crates

Production crates live under `crates/`:

- `synapse-core`: shared protocol, error, cache, fingerprint, scheduler, and tokenizer types.
- `synapse-engine-ort`: in-process ONNX Runtime embedding engine.
 - `synapse-module`: SubC management surface, model cache, durable jobs, and worker host. Builds the `ck-synapse` binary (fleet `ck-*` naming convention for Activity Monitor grouping; `module_id` stays `synapse`).
 - `synapse-worker-llama`, `synapse-worker-mlx`, `synapse-worker-ane`: supervised worker binaries (`ck-synapse-worker-*`) that speak the Synapse worker protocol over Unix sockets (named pipes on Windows).

### MLX / Metal build requirement

`synapse-worker-mlx` depends on `mlx-rs`/`mlx-sys`, which need the full Xcode Metal toolchain on macOS. Command Line Tools alone can make `xcrun` fail to find `metal` or `metallib`, producing noisy CMake output from dependencies.

Use an explicit developer directory when building the worker or full workspace on affected hosts:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo build -p synapse-worker-mlx
```

Do not auto-set `DEVELOPER_DIR` in scripts; configure the host or invocation so Cargo uses the intended Xcode installation.

### Module config (`synapse.jsonc`)

Example `~/.config/cortexkit/synapse.jsonc` (or project `.cortexkit/synapse.jsonc`):

```jsonc
{
  // performance | balanced | quiet
  "knob": "balanced",
  "microllm_max_tokens": 512,
  "grammar_enabled": false,
  "cache_max_bytes": 34359738368,
  "alias_admin_enabled": false,
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
    "bulk_quantum_tokens": 2048
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
