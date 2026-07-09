# Synapse workspace

This workspace contains the Synapse SubC module, supervised inference workers, and benchmark lanes described in `STRUCTURE.md` and `ARCHITECTURE.md`.

## Synapse crates

Production crates live under `crates/`:

- `synapse-core`: shared protocol, error, cache, fingerprint, scheduler, and tokenizer types.
- `synapse-engine-ort`: in-process ONNX Runtime embedding engine.
- `synapse-module`: SubC management surface, model cache, durable jobs, and worker host.
- `synapse-worker-llama`, `synapse-worker-mlx`, `synapse-worker-ane`: supervised worker binaries that speak the Synapse worker protocol over Unix sockets.

### MLX / Metal build requirement

`synapse-worker-mlx` depends on `mlx-rs`/`mlx-sys`, which need the full Xcode Metal toolchain on macOS. Command Line Tools alone can make `xcrun` fail to find `metal` or `metallib`, producing noisy CMake output from dependencies.

Use an explicit developer directory when building the worker or full workspace on affected hosts:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo build -p synapse-worker-mlx
```

Do not auto-set `DEVELOPER_DIR` in scripts; configure the host or invocation so Cargo uses the intended Xcode installation.
