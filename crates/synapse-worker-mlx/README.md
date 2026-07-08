# synapse-worker-mlx

Supervised Synapse worker for MLX/Metal embedding models. It speaks the v1
length-prefixed worker protocol over a module-owned Unix socket and keeps MLX's
abort-capable C++/Metal runtime out of the module process.

Supported v1 model families:

- BERT/MiniLM safetensors snapshots (`model_type = "bert"`) with mask-aware
  mean/CLS/last pooling and optional L2 normalization.
- Qwen-style safetensors snapshots with the hand-written MLX graph ported from
  `bench/lanes/mlx`.

Build note: `mlx-rs` builds Metal support through the Apple toolchain. On macOS
CI/dev hosts use:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo build -p synapse-worker-mlx
```

Packaging trap: `mlx.metallib` must ship next to the installed worker binary.
The MLX runtime looks beside the executable at run time; relocating only the
binary produces a late Metal load failure even though the Rust build succeeded.
