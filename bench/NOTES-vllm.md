# vLLM family MiniLM smoke notes

Machine: Apple M5 Max, 128 GB RAM, macOS 26.5.1. Rust wrap lane built from this repo with `cargo build --release -p lane-wrap-embed`.

## Summary

| Mode | Install | Serve MiniLM | Smoke result | Peak RSS | Venv | Cold start | Versions / SHA | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| vllm CPU | success | success | `bench/results/smoke-vllm-cpu.json` | 565440 KB | 1.3G | 13.005s | `vllm 0.24.0+cpu` | Needed `VLLM_ENABLE_V1_MULTIPROCESSING=0` and `--gpu-memory-utilization 0.4`. Default memory reservation failed. For smoke, I had to run a temporary 240-token HF-truncated copy of the 200-chunk corpus because the wrap lane hardcodes 512-token pre-truncation while MiniLM's sentence-transformers config advertises 256 max seq length. |
| vllm-metal | success after install workarounds | blocked | n/a | n/a | 1.6G | n/a | `vllm 0.24.0+cpu`, `vllm-metal 0.3.0`, plugin HEAD `72294be2c8aaa57871988290fa71b775f3d109a6` | Local install under this repo needed a temporary `[workspace]` stanza in the plugin `Cargo.toml` so maturin would stop treating the checkout as part of synapse's Rust workspace, and `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` so Metal artifact build could find Xcode. Serve failed on MiniLM with `ValueError: Model type bert not supported.` |
| vllm-mlx | success | success, but only with a supported LLM as the primary model | `bench/results/smoke-vllm-mlx.json` | 1873296 KB | 1.2G | 4.200s | `vllm-mlx 0.4.0`, repo HEAD `0dd115769ef1196a715b96b181353edacd2a4f69`, `transformers 5.12.1` | Fresh install with `transformers 5.13.0` crashed before startup; pinning `5.12.1` fixed that. `vllm-mlx serve sentence-transformers/all-MiniLM-L6-v2 --embedding-model sentence-transformers/all-MiniLM-L6-v2` still failed because the primary model path rejects BERT, so the working setup was `vllm-mlx serve Qwen/Qwen3-0.6B --embedding-model sentence-transformers/all-MiniLM-L6-v2`. Smoke used the same temporary 240-token HF-truncated corpus copy as the CPU run. |

## Commands and outcomes

### 1) vllm CPU

Install:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
uv venv --python 3.12 --seed --managed-python bench/lanes/vllm-ext/vllm-cpu-venv
VLLM_TARGET_DEVICE=cpu CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON uv pip install . --no-build-isolation
```

Exact default-memory blocker before the working flags:

> Available memory on node 0 (56.16/128.0 GiB) on startup is less than desired CPU memory utilization (0.92, 117.76 GiB).

Working serve command:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
vllm serve sentence-transformers/all-MiniLM-L6-v2 \
  --runner pooling \
  --served-model-name sentence-transformers/all-MiniLM-L6-v2 \
  --port 8014 \
  --gpu-memory-utilization 0.4
```

Probe:

```bash
curl -s http://127.0.0.1:8014/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"sentence-transformers/all-MiniLM-L6-v2","input":["hello world"]}'
```

Result: 384-dim vector returned.

Smoke run notes:

- The direct smoke corpus hit two MiniLM-specific problems under the existing wrap lane: 400s from the model's 256-token limit and later per-request timeouts when forcing 512.
- I therefore generated a temporary non-committed copy of the same 200 chunks, truncating text to 240 HF-tokenized tokens so the existing wrap lane could still run unchanged against the MiniLM endpoint.
- The committed JSON in `bench/results/smoke-vllm-cpu.json` is from that temporary 240-token copy.

### 2) vllm-metal

Install blocker 1, before workaround:

> error: current package believes it's in a workspace when it's not

Install blocker 2, before workaround:

> xcode-select: error: tool 'xcodebuild' requires Xcode, but active developer directory '/Library/Developer/CommandLineTools' is a command line tools instance

After adding a temporary `[workspace]` table to the plugin checkout and exporting `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`, installation completed.

Serve attempt:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
vllm serve sentence-transformers/all-MiniLM-L6-v2 \
  --runner pooling \
  --served-model-name sentence-transformers/all-MiniLM-L6-v2 \
  --port 8017
```

Exact MiniLM blocker:

> ValueError: Model type bert not supported.

This matches the plugin's own support matrix: text pooling support is presently documented for Qwen3-Embedding / Qwen3-Reranker, not BERT MiniLM.

### 3) vllm-mlx

Fresh-install startup blocker:

> AttributeError: 'str' object has no attribute '__module__'

That happened with `transformers 5.13.0`; pinning `transformers==5.12.1` fixed startup.

Serve attempt with MiniLM as both primary and embedding model still failed because the primary model load path is LLM-only:

> ValueError: Model type bert not supported.

Working serve command:

```bash
vllm-mlx serve Qwen/Qwen3-0.6B \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --port 8018
```

Probe with model `sentence-transformers/all-MiniLM-L6-v2` returned a 384-dim vector from `/v1/embeddings`.

Smoke run notes:

- Same temporary 240-token HF-truncated copy of the 200-chunk smoke corpus as the CPU run.
- Existing wrap lane used unchanged with `--batch 1`.

## Smoke LaneResult files

- CPU: `bench/results/smoke-vllm-cpu.json`
- MLX: `bench/results/smoke-vllm-mlx.json`

## Bottom line

- **vllm CPU**: works for MiniLM embeddings on macOS, but needed lower memory reservation and a truncated smoke-corpus copy to stay within MiniLM's effective 256-token limit under the existing wrap lane.
- **vllm-metal**: installable here, but **cannot serve MiniLM** because the Metal/MLX backend does not support BERT MiniLM (`Model type bert not supported`).
- **vllm-mlx**: **can serve MiniLM embeddings**, but only as an auxiliary embedding model attached to a supported primary LLM; standalone MiniLM-as-primary fails for the same BERT support reason.
