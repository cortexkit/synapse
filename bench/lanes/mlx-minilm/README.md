# mlx-minilm lane

Measure `all-MiniLM-L6-v2` embeddings on MLX (GPU).

## Install

```bash
python3.11 -m venv /tmp/synapse-mlx-minilm-venv
source /tmp/synapse-mlx-minilm-venv/bin/activate
python -m pip install -r bench/lanes/mlx-minilm/requirements.txt
```

## Run

```bash
python bench/lanes/mlx-minilm/main.py \
  --corpus bench/data/corpus-smoke.jsonl \
  --out /tmp/mlx-minilm-smoke.json \
  --vectors-out /tmp/mlx-minilm-smoke-vectors.jsonl \
  --model-label "all-MiniLM-L6-v2@mlx-bf16"
```

The lane first tries `mlx-community/all-MiniLM-L6-v2-bf16`. If that model is unavailable, it falls back to `sentence-transformers/all-MiniLM-L6-v2` and converts it to local MLX weights under `~/.cache/synapse/mlx-minilm/`.
