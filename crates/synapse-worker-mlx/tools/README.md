# MLX worker tools

- MiniLM parity reference: `uv venv /tmp/synapse-minilm-ref && . /tmp/synapse-minilm-ref/bin/activate && uv pip install torch transformers safetensors && python crates/synapse-worker-mlx/tools/dump_minilm_reference.py --model <sentence-transformers/all-MiniLM-L6-v2 snapshot> --tokenizer <tokenizer snapshot or tokenizer.json>` dumps Hugging Face per-layer hidden-state summaries for comparison with the worker's hidden `--debug-bert-model` dump.
