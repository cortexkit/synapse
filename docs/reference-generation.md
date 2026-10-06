# CPU fp32 reference generation

The single reference implementation is `bench/parity/reference/generate_reference.py`.
It uses Transformers 5.16.1, Torch 2.14.0, seed 0, CPU fp32, eager attention,
one intra-op thread, one inter-op thread, and deterministic algorithms.
The script sets `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before importing
Torch or Transformers; callers need not supply those settings. These settings
are recorded in each parity fixture's `reference` metadata.

Model files come from the pinned revisions in `bench/parity/models.json`.
The generator checks each declared file's SHA-256 before loading the model.
Use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` to prohibit model downloads.

## Deterministic re-seal comparison (2026-10-06)

Two complete generations into separate temporary directories, with a fresh
Python process for each model, produced byte-identical fixture files and
byte-identical `fixtures/index.json`. Both generations used:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-project --python 3.12 \
  --with transformers==5.16.1 --with torch==2.14.0 python ...
```

The Python harness copied `bench/parity/models.json` into each temporary
root, imported the generator, and called
`run(["--hf-cache", cache, "--model", slug], parity_dir=temporary_root)`.
The cache was `~/.cache/huggingface/hub`; no model files were copied or
modified. File reads follow the cache symlinks to their contents.

Compared with the previously committed fixtures:

| Model | Maximum absolute difference | Minimum cosine | Ranking changes |
| --- | ---: | ---: | --- |
| gte-modernbert-base | 8.016359061002731e-7 | 0.9999999999766092 | Not applicable |
| gte-reranker-modernbert-base | 8.255243301391602e-6 | 0.9999999999982356 | None |
| qwen3-embedding-0.6b | 5.513429641723633e-7 | 0.9999999999928043 | Not applicable |
| qwen3-reranker-0.6b | 1.1920928955078125e-7 | 0.9999999999999996 | None |

All non-output case fields, including token IDs, matched. Embedding cosine
was measured per case. Reranker cosine and descending score order were
measured separately for each category and each candidate pool, preserving
input order for ties. No group changed order. All absolute differences were
below 1e-5. The comparison includes every case, including the padded batch
and the 8192-token cases.

SHA-256 values reproduced in both runs:

| Model | Deterministic fixture SHA-256 |
| --- | --- |
| gte-modernbert-base | ec179ee4a8fb4dc759676751de717cfbabe30fc5009f732353bff1d226fe552f |
| gte-reranker-modernbert-base | 241c7c058148bba3a2e3f1d5a33b25013d688ed320c9bb3dc4f901b8f1a2c083 |
| qwen3-embedding-0.6b | fe401fff207c0b67841dbd6a653020bcf32d2cc5998c728b3252132698a27a3b |
| qwen3-reranker-0.6b | b34adc4106321bb00665d90503a28b4b1f1eaa5fe3deca5f6ca8ae05fb47ff95 |

The old fixtures did not record thread or attention settings. Explicit eager
attention and single-threaded reductions change fp32 rounding slightly; the
comparison above establishes that the deterministic re-seal retains numeric
and ranking behavior. The certification library embeds small subsets of these
cases for profile self-checks. Their source digests and copied outputs must be
regenerated when the full fixtures change. The direct-ANE padding golden also
records source digests that must be refreshed.
