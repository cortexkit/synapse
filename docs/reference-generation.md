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

After regenerating the direct-ANE padding golden with its existing generator,
only the 12 `source_sha256` provenance fields changed. All token IDs, padded
IDs and mask values were unchanged (maximum expected-value difference: 0).
Separately, `subsets_are_exact_copies_of_the_sealed_source_cases` passed,
confirming that the certification self-check subsets retain the full fixtures'
metadata and the exact selected case contents.

## Generating references

Run from the repository root. Stage the generated parity fixture JSON files
before comparing or re-sealing them; each run writes a model fixture under
`fixtures/<model>/` and its digest entry in `fixtures/index.json` under the
staging root. For example, to reproduce all four models twice:

```sh
first=$(mktemp -d)
second=$(mktemp -d)
for output in "$first" "$second"; do
  for model in gte-modernbert-base gte-reranker-modernbert-base \
    qwen3-embedding-0.6b qwen3-reranker-0.6b; do
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      uv run --no-project --python 3.12 \
      --with transformers==5.16.1 --with torch==2.14.0 \
      python bench/parity/reference/generate_reference.py \
      --hf-cache ~/.cache/huggingface/hub --model "$model" --output-dir "$output"
  done
done
diff -r "$first/fixtures" "$second/fixtures"
```

To stage `crates/synapse-module/src/catalog/models.json` (inputs and reference
outputs used to check newly loaded catalog lanes) and
`crates/synapse-module/src/fixtures/catalog_rerank_gte_modernbert_fp32.json`
(query/candidate scores used to verify the reranker before release):

```sh
output=$(mktemp -d)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-project --python 3.12 \
  --with transformers==5.16.1 --with torch==2.14.0 \
  python bench/parity/reference/generate_reference.py \
  --hf-cache ~/.cache/huggingface/hub --catalog --output-dir "$output"
```

The catalog mode reads its input texts and query/candidate groups from the
committed catalog and evidence JSON, preserves their order and unrelated
metadata, and writes those same relative paths under the output root. Omit
`--output-dir` only when ready to overwrite the committed outputs. Catalog mode
does not add a self-check to entries that have none, including Qwen3 reranker.

Catalog embedding references use the same CPU backend as parity, without
padding. Qwen3 catalog inputs include the terminal EOS required by the
catalog's owned tokenizer policy; parity cases retain their existing token
IDs. GTE reranker catalog scores use a double-precision sigmoid of the fp32
classifier logit: each stored score equals `1 / (1 + exp(-raw_logit))`.

## Catalog consolidation comparison (2026-10-06)

The measurements below compare the previously committed catalog self-check
outputs (produced by the libraries in the second column) with the unified
generator's deterministic Transformers 5.16.1 / Torch 2.14.0 CPU fp32 outputs.

| Model | Previous reference generator libraries | Maximum absolute difference | Minimum cosine |
| --- | --- | ---: | ---: |
| gte-modernbert-base | Transformers 5.17.0 / Torch 2.14.0 | 0 | 1.0 |
| gte-reranker-modernbert-base | Transformers 5.17.0 / Torch 2.14.0 | 0 | 1.0 |
| qwen3-embedding-0.6b | Candle Transformers 0.10.2 CPU f32 | 7.262907800661966e-7 | 0.9999999999901454 |

Cosine was measured per embedding vector and per reranker query's score row.
The GTE reranker release evidence's raw logits and scores were unchanged
(maximum absolute difference: 0 for each). Qwen3 reranker has no catalog
self-check and therefore no catalog comparison. The three catalog self-checks
listed in the table use fixture revision 2 so installed lanes re-check against
the new provenance and references. Catalog manifest digests depend only on upstream and file
pins, not self-check data, and remain unchanged.
