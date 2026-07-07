# CoIR retrieval eval harness

This harness measures retrieval quality for the exact embedding artifacts shipped by Synapse. It prepares CoIR data in the same `{id, text}` JSONL shape consumed by the bench lanes, lets a lane binary produce vectors, and then scores those vectors offline.

## Environment

This project is managed with `uv` and expects Python 3.12.

```bash
uv sync
```

The optional Python reference reranker uses a separate dependency group so the default harness environment stays lean.

```bash
uv sync --group reference
```

## 1. Prepare the CoIR task files

### cosqa

```bash
uv run prepare.py --task cosqa --out-dir work/cosqa
```

That command downloads the public `CoIR-Retrieval/cosqa` dataset from Hugging Face and writes:

- `work/cosqa/corpus.jsonl`
- `work/cosqa/queries.jsonl`
- `work/cosqa/qrels.tsv`

### csn-python

```bash
uv run prepare.py --task csn-python --out-dir work/csn-python --max-queries 2000
```

That command downloads the Python slice of `CoIR-Retrieval/CodeSearchNet`, keeps the full corpus, and optionally keeps only the first `N` sorted test-query ids plus matching qrels.

The JSONL files use the lane schema:

```json
{"id":"...","text":"..."}
```

## 2. Embed the corpus and queries with a bench lane

Concrete example with the ORT lane and the local MiniLM ONNX snapshot:

```bash
../../target/release/lane-ort-embed \
  --model ~/.cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/<snapshot>/model.onnx \
  --tokenizer ~/.cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/<snapshot>/tokenizer.json \
  --pooling mean \
  --max-length 512 \
  --model-label all-MiniLM-L6-v2-onnx \
  --corpus work/cosqa/corpus.jsonl \
  --vectors-out work/cosqa/corpus-vectors.jsonl \
  --out work/cosqa/corpus-lane-result.json

../../target/release/lane-ort-embed \
  --model ~/.cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/<snapshot>/model.onnx \
  --tokenizer ~/.cache/huggingface/hub/models--Qdrant--all-MiniLM-L6-v2-onnx/snapshots/<snapshot>/tokenizer.json \
  --pooling mean \
  --max-length 512 \
  --model-label all-MiniLM-L6-v2-onnx \
  --corpus work/cosqa/queries.jsonl \
  --vectors-out work/cosqa/query-vectors.jsonl \
  --out work/cosqa/query-lane-result.json
```

### Prefix caveat for retrieval models

Some embedding models expect different text prefixes for documents and queries. The lanes expose a single `--prefix-document` flag, so run the lane once for documents with the document prefix and again for queries with the query prefix.

| Model family | Document run | Query run |
| --- | --- | --- |
| Qwen3-Embedding | none | `Instruct: <one-sentence retrieval task>\nQuery:` |
| nomic-embed | `search_document: ` | `search_query: ` |
| GTE | none | none |
| MiniLM | none | none |
| jina-retrieval | none | none |

- Qwen3 reference: <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>
- nomic reference: <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>

For Qwen3 retrieval tasks, the model card recommends adding a one-sentence instruction to queries and leaving documents unprefixed. For nomic-embed, the model card requires `search_document:` for documents and `search_query:` for queries.

## 3. Score the emitted vectors

```bash
uv run score.py \
  --corpus-vectors work/cosqa/corpus-vectors.jsonl \
  --query-vectors work/cosqa/query-vectors.jsonl \
  --qrels work/cosqa/qrels.tsv
```

`score.py` performs exact brute-force cosine retrieval with NumPy and scores the top results offline with `pytrec_eval`. No network access is needed at score time.

Example report:

```json
{"mrr_at_10":0.0,"n_corpus":20604,"n_queries":500,"ndcg_at_10":0.0,"recall_at_10":0.0,"task":"cosqa"}
```

## 4. Reference rerank cross-checks

Score rerank requests with the Hugging Face reference `Alibaba-NLP/gte-reranker-modernbert-base` implementation:

```bash
uv run --group reference reference_rerank.py \
  --requests work/cosqa/rerank-requests-subset.jsonl \
  --out work/cosqa/reference-scores-subset.jsonl
```

Compare two rerank score files on the same request subset:

```bash
uv run compare_rerank_scores.py \
  --reference-scores work/cosqa/reference-scores-subset.jsonl \
  --candidate-scores work/cosqa/llama-scores-subset.jsonl \
  --requests work/cosqa/rerank-requests-subset.jsonl
```

## Tests

```bash
uv run pytest
```
