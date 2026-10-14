# ModernBERT reranking

## Scope

The owned runtime supports `Alibaba-NLP/gte-reranker-modernbert-base` as a rerank capability on `ModelFamily`. The embedding runner and the `gte-modernbert-base` CLS-plus-L2 output remain unchanged. Reranking is fp32-only on the CPU correctness path and the block-resident Metal path.

The CLI selects reranking with `--rerank-requests`. Each JSONL row has `id`, `query`, and a non-empty `documents` array. `--scores-out` writes one raw-logit array in document order. `--reference` accepts the same `{id, scores}` JSONL emitted by `bench/eval-coir/reference_rerank.py` and enforces Pearson and tie-aware top-1 gates.

## Token path verification

The checkpoint's `tokenizer_config.json` specifies right truncation, longest-first pair truncation, right padding, an 8,192-token model maximum, and the standard `[CLS] query [SEP] document [SEP]` post-processor. Runtime tokenizer padding is disabled; graph batches explicitly fill with token ID 50283 and mask every padded position from attention and masked-mean pooling. Buckets use the combined pair length.

The committed Transformers tokenizer fixture includes the following query/document input and token IDs:

- query: `what is rust?`
- document: `Rust is a systems programming language.`
- IDs: `[50281, 5371, 310, 20035, 32, 50282, 51, 461, 310, 247, 2718, 10717, 3448, 15, 50282]`

The committed `fixtures/rerank-reference.json` records this check and a second pair. The ignored end-to-end fixture test validates both IDs and raw scores when `MODERNBERT_RERANK_MODEL` points at the checkpoint.

## Classification head found in the checkpoint

The checkpoint differs from its embedding sibling in an important way: `gte-modernbert-base` embeds the CLS token, while the reranker config sets `classifier_pooling: "mean"`. The reranker masks padding, averages the final normalized encoder states, and then applies the sequence-classification head; CLS pooling would compute a different score.

The loaded tensors and exact inference path are:

| Stage | Tensor / behavior | Shape |
|---|---|---:|
| Pool | attention-mask-weighted mean | 768 |
| Dense | `head.dense.weight`, no bias | 768 x 768 |
| Activation | exact-erf GELU | 768 |
| Norm | `head.norm.weight`, epsilon 1e-5, no bias | 768 |
| Projection | `classifier.weight` | 1 x 768 |
| Bias | `classifier.bias` | 1 |

The output is the single f32 regression logit. The Python reference reads `model(...).logits[:, 0]`; it does **not** apply sigmoid. The owned runtime therefore writes raw logits too.

CPU runs the encoder and the same masked-mean/head math in Rust fp32. Metal extends the resident MPSGraph through masked mean, dense, GELU, norm, and classifier, returning one score per pair rather than reading encoder hidden states to the host. Rerank plans are separately keyed from embedding plans. Explicit O0 compilation is the default, graph revision 3 participates in cache identity, and one `.mpsgraphpackage` is stored per combined-length bucket shape.

## Scoring gates

The required thresholds are Pearson >= 0.999 and tie-aware top-1 >= 0.98. Supply `--reference` JSONL `{id, scores}` rows with the same request IDs and document order as the candidate scores. Constant-score requests do not have a defined per-request correlation.

## Tests and reproduction

- `modernbert_rerank_matches_transformers_reference_fixture_on_cpu` checks pair IDs and end-to-end CPU logits against the committed Transformers snippet.
- `modernbert_rerank_static_feeds_survive_multiple_calls` exercises different Metal shapes before rescoring the fixture and compares against a fresh-process invocation, guarding static-feed lifetime and graph-cache reuse.
- `rerank_gate_uses_pearson_and_tie_aware_top1` keeps the ranking gate semantics under ordinary CI.

Example:

```sh
spike-unified-rt \
  --model /path/to/gte-reranker-modernbert-base \
  --tokenizer /path/to/gte-reranker-modernbert-base/tokenizer.json \
  --rerank-requests /tmp/cosqa-rerank.jsonl \
  --scores-out /tmp/owned-rerank-scores.jsonl \
  --reference /tmp/transformers-rerank-scores.jsonl \
  --out /tmp/owned-rerank-result.json \
  --device metal --dtype f32 \
  --shapes bucketed --execution explicit \
  --package-cache /tmp/owned-rerank-packages \
  --max-length 512
```
