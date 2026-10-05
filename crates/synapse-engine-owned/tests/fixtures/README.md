# Metal rerank boundary fixtures

The `pairs` arrays are the first two cases' `input_ids` from the corresponding
committed `bench/parity/fixtures/<slug>/<slug>.ref-v1.transformers-5.16.1.seed-0.json`.
They are already composed inputs; the engine does not tokenize text.

`bucket_pairs` is their stable ascending-length order used by the Metal batch
planner. `ids` right-pads those pairs with the manifest pad id to a 2 × 128
bucket; `mask` marks the original lengths, including any special/template tokens.
The capture test checks the actual bucket-planner output and the padding helper
shared by both reranker families, then checks candidate-order score restoration.
Module text composition is upstream of this engine-owned test boundary.
