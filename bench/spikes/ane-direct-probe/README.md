# Direct-API GTE ModernBERT probe

This standalone Cargo workspace drives `_ANEInMemoryModel` directly. It is not a
member of the repository workspace because the private-API binding comes from a
separate MIT-licensed checkout.

## Full-model correctness probe

Build and run with a local Hugging Face snapshot and the committed pre-tokenized
rows:

```sh
cargo run --release --bin modernbert_full -- \
  "$MODEL_SNAPSHOT" rows.jsonl \
  --seq 512 \
  --layers-per-executable 1 \
  --warm-repetitions 5 \
  --report "$REPORT_PATH" \
  --vectors-out "$VECTORS_PATH"
```

`$MODEL_SNAPSHOT` must contain `config.json` and `model.safetensors`. The binary
never tokenizes. Each JSONL row supplies either `input_ids` for one shape or an
`input_ids_by_shape` object keyed by sequence length. IDs are padded with the
checkpoint's pad ID, matching the production engine boundary.

The default is one transformer layer per executable. `--layers-per-executable
2` is the bounded fusion control. Attention keeps heads on the channel axis and
uses one batched matrix multiplication per query tile rather than dispatching a
matrix multiplication per head.

The report contains every normalized 768-dimensional vector, row-set and model
digests, row-wise cosine against the built-in CPU fp32 reference, determinism,
checkpoint localization, warm wall time, and one-minute load average. A 512 run
exits unsuccessfully unless minimum cosine is at least 0.999 and repeated vectors
are byte-identical.

`rows.jsonl` contains four short real-text rows and four repeated-prose rows near
each of the 512, 1024, and 2048 shape limits. Its SHA-256 is
`f4889a38df77b9940ce973c4d9b82857d0c401987ae8e77b5ca25e6062808c39`.

## Direct-ANE sequence-classification spike

`modernbert_rerank` reuses the ModernBERT encoder graph builders for
`Alibaba-NLP/gte-reranker-modernbert-base`. It loads the checkpoint's `model.*`
encoder weights and performs masked mean pooling, dense, exact-erf GELU, layer
norm, and classifier (including classifier bias) on CPU in fp32. It is a spike,
not a served lane. The retained 20-pair evaluation **failed** the required
sigmoid-score fidelity gate; do not use this binary's scores as an accepted
benchmark arm. See the Neural Engine follow-up in
`docs/evidence/rerank-eval-2026-09/README.md`.

```sh
env -u TMPDIR cargo run --release --bin modernbert_rerank -- "$MODEL_SNAPSHOT" < pairs.jsonl
```

Input is one JSON object per line with `id` and `input_ids`: use Hugging Face
`tokenizer(query, document, truncation=False, padding=False)` including special
tokens. Every supplied position is attended; the binary adds masked padding to
the smallest multiple of 64 and rejects empty or over-limit inputs. It caches
one encoder bundle per width for the process lifetime. Output starts with a
`ready` model-identity record, followed by one JSON result per pair containing a
raw logit, token count, shape, encoder/CPU-head/total milliseconds, and compile
milliseconds only on a shape's first use. Encoder time includes CPU embedding
gather and IOSurface transfers; total excludes compilation, JSON I/O and model
loading. Pooling is included in CPU-head time. Optional `reference_pool` input
and returned `reference_head_logit` isolate head arithmetic from encoder error;
returned `pooled` values support private fidelity diagnosis.

The caller must check the read-only `campaign_rig_claim` table before inference
and between queries, waiting until no campaign holds the Neural Engine. All
reported timing is ambient wall time, not isolated device execution time.

### Formula and layer diagnosis

Add `--diagnose "$PRIVATE_DIAG_DIR"` after the snapshot argument to score each
input with two CPU reference encoders as well: the ANE graph's tanh-GELU formula
and exact-erf GELU. Both use the same fp32 pooling/classification head. The
embedding probe's default CPU reference already used exact erf; its default
behavior and the shared ANE graph are unchanged.

Set `capture_layers: true` on selected input rows to save all 22 post-residual,
pre-final-norm hidden states beneath `pair-N` in the diagnostic directory, where
N is the one-based input ordinal. Files are little-endian fp32: ANE files are
channel-major `[hidden, width]`; CPU files are token-major `[width, hidden]`.
The `cpu_logits` result object distinguishes `tanh` and `erf`. Diagnostic inputs
containing literal pad IDs are rejected because the existing embedding CPU
reference masks those IDs, while pair inference attends every supplied token.
Layer capture adds surface reads and disk I/O, so diagnostic timings are not
comparable to normal scoring timings. Keep states and token IDs private.
