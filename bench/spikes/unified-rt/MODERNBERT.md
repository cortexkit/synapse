# ModernBERT graph

## Scope

This spike adds `Alibaba-NLP/gte-modernbert-base` to the owned fp32 runtime. The CLI detects `model_type: "modernbert"` and selects the ModernBERT graph; the existing MiniLM path is unchanged. Both backends load the Hugging Face safetensors directly. The published gte checkpoint stores fp16 parameters, which the loader expands to fp32 before execution.

## Architecture verified from Hugging Face

The implementation was checked against the gte configuration and Transformers' `modeling_modernbert.py` / `configuration_modernbert.py`, rather than inferred from ordinary BERT:

- 22 layers, hidden size 768, 12 attention heads, and GeGLU intermediate size 1152.
- Token embeddings followed by bias-free layer normalization. There are no learned position or token-type embeddings.
- Pre-norm residual blocks. Layer 0 intentionally has no attention norm; later layers do. Every layer has an MLP norm, and the encoder ends with a final norm.
- Fused bias-free `Wqkv`, bias-free attention output, and bias-free MLP projections.
- GeGLU uses `gelu(first_half(Wi(x))) * second_half(Wi(x))` before `Wo`.
- RoPE rotates half-head pairs. Full-attention layers use theta 160,000 and sliding-attention layers use theta 10,000.
- Layers 0, 3, 6, ... are full attention. The other layers use an inclusive half-window of 64, corresponding to the configured total local window of 128.
- Padding is excluded from attention using `pad_token_id` and the tokenizer attention mask. Any tokenizer-level baked padding is disabled with `tokenizer.with_padding(None)`.
- The gte embedding is the final normalized hidden state at the CLS position, followed by L2 normalization. Padding is never considered by pooling.

The configuration contains the legacy `position_embedding_type: "absolute"` field, but the ModernBERT modeling source does not create an absolute-position table; it creates RoPE values for each attention type.

## CPU provider

`src/modernbert.rs` owns the model and graph. Dense operations call the same Accelerate SGEMM primitive as the MiniLM CPU provider. Layer norm, exact-erf GELU, RoPE, additive masking, softmax, residuals, CLS pooling, and L2 normalization are owned Rust fp32 operations.

The parity command checks both certification gates and fails the run if either misses:

- mean per-vector cosine >= 0.9999
- mean top-10 neighbor overlap >= 0.995, using every vector as a query

An ignored integration test, `modernbert_400_chunk_parity_gate`, asserts the same gates over exactly 400 vectors when `MODERNBERT_PARITY_VECTORS` and `MODERNBERT_REFERENCE_VECTORS` are supplied. This keeps the large, gitignored corpus and generated vectors out of normal unit tests while making the certification check executable rather than print-only.

## Metal block-resident graph

`src/modernbert_mpsgraph.m` builds one MPSGraph containing all 22 encoder layers and the final norm for each `(batch, sequence length)` shape. Hidden states are uploaded once after CPU embedding lookup/norm and read once after the final norm. QKV, both RoPE variants, attention scores, softmax, context, residuals, GeGLU, and every encoder norm remain on device.

Static weights are cached as Metal buffers by source pointer. Graph plans are cached by shape and attention pattern. RoPE tables and the additive local band are precomputed per sequence-length bucket in Rust. Each batch combines that band with its key-padding mask. Full and local masks are fed separately, so alternating layers select the right mask without a CPU/device boundary.

The Metal graph is intentionally fp32. The existing MiniLM MPSGraph f16 experiment was slower on M1, and this lane's required reference is fp32.

## Parity results

Reference vectors came from `lane-ort-embed` using the cached gte fp32 ONNX export, `--pooling cls`, max length 512, and the first 400 rows of `bench/data/corpus-v2.jsonl`. The subset contains 62,838 post-tokenization input tokens. Both owned paths used length-sorted batching with a 4,000,000 attention-unit budget.

| Backend | Mean cosine vs ORT fp32 | Mean top-10 overlap | Gate |
|---|---:|---:|---|
| CPU / Accelerate fp32 | 0.9999999999973957 | 1.0000 | pass |
| Metal / MPSGraph resident fp32 | 0.9999999999970659 | 1.0000 | pass |

## M1 throughput

Timed runs were made only on `[bench-host-alias]`, under `[bench-user-home]/bench.lock`, over the same 400 chunks and length-sorted batching policy. “Warm” is an immediate second process run, so it includes the effect of OS and Metal shader caches while rebuilding process-local graph plans.

| Backend | First-run tok/s | Warm tok/s | First infer time | Warm infer time |
|---|---:|---:|---:|---:|
| CPU / Accelerate fp32 | 1,066.2 | 1,068.9 | 58.934 s | 58.787 s |
| Metal / MPSGraph resident fp32 | 12,405.3 | 14,564.3 | 5.065 s | 4.315 s |

Metal was about 11.6x faster than CPU on the first run and 13.6x faster warm. The first/warm difference is material for Metal and negligible for CPU.

## Reproduction notes

The gte Hugging Face snapshot must contain `config.json`, `tokenizer.json`, and `model.safetensors`. A parity run also needs the ORT vectors as JSONL `{id, vec}` rows. Example:

```sh
spike-unified-rt \
  --model /path/to/gte-modernbert-base \
  --tokenizer /path/to/gte-modernbert-base/tokenizer.json \
  --corpus bench/data/corpus-v2.jsonl \
  --limit 400 \
  --device metal --dtype f32 \
  --reference /tmp/gte-ort-400.jsonl \
  --vectors-out /tmp/gte-owned-400.jsonl \
  --out /tmp/gte-owned-result.json
```

The largest implementation traps were the layer-0 identity attention norm, different RoPE theta values for local and global layers, the inclusive 64-token half-window, GeGLU half ordering, and padding embedded as the configured pad token rather than token zero.
