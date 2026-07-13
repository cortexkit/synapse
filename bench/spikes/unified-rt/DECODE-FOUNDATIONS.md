# Qwen3 decode foundations on the owned Metal runtime

## Status

Qwen3-0.6B raw-completion generation now runs through the owned Metal runtime with a resident grouped-query KV cache, a bucket-specialized prefill graph, and a bucket-specialized one-token graph. The final gate matched Transformers CPU fp32 for all 64 generated tokens on all 20 prompts (`20/20`, 1,280 compared tokens, no accepted near ties).

This is a research-spike implementation. It proves decode correctness and the intervention seams before production surface design.

## Graph and cache architecture

The cache has one key buffer and one value buffer per transformer layer. Each f16 Metal buffer has logical shape `[1, 8, max_seq, 128]`: batch, the eight physical KV heads, absolute position, and head dimension. Query-head broadcast is performed only inside attention; the 16 query-head representation is never cached. Layer-major buffers keep `(layer, K|V)` directly addressable and make pause-time inspection or a future PAQ intervention independent of a monolithic allocation format.

| Cache bucket | K + V bytes, 28 layers |
|---:|---:|
| 512 | 58,720,256 (56 MiB) |
| 1,024 | 117,440,512 (112 MiB) |
| 2,048 | 234,881,024 (224 MiB) |

The three capacities are fixed compile-at-load buckets. Package paths include the model identity, precision, OS build, graph revision, pass kind, and bucket. Both passes use `MPSGraphOptimizationLevel0`, synchronous specialization, and one immutable package per shape.

### Prefill

The prefill graph has the full bucket shape. The prompt occupies the prefix and the remaining hidden rows are zero padded. Its fp32 additive mask admits only real causal keys; a one-hot selector chooses the final real prompt row for the LM head. Every layer emits its normalized, RoPE-rotated K and V before GQA broadcast. Those outputs initialize the preallocated layer caches.

### Single-token step

The step graph always has query length one and reads the fixed cache bucket. For each layer it computes per-head Q/K RMSNorm, applies RoPE at the absolute cache position, and attends over `[cache, current_KV]`. The mask admits cache positions below the absolute position plus the current token at the fixed tail slot. The graph emits the current unbroadcast K and V; the runtime copies each KV head into the addressed cache position. The same step executable is reused for every position in a bucket.

All weights and cache entries remain f16. RMSNorm, attention scores, softmax, and matrix accumulations are fp32; projection outputs return to f16 between operations. The final norm is followed by a tied `embed_tokens.weight` projection when `tie_word_embeddings` is true, or `lm_head.weight` when it is false. LM-head accumulation and the returned logits are fp32. Greedy selection breaks exact logit ties by the lowest token id, matching first-index argmax behavior.

## Token-exact gate

Reference script: [`reference_qwen3_decode.py`](reference_qwen3_decode.py). It rejects any environment other than `transformers==4.51.0` and `torch==2.13.0`, loads `Qwen/Qwen3-0.6B` on CPU in fp32, and records every generated token plus the top five logits at every step.

Exact generation contract:

- prompt set: [`decode-prompts.jsonl`](decode-prompts.jsonl), 20 varied raw-completion prompts;
- tokenizer: model-canonical `tokenizer.json`, `add_special_tokens=true`;
- no chat template, system prompt, terminal embedding EOS rewrite, padding, or truncation;
- `do_sample=false`, `max_new_tokens=64`, `use_cache=true`; inherited temperature/top-k/top-p values are recorded but ignored by greedy generation;
- model `generation_config.json` EOS list (`151645`, `151643`) and pad id (`151643`);
- candidate: f16 storage with fp32 accumulation, cache bucket 512, greedy argmax;
- divergence report: owned and reference top-five logits at the first differing step;
- a mismatch may only be classified as an f16 near tie when either top-two gap is below `1e-3`.

Command used for the oracle:

```sh
MODEL=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
uv run --python 3.12 \
  --with 'transformers==4.51.0' --with 'torch==2.13.0' --with 'accelerate==1.14.0' \
  bench/spikes/unified-rt/reference_qwen3_decode.py \
  --model "$MODEL" \
  --prompts bench/spikes/unified-rt/decode-prompts.jsonl \
  --out target/qwen3-reference-20x64.jsonl \
  --max-new-tokens 64 --top-k-logits 5
```

Candidate command:

```sh
target/alfonso-decode/debug/spike-unified-rt \
  --model "$MODEL" --tokenizer "$MODEL/tokenizer.json" \
  --generate-prompts bench/spikes/unified-rt/decode-prompts.jsonl \
  --decode-reference target/qwen3-reference-20x64.jsonl \
  --max-new-tokens 64 --decode-cache-bucket 512 --decode-top-k 5 \
  --device metal --dtype f16 --execution explicit \
  --package-cache target/qwen3-decode-packages \
  --decode-tap-out target/qwen3-tap-20x64.jsonl \
  --out target/qwen3-decode-20x64.json
```

Final result: **20/20 prompts and 1,280/1,280 tokens exact; zero near-tie exemptions**.

The first full run found one real failure at `completion-06`, step 7. The owned top two were `13079=16.741596` and `61686=16.739628`; CPU fp32 had `61686=16.739162` and `13079=16.737160`. Both gaps exceeded `1e-3`, so the gate correctly rejected it. Casting every matmul operand to fp32 for accumulation, while retaining f16 weight/cache/intermediate storage, repaired that failure. The complete gate then passed without exemptions.

## Instrumentation seams

The controller in `qwen3_decode.rs` owns intervention semantics rather than hiding them in the CLI.

### Pre-commit token-stream tap

`TokenStreamTap::before_commit` receives `(step, token_id, &[TopLogit])` after argmax and before sequence/cache commitment. The callback has an immutable top-k view and cannot change selection.

Test: `qwen3_decode::tests::token_stream_tap_observes_before_commit_without_changing_tokens` compares tapped and untapped runs and verifies every event's winner.

### Pausable, inspectable state

`DecodeSession` retains the sequence, generated suffix, next-step fp32 logits, backend cache, and absolute cache position. Stopping after any `generate` call is a pause; calling it again resumes from the same state. `position` and `inspect_cache_layer` expose progress and a layer's K-then-V values without rebuilding the graph.

Test: `qwen3_decode::tests::paused_state_resumes_to_uninterrupted_tokens` pauses after six steps, inspects the cache, and proves the resumed 16-token result equals an uninterrupted run.

### Forced token splice

`DecodeSession::splice` commits external token ids one at a time through the same absolute-position step graph. Q/K per-head RMSNorm, RoPE, and every layer cache therefore advance exactly as they do for normal generated tokens. Generation resumes from logits produced after the final forced token.

Test: `qwen3_decode::tests::splice_matches_prefilling_the_concatenated_sequence` proves that splice-then-continue equals a fresh prefill of the concatenated sequence followed by continuation, including byte-for-value cache inspection in the deterministic test kernel.

### Addressable weight regions

`Model::weight_regions` returns an ordered `(Option<layer>, tensor_name) -> WeightRegion` map. Each entry carries the model-owned pointer used as the Metal static-buffer handle, byte length, and deterministic checksum. Global entries cover embeddings, final norm, and an untied LM head; each layer exposes both norms and all attention/MLP projections. The pointers stay stable for the loaded model's lifetime.

Test: `qwen3_decode::tests::addressable_weight_regions_are_byte_identical_across_loads` writes a complete tiny Qwen3 safetensor fixture, loads it twice through the normal model loader, enumerates both maps, and byte-compares every region.

## Informational development-host performance

These are contended Apple M5 Max development-host numbers from the correctness command above, using a debug Rust binary and already-built graph-v6 packages. They are not locked-hardware or graduation results.

| Cache | Prompts | Prefill tokens | Generated tokens | Prefill tok/s | Decode tok/s |
|---:|---:|---:|---:|---:|---:|
| 512 | 20 | 203 | 1,280 | 41.9 | 3.24 |
| 1,024 | — | — | — | not run | not run |

The 512 decode number includes package loading on the first step and synchronous readback of top-five logits plus current K/V outputs. The 1,024 snapshot was intentionally skipped rather than presenting another contended compile run as steady-state evidence. Graduation still requires locked hardware and the same harness against llama-Metal and MLX.

## Before production

Decode still needs:

- continuous and multi-request batching, admission control, cancellation, and cache reclamation;
- temperature, top-k/top-p, repetition controls, deterministic seeded sampling, and log-prob policy;
- chat templates and role/system-message policy (this wave is raw completion only);
- an async streaming API with backpressure rather than the research JSON output and tap file;
- cache paging/compaction and long-context policy beyond fixed 512/1,024/2,048 buckets;
- device-side KV writes and token selection to remove synchronous K/V and logits readback;
- locked-hardware latency, throughput, energy, and memory comparisons against llama-Metal and MLX;
- native-model intervention tests for pause/splice under production scheduling, plus PAQ mutation safety and rollback policy.
