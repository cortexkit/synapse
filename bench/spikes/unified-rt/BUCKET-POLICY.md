# Serving bucket policies

## Status

Bucket policy v1 remains the default serving shape policy. `--bucket-policy 1|2` selects a version while `--shapes exact` retains corpus-dependent shapes for A/B measurements. Both bucket policies produce 10 shapes for `--max-length 512 --attention-units 4000000`, pre-discover every accelerator shape before timed inference, and record in-process `first`, `warm`, and `steady` rows with `--passes N`. Policy v2 is opt-in and varies the batch-row limit by sequence bucket, using the row caps listed below; it is not the default.

## Pure policy function

The shape set depends only on `(max_length, attention_units, policy_version)`; corpus contents and load order are not inputs.

1. Start with the v1 sequence ladder `64, 96, 128, 160, 192, 256, 320, 384, 448, 512`.
2. Remove entries at or above `max_length`, append `max_length`, sort, and deduplicate. This keeps the cap exact for non-ladder maximums.
3. For every sequence bucket `S`, set batch rows to `min(8, floor(attention_units / S²))`. The CLI rejects a budget that cannot hold one `max_length` row.
4. Sort incoming rows by real token length. Add rows until the batch-row limit of the smallest sequence bucket covering the candidate longest row would be exceeded. Dispatch the resulting batch in that covering `(batch, sequence)` shape.
5. Right-pad real rows to the sequence bucket and add fully masked rows to the batch bucket. MiniLM uses its configured pad token, ModernBERT uses `pad_token_id`, and Qwen3 uses zero hidden states for inactive rows. Dummy outputs are discarded. Tokenizer-carried padding is disabled before length calculation and inference; Qwen3 also strips carried padding before adding its required terminal token.

The fixed row cap bounds tail-row padding without making the policy depend on a corpus-specific preferred batch size. Lower attention budgets reduce rows deterministically for the longer buckets.

The complete default set is:

| sequence | batch rows | attention units |
|---:|---:|---:|
| 64 | 8 | 32,768 |
| 96 | 8 | 73,728 |
| 128 | 8 | 131,072 |
| 160 | 8 | 204,800 |
| 192 | 8 | 294,912 |
| 256 | 8 | 524,288 |
| 320 | 8 | 819,200 |
| 384 | 8 | 1,179,648 |
| 448 | 8 | 1,605,632 |
| 512 | 8 | 2,097,152 |

The set has 10 entries, below the 12-shape target. Changing the ladder, row cap, mapping, or padding semantics requires a policy-version bump.

## Policy v2 row ladder

Policy v2 retains the v1 sequence ladder and padding semantics but makes the row cap a pure function of sequence-bucket index. For the default ten buckets, the caps are `16/16/16/16/16/16/12/12/8/8`. The attention-unit budget remains an upper bound, so the final row count is `min(indexed_cap, floor(attention_units / S²))`.

| index | sequence | v2 batch rows | attention units |
|---:|---:|---:|---:|
| 0 | 64 | 16 | 65,536 |
| 1 | 96 | 16 | 147,456 |
| 2 | 128 | 16 | 262,144 |
| 3 | 160 | 16 | 409,600 |
| 4 | 192 | 16 | 589,824 |
| 5 | 256 | 16 | 1,048,576 |
| 6 | 320 | 12 | 1,228,800 |
| 7 | 384 | 12 | 1,769,472 |
| 8 | 448 | 8 | 1,605,632 |
| 9 | 512 | 8 | 2,097,152 |

Policy v1 remains the default. Select v2 explicitly to compare its row ladder and distinct `bucket-policy-v2` cache identity.

## Load ownership and cache identity

Metal and CUDA use the same `BatchShape` policy and model-family `embed_batch` seam. During load, accelerator providers execute one fully padded request for every policy shape. Metal therefore compiles or loads all MPSGraph packages and CUDA constructs all enabled per-shape graphs before `cold_load_s` stops. Timed pass 1 performs inference only. CPU has no shape compilation and does not run the accelerator pre-discovery loop.

Metal package roots include both graph revision and shape identity: bucketed roots contain `bucket-policy-v1` or `bucket-policy-v2`, while exact roots contain `shapes-exact`. A policy change cannot reuse another policy's packages. Result JSON reports the selected version and shapes plus package count and recursive bytes.

Exact mode intentionally keeps synchronous first-use discovery so it remains an honest baseline for the cost that bucketing removes. Its pass-1 inference includes compilation on a cache miss; its warm and steady passes reuse in-process plans.

## In-process timing and accounting

`--passes N` emits one row per corpus pass. Pass 1 is `first`; the final pass is `steady` when `N > 2`; intervening rows are `warm`. The top-level lane timing remains compatible with the shared result schema and mirrors the final pass. Parity is checked after every pass, outside the inference timer. Corpora above 1,000 rows deterministically sample at most 100 rank queries against the full vector set; all 400 standard rows remain rank queries.

`real_tokens` counts non-padding input tokens. `padded_tokens` is total dispatched bucket area, including sequence padding and inactive batch rows. `padding_waste_fraction` is `(padded_tokens - real_tokens) / padded_tokens`. Result JSON reports `padding_waste_gate_passed`; bucketed runs write the auditable result and then return a failure if waste reaches 15%. Exact-mode results report no bucket gate.
