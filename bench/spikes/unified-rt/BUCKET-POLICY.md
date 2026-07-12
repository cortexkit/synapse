# Serving bucket policy v1

## Status

Bucket policy v1 is the default serving shape policy. `--shapes exact` retains the former corpus-dependent behavior for A/B measurements. The default policy produces 10 shapes for `--max-length 512 --attention-units 4000000`, pre-discovers every accelerator shape before timed inference, and records in-process `first`, `warm`, and `steady` rows with `--passes N`.

## Pure policy function

The shape set depends only on `(max_length, attention_units, policy_version)`; corpus contents and load order are not inputs.

1. Start with the v1 sequence ladder `64, 96, 128, 160, 192, 256, 320, 384, 448, 512`.
2. Remove entries at or above `max_length`, append `max_length`, sort, and deduplicate. This keeps the cap exact for non-ladder maximums.
3. For every sequence bucket `S`, set batch rows to `min(8, floor(attention_units / S²))`. The CLI rejects a budget that cannot hold one `max_length` row.
4. Sort incoming rows by real token length. Add rows until the batch-row limit of the smallest sequence bucket covering the candidate longest row would be exceeded. Dispatch the resulting batch in that covering `(batch, sequence)` shape.
5. Right-pad real rows to the sequence bucket and add fully masked rows to the batch bucket. MiniLM uses its configured pad token, ModernBERT uses `pad_token_id`, and Qwen3 uses zero hidden states for inactive rows. Dummy outputs are discarded. Tokenizer-carried padding is disabled before length calculation and inference; Qwen3 also strips carried padding before adding its required terminal token.

The fixed row cap is intentionally conservative. It keeps tail-row waste below the serving gate on 400-row workloads and avoids making the policy depend on a corpus-specific preferred batch size. Lower attention budgets reduce rows deterministically for the longer buckets.

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

## Distribution justification

The ladder was selected from the retained standard corpora and the 11,293-row magic-context (MC) corpus, not from shapes encountered during serving. Lengths below are after each family's production tokenizer rules and a 512-token cap.

| corpus | rows | real tokens | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM standard | 400 | 66,783 | 146 | 176 | 199 | 215 | 226 | 240 | 300 |
| gte-ModernBERT standard | 400 | 62,838 | 139 | 164 | 187 | 203 | 210 | 225 | 257 |
| Qwen3 standard | 400 | 46,716 | 107 | 123 | 135 | 145 | 150 | 167 | 172 |
| gte-ModernBERT MC | 11,293 | 4,172,183 | 346 | 381 | 415 | 436 | 444 | 450 | 512 |

The initial seven-step candidate `64/96/128/192/256/384/512` had sequence-only waste of 16.47% on standard ModernBERT and 19.32% on standard Qwen3, so it failed the 15% requirement before tail-row padding. Adding 160 resolves the standard-corpus gap; 320 and 448 resolve the dense MC clusters. The measured v1 waste, including inactive tail rows, is 13.62% MiniLM, 13.57% ModernBERT, 12.69% Qwen3, and 7.27% MC.

## Load ownership and cache identity

Metal and CUDA use the same `BatchShape` policy and model-family `embed_batch` seam. During load, accelerator providers execute one fully padded request for every policy shape. Metal therefore compiles or loads all MPSGraph packages and CUDA constructs all enabled per-shape graphs before `cold_load_s` stops. Timed pass 1 performs inference only. CPU has no shape compilation and does not run the accelerator pre-discovery loop.

Metal package roots include both graph revision and shape identity: bucketed roots contain `bucket-policy-v1`, while exact roots contain `shapes-exact`. A policy change cannot reuse an older policy's packages. Result JSON reports the selected shapes plus package count and recursive bytes.

Exact mode intentionally keeps synchronous first-use discovery so it remains an honest baseline for the cost that bucketing removes. Its pass-1 inference includes compilation on a cache miss; its warm and steady passes reuse in-process plans.

## In-process timing and accounting

`--passes N` emits one row per corpus pass. Pass 1 is `first`; the final pass is `steady` when `N > 2`; intervening rows are `warm`. The top-level lane timing remains compatible with the shared result schema and mirrors the final pass. Parity is checked after every pass, outside the inference timer. Corpora above 1,000 rows deterministically sample at most 100 rank queries against the full vector set; all 400 standard rows remain rank queries.

`real_tokens` counts non-padding input tokens. `padded_tokens` is total dispatched bucket area, including sequence padding and inactive batch rows. `padding_waste_fraction` is `(padded_tokens - real_tokens) / padded_tokens`. Bucketed runs fail if this reaches 15%.

## Contended-local tradeoff

These measurements were taken on `Ismets-MacBook-Pro.local`, Apple M5 Max, macOS 26.5.2 build 25F84. The host was not locked or isolated; values are labeled contended-local and are not M1 graduation numbers. All rows use Metal fp32, an empty package directory, the 4,000,000-unit budget, length sorting, and three in-process passes. Standard rows passed the external mean-cosine and all-query top-10 gates on every pass. MC bucketed output passed mean cosine and 100 deterministic top-10 queries against exact-mode output; a separate exact cache-hit parity run passed the same gate.

### Magic-context corpus

| shapes | cold load | packages | package bytes | first tok/s | steady tok/s | padding waste | cosine | top-10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 5.547 s | 158 | 20,491,457 | 16,980 | 17,658 | 0.42% | 1.00000000* | 1.000* |
| bucketed v1 | 11.126 s | 10 | 1,296,600 | 23,678 | 21,220 | 7.27% | 1.00000000 | 1.000 |

`*` The exact three-pass miss run generated the MC reference. Its separate one-pass cache-hit gate measured cosine 1.00000000 and top-10 1.000. This local corpus produced 158 exact packages; the earlier retained f16 run produced 162 packages and 30.0 MiB. The bounded result is stable at 10 regardless of either corpus's exact-shape count.

Bucketing reduced this run from 158 packages and 20.49 MB to 10 packages and 1.30 MB (93.7% fewer packages, 93.7% fewer bytes). It moved preparation into cold load, improved first-pass throughput by 39.4%, and improved the contended steady observation by 20.2%, at 7.27% token-area waste.

### Standard 400-row corpora

| family | shapes | cold load | packages | bytes | first tok/s | steady tok/s | waste | cosine | top-10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM | exact | 2.500 s | 6 | 307,218 | 25,828 | 120,568 | 12.71% | 1.00000000 | 1.000 |
| MiniLM | bucketed | 3.160 s | 10 | 511,878 | 83,104 | 76,749 | 13.62% | 1.00000000 | 1.000 |
| gte-ModernBERT | exact | 6.547 s | 5 | 648,406 | 10,574 | 30,840 | 13.60% | 1.00000000 | 1.000 |
| gte-ModernBERT | bucketed | 8.982 s | 10 | 1,296,600 | 29,985 | 25,296 | 13.57% | 1.00000000 | 1.000 |
| Qwen3 | exact | 4.964 s | 4 | 834,572 | 6,613 | 7,030 | 17.19% | 1.00000000 | 1.000 |
| Qwen3 | bucketed | 12.821 s | 10 | 2,086,264 | 9,868 | 7,879 | 12.69% | 1.00000000 | 1.000 |

The standard corpora use only a subset of possible exact shapes, so bucket pre-discovery stores more packages than these particular exact runs. That is the deliberate bounded-serving trade: package count is fixed before requests arrive, pass 1 has no compilation, and unseen corpora cannot grow the cache. Batch row 8 also lowers steady throughput for MiniLM and ModernBERT on this host; Qwen3 and the long-sequence MC workload benefit. These contended observations justify preserving `--shapes exact` and rerunning both modes under the locked protocol rather than claiming a universal throughput win.

Raw result JSON is retained in `results/bucket-policy/`.

## Locked-M1 follow-up

The next locked-M1 matrix should:

1. Clear each versioned package root and record bucketed miss `cold_load_s`, package count/bytes, and all three in-process pass rows; repeat from a fresh process with cache hits.
2. Run exact mode beside bucketed mode for the three standard corpora and MC, keeping compilation in exact pass 1 visible rather than folding it into load.
3. Verify no package count or modification-time change during bucketed inference.
4. Recheck real/padded tokens and the `<15%` gate against the canonical M1 corpora; the local MiniLM input differs from the prior M1 staging copy.
5. Record GPU occupancy and power for the fixed eight-row policy. If the locked data confirms the MiniLM/ModernBERT steady regression, evaluate a policy-v2 row ladder without changing v1 cache identity or results.
6. Preserve external parity gates for all 400-row cells and deterministic large-corpus rank sampling for MC.
