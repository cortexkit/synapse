# Spike A — llama.cpp in-process embeddings

## RESOLUTION (post-spike, 2026-07-08): MiniLM parity failure root-caused and fixed

The "owned pretokenized token-id path" suspect below was correct, and the exact
mechanism is now proven: the Qdrant `all-MiniLM-L6-v2-onnx` `tokenizer.json`
ships with **baked-in fixed padding to 128 tokens** (`padding.strategy =
{"Fixed": 128}`). Our lanes override truncation at load time but never touched
padding, so every short chunk carried up to 124 `[PAD]` ids. The ort lane is
immune because it passes the attention mask (pads are masked out of pooling);
llama.cpp batches have no attention-mask concept, so the pads entered the
forward pass as real tokens and poisoned the pooled vectors. The child was
immune because llama-server tokenizes raw text itself from the GGUF vocab.

Fix: `tokenizer.with_padding(None)` in `load_tokenizer` (committed). Re-measured
on the 1,000-chunk subset vs ort fp32 reference:

| Path | mean cosine | rank overlap (k=10) | worst decile |
|---|---:|---:|---:|
| in-process manual mean pool, padding stripped | **0.9999977** | 0.995 | 0.95 |

Real token count 166,697 (was 172,746 with pads — note: lanes tokenizing with
this artifact counted pads in tok/s, a ~3.5% inflation on MiniLM rows; vectors
were always correct where attention masks were honored).

## Qwen latency: resolved (post-spike, 2026-07-08)

The `17.7 ms` vs `7.4 ms` Qwen3 gap was **not** a raw in-process penalty. The
child benchmark was reusing one hot `llama-server` slot across 200 identical
requests, and the server kept the first `N-1` tokens resident between calls.
Verbose server logs show the exact mechanism on every repeated request:
`n_past = 47`, then `n_past was set to 46`, so only the **last token** is
re-evaluated. Our in-process loop reset the sequence and re-ran all `47`
tokens every time, so the comparison mixed **fresh full-query embed** with
**cached suffix replay**.

### Measurement trail on the M1 Max

| Variant | Long p50 / p95 | Short p50 / p95 | Read |
|---|---:|---:|---|
| in-process, full query, reset seq, fixed slot, 8 threads | 17.700 / 17.884 ms | 13.852 / 13.934 ms | baseline fresh embed |
| in-process, fresh seq ids, no reset, rotate across 201 slots | 17.680 / 17.762 ms | 13.788 / 13.877 ms | clearing cost is not the gap |
| in-process, keep prefix KV and re-evaluate only the last token (`--reuse-prefix-tokens 46` long / `15` short) | **6.090 / 6.718 ms** | **6.080 / 6.187 ms** | mirrors server reuse path |
| child `llama-server` default | 7.373 / 8.044 ms | 7.303 / 7.697 ms | one hot slot, repeated identical prompt |
| child `llama-server --slot-prompt-similarity 0.0 --no-cache-prompt --cache-ram 0` | 7.378 / 8.109 ms | 7.306 / 7.978 ms | still reuses the live slot by LRU |

Two useful negatives:

- changing threads from `8` to `10` did nothing meaningful (`17.700 ms` ->
  `17.721 ms` on the long query),
- rotating across fresh sequence ids without clearing changed almost nothing,
  so the missing ~10 ms was not sequence-reset overhead or graph-shape churn.

The step timings from the in-process long-query run make that visible: batch
build is ~`0.001 ms`, explicit KV reset is ~`0.003 ms`, the `decode()` call
returns in ~`1.09 ms`, and the remaining wall time lands when pooled
embeddings are materialized back to host memory. When we keep the first 46
Qwen tokens resident and replay only the last token, that pool/materialization
phase drops from ~`16.6 ms` to ~`5.0 ms` and total p50 drops below the child.

### Root cause and consequence

- **Root cause:** the old benchmark compared two different workloads.
  `llama-server` was serving an identical-query **prefix-cache replay** path;
  the in-process lane was serving a **fresh full-query embed** path.
- **Consequence for Synapse:** for ordinary one-off search queries on this M1,
  the honest Qwen3 full-query cost is still about `17-14 ms` depending on query
  length. If Synapse has an exact-prefix reuse opportunity, in-process llama.cpp
  can exploit it too and is actually faster than the child on that path.
- **Benchmarking consequence:** repeated-same-text latency loops understate the
  child's real single-query cost. To compare fresh-query latency fairly, the
  harness must either vary the query text each iteration or explicitly clear the
  server slot state between requests.

One issue still remains open from this spike:
1. **builtin-pooling + encode path aborts** with
   `GGML_ASSERT(ggml_can_mul_mat)` on unpadded inputs (llama-cpp-2 0.1.151) —
   manual pooling is the working encoder path in-process. Also a live example
   of the in-process crash-domain concern: the assert is a process abort, not
   an error return.

The verdict below is the honest pre-resolution record; with parity fixed and
Qwen latency root-caused, the remaining open question is the builtin-pooling
encoder abort.

## Status (pre-resolution record)

**Result: still no clear replacement for the child path.**

This follow-up changed the question from:

- "does in-process with our own manual pooling clear the bar?"

to:

- "does in-process with llama.cpp's builtin sequence pooling clear the bar?"

The answer is still **no** on the clean **M1 Max** bench box.

- **MiniLM**: builtin pooling did **not** fix parity. It landed at **0.9736516294** vs ORT fp32, essentially identical to the manual path, while the child lane stayed at **0.9999959826**.
- **Qwen3**: builtin pooling kept parity at **0.9999996803** and improved throughput over the child, but the in-process single-call latency gap stayed large: **~17.7 ms p50** vs **~7.39 ms p50** for the child on the same box.

So the new finding is sharper than the first pass: **the MiniLM failure is not manual-pooling-specific on a clean machine**. Builtin pooling and manual pooling behave the same under the owned pretokenized token-id path.

## Path decision: `llama-cpp-2` vs direct FFI

I stayed on **`llama-cpp-2` 0.1.151** with Metal enabled and did **not** fall back to hand-rolled `llama.h` bindings.

Why:

- the crate built cleanly,
- it exposes context pooling type, sequence embeddings, and flash-attention policy,
- the blocker was not missing API or build failure,
- the blocker was the **measured behavior** of the in-process token-id path.

## Measurement setup

All timed measurements in this follow-up ran on the **M1 Max bench box over LAN SSH**, not the contended M5.

- Binaries were built locally for arm64 and rsynced to `~/bench-tools/llama-inproc-followup/`.
- The 2,000-item subset was regenerated on the M1 from `~/Work/synapse/corpus/aft-chunks.jsonl` using `text = embed_text` and stable line-number ids.
- Child-process comparison used `~/bench-tools/bin/llama-server-wrap.sh`.
- Single-call latency used two query shapes:
  - a longer code-search query (the same one as the first pass),
  - a shorter search-like query.
- Caveat: under the MiniLM tokenizer, both query strings ended up at **128 tokens**, so the "short" query was not actually a shorter model input for MiniLM. Under Qwen3, the short query was **16 tokens** vs **47 tokens** for the long query.

## Implementation changes in this follow-up

The crate now supports:

- `--pooling-implementation builtin|manual`
- builtin sequence pooling via `embeddings_seq_ith`
- manual token pooling via `embeddings_ith`
- `--flash-attention auto|enabled|disabled`
- `--reset-policy sequence|context|none`
- `--sequence-strategy fixed|rotate`
- `--reuse-prefix-tokens N` for server-style identical-prefix replay experiments
- latency tests that reuse one context and can either reset, rotate, or replay only a suffix instead of rebuilding contexts

## 1) Parity + throughput over the first 2,000 chunks

### all-MiniLM-L6-v2

| Path | tok/s | Mean cosine vs ORT fp32 | Cold load | Result |
|---|---:|---:|---:|---|
| ORT fp32 reference | 11,737.63 | reference | 0.0766 s | baseline |
| in-process, manual mean pool | 107,963.40 | 0.9736516294 | 0.0660 s | **FAIL** |
| in-process, builtin mean pool | 106,517.29 | 0.9736516294 | 0.0661 s | **FAIL** |
| child (`llama-server`) | 56,099.94 | 0.9999959826 | 0.2707 s | PASS |

### Qwen3-Embedding-0.6B

| Path | tok/s | Mean cosine vs ORT fp32 | Cold load | Result |
|---|---:|---:|---:|---|
| ORT fp32 reference | 319.74 | reference | 2.4967 s | baseline |
| in-process, manual last pool | 4,078.66 | 0.9999996803 | 1.4038 s | PASS |
| in-process, builtin last pool | 4,623.60 | 0.9999996803 | 1.0074 s | PASS |
| child (`llama-server`) | 3,785.74 | 0.9999996788 | 0.5326 s | PASS |

### Parity verdict

The requested builtin-pooling follow-up **did not rescue MiniLM**.

Manual and builtin MiniLM parity are numerically the same to the printed precision, which changes the diagnosis:

- the clean-box failure is **not** explained by manual pooling,
- the remaining suspect is the broader **owned pretokenized token-id path** for encoder models.

That is an inference from the measurements, not a proven internal root cause, but it is the strongest supported read.

## 2) Single-call latency, 200 warm sequential single-text embeds

### MiniLM

| Path | Query shape | p50 | p95 | Cold load |
|---|---|---:|---:|---:|
| in-process, manual | long | 2.536 ms | 3.115 ms | 0.0788 s |
| in-process, builtin | long | 2.547 ms | 2.595 ms | 0.0620 s |
| child | long | 2.734 ms | 5.899 ms | 0.2642 s |
| in-process, builtin | short | 2.533 ms | 2.564 ms | 0.0630 s |
| child | short | 2.619 ms | 6.292 ms | 0.2641 s |

MiniLM in-process is slightly better than the child on latency, especially in the tail, **but the parity miss makes that win unusable**.

### Qwen3 regression chase

Long-query results on the same M1 box:

| In-process variant | ctx / batch / ubatch | flash | p50 | p95 |
|---|---|---|---:|---:|
| manual last pooling | 1024 / 4096 / 1024 | auto | 17.127 ms | 17.195 ms |
| builtin last pooling | 1024 / 4096 / 1024 | auto | 17.700 ms | 17.790 ms |
| builtin last pooling | 256 / 256 / 256 | auto | 17.677 ms | 17.765 ms |
| builtin last pooling | 128 / 128 / 128 | auto | 17.680 ms | 17.774 ms |
| builtin last pooling | 256 / 256 / 256 | enabled | 17.670 ms | 17.764 ms |
| child | 1024 / 4096 / 1024 | server default | **7.386 ms** | **8.012 ms** |

Short-query result:

| Path | Query shape | p50 | p95 |
|---|---|---:|---:|
| in-process best tried | short | 13.726 ms | 13.823 ms |
| child | short | **7.302 ms** | **7.693 ms** |

### Qwen latency verdict

The gap **did not close**.

What I tried:

1. builtin last pooling instead of manual,
2. much smaller `ctx` / `batch` / `ubatch` sized for single queries,
3. a latency loop that reuses one context and only resets the sequence cache between calls,
4. explicit flash attention enable.

What happened:

- builtin pooling was actually a little **slower** than manual on the long query,
- smaller context and batch sizes changed almost nothing,
- explicit flash enable changed almost nothing,
- the child stayed around **2.3x faster** on long-query p50.

So the supported conclusion is: **the Qwen single-call latency regression remains unexplained by the obvious config knobs and is real on the M1**.

## 3) KV-reservation footgun

This remains prominent and is a real production concern.

Observed in the full-corpus Qwen in-process run on the M1:

- `n_seq_max = 101`
- Metal KV buffer size = **11,312 MiB**

MiniLM was tame by comparison:

- `n_seq_max = 8`
- no large KV cache because the encoder path is non-causal

This means the in-process runner cannot naively size `n_seq_max` from the largest corpus batch shape and then keep that context hot forever for query latency. The design needs explicit separation between:

- **throughput contexts** sized for corpus batches, and
- **hot single-query contexts** sized narrowly for the latency path.

Even after trying smaller single-query contexts, the Qwen latency gap to the child still held, but the KV sizing issue is independently real and must stay visible in the design.

## 4) Updated interpretation

The first-pass M5 diagnosis was:

- "MiniLM parity failure is manual-pooling-specific."

The clean M1 follow-up changed that.

What the new evidence supports:

- **MiniLM parity failure survives builtin pooling unchanged**, so manual pooling is not the root of the problem.
- **Qwen parity is fine** in both manual and builtin forms.
- **Qwen latency gap survives** builtin pooling, smaller contexts, smaller batch sizes, and explicit flash-attention enable.

The most useful architecture read from this spike now is:

1. llama.cpp **can** be embedded in-process cleanly,
2. decoder-style embedding models like Qwen3 can be **numerically faithful** in-process,
3. encoder-style MiniLM under the owned pretokenized token-id path is **not numerically faithful** here,
4. the child path still has a real latency advantage for Qwen single-query embeds on the M1,
5. large causal-batch contexts create a substantial **KV residency hazard** for a hot in-process daemon.

## Verdict

**No — even with builtin pooling, in-process llama.cpp does not clear the replacement bar yet.**

Why:

- the MiniLM parity gate still fails badly (**0.9736516294** instead of `>= 0.9999`),
- the Qwen single-call latency gap to the child remains large (**17.7 ms vs 7.39 ms p50** on the long query),
- and the KV-reservation behavior for causal batch contexts is a real operational footgun.

What improved:

- builtin pooling improved the Qwen throughput/cold-load story,
- MiniLM and Qwen both still beat the child on batch throughput in-process,
- MiniLM single-call latency is slightly better in-process.

But the bar for replacement was parity-first, and the clean follow-up says **builtin pooling is not enough**.

## Next fork

If this spike continues, the next decision is no longer about manual vs builtin pooling. That fork is settled.

The next fork is:

1. **accept that the external-tokenizer token-id path is the problem** and either
   - let llama.cpp own tokenization for encoder models, or
   - prove/token-diff why the HF-tokenizer ids diverge from the child path, or
2. keep the child for embeddings and treat in-process llama.cpp as an interesting but not yet production-safe lane.
