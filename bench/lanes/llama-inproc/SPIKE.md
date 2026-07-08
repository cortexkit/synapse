# Spike A — llama.cpp in-process embeddings

## Status

**Result: mixed, not a clear replacement for the child path.**

- **Qwen3-Embedding-0.6B f16 GGUF**: in-process worked, matched ORT numerically, and beat the child on 2,000-item corpus throughput.
- **all-MiniLM-L6-v2 f16 GGUF**: the owned tokenize → forward → manual mean-pool pipeline **missed parity badly** against ORT fp32, while the child lane stayed effectively perfect.
- **Headline latency claim** also split by model: MiniLM single-call latency improved vs the child, Qwen3 got worse.

My bottom line: **llama.cpp as an in-process library does not clear the bar for a production replacement of the child for embeddings today**, because the encoder/manual-pooling path failed the parity gate.

## Path decision: `llama-cpp-2` vs direct FFI

I stayed on **`llama-cpp-2` 0.1.151** with Metal enabled and did **not** fall back to hand-rolled `llama.h` bindings.

Why:

- the crate built cleanly here,
- it exposed enough surface to load GGUF models, create contexts, submit batches, and read embeddings,
- the blocker was **not** missing API or build failure; it was the **numerical behavior** of manual pooling for MiniLM.

So the spike result is about the runtime/API semantics, not a packaging failure.

## Measurement setup / caveats

- Machine: **M5 Max**, macOS, with light background activity allowed.
- These are **correctness + relative-latency** numbers, not idle-gated publish-table numbers.
- The worktree did **not** contain `bench/data/corpus-v2.jsonl`. I generated a temporary equivalent subset from the first 2,000 rows of `corpus/aft-chunks.jsonl` using `text = embed_text` and stable line-number ids. `bench/run-matrix.sh` already documents that `corpus-v2` is converted from `corpus/aft-chunks.jsonl`, so this is close in content, but it is still a caveat.
- The local cache under `models--mradermacher--Qwen3-Embedding-0.6B-GGUF` did **not** contain an f16 file here (only Q4/Q6). For the Qwen f16 run I used the locally cached official snapshot under `models--Qwen--Qwen3-Embedding-0.6B-GGUF`.
- Single-call latency used the same 42-word query for both runtimes. That query tokenized to **128 MiniLM tokens** and **47 Qwen tokens**.

## Implementation summary

The new crate is `bench/lanes/llama-inproc/`.

It does:

- Hugging Face `tokenizers` preprocessing,
- truncation before inference,
- length-sorted token-budget batching,
- explicit `encode` (MiniLM) vs `decode` (Qwen3) forward paths,
- manual pooling from per-token embeddings,
- L2 normalization,
- LaneResult JSON output plus optional raw vectors,
- a small latency helper binary for 200 sequential warm single-text calls.

## 1) Parity vs ORT fp32 (`synapse-bench parity`, first 2,000 chunks)

| Model | In-process mean cosine vs ORT fp32 | Gate (>= 0.9999) | Child mean cosine vs ORT fp32 |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 f16 GGUF | **0.9736238726** | **FAIL** | 0.9999953568 |
| Qwen3-Embedding-0.6B f16 GGUF | **0.9999994553** | **PASS** | 0.9999994527 |

This is the key spike result.

- **Qwen3**: manual last-token pooling is numerically fine.
- **MiniLM**: manual mean pooling over token embeddings is **not** equivalent to the child / ORT reference path.

## 2) Single-call latency, 200 warm sequential single-text embeds

| Model | In-process p50 | In-process p95 | Child p50 | Child p95 | Winner |
|---|---:|---:|---:|---:|---|
| all-MiniLM-L6-v2 | **1.451 ms** | **1.970 ms** | 1.969 ms | 2.367 ms | in-process |
| Qwen3-Embedding-0.6B | 14.010 ms | 15.878 ms | **5.258 ms** | **5.734 ms** | child |

Claim #1 only holds for MiniLM here. It fails for Qwen3.

## 3) Batch throughput over the 2,000-chunk subset

| Model | In-process tok/s | Child tok/s | Delta |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 | **235,334** | 93,776 | **2.51x faster** |
| Qwen3-Embedding-0.6B | **7,176** | 5,489 | **1.31x faster** |

Claim #2 is only half true:

- throughput is at least equal, often materially better,
- but parity is **not** at least equal because MiniLM failed hard.

## 4) Cold load wall time

These came from the full lane runs in the same warmed development session. Treat them as **process-cold, shader-cache-warm** numbers, not first-boot numbers.

| Model | In-process cold load | Child cold load |
|---|---:|---:|
| all-MiniLM-L6-v2 | **0.064 s** | 0.258 s |
| Qwen3-Embedding-0.6B | 1.036 s | **1.031 s** |

So in-process only showed a meaningful cold-load win for MiniLM.

## API / runtime sharp edges discovered

1. **`n_ctx` is total context, not per-sequence context.**
   For multi-sequence batching I had to set `total_n_ctx = per_sequence_ctx * n_seq_max`. If I passed `512` with `n_seq_max > 1`, llama.cpp silently reduced `n_ctx_seq`, and MiniLM parity cratered further because sequences were effectively truncated.

2. **Encoder batches are effectively capped by `n_ubatch`.**
   The encoder path hit a GGML assert requiring `n_ubatch >= n_tokens`. The child/server path hides this by internally managing work, but the in-process path did not. For MiniLM the effective batch token budget became `min(batch_size, ubatch_size) = 1024`.

3. **Manual token-embedding pooling is not equivalent to sequence embeddings on MiniLM.**
   The owned pipeline used `pooling_type = none`, read per-token embeddings, mean-pooled, and normalized. That landed at **0.9736** instead of ~1.0. The child lane using llama.cpp's built-in mean pooling stayed at **0.999995** on the same inputs.

4. **Decoder memory balloons with large `n_seq_max`.**
   On the 2,000-chunk Qwen run, the in-process context chose `n_seq_max = 101`, which made llama.cpp reserve about **11.3 GiB** of Metal KV cache. That is a real production footgun for a hot in-process runner.

5. **`llama-cpp-2` is usable but thin.**
   Important gaps for this spike:
   - no safe `has_encoder` / `has_decoder` helper,
   - no direct `n_embd_out` helper,
   - `LlamaContext` borrows `LlamaModel`, so a long-lived owned `model + context` object needs self-referential structure patterns or a different abstraction.

## Verdict

**No — not as specified.**

If the requirement is:

- in-process,
- our own Hugging Face tokenization,
- our own forward selection,
- our own manual pooling,
- and parity at least equal to the child,

then this spike does **not** clear the bar.

What it proved:

- llama.cpp **can** be embedded in-process on Metal cleanly,
- the decoder/last-token path (Qwen3) is good,
- throughput can beat the child.

What killed it:

- the encoder/manual-mean path for MiniLM missed parity by a mile,
- the Qwen single-call latency headline was worse than the child,
- multi-sequence decoder contexts reserve much more GPU memory than the child architecture exposes at the HTTP boundary.

If this leg is revisited later, the next fork is explicit:

1. **accept llama.cpp built-in pooling** and stop insisting on fully owned pooling semantics, or
2. **drop below `llama-cpp-2` into lower-level FFI / upstream investigation** and prove whether MiniLM token embeddings can be made numerically equivalent to the child path.

Until then, the child remains the safer architecture for embeddings.