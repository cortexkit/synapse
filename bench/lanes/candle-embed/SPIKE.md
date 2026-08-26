# Spike B: candle as an owned-runner foundation for MiniLM embeddings

## Setup

- Machine class: Apple M5 Max.
- Measurements here are **not idle-gated table-grade numbers**; light background activity may have existed.
- The task corpus file was unavailable in this checkout, so the runs below used a separately managed, read-only benchmark fixture.
- Subset measured: first 2,000 chunks (`/tmp/corpus-v2-2000.jsonl`).
- Reference lane for parity: `lane-ort-embed` fp32 MiniLM on the same 2,000-chunk subset.
- Candle implementation choices: stock `candle-transformers` BERT for `sentence-transformers/all-MiniLM-L6-v2`, `tokenizers` truncation at 512, length-sorted attention-unit batching (4M), mean pooling, L2 normalization, vectors restored to original chunk order.

## Results

### Same-run reference

| Lane | Status | Tok/s | Cold load (s) | Notes |
| --- | --- | ---: | ---: | --- |
| ort-cpu fp32 MiniLM | passed | 30,264 | 0.085 | Same 2,000-chunk subset; lines up closely with the existing 29.6k context number. |

### Candle lane

| Backend | DType | Status | Mean cosine vs ort fp32 | Tok/s | Cold load (s) | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| CPU | fp32 | passed | 1.0000000000 | 2,557 | 0.071 | Correct output, but very slow. |
| Metal | fp32 | **failed** | n/a | n/a | n/a | Warmup fails: `no metal implementation for layer-norm`. |
| Metal | f16 | **failed** | n/a | n/a | n/a | Same failure: `no metal implementation for layer-norm`. |

### Against the decision-context numbers

| Runtime / lane | Tok/s | Relative to candle CPU fp32 |
| --- | ---: | ---: |
| candle CPU fp32 | 2,557 | 1.0x |
| ort CPU fp32 (context: 29.6k) | 29,600 | 11.6x faster |
| ort CPU fp32 (same-run ref: 30.3k) | 30,264 | 11.8x faster |
| llama-server Metal f16 (context: 92.8k) | 92,800 | 36.3x faster |
| MLX Metal (context: 130k+) | 130,000+ | 50.8x+ faster |

## Parity

- CPU fp32 parity gate: **passed**.
- `synapse-bench parity` on the 2,000-chunk subset reported:
  - matched ids: 2,000
  - mean cosine: `0.9999999999987877`
  - top-k overlap (`k=10`, `stride=50`): perfect (`1.0` across the report)

Interpretation: the lane is numerically correct on CPU fp32. Accuracy is not the problem.

## Kernel-quality verdict

For this workload, candle is **not in the same league** as the measured engines.

- On CPU, the stock candle BERT path is about **an order of magnitude slower than ORT CPU** on the same MiniLM subset.
- On Metal, the stock path is not merely slower; it **does not run at all** on this machine/build because BERT hits a missing Metal `layer-norm` implementation during warmup.

That means the strongest possible positive outcome for the “owned Rust runner on existing Rust kernels” argument did **not** materialize here. The CPU result is far off the pace, and the Metal story is currently blocked by op coverage before throughput can even be measured.

## Sharp edges found

1. **Metal op coverage gap**
   - Both Metal fp32 and Metal f16 fail immediately in warmup with:
     - `Metal error no metal implementation for layer-norm`
   - This is a hard blocker for stock MiniLM/BERT encoder execution on Metal in this setup.

2. **CPU performance gap despite exact parity**
   - The lane reproduces the ORT vectors essentially exactly, but throughput is only **~8.5% of ORT CPU**.
   - So correctness is achievable, but the kernel/runtime stack is not competitive here.

3. **512-token configuration is easy; competitiveness is not**
   - Loading from HF cache / `hf-hub`, tokenizer truncation, sorted batching, and sentence-transformer-style pooling were all straightforward.
   - The hard part is kernel quality and backend completeness, not wiring.

4. **Benchmark-data availability in task worktrees**
   - `bench/data/corpus-v2.jsonl` was absent from this worktree, so the spike had to read the parent repo copy.
   - That did not affect correctness, but it is a practical friction point for repeatable spike runs.

## Honest one-paragraph verdict

As of this spike, candle is **not a viable kernel foundation for an owned encoder runner** if the goal is to be within striking distance of ORT/ggml/MLX on Apple hardware without substantial extra backend work. The CPU path is numerically solid but much too slow, and the Metal path for stock BERT MiniLM is blocked by missing `layer-norm` support before any meaningful throughput comparison is possible. The result is a clean negative: candle remains attractive as a Rust-native tensor runtime in principle, but for Synapse’s encoder workloads on this machine class, it does not currently provide the combination of **performance + Metal completeness** needed to support the “build our own runner on candle kernels” branch of Decision #1.
