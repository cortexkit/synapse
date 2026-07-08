# Spike C viability: owned unified tensor runtime

## Verdict

The owned-runtime path is technically viable for a narrow fp32 encoder, but the first useful seed is already much slower than the mature CPU engine and did not reach a working Metal backend in this session. The CPU path is real: it loads stock `all-MiniLM-L6-v2` safetensors, executes the BERT encoder with our tensor/runtime code, batches by token length and attention budget, mean-pools, L2-normalizes, and passes the ORT fp32 parity gate.

For Synapse v1, full ownership is not the economical default. A realistic owned path to MiniLM + ModernBERT-class + Qwen3-0.6B embed/rerank encoders + one tiny decoder across CPU/Metal/CUDA/Vulkan with q8/q6k quantization is roughly **60-90 engineer-weeks** before it is a dependable runtime surface. Linking llama.cpp/MLX as libraries is still the honest near-term choice unless owning kernels/model support is itself the product strategy.

## What works

Implemented at `bench/spikes/unified-rt/`:

- A workspace crate and CLI-compatible lane binary (`spike-unified-rt`).
- Owned tensor type: shape, row-major stride metadata, dtype, and owned f32 buffers.
- Direct eager BERT execution rather than a graph IR. For this spike that was the right trade: the hard evidence was model/kernels/scheduling, not graph optimizations.
- Safetensors loader for f32/f16/bf16 float tensors, with non-float tensors ignored.
- CPU kernel provider using Apple Accelerate `cblas_sgemm` for all dense and attention matmuls; layernorm, softmax, GELU, pooling, and L2 normalization are owned Rust loops.
- A provider trait separating the model graph from kernels. A CUDA/Metal/Vulkan provider would implement the same `matmul` and `layer_norm` boundary without changing BERT graph code.
- Tokenizers preprocessing with 512 truncation, mean pooling, L2 normalization, and length-sorted greedy batching by attention-unit budget.

Not working:

- Metal execution. The `--device metal` path is a deliberate provider-boundary stub and errors before inference. CPU parity was prioritized per supervision; MPSGraph/Metal measurement remains the next step.

## Correctness and throughput evidence

The committed repo does not include `bench/data/corpus-v2.jsonl` because `bench/data/*.jsonl` is gitignored. For verification I generated an uncommitted local `target/unified-rt-corpus-v2.jsonl` from the first 1,000 rows of `corpus/aft-chunks.jsonl`, using each row's `embed_text` as lane `text`, matching `bench/run-matrix.sh`'s note that corpus-v2 is converted from that export.

| Lane | Items | Tokens | Mean cosine vs ORT | Rank overlap | Tok/s | Notes |
|---|---:|---:|---:|---:|---:|---|
| ORT MiniLM fp32 reference | 1,000 | 172,746 | reference | reference | 24,520.9 | Local release run on the generated subset |
| Owned RT CPU/Accelerate | 1,000 | 172,746 | 0.9999999999985607 | 1.0 top-10 | 10,557.2 | Passes the >= 0.9999 gate |
| Owned RT Metal | n/a | n/a | n/a | n/a | n/a | Stub only; no measured Metal throughput |

Context numbers from the decision bench prompt/machine class:

| Engine context | Tok/s |
|---|---:|
| ORT CPU | 29.6k |
| llama-server Metal | 92.8k |
| MLX Metal | 130k+ |
| Owned RT CPU, this spike | 10.6k |

Interpretation: correctness is excellent. CPU throughput is about 43% of the local ORT MiniLM run and about 36% of the prompt's ORT CPU context. The gap is plausible: the owned path repacks heads for every attention matmul, runs pointwise ops serially, has no fusion, and has no thread-level scheduling beyond Accelerate inside SGEMM.

## Time spent / velocity datum

Approximate wall-clock engineering allocation in this session:

| Layer | Hours | Outcome |
|---|---:|---|
| Tensor core + safetensors I/O | 0.7 | Shape/stride/dtype tensor, f32/f16/bf16 conversion, config/key loading |
| CPU kernels | 0.8 | Accelerate SGEMM trait boundary plus owned layernorm/softmax/GELU |
| BERT model graph | 1.4 | Embeddings, 6 encoder layers, MHA, FFN, residuals, pooling |
| CLI + batching | 0.5 | Lane result schema, vectors, reference parity, length-sorted attention budget |
| Debugging/parity | 0.7 | Dtype skip, local corpus conversion, ORT reference generation, parity run |
| Metal | 0.2 | Trait stub and wall characterization only |
| Memo | 0.5 | Viability and extrapolation |

Total: about **4.8 hours**.

## Walls hit

1. **Metal is not a small additive backend.** The trait boundary is clean, but making Metal real means choosing between MPSGraph block execution and lower-level Metal kernels. MPSGraph is probably the cheapest next route, but Rust bindings and tensor lifetime/dtype management still need dedicated time. Custom shaders are not justified yet.
2. **Attention layout dominates owned-runtime complexity.** BERT weights are easy; efficient attention is not. This spike packs per-head Q/K/V into contiguous scratch buffers, which is correct but costs memory traffic and leaves fusion on the table.
3. **Numerical parity requires exact boring details.** GELU variant, layernorm epsilon, mask value, mean-pooling mask, tokenization/truncation, and position/type embeddings all had to match. A model family change repeats this work.
4. **Scheduling is real even for embeddings.** Length sorting helped avoid pathological padding. Production needs scheduler-level queues, cancellation, backpressure, warm pools, and memory budget accounting.
5. **Corpus/model artifacts are part of the benchmark contract.** The required corpus file was absent from the checkout; the run used a reproducible local conversion from the checked-in source export. Published numbers should run against the exact `bench/data/corpus-v2.jsonl` artifact.

## What would be hard next

Engineering-hard, not research-hard:

- MPSGraph/Metal provider for matmul first, then layernorm/softmax/GELU fusion.
- CUDA and Vulkan providers with common layout contracts and per-provider autotuning.
- Runtime graph representation if multiple architectures share optimizations.
- CI matrix for model artifacts, parity fixtures, backend availability, and macOS/Xcode drift.
- Proper memory planner/scratch allocator to avoid per-layer/per-head allocations.

Harder / partially research-shaped:

- q8/q6k quantization across every backend. The algorithms are known, but the kernel zoo and per-device performance cliffs are the cost center.
- Attention variants: GQA/MQA, sliding window, RoPE/YaRN, ALiBi, ModernBERT-style changes, cross-encoder rerank heads.
- New hardware enablement such as M5 tensor accelerators or ANE/CoreML. This is mostly integration research until public APIs and performance contracts stabilize.
- Decoder KV-cache: not mathematically hard, but production quality requires cache layout, paging/eviction, prompt batching, prefill/decode split, and sampling correctness.

## Grounded extrapolation to Synapse v1

Estimated effort to turn this seed into Synapse's v1 local inference surface:

| Area | Engineer-weeks | Confidence |
|---|---:|---|
| Runtime core: graph/eager IR, memory planner, scheduler, batching, cancellation | 6-9 | Medium |
| CPU provider: f32/f16, quant matmuls, thread policy, fusion | 4-6 | Medium |
| Metal provider: MPSGraph or kernels, f16, fusion, profiling | 8-12 | Medium-low |
| CUDA provider | 6-9 | Medium |
| Vulkan provider | 8-12 | Low |
| Quant formats q8/q6k across providers | 12-18 | Low |
| Model families: MiniLM, ModernBERT, Qwen3 embed/rerank encoders | 10-16 | Medium-low |
| Tiny decoder: RoPE/GQA/RMSNorm, KV-cache, sampling, chat templates | 8-12 | Medium-low |
| Test/parity harness, packaging, artifact management, docs | 6-10 | Medium |
| Performance tuning to approach llama.cpp/MLX | 8-14 | Low |

Total: **60-90 engineer-weeks**, with meaningful risk that Vulkan/quant/Metal tuning expands the upper bound.

Adopt-path comparison: linking llama.cpp/MLX as libraries still looks like **6-12 engineer-weeks** for robust packaging, lifecycle, scheduling, API wrapping, and parity/power measurement. Owned runtime therefore costs roughly **5-10x** more before it reaches comparable breadth, and it starts behind on throughput.

## Recommendation

Keep this crate as evidence and possibly as a small CPU reference harness. Do not choose full ownership for v1 unless the strategy explicitly values long-term kernel/control ownership over delivery speed. If the team wants one more owned-runtime data point, spend the next 8 hours only on **MPSGraph matmul provider + measured mixed CPU/Metal MiniLM**, not custom shaders or a general graph compiler.
