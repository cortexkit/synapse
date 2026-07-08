# Spike C viability: owned unified tensor runtime

## Verdict

The owned-runtime path is technically viable for a narrow fp32 encoder, and the Metal provider now executes the full all-MiniLM-L6-v2 forward pass with **all dense and attention matmuls dispatched through MPSGraph**. Correctness is not the blocker: the mixed Metal path passes the 1,000-chunk ORT fp32 parity gate with mean cosine `0.9999999999985539`.

Performance is the blocker. The current eager provider keeps layernorm, softmax, GELU, residual adds, pooling, head packing, and tensor ownership on the CPU. Even with cached MPSGraph matmul plans and cached dense-weight buffers, every non-static matmul boundary uploads activations and reads results back. On the M1 bench box this mixed path reaches only **2.34k tok/s**, slower than the owned CPU/Accelerate path at **6.96k tok/s** and far behind the context Metal engines.

For Synapse v1, full ownership is still not the economical default. MPSGraph is a viable correctness-oriented kernel source for a first Metal provider, but using it one eager op at a time is not a viable performance architecture. The next useful owned-runtime step would be either a larger MPSGraph subgraph that keeps attention/FFN blocks resident on device, or custom/MLX-style kernels with explicit buffer residency and fusion.

## What works

Implemented at `bench/spikes/unified-rt/`:

- A workspace crate and CLI-compatible lane binary (`spike-unified-rt`).
- Owned tensor type: shape, row-major stride metadata, dtype, and owned f32 buffers.
- Direct eager BERT execution rather than a graph IR. For this spike that was the right trade: the hard evidence was model/kernels/scheduling, not graph optimizations.
- Safetensors loader for f32/f16/bf16 float tensors, with non-float tensors ignored.
- CPU kernel provider using Apple Accelerate `cblas_sgemm` for all dense and attention matmuls; layernorm, softmax, GELU, pooling, and L2 normalization are owned Rust loops.
- Provider trait separating the model graph from kernels.
- Metal provider using a small Objective-C MPSGraph shim compiled by `cc`:
  - `--device metal` initializes a Metal device and command queue.
  - Dense layer matmuls call `matmul_static_rhs`, which lets the Metal provider cache RHS weight buffers. The existing warmup pass uploads these weights before timed inference.
  - Dynamic attention matmuls also run on MPSGraph, but their per-head Q/K/V/scores buffers are uploaded per call.
  - MPSGraph matmul plans are cached by shape/layout so a batch reuses graphs instead of rebuilding 180 graphs per MiniLM forward.
- Tokenizers preprocessing with 512 truncation, mean pooling, L2 normalization, and length-sorted greedy batching by attention-unit budget.
- Unit tests compare MPSGraph matmul against the CPU provider for both supported RHS layouts.

Not working / deliberately out of scope for this lane:

- Fully GPU MiniLM. Pointwise ops, residual adds, masking/softmax, pooling, and the attention head packing remain on CPU.
- f16 Metal. The delivered provider is fp32 only so it remains parity-comparable with the ORT fp32 reference.
- General graph compilation, custom Metal shaders, quantization, or new model families.

## Binding choice

I checked the objc2 ecosystem first. `objc2-metal-performance-shaders-graph` exists at `0.3.2`, but using the generated API directly would have required pulling in a broad generated binding surface and proving the exact MPSGraph selector set under the time box. The SDK headers were available locally in Xcode 26.6, so this lane uses a minimal Objective-C shim instead. That kept the Rust surface small and made the selectors compile against Apple's canonical headers while still using MPSGraph rather than custom shaders.

## Correctness and throughput evidence

The committed repo does not include `bench/data/corpus-v2.jsonl` because `bench/data/*.jsonl` is gitignored. Verification used a generated `target/unified-rt-corpus-v2-1000.jsonl` from the first 1,000 rows of `corpus/aft-chunks.jsonl`, using each row's `embed_text` as lane `text`, matching `bench/run-matrix.sh`'s note that corpus-v2 is converted from that export.

Reference vectors were generated with `lane-ort-embed` using `sentence-transformers/all-MiniLM-L6-v2` ONNX fp32, mean pooling, max length 512. The M1 timed run used the shared timed-run lock at `[bench-user-home]/bench.lock`.

| Lane / variant | Machine | Items | Tokens | Mean cosine vs ORT | Tok/s | Notes |
|---|---|---:|---:|---:|---:|---|
| ORT MiniLM fp32 reference | local M5 | 1,000 | 172,746 | reference | 30,379.7 | Reference vectors only; not the M1 context number |
| Owned RT CPU/Accelerate | M1 bench | 1,000 | 172,746 | 0.9999999999985539 | 6,955.6 | Same eager graph, Accelerate SGEMM |
| Owned RT Metal MPSGraph matmul + CPU pointwise | M1 bench | 1,000 | 172,746 | 0.9999999999985539 | 2,338.8 | All matmuls on MPSGraph; pointwise/head packing on CPU |
| Owned RT Metal MPSGraph matmul + CPU pointwise | local M5 dev run | 1,000 | 172,746 | 0.9999999999985539 | 3,513.1 | Non-authoritative dev-loop run |

Context numbers from the lane prompt for the same M1 bench box and corpus:

| Engine context | Tok/s |
|---|---:|
| llama.cpp Metal | 51.8k |
| MLX Metal | 132.6k |
| ORT CPU | 11.5k |
| Owned RT CPU seed | ~5-10k |

Interpretation: Metal correctness is excellent, but the mixed provider is slower than CPU because it forces many eager CPU/GPU boundaries. Static dense weights are cached after warmup, but activations still move for every matmul, and every MPSGraph result is read back so CPU pointwise code can continue.

## Transfer-cost analysis

What the current provider saves:

- Dense layer weights are uploaded lazily through `matmul_static_rhs`; the warmup inference touches every linear layer, so timed batches reuse cached Metal buffers for those RHS weights.
- MPSGraph plans are cached by `(m, n, k, rhs-layout)`, avoiding per-call graph construction for repeated Q/K/V/output and attention shapes inside a batch.

What it still pays:

- Each dense matmul uploads the LHS activation matrix and reads the output matrix back to CPU.
- Each attention matmul uploads packed per-head Q/K/scores/V buffers and reads scores/context back.
- Softmax, masking, GELU, layernorm, residual adds, mean pooling, and L2 normalization run on CPU, so there is no long-lived device-resident hidden state.
- Head packing/unpacking is CPU memory traffic before and after the GPU calls.

For a MiniLM batch this creates hundreds of upload/execute/readback boundaries. Fusing even one encoder layer as an MPSGraph subgraph would keep Q/K/V, scaled masking, softmax, attention output, FFN matmuls, residuals, and layernorm intermediates resident until the layer output. That is the transfer-saving step that would matter; simply replacing more pointwise loops one by one with separate MPSGraph calls would likely make boundary overhead worse.

## Time spent / velocity datum

Approximate additional Lane 2 allocation:

| Layer | Hours | Outcome |
|---|---:|---|
| Binding assessment + SDK header check | 0.3 | Chose minimal Objective-C MPSGraph shim over direct objc2 bindings for this time box |
| MPSGraph shim + build integration | 1.1 | Metal device/queue, matmul, transpose-RHS layout, graph plan cache, RHS weight-buffer cache |
| Provider trait changes | 0.5 | Added static-RHS matmul hook while leaving CPU behavior unchanged |
| Correctness tests and local parity | 0.8 | Unit matmul tests plus 1,000-chunk local Metal parity run |
| M1 transfer/locked measurement | 0.6 | Uploaded arm64 binary, used timed-run lock, measured CPU and Metal variants |
| Memo update | 0.4 | Results, transfer analysis, revised verdict |

Total additional time: about **3.7 hours**. Combined with the first CPU/runtime spike, the evidence base is about **8.5 hours**.

## Walls hit

1. **MPSGraph is easy to call but expensive at eager granularity.** Graph plan caching fixes the worst compile overhead, but op-by-op CPU/GPU synchronization dominates.
2. **Buffer residency needs a graph-level contract.** A matmul-only trait can cache dense weights, but it cannot keep hidden states, attention scores, or layer outputs on device across pointwise ops.
3. **Attention layout still dominates owned-runtime complexity.** This spike packs per-head Q/K/V into contiguous CPU scratch buffers, which is correct but causes extra traffic and prevents fused attention.
4. **Numerical parity requires exact boring details.** GELU variant, layernorm epsilon, mask value, mean-pooling mask, tokenization/truncation, and position/type embeddings all had to match. A model family change repeats this work.
5. **Scheduling is real even for embeddings.** Length sorting helped avoid pathological padding. Production needs scheduler-level queues, cancellation, backpressure, warm pools, and memory budget accounting.

## What would be hard next

Engineering-hard, not research-hard:

- Express each MiniLM encoder layer as one MPSGraph subgraph, or introduce a tiny graph IR that can hand MPSGraph whole blocks instead of scalar provider calls.
- Implement GPU layernorm/softmax/GELU/residual/pooling with real buffer residency, not separate readback-heavy MPSGraph calls.
- Replace CPU head packing with device-resident layout transforms or a fused attention path.
- CUDA and Vulkan providers with common layout contracts and per-provider autotuning.
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
| Metal provider: MPSGraph subgraphs or kernels, f16, fusion, profiling | 8-12 | Medium-low |
| CUDA provider | 6-9 | Medium |
| Vulkan provider | 8-12 | Low |
| Quant formats q8/q6k across providers | 12-18 | Low |
| Model families: MiniLM, ModernBERT, Qwen3 embed/rerank encoders | 10-16 | Medium-low |
| Tiny decoder: RoPE/GQA/RMSNorm, KV-cache, sampling, chat templates | 8-12 | Medium-low |
| Test/parity harness, packaging, artifact management, docs | 6-10 | Medium |
| Performance tuning to approach llama.cpp/MLX | 8-14 | Low |

Total remains **60-90 engineer-weeks**, with meaningful risk that Vulkan/quant/Metal tuning expands the upper bound.

Adopt-path comparison: linking llama.cpp/MLX as libraries still looks like **6-12 engineer-weeks** for robust packaging, lifecycle, scheduling, API wrapping, and parity/power measurement. Owned runtime therefore costs roughly **5-10x** more before it reaches comparable breadth, and it starts behind on throughput.

## Recommendation

Keep this crate as evidence and possibly as a CPU/Metal correctness harness. Do not choose full ownership for v1 unless the strategy explicitly values long-term kernel/control ownership over delivery speed. If the team wants one more owned-runtime data point, do not spend it on more eager matmul plumbing. Spend it on a single device-resident encoder-layer implementation, then decide whether MPSGraph subgraphs are enough or whether custom kernels/MLX-style integration is the only credible Metal path.
