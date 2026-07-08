# Spike C viability: owned unified tensor runtime

## Verdict

The owned-runtime path is technically viable for a narrow fp32 encoder, and the Metal provider now has the backend contract this spike needed: a block-level MiniLM encoder entry point that runs all 6 encoder layers as one MPSGraph per batch shape while keeping hidden states resident on device. Correctness is not the blocker: the resident Metal path passes the 1,000-chunk ORT fp32 parity gate with mean cosine `0.9999999999985806`.

Performance changed qualitatively once the CPU/GPU boundary moved from eager matmuls to whole encoder blocks. On the M1 bench box, the old mixed provider reached only **2.34k tok/s** because it read activations back for CPU pointwise work. The resident encoder reached **75.996k tok/s** on the first locked run after upload and **116.534k tok/s** on the final locked rerun after MPSGraph/system caches were warm, versus **6.955k tok/s** for the owned CPU/Accelerate path on the same 1,000 chunks. Even the conservative first resident run is above the lane's llama.cpp-Metal context number (**51.8k tok/s**); the honest remaining gap is to MLX (**132.6k tok/s**), where resident MPSGraph is about **1.7x slower cold-ish** and **1.14x slower warm**.

For Synapse v1, full ownership is still not automatically the economical default, but the Metal conclusion is now different: **MPSGraph is a viable Metal path when the provider receives whole encoder blocks with explicit residency**. It is not viable one eager op at a time. The remaining owned-runtime question is whether building and maintaining graph-level contracts, f16/quant variants, schedulers, and additional model families is worth the product schedule cost compared with linking llama.cpp/MLX.

## What works

Implemented at `bench/spikes/unified-rt/`:

- A workspace crate and CLI-compatible lane binary (`spike-unified-rt`).
- Owned tensor type: shape, row-major stride metadata, dtype, and owned f32 buffers.
- Direct BERT execution rather than a general graph IR. For this spike that was the right trade: the hard evidence was model/kernels/scheduling, not graph optimizations.
- Safetensors loader for f32/f16/bf16 float tensors, with non-float tensors ignored.
- CPU kernel provider using Apple Accelerate `cblas_sgemm` for all dense and attention matmuls; layernorm, softmax, GELU, pooling, and L2 normalization are owned Rust loops.
- Provider trait separating the model graph from kernels, with an optional block-level `encoder_forward` override. CPU keeps the scalar fallback path; Metal overrides the block.
- Metal provider using a small Objective-C MPSGraph shim compiled by `cc`:
  - `--device metal` initializes a Metal device and command queue.
  - The legacy eager `matmul` / `matmul_static_rhs` hooks remain for unit coverage and fallback experiments.
  - Dense weights, biases, and layernorm parameters are cached as Metal buffers by pointer and byte length.
  - The resident path builds one MPSGraph per `(batch, seq, hidden, heads, intermediate, layers)` shape. The graph contains Q/K/V projections, scaled masked attention, softmax, attention output, residual adds, exact-erf GELU, FFN, and both encoder layernorms for all 6 layers.
  - CPU touches hidden states once to upload embedding+embedding-layernorm output and once to read back final hidden states. Mean pooling and L2 normalization remain CPU-side.
- Tokenizers preprocessing with 512 truncation, mean pooling, L2 normalization, and length-sorted greedy batching by attention-unit budget.
- Unit tests compare MPSGraph matmul against the CPU provider for both supported RHS layouts and compare a tiny resident encoder block against the scalar CPU path.

Not working / deliberately out of scope for this lane:

- GPU embedding lookup, embedding layernorm, mean pooling, and L2 normalization. They are once-per-batch boundaries and were not the measured bottleneck.
- f16 Metal. The delivered provider is fp32 so it remains parity-comparable with the ORT fp32 reference; f16 should be measured separately before treating the speedup as a production number.
- General graph compilation, custom Metal shaders, quantization, or new model families.

## Binding choice

I checked the objc2 ecosystem first. `objc2-metal-performance-shaders-graph` exists at `0.3.2`, but using the generated API directly would have required pulling in a broad generated binding surface and proving the exact MPSGraph selector set under the time box. The SDK headers were available locally in Xcode 26.6, so this lane uses a minimal Objective-C shim instead. That kept the Rust surface small and made the selectors compile against Apple's canonical headers while still using MPSGraph rather than custom shaders.

## Correctness and throughput evidence

The committed repo does not include `bench/data/corpus-v2.jsonl` because `bench/data/*.jsonl` is gitignored. Verification used `target/unified-rt-corpus-v2-1000.jsonl` from the first 1,000 rows of `bench/data/corpus-v2.jsonl` on the M1 bench box, matching `bench/run-matrix.sh`'s note that corpus-v2 is converted from the AFT chunk export.

Reference vectors were generated with `lane-ort-embed` using `sentence-transformers/all-MiniLM-L6-v2` ONNX fp32, mean pooling, max length 512. The M1 timed run used the shared timed-run lock at `[bench-user-home]/bench.lock`.

| Lane / variant | Machine | Items | Tokens | Mean cosine vs ORT | Tok/s | Notes |
|---|---|---:|---:|---:|---:|---|
| ORT MiniLM fp32 reference | M1 bench | 1,000 | 172,746 | reference | 11,488.2 | Reference vectors for this run |
| Owned RT CPU/Accelerate | M1 bench | 1,000 | 172,746 | 0.9999999999985222 | 6,955.5 | Scalar provider, Accelerate SGEMM |
| Owned RT Metal MPSGraph matmul + CPU pointwise | M1 bench | 1,000 | 172,746 | 0.9999999999985539 | 2,338.8 | Previous eager mixed provider; boundary dominated |
| Owned RT Metal resident encoder MPSGraph | M1 bench | 1,000 | 172,746 | 0.9999999999985806 | 75,996.3 | First locked resident run after upload; one full 6-layer encoder graph per batch shape |
| Owned RT Metal resident encoder MPSGraph | M1 bench | 1,000 | 172,746 | 0.9999999999985806 | 116,534.4 | Final locked rerun after MPSGraph/system caches were warm |
| Owned RT Metal MPSGraph matmul + CPU pointwise | local M5 dev run | 1,000 | 172,746 | 0.9999999999985539 | 3,513.1 | Older non-authoritative dev-loop run |

Context numbers from the lane prompt for the same M1 bench box and corpus:

| Engine context | Tok/s | Resident Metal gap |
|---|---:|---:|
| llama.cpp Metal | 51.8k | resident MPSGraph is ~1.47x faster on the conservative run, ~2.25x faster warm |
| MLX Metal | 132.6k | resident MPSGraph is ~0.57x as fast cold-ish, ~0.88x as fast warm |
| ORT CPU | 11.5k | resident MPSGraph is ~6.6x to ~10.1x faster |
| Owned RT CPU seed | ~5-10k | resident MPSGraph is above this band |

Interpretation: Metal correctness stayed excellent, and moving pointwise work into the same MPSGraph eliminated the dominant transfer cost. The two resident rows show the likely MPSGraph compile/system-cache sensitivity; both clear the CPU and mixed-provider bars. This is still a narrow fp32 MiniLM result, not a claim that the owned runtime is broadly competitive with MLX across models or dtypes.

## Transfer-cost analysis

What the eager provider saved:

- Dense layer weights were uploaded lazily through `matmul_static_rhs`; warmup touched every linear layer, so timed batches reused cached Metal buffers for those RHS weights.
- MPSGraph plans were cached by `(m, n, k, rhs-layout)`, avoiding per-call graph construction for repeated matmul shapes.

What it still paid:

- Each dense matmul uploaded the LHS activation matrix and read the output matrix back to CPU.
- Each attention matmul uploaded packed per-head Q/K/scores/V buffers and read scores/context back.
- Softmax, masking, GELU, layernorm, residual adds, mean pooling, and L2 normalization ran on CPU, so there was no long-lived device-resident hidden state.
- Head packing/unpacking was CPU memory traffic before and after the GPU calls.

What the resident encoder saves:

- Per batch, hidden states are uploaded once after CPU embedding/embedding-layernorm and read back once after the final encoder layer.
- Q/K/V, attention scores, masks, softmax probabilities, context, FFN intermediates, residuals, and both encoder layernorms stay inside MPSGraph for all 6 layers.
- Head layout changes are MPSGraph reshapes/transposes rather than CPU scratch packing.
- Static weights and biases are still cached as Metal buffers and fed to the per-shape graph.

The measured effect is the core lesson of the spike: replacing CPU loops with separate GPU ops would have increased synchronization pressure, while changing the provider contract to whole encoder blocks removes hundreds of per-batch upload/readback boundaries.

## Time spent / velocity datum

Approximate additional Lane 2 allocation:

| Layer | Hours | Outcome |
|---|---:|---|
| Binding assessment + SDK header check | 0.3 | Chose minimal Objective-C MPSGraph shim over direct objc2 bindings for this time box |
| Initial MPSGraph shim + build integration | 1.1 | Metal device/queue, matmul, transpose-RHS layout, graph plan cache, RHS weight-buffer cache |
| Initial provider trait changes | 0.5 | Added static-RHS matmul hook while leaving CPU behavior unchanged |
| Initial correctness tests and local parity | 0.8 | Unit matmul tests plus 1,000-chunk local Metal parity run |
| Initial M1 transfer/locked measurement | 0.6 | Uploaded arm64 binary, used timed-run lock, measured CPU and eager Metal variants |
| Resident encoder provider contract | 0.4 | Added optional block-level `encoder_forward`; CPU uses scalar fallback |
| Full-stack MPSGraph encoder shim | 2.0 | One 6-layer graph per batch shape with resident attention, GELU, residual, and layernorm intermediates |
| Resident encoder tests and debug | 0.7 | Added tiny encoder CPU-vs-Metal unit coverage and fixed graph shape/feed details |
| Resident M1 parity/throughput measurement | 0.6 | Uploaded arm64 binary, regenerated ORT references, measured CPU and resident Metal under lock, then reran final Metal binary after caches were warm |
| Memo update | 0.5 | Results, transfer analysis, revised verdict |

Total Lane 2 time is about **7.5-8 hours**. Combined with the first CPU/runtime spike, the evidence base is about **12-13 hours**.

## Walls hit

1. **Eager op granularity was the wrong contract.** Matmul plan caching fixed graph construction overhead but not synchronization. Whole-block residency fixed the measured bottleneck.
2. **MPSGraph can express the full MiniLM encoder block.** Exact-erf GELU, layernorm via mean/variance primitives, softmax, reshapes/transposes, residuals, and all matmuls fit in a single graph.
3. **Shape specialization remains a scheduling concern.** The implementation caches one graph per batch/sequence shape; production should bucket or prewarm shapes to control compile/cache churn.
4. **Numerical parity requires exact boring details.** GELU variant, layernorm epsilon, mask value, mean-pooling mask, tokenization/truncation, and position/type embeddings all had to match. A model family change repeats this work.
5. **This does not solve the runtime product surface.** Production still needs scheduler-level queues, cancellation, backpressure, warm pools, memory budget accounting, f16/quant coverage, and model-family coverage.

## What would be hard next

Engineering-hard, not research-hard:

- Generalize the block-level provider contract or introduce a tiny graph IR so other encoders can hand MPSGraph whole blocks without bespoke FFI structs.
- Add f16 and measure the speed/parity trade, including whether MPSGraph's reduced-precision paths beat fp32 on the M1/M-series matrix units.
- Move mean pooling/L2 normalization to the device when returning pooled vectors directly, while preserving the option to read final hidden states.
- Add shape bucketing, graph prewarming, memory planning, and scratch-buffer accounting to avoid per-shape surprises.
- CUDA and Vulkan providers with common layout contracts and per-provider autotuning.
- CI matrix for model artifacts, parity fixtures, backend availability, and macOS/Xcode drift.

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
| Metal provider: MPSGraph subgraphs or kernels, f16, fusion, profiling | 7-10 | Medium |
| CUDA provider | 6-9 | Medium |
| Vulkan provider | 8-12 | Low |
| Quant formats q8/q6k across providers | 12-18 | Low |
| Model families: MiniLM, ModernBERT, Qwen3 embed/rerank encoders | 10-16 | Medium-low |
| Tiny decoder: RoPE/GQA/RMSNorm, KV-cache, sampling, chat templates | 8-12 | Medium-low |
| Test/parity harness, packaging, artifact management, docs | 6-10 | Medium |
| Performance tuning to approach llama.cpp/MLX broadly | 8-14 | Medium-low |

Total remains **60-90 engineer-weeks**, with meaningful risk that Vulkan/quant/Metal tuning expands the upper bound. The resident MiniLM result lowers the Metal technical-risk line item, but it does not remove the breadth cost.

Adopt-path comparison: linking llama.cpp/MLX as libraries still looks like **6-12 engineer-weeks** for robust packaging, lifecycle, scheduling, API wrapping, and parity/power measurement. Owned runtime therefore costs roughly **5-10x** more before it reaches comparable breadth, even though MPSGraph block residency is now a credible Metal implementation strategy for encoder models.

## Recommendation

Keep this crate as evidence and as a CPU/Metal correctness harness. If the team wants an owned MiniLM-like encoder provider, MPSGraph block subgraphs are now a viable implementation path. Do not choose full ownership for all v1 local inference solely from this result: make that choice only if the strategy explicitly values long-term kernel/control ownership enough to pay the runtime, scheduler, dtype, quantization, and model-family cost.
