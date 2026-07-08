# Spike C viability: owned unified tensor runtime

## Verdict

The owned-runtime path is still technically viable for a narrow fp32 encoder, and the Metal provider still has the backend contract this spike needed: a block-level MiniLM encoder entry point that runs all 6 encoder layers as one MPSGraph per batch shape while keeping hidden states resident on device. Correctness remains strong: the resident fp32 Metal path still passes the 1,000-chunk ORT fp32 parity gate with mean cosine `0.9999999999986154`, mean top-10 rank overlap `1.0`, and worst-decile overlap `1.0`.

The new f16 variant answers a different question: **quality stayed inside the certified-equivalence band, but throughput collapsed on the M1 bench box**. On the same 1,000-chunk subset, resident f16 measured mean cosine `0.9999993204962262`, mean top-10 rank overlap `0.9995`, and worst-decile overlap `0.995`, so the vectors stayed extremely close to ORT fp32. But the timed locked runs reached only **4.236k tok/s** on the first run and **4.234k tok/s** warm. That is below the owned CPU/Accelerate baseline (**6.955k tok/s**) and nowhere near resident fp32 (**76.0k/116.5k**) or MLX (**132.6k**).

The public MPSGraph matrix-multiplication API in Xcode 26.6 exposes operand dtypes, but not a separate accumulation-precision control. This implementation therefore keeps hidden states and static encoder weights resident in f16, upcasts layernorm / GELU / softmax reductions to f32 where explicit casts are available, and leaves embeddings + pooling on CPU fp32. That was enough to keep quality strong, but not enough to make MPSGraph float16 fast on this M1 path.

For Synapse v1, the Metal conclusion is now split: **MPSGraph block residency is a viable fp32 Metal path for MiniLM-like encoders, but this measured f16 path is not a performance win and does not narrow the MLX gap**. Treat `--dtype f16` as an honest negative result, not an optimization.

## What works

Implemented at `bench/spikes/unified-rt/`:

- A workspace crate and CLI-compatible lane binary (`spike-unified-rt`).
- Owned tensor type: shape, row-major stride metadata, dtype, owned f32 buffers, and optional f16 mirrors for Metal-only static encoder parameters.
- Direct BERT execution rather than a general graph IR. For this spike that was the right trade: the hard evidence was model/kernels/scheduling, not graph optimizations.
- Safetensors loader for f32/f16/bf16 float tensors, with non-float tensors ignored.
- CPU kernel provider using Apple Accelerate `cblas_sgemm` for all dense and attention matmuls; layernorm, softmax, GELU, pooling, and L2 normalization are owned Rust loops.
- Provider trait separating the model graph from kernels, with an optional block-level `encoder_forward` override. CPU keeps the scalar fallback path; Metal overrides the block.
- Metal provider using a small Objective-C MPSGraph shim compiled by `cc`:
  - `--device metal --dtype f32|f16` initializes a Metal device and command queue.
  - The legacy eager `matmul` / `matmul_static_rhs` hooks remain for unit coverage and fallback experiments.
  - Dense weights, biases, and layernorm parameters are cached as Metal buffers by pointer and byte length.
  - The resident path builds one MPSGraph per `(batch, seq, hidden, heads, intermediate, layers, dtype)` shape. The graph contains Q/K/V projections, scaled masked attention, softmax, attention output, residual adds, exact-erf GELU, FFN, and both encoder layernorms for all 6 layers.
  - CPU touches hidden states once to upload embedding+embedding-layernorm output and once to read back final hidden states. Mean pooling and L2 normalization remain CPU-side.
  - `--dtype f16` converts encoder-layer static parameters once at load while keeping fp32 masters on CPU, uploads hidden states as f16, reads them back as f16, and converts only at the CPU boundary. Layernorm / GELU / softmax reductions are explicitly cast to f32 inside the graph where MPSGraph exposes that control. The public matmul API does not expose a separate accumulation-precision knob, so f16 matmuls use native MPSGraph behavior as measured.
- Tokenizers preprocessing with 512 truncation, mean pooling, L2 normalization, and length-sorted greedy batching by attention-unit budget.
- Unit tests compare MPSGraph matmul against the CPU provider for both supported RHS layouts and compare tiny resident encoder blocks against the scalar CPU path for both fp32 and f16.

Not working / deliberately out of scope for this lane:

- GPU embedding lookup, embedding layernorm, mean pooling, and L2 normalization. They are once-per-batch boundaries and were not the measured bottleneck.
- MPSGraph float16 as a production speedup on the M1 bench box. The measured `--dtype f16` path keeps quality, but it is much slower than resident fp32 and slower than the owned CPU baseline.
- General graph compilation, custom Metal shaders, quantization, or new model families.

## Binding choice

I checked the objc2 ecosystem first. `objc2-metal-performance-shaders-graph` exists at `0.3.2`, but using the generated API directly would have required pulling in a broad generated binding surface and proving the exact MPSGraph selector set under the time box. The SDK headers were available locally in Xcode 26.6, so this lane uses a minimal Objective-C shim instead. That kept the Rust surface small and made the selectors compile against Apple's canonical headers while still using MPSGraph rather than custom shaders.

## Correctness and throughput evidence

The committed repo does not include `bench/data/corpus-v2.jsonl` because `bench/data/*.jsonl` is gitignored. Verification used `~/bench-tools/unified-rt-metal/corpus-1000.jsonl` on the M1 bench box, matching the previously documented 1,000-row corpus-v2 subset.

Reference vectors were generated with `lane-ort-embed` using `sentence-transformers/all-MiniLM-L6-v2` ONNX fp32, mean pooling, max length 512. Rank-overlap checks used `synapse-bench parity --k 10 --stride 1` so every shared item acted as a query.

### Quality / parity on the 1,000-chunk subset

| Variant | Mean cosine vs ORT fp32 | Mean top-10 overlap | Worst-decile overlap | Notes |
|---|---:|---:|---:|---|
| ORT MiniLM fp32 reference | reference | reference | reference | Reference vectors for this run |
| Owned RT Metal resident encoder fp32 | 0.9999999999986154 | 1.0000 | 1.0000 | Current fp32 recheck on the resident path |
| Owned RT Metal resident encoder f16 | 0.9999993204962262 | 0.9995 | 0.9950 | `--dtype f16` resident path |

The f16 path cleared the tail-sensitive gate shape the project uses elsewhere (`>= 0.999` cosine, `>= 0.95` rank overlap, `>= 0.9` worst-decile overlap). The f16-vs-fp32 quality delta was tiny: about `-6.80e-7` mean cosine, `-0.0005` mean top-10 overlap, and `-0.0050` worst-decile overlap.

### Throughput on the M1 bench box

Timed runs used the shared lock at `[bench-user-home]/bench.lock`. The new f16 numbers were measured after the parity pass above, then rerun immediately under the same lock to capture the warm-cache state identically to the published fp32 rows.

| Variant | First locked tok/s | Warm locked tok/s | Notes |
|---|---:|---:|---|
| Owned RT CPU/Accelerate | 6,955.5 | — | Scalar provider, Accelerate SGEMM |
| Owned RT Metal MPSGraph matmul + CPU pointwise | 2,338.8 | — | Previous eager mixed provider; boundary dominated |
| Owned RT Metal resident encoder MPSGraph fp32 | 75,996.3 | 116,534.4 | Published resident fp32 locked pair |
| Owned RT Metal resident encoder MPSGraph f16 | 4,236.4 | 4,234.3 | New locked pair; essentially no warm-cache gain |

### Context numbers from the lane prompt for the same M1 bench box and corpus

| Engine context | Tok/s | Resident fp32 gap | Resident f16 gap |
|---|---:|---:|---:|
| llama.cpp Metal | 51.8k | fp32 is ~1.47x faster cold-ish, ~2.25x faster warm | f16 is ~0.08x as fast |
| MLX Metal | 132.6k | fp32 is ~0.57x as fast cold-ish, ~0.88x as fast warm | f16 is ~0.03x as fast |
| ORT CPU | 11.5k | fp32 is ~6.6x to ~10.1x faster | f16 is ~0.37x as fast |
| Owned RT CPU seed | ~5-10k | fp32 is above this band | f16 falls below the measured CPU baseline |

Interpretation: the fp32 resident result still says block-level residency matters, and the f16 result says dtype alone is not a free speed win on this backend. **f16 kept the vectors but lost the speed**: versus resident fp32, the new path is about **17.9x slower** on the first locked run and **27.5x slower** on the warm rerun.

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

What the f16 resident variant changed:

- Encoder-layer static parameters are converted once to f16 mirrors at load and uploaded from those mirrors rather than from fp32 buffers.
- Hidden states cross the CPU/GPU boundary as f16 instead of fp32.
- Layernorm / GELU / softmax are explicitly cast to f32 inside the graph, then cast back to f16, because those are the cheap reduction-heavy spots where MPSGraph gives an explicit control surface.
- The public MPSGraph matmul API does **not** expose a separate accumulation-precision control, so the measured performance difference comes from MPSGraph's native float16 execution path rather than from extra synchronization boundaries.

The measured effect remains the core lesson of the spike: replacing CPU loops with separate GPU ops would have increased synchronization pressure, while changing the provider contract to whole encoder blocks removes hundreds of per-batch upload/readback boundaries. But the new result adds an important qualifier: **cutting boundary bytes in half was not enough to make MPSGraph float16 faster on this M1 path**.

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
| Resident f16 variant + M1 measurement | 1.6 | Added load-time f16 mirrors, upload/readback boundary casts, full-tail parity checks, and honest locked throughput numbers |
| Memo update | 0.5 | Results, transfer analysis, revised verdict |

Total Lane 2 time is about **9-9.5 hours**. Combined with the first CPU/runtime spike, the evidence base is about **13.5-14.5 hours**.

## Walls hit

1. **Eager op granularity was the wrong contract.** Matmul plan caching fixed graph construction overhead but not synchronization. Whole-block residency fixed the measured bottleneck.
2. **MPSGraph can express the full MiniLM encoder block.** Exact-erf GELU, layernorm via mean/variance primitives, softmax, reshapes/transposes, residuals, and all matmuls fit in a single graph.
3. **Shape specialization remains a scheduling concern.** The implementation caches one graph per batch/sequence shape; production should bucket or prewarm shapes to control compile/cache churn.
4. **Numerical parity requires exact boring details.** GELU variant, layernorm epsilon, mask value, mean-pooling mask, tokenization/truncation, and position/type embeddings all had to match. A model family change repeats this work.
5. **This does not solve the runtime product surface.** Production still needs scheduler-level queues, cancellation, backpressure, warm pools, memory budget accounting, f16/quant coverage, and model-family coverage.
6. **Float16 quality and performance decoupled.** Tail metrics stayed excellent, but throughput collapsed. Until MPSGraph's float16 behavior is explained and improved, a smaller dtype does not imply a faster Metal lane.

## What would be hard next

Engineering-hard, not research-hard:

- Generalize the block-level provider contract or introduce a tiny graph IR so other encoders can hand MPSGraph whole blocks without bespoke FFI structs.
- Diagnose why MPSGraph float16 is dramatically slower than the resident fp32 graph on M1, and test whether a different graph formulation or newer runtime/API surface changes that result.
- Move mean pooling/L2 normalization to the device when returning pooled vectors directly, while preserving the option to read final hidden states.
- Add shape bucketing, graph prewarming, memory planning, and scratch-buffer accounting to avoid per-shape surprises.
- CUDA and Vulkan providers with common layout contracts and per-provider autotuning.

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

Total remains **60-90 engineer-weeks**, with meaningful risk that Vulkan/quant/Metal tuning expands the upper bound. The resident fp32 MiniLM result lowers the block-residency risk line item, but the f16 regression shows dtype-path risk is still real.

Adopt-path comparison: linking llama.cpp/MLX as libraries still looks like **6-12 engineer-weeks** for robust packaging, lifecycle, scheduling, API wrapping, and parity/power measurement. Owned runtime therefore costs roughly **5-10x** more before it reaches comparable breadth, even though MPSGraph block residency is now a credible fp32 Metal implementation strategy for encoder models.

## Recommendation

Keep this crate as evidence and as a CPU/Metal correctness harness. If the team wants an owned fp32 MiniLM-like encoder provider, MPSGraph block subgraphs are now a viable implementation path. Do **not** treat `--dtype f16` as a Metal speedup on the current M1 path: it preserves quality but destroys throughput. Do not choose full ownership for all v1 local inference solely from this result; make that choice only if the strategy explicitly values long-term kernel/control ownership enough to pay the runtime, scheduler, dtype, quantization, and model-family cost.