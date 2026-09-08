# M5 full-context ModernBERT spike evidence

## Static implementation checkpoint — 2026-09-08

Hardware and tools observed on the target before conversion or inference:

- MacBook Pro `Mac17,6`, Apple M5 Max, 18 cores, 128 GB memory
- macOS 27.0 build `26A5425a`
- Python 3.12.12
- Swift 6.4 (`swiftlang-6.4.0.33.1`, target arm64-apple-macosx27.0.0)
- `uv` 0.12.2
- PyTorch 2.14.0
- coremltools 9.0
- transformers 5.16.1
- tokenizers 0.23.2
- safetensors 0.8.0
- NumPy 2.5.3
- psutil 7.2.2

The cached GTE snapshot is `e7f32e3c00f91d699e8c43b53106206bcc72bb22`. Runtime reports compute content digests rather than treating that directory name as proof of checkpoint identity. coremltools 9.0 warned that PyTorch 2.14.0 was newer than its tested 2.7.0 ceiling; the spike records that compatibility warning and does not silently downgrade the installed current toolchain.

Baseline memory before implementation checks: physical 137,438,953,472 bytes; free 15,979,544,576 bytes; wired 8,447,852,544 bytes; `memory_pressure -Q` reported 78% system-wide free. Free pages alone intentionally are not presented as available-memory capacity.

## Source inspection

The implementation was designed after inspecting `smpanaro/ModernBERT-AppleNeuralEngine` at commit `d0268940c0af4ffe2bca0f9eb3842fd05984215c`. Its `model.py` sets `query_chunk_size=8192`; each query chunk contracts against full-length keys and values, and softmax normalizes over the full key dimension. Its local mask uses distance `<= local_attention_window_size // 2`. Its floor-derived chunk count can omit a final non-multiple tile, and its precomputed 8192 local mask is approximately 66 MB. The implementation here retains the full K/V property, fixes the tail, and emits only tile-local local masks.

The historical negative in `john-rocky/CoreML-LLM` PR 169 concerns Perplexity's bidirectional Qwen3-0.6B `pplx-embed` models, an un-tiled fixed 8192 graph, Apple M4 Max, and macOS 26. The reported runtime error was `ANEProgramProcessRequestDirect status=0x15: Program Inference error`. No broad ModernBERT or M5 hardware limit is inferred from it.

## Measurement result: stopped at 1024

The primary granted a non-overlapping hardware measurement slot after the matched-ANE worker completed its conflicting interval. The 1024 export bounded the largest PyTorch and MIL attention score output to `[1, 1024, 1, 256]`; conversion, separate reload, prediction, and placement inspection completed. The three CPU_AND_NE vectors had cosines `0.9984976053`, `0.9999022484`, and `0.9987312555` against the bounded full-context reference. The minimum is below the unchanged `0.999` gate, so this spike does **not** establish acceptable 1024, 4096, or 8192 Core ML parity.

Same-input diagnostics localize the behavior:

- custom query-tiled versus Hugging Face pooling: minimum cosine `1.0`, max absolute error `6.2584877e-7`;
- exported versus eager custom checkpoints and pooling: max absolute error `0.0`;
- independently assembled global/local masks and RoPE: max absolute error `0.0`;
- CPU_AND_NE raw token embeddings: exact; embedding normalization remains at cosine `0.9999998212`; active CLS first crosses `0.999` at layer 16's attention residual for the full and partial-tile rows, where sampled reference activations reach `18396.0176`;
- CPU_ONLY raw token embeddings: exact, but embedding normalization immediately falls to minimum cosine `0.3606245518`, before attention. The lowered MIL layer norm has the intended final axis on `[1,1024,768]`, so CPU_ONLY's final negative cosine exposes a Core ML CPU execution-path problem rather than disproving tiled attention.

`MLComputePlan` preferred all 2,112 einsums, 1,056 softmaxes, 88 convolutions, and 45 layer normalizations on the Neural Engine for CPU_AND_NE. This is placement-plan evidence, not a runtime trace. Repeated CPU_AND_NE prediction was deterministic (`max_abs=0.0`). Conversion took 41.059 s, separate load 33.796 s, first predict 47.531 ms, and the measured warm predict 34.611 ms.

An interim driver revision incorrectly treated the parity miss as a warning and began later stages before diagnosis. Its external 2048/4096 logs are preserved exactly under `/tmp/modernbert-full-context-m5`; the 4096 placement child was terminated, no 8192 stage started, and those later artifacts are excluded from feasibility claims. The committed driver now treats any sub-gate parity result as a hard stage stop. Full compact metrics, digests, placement, memory observations, and the artifact pointer are in `EVIDENCE-1024.json`.
