# Q8 embedding: quality measured; speed still unresolved

**Decision:** block-32 Q8_0 is the best of the measured compression schemes.
It is **not numerically equivalent to f16**, but it changes less than ~1.3% of
non-empty pools' top-10 membership relative to f16, with no observed mean Sol-quality
loss. Per-channel W8 is worse, and unconditioned dynamic W8A8 changes too many
rankings to adopt on the promise of speed. **Do not design four speed kernels
from this evidence:** no permitted quiet-window serving measurement ran.

[Full tables](TABLES.md) · [machine-readable aggregates](aggregate.json) ·
[quiet-window refusal](speed-status.json). Only scripts and aggregates are public;
queries, candidate text, identities, token IDs, vectors and logs stay private.

## Part A — protocol and interpretation

Measured on an **Apple M5 Max, 128 GiB unified memory, macOS 27.0.1**, via
PyTorch MPS, sequentially, with CPU threads capped at four, batch 8, eager attention, no autocast. Versions are pinned in
`requirements.txt`. This is a numerical experiment, **not** a serving throughput
benchmark; `runs.elapsed_s` in the JSON is progress accounting only.

Both HF revisions and every listed file hash were checked against
`bench/parity/models.json`:

| Model | Revision | Readout |
|---|---|---|
| Qwen3-Embedding-0.6B | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | last active token, explicit EOS **151643**, L2 |
| gte-modernbert-base | `e7f32e3c00f91d699e8c43b53106206bcc72bb22` | CLS, L2 |

The manifest owns special-token IDs, rather than optional HF tokenizer attributes.
Qwen query text is `Instruct: {task}\nQuery: {query}`; documents have no prefix.
The public, established code-retrieval task is “Given a web search query, retrieve
relevant code snippets that answer the query”, used for both query populations.
The reference computes in fp32 from the published checkpoint values; those values
were not originally trained/stored in fp32. F16 means the same PyTorch model in
f16, not certification of our separate MPSGraph implementation.

All **85 AFT / 92 context-search queries**, including empty/small pools, and
**3,613 / 3,730** frozen candidates from the September rerank evaluation were
retained. Candidate hashes and both full-pool Sol partitions were checked. There
are **5,744 distinct encoded sequences per model** after deduplication. The frozen
text is reused unchanged, then each model reserves its terminal token within a
512-token ceiling; 15 Qwen / 14 ModernBERT unique sequences need further clipping.
There was no retrieval replay, new labeling, text enrichment or external text upload.

Quantization covers all 2-D weights, **including token embeddings**; 1-D norms
remain floating point. Q8_0 uses 32-value blocks, f32 absmax/127 division,
round-away-from-zero codes and a stored f16 scale; dequantized weights compute in
f16. Channel W8 uses symmetric per-output-row int8 and f32 scales. W8A8 uses those
weights, per-token dynamic absmax int8 **at every linear layer** (196 Qwen / 66
ModernBERT), exact int32 accumulation, rescaling and f16 residual/nonlinear paths.
Embedding lookups in W8A8 are weight-only; attention QK/AV products are not linear
modules and are not quantized. On MPS, bounded f32 partial dot products are exact
integer sums, converted to int32 before further accumulation; extreme/cancellation
controls compare with independent CPU int64 products. This is not an int8 speed test.

**Metric definitions:** cosine min/p1/p50 uses unique queries and documents within
each tool, recomputed in float64. Rankings use cosine, with exact ties preserving
frozen pool order. Overlap@10 is intersection of the two displayed top-10 prefixes,
divided by `min(10, reference-relevant count, candidate count)`, or zero if the
reference has no relevant items. Against an embedding rank, all candidates count
as reference-relevant. Against Sol, only its **ordered relevant top 10** count,
not any relevant item outside that prefix. This is the rerank README's definition.
Kendall τ uses strict full-pool orders, excluding n<2 pools (AFT n=79, context n=85).
Empty pools make the self-overlap ceilings **83/85 = .976471** and
**90/92 = .978261**, not 1. Judge deltas average A and B per query and use 10,000
paired query-bootstrap resamples, seed 2609 (NumPy generator).

### The actual consumer GGUF matches, exactly

`Qwen/Qwen3-Embedding-0.6B-GGUF`, revision
`370f27d7550e0def9b39c1f16d3fbaa13aa67728`, file
`Qwen3-Embedding-0.6B-Q8_0.gguf`:

**SHA-256 `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`.**

The `gguf` package reads/dequantizes every mapped tensor: **197 Q8_0 tensors,
595,710,976 quantized values, 113 F32 norm tensors**. Zero values differ from our
independent simulation, zero differ from an f16-source simulation, and the own
implementation agrees exactly with the package's independent quantize/dequantize
path. The real-GGUF and simulated-Q8 embedding tables are identical. All F32
norm tensors also equal the pinned HF values; no tensor was silently omitted or
permuted. The paired f16 file's SHA-256 is
`421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340`.

### Inside or outside the numerical and judge rulers?

**Every quantized scheme is outside the f16-vs-fp32 numerical difference** on
cosine and ranking overlap/τ, for both models and tools. Q8_0 is not an f16 alias.
F16/fp32 lose only .00118–.00326 aggregate overlap; Q8_0 loses .00706–.01413 from
the empty-pool-adjusted self ceiling. Channel W8 and W8A8 lose substantially more.
ModernBERT's Q8 minimum cosine (.993732) warrants an explicit tail gate, not just
an average. Qwen W8A8 p1 is only .9705–.9720.

Sol overlap happens to be **identical for f16 and fp32 in all four populations**.
Relative to that zero judge-delta ruler, Qwen Q8_0/channel context deltas are inside
(also zero); all other quantized judge deltas are outside. But **all are inside the
observed judge movement**: per-query mean absolute f16 A/B changes are .0541–.0648,
versus scheme mean changes of −.0051 to +.0101. Sol B/A agreement reproduces the
published **.770350 (AFT, n=85)** and **.620118 (context, n=89)**, excluding only
the three same-order two-item context repeats. Judge noise is a sensitivity ruler,
not an upper bound or proof of equivalence. Most paired intervals include zero;
Q8's small nonnegative intervals are not evidence of a robust quality improvement.
In particular, noisy judge agreement does not excuse W8A8's large numerical drift.

## Part B — deferred, with a runnable harness

Every observed 1-minute load was over 16 (initial 41.69, intermediate probe 60.87;
the final probe is recorded in `speed-status.json`). No llama build, serving arm
or AFT fill was run, and no live module was called. The table is intentionally blank:

| Serving arm | 128/512 tokens × batch 1/16/64 | AFT fill rows/min | call p50/p90 | peak memory |
|---|---|---|---|---|
| llama.cpp f16 GGUF | deferred | not requested | — | — |
| llama.cpp consumer Q8_0 | deferred | deferred | — | — |
| owned f16 Metal, scratch production module | deferred | deferred | — | — |

`speed.py` pins llama.cpp **`680a036285273a3ff56032ec5d7f3352609eba4f`**, requires a
clean source checkout and Metal CMake configuration, records binary/model hashes,
and confirms GPU offload. Five interleaved/reversed rounds cover every sweep
cell, with independent process/warmup lifetimes. The three interleaved AFT fill
rounds per lane last at least **180 seconds** each: 64 rows/call, two callers,
100–150 tokens including EOS. Synthetic code averages ~395 characters including
its short header. Text goes through both serving tokenizers; llama `/tokenize`
is checked against the common HF IDs before timing, and completed token counts
are validated on both paths. There is no pre-tokenized llama timing shortcut.

Synapse uses the existing **`subc_call` → management `embed.batch`** path, launching
its own `ck-synapse` under a unique scratch module ID with isolated data/store/lease
roots. It never targets `synapse`. Production inline admission and 3,072-token bulk
quanta are retained; job responses include **all** `embed.result` pages and item IDs.
Latency includes transport, Synapse CLI startup, queuing, compute and vector
readback. Fill rows/min counts completed rows over total wall time, including the
final drain; per-call p50/p90 and load are recorded. Every arm/request checks load
<16 and aborts rather than waiting or producing contaminated medians. Darwin
`time -l` captures lifetime high-water RSS and, when available, peak physical
footprint; this includes model loading/warmup/KV allocations, not just timed calls.
The client and shared daemon are outside that process-memory boundary. Llama uses
64 sequence slots for sweeps / 128 for fills, so two 64-row requests can actually
batch rather than being serialized through two sequence slots.

### Reproduction

Run from the repository root. Keep `.private/` ignored; never publish its contents.
Use Python 3.12 and the pinned numerical environment:

```sh
E=docs/evidence/q8-embedding
uv venv "$E/.private/venv" --python 3.12
uv pip install --python "$E/.private/venv/bin/python" -r "$E/requirements.txt"
PY="$E/.private/venv/bin/python"
"$PY" "$E/test_quality.py"
"$PY" "$E/test_speed.py"
"$PY" "$E/quality.py" --device mps --batch-size 8 \
  --private-dir "$E/.private/vectors" --output "$E/aggregate.json"
"$PY" "$E/render_tables.py" --input "$E/aggregate.json" --output "$E/TABLES.md"
"$PY" "$E/speed.py" --probe --private-dir "$E/.private/speed" --output "$E/speed-status.json"
```

Quality defaults to offline pinned HF cache and the private September evaluation
root. `--allow-download` fetches public model files only. Missing private data is
an error, not permission to replace it with synthetic quality queries.

During a quiet window, prepare llama inside `.private/` (the fetched source was
inspected to confirm the existing server flags and text/token contracts):

```sh
git clone https://github.com/ggml-org/llama.cpp.git "$E/.private/llama.cpp"
git -C "$E/.private/llama.cpp" checkout --detach 680a036285273a3ff56032ec5d7f3352609eba4f
# Refuse heavy preparation on a busy machine too.
"$PY" -c 'import os; assert os.getloadavg()[0] < 16'
cmake -S "$E/.private/llama.cpp" -B "$E/.private/llama.cpp/build" -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$E/.private/llama.cpp/build" --target llama-server -j 4
"$PY" "$E/speed.py" --private-dir "$E/.private/speed" --output "$E/speed-status.json" \
  --llama-source "$E/.private/llama.cpp" --llama-server "$E/.private/llama.cpp/build/bin/llama-server" \
  --snapshot "$PINNED_HF_SNAPSHOT" --f16-gguf "$PINNED_F16_GGUF" --q8-gguf "$CONSUMER_Q8_GGUF" \
  --synapse-binary "$CK_SYNAPSE" --subc-call "$SUBC_CALL" --subc "$SUBC_CONNECTION_FILE"
```

The last three variables name existing, matching production binaries and an
explicit daemon connection. No management operation or transport shim is added.
Only the newly launched scratch module is called; only owned child PIDs are stopped.

## Recommendation and quality gate

| Backend | What to build, conditional on a measured benefit |
|---|---|
| Metal | Q8_0 **resident compressed weights with fused dequantization**, initially for memory/capacity. Weight-only speed is unproven; finish Part B and AFT fills before optimizing GEMM. Do not adopt raw dynamic W8A8. |
| CUDA | Q8_0 weight-only first for the best measured quality. Native int8 Tensor Core W8A8 might buy compute speed, but this unconditioned scheme fails the proposed ranking gate; calibrate/condition and remeasure before committing to it. |
| Vulkan | Q8_0 weight-only for capacity, with a device-specific latency gate. No evidence here establishes portable int8 dot-product speed or makes per-channel weights preferable. |
| ANE | Through our fp16-only API, Q8 can mean **compressed storage/host staging only**, expanded at load. ANE compute and resident expanded tensors remain fp16: no int8 speedup or promised halving of ANE resident memory. |

Weight-only block storage is ~**53.1%** of f16 matrix bytes (34 bytes/32 weights),
not a 2× runtime speed promise. Expanding *all* weights into persistent f16 buffers
at load forfeits resident-memory savings. Activation quantization is **not known
to be necessary for speed**; it is a separate quality-costing option to test only
if weight-only throughput proves compute-bound. Nothing measured here warrants
per-channel W8 instead of Q8_0 for these embedding lanes.

Proposed **separate Q8 numeric identity**, not f16 equivalence: on each population,
non-empty overlap@10 vs f16 ≥.985; full-pool τ ≥.975; cosine vs fp32 min ≥.99,
p1 ≥.9994, p50 ≥.99965; paired mean Sol-overlap change lower 95% bound ≥−.005 on
both reference passes together, with neither pass hiding a material regression.
The observed Q8_0 passes these provisional gates; channel W8/W8A8 do not. Require
backend parity to the **dequantized Q8 reference**, a fresh held-out query set,
plus a quiet throughput/memory gate and peak-load headroom before release.
These are proposed rollout tolerances derived from this sample, not a claimed
existing f16 certification pass. Apply them separately to each model and backend.

## UNRESOLVED

- Apple serving speed, peak memory and **AFT re-embed wall time** remain unmeasured.
  After fills, estimate `minutes = corpus rows / measured sustained rows per minute`;
  do not substitute short PyTorch arm elapsed times or extrapolate from tokens/s.
- The scratch production and llama launch paths are source-checked and covered by
  transport/page/token controls, **not end-to-end exercised** without a quiet window.
  Native Metal build and GPU memory-accounting scope still need live verification.
- The retrieval pools are conditional on the September candidate retrieval, not a
  full-index recall test. Many short metadata-only candidates and small context
  pools make membership agreement forgiving; a new held-out retrieval test matters.
- MPS PyTorch fp32/f16 is not CPU-oracle or owned-kernel certification. This study
  does not measure activation-conditioned W8A8, attention-matmul quantization,
  larger Qwen 4B/8B models, calibration costs, or other backend performance.
- The ModernBERT low-cosine tail and judge instability need attention before broad
  deployment. Mean improvements inside judge movement are not proof of no loss.
