# Warm 512-token forward stages

Measured after the reboot, OS build 26A434, release build with manifest-pinned
packages and the committed 512-token fixtures. Each model runs in its own fresh
process: one initializing forward, five untraced timed warm forwards, then one
traced warm forward. The stage table is from that last forward, not the median.
Compilation and traced-forward printing are excluded from the median.

## Cause and fix

The bottleneck was two dense scalar fp32 CPU projections, not per-layer fp16
conversion, surface allocation, or request recreation. ModernBERT's residual
`rotation_in.weight` is the converted dense `Qᵀ C diag(gamma) Q` projection;
it cannot be replaced by a bare fast Hadamard transform. It now runs as one
CPU Accelerate SGEMM over all sequence columns, using the exact committed fp32
matrix. Rotation-out also uses batched fp32 SGEMM. CLS embedding and Qwen's
last-token readout normalize/project only the row they consume; the ModernBERT
classifier still processes every unpadded row for mean pooling. No transformer
layer moved to CPU, and no ANE graph, grouping or executable was changed.

The probe retains persistent alternating IOSurfaces and cached requests just
like this worker. Its graph uses unrotated original weights, with embedding and
final normalization on ANE, rather than these folded rotated weights and CPU
boundary projections. The current probe also groups two layers per executable
at 512; the worker still dispatches one executable per layer. The optimized gte
median nevertheless beats the supplied 31.6ms direct-probe reference (and is
close to the supplied 25.5ms Core ML reference).

## Before and after medians (milliseconds)

Fixture metrics compare the optimized worker output to the pinned Transformers
reference for the same composed 512-token input. Embedding cosine measures
vector agreement (1 is identical direction); reranker `abs` is the absolute
difference between the worker's probability and the reference probability.

| Model | Before | After | After fixture metric |
|---|---:|---:|---|
| gte-modernbert-base | 403.621 | 26.800 | cosine 0.999983791 |
| gte-reranker-modernbert-base | 469.290 | 31.387 | score 0.9543699; expected 0.9541642069816588; abs 0.000205696 |
| qwen3-embedding-0.6b | 84.769 | 84.128 | cosine 0.999957070 |
| qwen3-reranker-0.6b | 85.062 | 83.329 | score 0.998698; expected 0.9987668991088868; abs 0.000068903 |

FP32 BLAS changes accumulation order versus scalar summation, so ModernBERT
outputs need not be bit-identical to the old implementation. Warm repeat outputs
were byte-identical within each run. The independent fixture metrics are above;
these are not a full padded-golden certification. Qwen has no rotation boundary
and remains dominated by ANE evaluation.

One-minute load averages varied during before/after runs: before gte 19.71–22.68,
before reranker 32.88–24.10, before Qwen embedding 24.10–28.80, before Qwen reranker
28.80–27.59; after gte 41.98–18.65, after reranker 18.65–17.30, after Qwen embedding
17.30–32.25, after Qwen reranker 32.25–37.33. These are observed-at-load timings,
not a controlled load-matched speedup experiment. No full stress test was run
at those loads.

## One traced forward, host stages (milliseconds)

| Stage | gte before | gte after | reranker before | reranker after | Qwen embed before | Qwen embed after | Qwen rerank before | Qwen rerank after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Host prologue | 1.569208 | 1.641250 | 1.858833 | 1.524250 | 1.767583 | 1.734667 | 1.787167 | 1.722167 |
| Initial fp32→fp16 surface copy | 0.035000 | 0.083334 | 0.035875 | 0.023041 | 0.031042 | 0.032708 | 0.034083 | 0.033084 |
| rotation_in fp32 | 239.148125 | 2.302042 | 288.371333 | 2.045459 | — | — | — | — |
| Residual fp32→fp16 surface copy | 0.032292 | 0.027750 | 0.029084 | 0.024417 | — | — | — | — |
| Mask surface copy | 0.003166 | 0.005083 | 0.003834 | 0.003583 | 0.003458 | 0.003416 | 0.002625 | 0.003250 |
| Final fp16→fp32 readback | 0.049417 | 0.053625 | 0.050125 | 0.054500 | 0.068834 | 0.066542 | 0.064250 | 0.063042 |
| Final norm + rotation_out fp32 | 122.300375 | 0.030500 | 149.171125 | 3.625208 | 0.990750 | 0.009042 | 0.969000 | 0.009000 |
| CPU head | 0.001875 | 0.001000 | 0.713250 | 0.415291 | 0.001209 | 0.000958 | 0.003458 | 0.002416 |
| All cached-request preparation | 0.001539 | 0.002040 | 0.001669 | 0.000832 | 0.000750 | 0.001293 | 0.002000 | 0.001207 |
| All synchronous ANE evaluation | 23.299334 | 22.796542 | 22.741208 | 22.222500 | 81.324748 | 81.287167 | 82.711040 | 80.903876 |

Every traced warm dispatch reused its existing cached request: zero new
requests, no IOSurface allocation, and no host copy/conversion between layers.
The latter two follow directly from borrowing alternating resident surfaces in
the dispatch loop. Copies occur only at the explicit boundary stages above.
The vendor's profiling API measures actual cached-slot state for request reuse.

Apple's bound `evaluateWithQoS:options:request:error:` is synchronous. Cached
request preparation is timed separately, but submission and waiting occur
inside one private call and are **not separably observable** through this API.
The per-layer numbers below are explicitly submit-plus-wait, not invented
asynchronous submit/wait timings.

## Per-layer synchronous submission plus wait (milliseconds)

| Layer | gte before | gte after | reranker before | reranker after | Qwen embed before | Qwen embed after | Qwen rerank before | Qwen rerank after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.252667 | 1.220459 | 1.204417 | 1.167000 | 3.027750 | 3.059417 | 3.033958 | 3.025208 |
| 1 | 1.018833 | 0.955958 | 0.952375 | 0.931042 | 2.892208 | 3.009209 | 2.983458 | 2.882083 |
| 2 | 1.012083 | 0.963333 | 0.948750 | 0.931334 | 2.890209 | 2.969042 | 2.984958 | 2.885750 |
| 3 | 1.202583 | 1.151208 | 1.133917 | 1.128333 | 2.902500 | 2.844708 | 3.103458 | 2.867041 |
| 4 | 1.002750 | 0.982708 | 1.034250 | 0.931333 | 2.877334 | 2.846542 | 2.960791 | 2.904250 |
| 5 | 0.998833 | 0.994375 | 1.051584 | 0.925834 | 2.895042 | 2.869167 | 2.939917 | 2.879417 |
| 6 | 1.191834 | 1.130000 | 1.159083 | 1.107291 | 2.916833 | 2.932000 | 2.964417 | 2.881500 |
| 7 | 0.981500 | 0.962250 | 0.999625 | 0.914667 | 2.925375 | 2.903875 | 2.962834 | 2.876417 |
| 8 | 0.969500 | 0.982625 | 0.931042 | 0.914041 | 2.901083 | 2.855916 | 2.968083 | 2.895042 |
| 9 | 1.205834 | 1.144458 | 1.153500 | 1.129833 | 2.944708 | 2.887667 | 2.978584 | 2.876084 |
| 10 | 1.009833 | 0.953583 | 0.943250 | 0.933792 | 2.918666 | 2.865791 | 2.965916 | 2.869250 |
| 11 | 1.009833 | 0.965125 | 0.950458 | 0.951750 | 2.915083 | 2.848000 | 2.923916 | 2.862708 |
| 12 | 1.161167 | 1.155250 | 1.147125 | 1.139958 | 2.892583 | 2.900167 | 2.948166 | 2.876042 |
| 13 | 0.960250 | 0.967167 | 0.973000 | 0.960875 | 2.876375 | 2.881667 | 2.997792 | 2.877042 |
| 14 | 0.969500 | 0.945458 | 0.929750 | 0.954250 | 2.880750 | 2.841917 | 2.973167 | 2.884209 |
| 15 | 1.152959 | 1.138375 | 1.152500 | 1.131250 | 2.907000 | 2.867417 | 2.962875 | 2.869417 |
| 16 | 0.963708 | 0.948584 | 0.938458 | 0.960375 | 2.865959 | 2.889583 | 2.947541 | 2.909750 |
| 17 | 0.981250 | 0.960792 | 0.939458 | 0.953500 | 2.870917 | 2.867125 | 2.930667 | 2.931916 |
| 18 | 1.157125 | 1.241542 | 1.130334 | 1.120417 | 2.878875 | 2.858750 | 2.912208 | 2.886792 |
| 19 | 0.965458 | 0.944750 | 0.993541 | 0.947583 | 2.873333 | 2.866042 | 2.889125 | 2.902542 |
| 20 | 0.971667 | 0.947458 | 0.949625 | 0.943667 | 2.870833 | 2.860583 | 2.921084 | 2.922417 |
| 21 | 1.160167 | 1.141084 | 1.125166 | 1.144375 | 2.917416 | 2.994791 | 2.906541 | 2.910083 |
| 22 | — | — | — | — | 2.894083 | 2.868166 | 2.908042 | 2.866458 |
| 23 | — | — | — | — | 2.899541 | 2.874792 | 2.933958 | 2.902125 |
| 24 | — | — | — | — | 2.887417 | 2.870333 | 2.944708 | 2.861208 |
| 25 | — | — | — | — | 2.943833 | 3.098042 | 2.921917 | 2.869875 |
| 26 | — | — | — | — | 2.926625 | 2.922333 | 2.918542 | 2.862625 |
| 27 | — | — | — | — | 2.932417 | 2.934125 | 2.924417 | 2.866625 |

Reproduce through the package's ignored hardware latency test with
`ANE_TEST_MODEL=<slug> ANE_TRACE_FORWARD=1 ANE_DIAGNOSTICS=1`, the converted
packages in `ANE_TEST_PACKAGES`, and `cargo test --release --locked -p
synapse-worker-ane-direct real_weight_512_warm_latency -- --ignored --nocapture`.
Leave TMPDIR unset or exactly the normal Darwin user temp directory.
