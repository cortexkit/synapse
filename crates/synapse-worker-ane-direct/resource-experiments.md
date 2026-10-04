# Fresh-process ANE resource experiments

## Scope and interpretation

Measured on Apple M5 Max with cargo 1.99.0 / rustc 1.99.0, release profile,
using the manifest-pinned converted gte-modernbert-base package. Each width was
run in a separate process, with no earlier resident shapes in that process.
Admission timings include graph construction, compilation, loading and the
payload-check instrumentation; they are not isolated private `load` call times.
The single-layer controls time compilation plus loading after graph construction.
The tests leave serving behavior unchanged; instrumentation is test-only. A
read-only binding accessor reports the exact MIL and packed weight bytes that
the existing compiler submits. The tests verify those lengths against the
emitted files and remove their exact artifacts after dropping the executables.
Apple's unload can already remove those files; cleanup tolerates that case.

`TMPDIR` was unset. Redirecting it into the worktree caused an unrelated
`verifyBundleAtPath: invalid model` failure and is not a resource-limit result.

The earlier failing ladder explicitly cleared `model.resident` before every
admission: it did **not** retain six shapes. That experiment failed while
accumulating executables within one shape at layer 14. However, this fresh
release experiment does **not reproduce** a resource ceiling:

| Width | Loaded layers retained together | Admission milliseconds | Final resident set | Load averages before → after (1/5/15 min) |
|---:|---:|---:|---|---|
| 2048 | 22 / 22 | 20841.473 | `[2048]` | 9.66/15.30/21.85 → 8.20/14.49/21.37 |
| 4096 | 22 / 22 | 40572.766 | `[4096]` | 8.20/14.49/21.37 → 9.37/13.95/20.85 |
| 8192 | 22 / 22 | 85424.919 | `[8192]` | 9.37/13.95/20.85 → 9.02/12.73/19.69 |

An earlier fresh 4096 release trial also admitted all 22 layers in 51329.792ms;
its diagnostic cleanup assertion subsequently failed because Apple's unload
had already removed a directory. That harness issue was corrected before the
three successful runs above. No serving algorithm was changed.

Consequently neither a hard per-program size limit at 4096 nor a hard
within-shape executable-count limit is established. The earlier resource error
could reflect transient/lingering ANE allocations or another resource consumer;
CPU load averages do not identify ANE memory usage. This experiment cannot
attribute the old failure to either resident bytes or executable count. No
resource-management fix or graph redesign has been implemented.

## Single mid-depth executable controls

Layer 14 (local attention) also loaded by itself in separate fresh release
processes:

| Width | Load milliseconds | MIL bytes | Weight bytes | Outcome | Load averages before → after |
|---:|---:|---:|---:|---|---|
| 4096 | 2039.822 | 127758 | 14195094 | LOADED | 9.44/12.64/19.53 → 9.48/12.60/19.48 |
| 8192 | 3400.576 | 236229 | 18391446 | LOADED | 9.48/12.60/19.48 → 10.08/12.67/19.46 |

## Payload of each resident executable

All 22 loads succeeded at each width. Values below are **bytes of compiler input**,
not opaque compiled-program allocation sizes. Packed weights include learned
weights, RoPE tables, local-distance masks, scalars and packing headers. MIL text
size can vary slightly with emitter iteration order without changing the graph.
Layers divisible by three use global attention; the others use local attention.

| Layer | Attention | MIL 2048 | Weights 2048 | MIL 4096 | Weights 4096 | MIL 8192 | Weights 8192 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | global | 66241 | 11078354 | 115441 | 12126930 | 213853 | 14224082 |
| 1 | local | 73541 | 12096918 | 127766 | 14195094 | 236226 | 18391446 |
| 2 | local | 73532 | 12096918 | 127765 | 14195094 | 236203 | 18391446 |
| 3 | global | 68510 | 11080086 | 117722 | 12128662 | 216123 | 14225814 |
| 4 | local | 73541 | 12096918 | 127773 | 14195094 | 236205 | 18391446 |
| 5 | local | 73541 | 12096918 | 127768 | 14195094 | 236221 | 18391446 |
| 6 | global | 68514 | 11080086 | 117724 | 12128662 | 216127 | 14225814 |
| 7 | local | 73540 | 12096918 | 127769 | 14195094 | 236226 | 18391446 |
| 8 | local | 73540 | 12096918 | 127759 | 14195094 | 236214 | 18391446 |
| 9 | global | 68497 | 11080086 | 117719 | 12128662 | 216125 | 14225814 |
| 10 | local | 73533 | 12096918 | 127769 | 14195094 | 236225 | 18391446 |
| 11 | local | 73538 | 12096918 | 127767 | 14195094 | 236221 | 18391446 |
| 12 | global | 68507 | 11080086 | 117728 | 12128662 | 216126 | 14225814 |
| 13 | local | 73537 | 12096918 | 127770 | 14195094 | 236211 | 18391446 |
| 14 | local | 73534 | 12096918 | 127756 | 14195094 | 236210 | 18391446 |
| 15 | global | 68512 | 11080086 | 117726 | 12128662 | 216128 | 14225814 |
| 16 | local | 73528 | 12096918 | 127754 | 14195094 | 236211 | 18391446 |
| 17 | local | 73532 | 12096918 | 127766 | 14195094 | 236227 | 18391446 |
| 18 | global | 68519 | 11080086 | 117724 | 12128662 | 216130 | 14225814 |
| 19 | local | 73532 | 12096918 | 127756 | 14195094 | 236225 | 18391446 |
| 20 | local | 73535 | 12096918 | 127769 | 14195094 | 236223 | 18391446 |
| 21 | global | 68498 | 11080086 | 117715 | 12128662 | 216130 | 14225814 |

## Observable I/O allocations and activation estimates

The binding exposes IOSurface allocation sizes for the worker's persistent I/O,
but not the private compiler's internal activation allocations or compiled
program sizes. The worker allocates these shared I/O surfaces only **after all
22 executables load**; therefore those surfaces were not present at the earlier
layer-14 failure.

| Width | Three hidden IOSurfaces, each | Mask IOSurface | Total IOSurface bytes | Total fp32 conversion-scratch bytes |
|---:|---:|---:|---:|---:|
| 2048 | 3145728 | 16384 | 9453568 | 18882560 |
| 4096 | 6291456 | 16384 | 18890752 | 37765120 |
| 8192 | 12582912 | 16384 | 37765120 | 75530240 |

Static graph-shape estimates (fp16, not measured peak allocations):

- Global attention has 12 heads and query tiles of 128. Each score-sized tensor
  is `[1,12,128,width]`: 12 MiB at 4096 and 24 MiB at 8192. Across all query tiles,
  one full score/probability family is 384 MiB or 1536 MiB respectively. Actual
  allocator reuse and tile liveness are private-compiler decisions.
- Local attention keys cover at most 256 positions, so one interior score-sized
  tensor is 0.75 MiB at either width. Layer 14 is a **local**, not global, layer.
- QKV and the combined MLP activation/gate projection both have 2304 channels:
  each sequence-wide tensor is 18 MiB at 4096 and 36 MiB at 8192.

Global attention is the quadratic tensor family, but these estimates cannot
prove it dominated the old load failure. Splitting attention and FFN has not
been implemented or measured. A complete layer already loads alone and all
22 full layers load together in these fresh trials; a split is not required to
fit the observed one-shape experiment. Whether it improves multi-shape headroom
needs a separate design/measurement decision.

## Reproduction

Run each width in its own process, with an explicit external timeout:

```
env -u TMPDIR ANE_DIAGNOSTICS=1 ANE_TEST_PACKAGES=../../target/ane-direct-packages ANE_TEST_SHAPE=4096 cargo test --release --locked -p synapse-worker-ane-direct fresh_process_gte_single_shape_admission -- --ignored --nocapture
env -u TMPDIR ANE_DIAGNOSTICS=1 ANE_TEST_PACKAGES=../../target/ane-direct-packages ANE_TEST_SHAPE=8192 ANE_TEST_LAYER=14 cargo test --release --locked -p synapse-worker-ane-direct fresh_process_gte_single_layer_load -- --ignored --nocapture
```

## Warm inference at 512 tokens

Each model used the committed fixture with exactly 512 composed tokens, a fresh
process and its manifest-pinned converted package. One untimed-for-median first
inference initializes the cached requests, followed by five timed full worker
forward passes; the reported warm value is their median. Compilation is excluded
from inference times. Each timed warm output was byte-identical to the initial
inference output that initialized the cached requests.
The fixture metrics below were identical in release and debug builds.

| Model | Release compile ms | Release first ms | Release warm median ms | Debug first ms | Debug warm median ms | Earlier debug first at **128**, ms |
|---|---:|---:|---:|---:|---:|---:|
| gte-modernbert-base | 10169.353 | 396.985 | 394.883 | 9469.445 | 7862.136 | 1068.531 |
| gte-reranker-modernbert-base | 9552.709 | 408.962 | 378.843 | 8444.831 | 8215.912 | 1394.890 |
| qwen3-embedding-0.6b | 25206.793 | 111.688 | 85.029 | 269.529 | 255.872 | 29.424 |
| qwen3-reranker-0.6b | 25453.085 | 110.545 | 85.871 | 744.664 | 182.304 | 32.431 |

These are not controlled release-versus-debug speedup measurements: release
runs had one-minute load averages 11.42–16.61, while debug runs reached 40.01.
The earlier debug measurements used short inputs at rung 128 and measured the
first inference, so they are explicitly not equivalent to the 512-token warm
medians. No comparison against Metal was run.

Release warm samples, verbatim from the tests (milliseconds):

- gte-modernbert-base: `[393.488875, 394.882792, 400.41133299999996, 400.656125, 390.907792]`; cosine `0.999984522`.
- gte-reranker-modernbert-base: `[373.806125, 381.828708, 385.02333300000004, 371.754084, 378.84325]`; score `0.9543079`, expected `0.9541642069816588`, absolute error `0.000143707`.
- qwen3-embedding-0.6b: `[84.827458, 85.049125, 85.02908400000001, 85.290375, 84.752625]`; cosine `0.999957070`.
- qwen3-reranker-0.6b: `[86.477917, 86.120667, 85.314334, 85.870584, 85.25654200000001]`; score `0.998698`, expected `0.9987668991088868`, absolute error `0.000068903`.

Debug warm samples, verbatim (milliseconds):

- gte-modernbert-base: `[7650.489624999999, 9586.259917, 7862.1356670000005, 8247.432333, 7828.144625]`.
- gte-reranker-modernbert-base: `[7676.138792, 8374.908375, 11328.45125, 8215.911708, 7052.462125]`.
- qwen3-embedding-0.6b: `[177.86475, 230.32512499999999, 255.872167, 293.975541, 345.912875]`.
- qwen3-reranker-0.6b: `[167.0455, 175.318542, 186.696834, 191.45387499999998, 182.303666]`.

The diagnostic changes intentionally make no residency-cap fix: one-shape
fresh-process admission passed, the older ladder already cleared previous
shapes, and no deterministic count/byte ceiling was reproduced. Full
simultaneous multi-model residency stress, padded-golden certification and
replacement-module lock testing remain outside this measurement report. The
latter kills the module holding the ANE lane lock and verifies that a replacement
cannot acquire the lane until all old workers have exited.
