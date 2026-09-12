# Direct Neural Engine access on M5 Max / macOS 27

Measured 2026-09-12 on an Apple M5 Max, macOS 27.0 (26A5425a), driving the
private `_ANEInMemoryModel` API through the Rust bindings published at
`mutable-state-inc/siliconswarm-at-ensue-plugin` (MIT).

Probes live in `bench/spikes/ane-direct-probe/`. They keep their own workspace
and reach the bindings by relative path, because vendoring an external checkout
into this repository's workspace would make every build depend on a clone that
is not part of it.

Load average was 8 to 12 throughout, not idle. Absolute figures carry that; the
ratios between arms measured in the same run are what the conclusions rest on.

## Verdict

A faithful port of gte-modernbert-base to this API lands at roughly **30 ms** for
a 512-position row, against **25.5 ms** for the same model through the shipping
Core ML lane. So direct access reaches Core ML's league but does not beat it by
being faithful. Any win has to come from doing something Core ML does not.

The reason is concentrated in one place: ops with compile-time weights run at
4,800 to 10,500 GFLOP/s, and attention, which needs runtime operands on both
sides, runs at about 1,000. Attention is over half the remaining cost.

## The API works here, and computes exactly

The published compatibility matrix for these bindings covers M1 through M4 on
macOS 15. Neither this chip nor this OS is in it, and the bindings resolve the
private framework by `dlopen` at runtime, which a successful build does not
prove.

A projection through an identity matrix returned its input with a worst absolute
error of **exactly 0**. The input was distinct per (channel, position) rather
than constant, so a collapsed projection or a transposed axis would have failed
rather than passed quietly.

One layout fact is easy to get backwards: in this API's NCHW, **the sequence axis
is WIDTH** and the hidden dimension goes on channels. A placeholder narrower than
64 is refused at compile time with `SpatialWidthTooSmall`. With height 1, the
element for channel `c` at position `w` sits at `c * width + w`.

## Cost of a ModernBERT layer, by part

512 positions, model geometry as configured (768 hidden, 12 heads of 64, 1152
intermediate), random weights. Only shapes affect timing, so random values time
identically to trained ones; nothing here says anything about numerics.

| part | median | GFLOP | GFLOP/s |
|---|---:|---:|---:|
| one projection, constant weights | 0.13 ms | 0.60 | 4,825 |
| gated feed-forward, constant weights | 0.26 ms | 2.72 | 10,482 |
| attention, batched over heads | 0.79 ms | 0.81 | 1,023 |
| attention, 128-token window | 0.50 ms | 0.40 | 810 |
| attention, sliced per head | 41.47 ms | 0.81 | 19 |

Assembling those: a global layer is about 1.57 ms and a windowed layer about
1.28 ms. ModernBERT runs global attention on every third layer, so 8 of its 22
layers are global and 14 windowed, giving **30.5 ms** for the model.

### Batching heads matters more than anything else measured

Expressing attention as one matmul per head — slice query, key and value per
head, then two small matmuls each — costs **41 ms**. Reshaping so heads sit on
the channel axis and issuing a single matmul across all of them costs **0.79
ms**, for identical arithmetic. That is a **53x** difference from graph
construction alone, and it was the entire content of an earlier conclusion in
this document that a direct port was 44x slower than Core ML. It was not; the
port was wrong. The reference implementation's own encoder example expresses it
the batched way.

### Windowing is worth less than sequence arithmetic suggests

A 128-token window at 512 positions touches a quarter of the score matrix, but
measured only 1.6x cheaper rather than 4x. Each 128-query tile attends to a
256-position halo, so the saving is 2x in arithmetic before accounting for four
tile matmuls running slightly slower per FLOP than one large one.

## No SRAM cliff on this chip

The binding reference reports ~32 MB of SRAM, with weights under 16 MB streaming
at ~15,000 GB/s and larger ones dropping to ~51 GB/s — a ~300x discontinuity.

It does not reproduce. Growing a square projection's weights from 2 MiB to
72 MiB, past the claimed SRAM size, gave a smooth monotonic curve: 105 µs at
2 MiB, 187 µs at 15.1 MiB, 208 µs at 18.0 MiB, 309 µs at 32 MiB, 565 µs at
72 MiB. Either side of the claimed threshold the curve is continuous.

Two sweeps with different sample points agree on a fixed cost of **~92 µs per
dispatch**, matching the same reference's dispatch-overhead figure. So that
document's dispatch number reproduces here and its memory-hierarchy number does
not.

That sweep cannot attribute its slope: a square projection grows weight bytes
and multiply-accumulates together. Holding weights at 8 MiB and growing only the
sequence separates them — 64 and 128 positions cost the same 145 to 154 µs,
after which cost rises linearly at about 0.48 µs per position. So a dispatch is
fixed-cost bound at short sequences and arithmetic bound at long ones.

## Fusion is not the lever

An earlier reading of the missing cliff concluded that layers should be fused as
deeply as the compiler allows. Measuring real layers retired that: fusing one,
two and three ModernBERT layers changed per-layer time by under 1%, because the
~92 µs dispatch cost is about 0.2% of a 1.5 ms layer.

Fusion is worth pursuing only where the fixed cost is a large share of the work,
which means very short sequences. At serving lengths it is noise.

## Instrumentation gap

`run_cached_with_stats` returns 0 ns at every size on this machine. The bindings
set the statistics mask on the request after compilation; the accompanying
comment suggests it belongs on the model before load. Every timing here is
therefore wall clock and includes dispatch overhead.

This is worth fixing if the path is pursued: it is the only tool that would
separate hardware time from framework overhead, including for the Core ML lane
this would be compared against.

## What is not measured

Numerics. Every arm uses random weights and checks no output, so none of this
says a ported model would produce correct vectors. A correctness gate against
the existing reference is required before any of it is believed as a model.

Two arms in the attribution probe report `execute failed`: four projections in
one graph, and a single dynamic matmul. Both build graphs whose terminal tensors
do not match the single output buffer the harness binds. That is a harness
limitation rather than an API one, and it leaves the dynamic matmul unpriced in
isolation.

Everything here is one machine, one chip, one OS build. The private API is
undocumented and can change without notice.
