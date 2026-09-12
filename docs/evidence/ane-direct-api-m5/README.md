# Direct Neural Engine access on M5 Max / macOS 27

Measured 2026-09-12 on an Apple M5 Max, macOS 27.0 (26A5425a), driving the
private `_ANEInMemoryModel` API through the Rust bindings published at
`mutable-state-inc/siliconswarm-at-ensue-plugin` (MIT).

Probes live in `bench/spikes/ane-direct-probe/`. They keep their own workspace
and reach the bindings by relative path, because vendoring an external checkout
into this repository's workspace would make every build depend on a clone that
is not part of it.

## The API works here, and computes exactly

The published compatibility matrix for those bindings covers M1 through M4 on
macOS 15. Neither this chip nor this OS is in it, and the bindings resolve the
private framework by `dlopen` at runtime, which a successful build does not
prove.

A projection through an identity matrix returned its input with a worst absolute
error of **exactly 0**. Compilation, dispatch, and hardware execution all hold.
The input was distinct per (channel, position) rather than constant, so a
collapsed projection or a transposed axis would have failed rather than passed
quietly.

One layout fact is worth stating because it is easy to get backwards: in this
API's NCHW, **the sequence axis is WIDTH** and the hidden dimension goes on
channels. A placeholder narrower than 64 is refused at compile time with
`SpatialWidthTooSmall`. With height 1, the element for channel `c` at position
`w` sits at `c * width + w`.

## There is no SRAM cliff on this chip

The binding reference reports ~32 MB of SRAM, with weights under 16 MB streaming
at ~15,000 GB/s and larger ones dropping to ~51 GB/s — a ~300x discontinuity
that would dominate any fusion decision.

It does not reproduce. Growing a square projection's weights from 2 MiB to
72 MiB, well past the claimed SRAM size, produced a smooth monotonic curve with
no discontinuity anywhere:

| weights | median | weights | median |
|---:|---:|---:|---:|
| 2.0 MiB | 105 µs | 32.0 MiB | 309 µs |
| 8.0 MiB | 139 µs | 40.5 MiB | 358 µs |
| 15.1 MiB | 187 µs | 50.0 MiB | 413 µs |
| 18.0 MiB | 208 µs | 60.5 MiB | 494 µs |
| 24.5 MiB | 238 µs | 72.0 MiB | 565 µs |

Either side of the claimed 16 MB threshold — 15.1 MiB at 187 µs and 18.0 MiB at
208 µs — the curve is continuous. Two sweeps with different sample points agree
on a fixed cost of **~92 µs per dispatch**, which matches the same reference's
dispatch-overhead figure closely. So that document's dispatch number reproduces
here and its memory-hierarchy number does not.

Note that this sweep alone cannot attribute the slope: a square projection grows
weight bytes and multiply-accumulates together, so memory and arithmetic are
perfectly confounded in it. The absence of a discontinuity is robust to that;
any bandwidth read from the slope is not.

## Short sequences are bound by dispatch, long ones by arithmetic

Holding weights fixed at 8 MiB and growing only the sequence separates them,
because arithmetic then scales while weight movement does not:

| seq | median | µs per position |
|---:|---:|---:|
| 64 | 153.9 | 2.404 |
| 128 | 145.2 | 1.134 |
| 256 | 189.0 | 0.738 |
| 512 | 343.8 | 0.672 |
| 1024 | 554.6 | 0.542 |

Doubling from 64 to 128 positions costs nothing — the time falls slightly, which
is noise around a flat region. Past 256 the curve is linear at roughly 0.48 µs
per position. So a dispatch is dispatch-and-weight bound at short sequences and
arithmetic bound at long ones, crossing over somewhere between 128 and 256 for
this shape.

The minimum width is also not the fastest: 64 positions cost more in total than
128. Whatever the hardware's internal tiling is, the floor is not an optimum.

## What this means for fusing transformer layers

Fusing layers into one graph saves the fixed per-dispatch cost. It does not save
arithmetic, and on this chip it does not rescue anything from a memory cliff,
because there is no cliff to fall off.

So the payoff is bounded and depends on sequence length:

- At serving lengths of 512 tokens the work is arithmetic bound, and fusion buys
  back only ~92 µs per dispatch eliminated. For a 22-layer model dispatched one
  layer at a time, that is about 2 ms of overhead in total, and halving the
  dispatch count recovers about 1 ms of it.
- At short sequences the same fusion is worth proportionally far more, because
  the fixed cost is most of the time spent.

This corrects a prediction made before measuring. The reference's SRAM cliff
implied an optimum fusion depth — fuse until a graph's weights cross 16 MB, then
stop. With no cliff, there is no such optimum from memory, and fusion should go
as deep as the compiler tolerates. The reported depth limit (~60 ops, two
transformer layers running and three crashing) is then the binding constraint,
and it is a compiler limit rather than a bandwidth one.

## Instrumentation gap

`run_cached_with_stats` returns 0 ns at every size on this machine. The bindings
set the statistics mask on the request after compilation; the accompanying
comment suggests it belongs on the model before load. Every timing here is
therefore wall clock and includes dispatch overhead, which is why the fixed cost
appears as a fit intercept rather than a direct reading.

This is worth fixing if this path is pursued: it is the only tool that would
separate hardware time from framework overhead, including for the Core ML lane
we would be comparing against.

## Standing caveats

Load average was 7 to 12 during these runs, not idle. The absolute numbers carry
that; the shapes of the curves are what the conclusions rest on. Every figure is
a single machine, one chip, one OS build, and a single `inner_product` rather
than a transformer layer. The private API is undocumented and can change without
notice.
