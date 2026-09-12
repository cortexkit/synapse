---
name: ANE direct kernels (private API, bypassing Core ML)
description: Drive Apple's Neural Engine directly from Rust through the private _ANEInMemoryModel API instead of Core ML — op set, hardware limits, and the dispatch-versus-SRAM tradeoff. Use when building or optimizing an ANE execution path in synapse.
keywords: ANE, Neural Engine, CoreML, private API, _ANEClient, _ANEInMemoryModel, IOSurface, MIL, kernels, fusion, dispatch, SRAM, fp16, embedding, ModernBERT, Espresso, silicon swarm
---

# ANE direct kernels

Use this when you are building or optimizing an execution path on the Apple
Neural Engine and Core ML's behaviour is the constraint — fixed input shapes,
opaque per-shape compilation cost, duplicated weights per compiled bundle, or
no visibility into where time actually goes.

Do not use it for the shipping embed lane. That runs on Core ML today and is
certified; this is the research path beside it.

## Reference implementation

`mutable-state-inc/siliconswarm-at-ensue-plugin` on GitHub (MIT). Clone it
wherever this machine keeps third-party checkouts; paths below are relative to
that clone.

- `skills/ane-private-api/SKILL.md` — the complete binding reference. Read this
  first and in full; it is 200 lines and replaces reading the source.
- `ane_kernel/crates/ane/` — Rust bindings. `graph/` builds and compiles,
  `ops/` has the op implementations, `ane_client.rs` and
  `ane_in_memory_model.rs` are the private-API bridge.
- `ane_kernel/crates/ane/examples/distilbert_model.rs` — a full encoder built
  with these primitives. The closest thing to a worked example for a BERT-family
  model; read `compile_layer` and the hand-built `layer_norm` and `gelu`.
- `ane_kernel/crates/ane/examples/distilbert_verify.rs` — how they gate accuracy
  before accepting a speedup.

A second implementation exists in Swift at `christopherkarani/Espresso` (MIT),
which generates MIL text and compiles it through `_ANEClient`. Useful as a
cross-check on API shape; the Rust one is the one to build on here.

## Shape of the API

```rust
let mut g = Graph::new();
let x = g.placeholder(shape);
let y = g.inner_product(x, &weights, in_ch, out_ch);
let exe = g.compile(NSQualityOfService::UserInteractive)?;
exe.run_cached(&[&input], &[&output])?;
```

Tensors are 4D NCHW and live in IOSurface buffers. Compute is fp16; the API
reads and writes f32 and converts at the boundary. Weights passed to
`inner_product` are `[out_channels, in_channels]` row-major, which is PyTorch's
`nn.Linear` layout, and are baked into the compiled program as fp16.

There is no fused GELU, SiLU, or layer norm. Compose them from primitives — the
DistilBERT example does exactly this and is the reference for how.

## Hardware limits that decide the design

| Property | Value |
|---|---|
| Compute precision | fp16 only |
| Dispatch overhead | ~0.095 ms per `run()` (XPC/IOKit) |
| SRAM | ~32 MB. Weights under 16 MB stream at ~15,000 GB/s; larger fall to ~51 GB/s DRAM |
| Graph depth | ~60 ops compiles. 2 fused transformer layers run; 3 compiles then crashes at runtime |
| Placeholder width | must be >= 64; pad shorter sequences |
| QoS | `UserInteractive` is lowest latency |

Treat every number here as a hypothesis to re-measure on the target chip, not
as a constant: they were measured on other people's hardware, and the SRAM
figures did NOT reproduce on ours — see the section below before designing
around them.

## What we measured on this hardware, and what did not hold

Measured on an M5 Max under macOS 27; full numbers and method in
`docs/evidence/ane-direct-api-m5/`. These supersede the reference table above
where they disagree, for this chip.

**Batch attention over heads. This is the single largest lever found.** A
ModernBERT-shaped layer whose attention is built as one matmul pair per head
costs 41 ms at 512 positions; reshaping so heads sit on the channel axis and
issuing one matmul across all of them costs 0.79 ms, for identical arithmetic.
53x from graph construction alone. The DistilBERT example expresses it the
batched way, with `reshape` to `[1, heads, head_dim, seq]`, a `transpose`, one
`matrix_multiplication`, and the inverse afterwards.

**There is no SRAM cliff here.** Weights from 2 MiB to 72 MiB, well past the
claimed 32 MB, produced a smooth monotonic curve with no discontinuity either
side of the reported 16 MB threshold. So do not design a fusion depth around
fitting weights into cache; that optimum does not exist on this chip.

**Fusion is not the lever.** Fusing one, two and three real layers changed
per-layer time by under 1%, because the fixed per-dispatch cost is about 0.2% of
a 1.5 ms layer. Fusion is worth pursuing only where sequences are short enough
that the fixed cost dominates the work.

**A new sequence shape costs about 0.2 s to compile**, flat from 128 to 2048
positions, against Core ML's 32 s at 1024 and 579 s at 8192. This is the real
prize: shapes can be compiled at startup rather than shipped as artifacts, so a
bucket ladder's step count stops being a cost worth designing around.

**Constant-weight ops reach 4,800 to 10,500 GFLOP/s; attention reaches about
1,000.** Anything with weights baked at compile time flies. The dynamic path,
where both matmul operands are runtime tensors, is roughly ten times worse per
FLOP and is where the remaining headroom is.

**A faithful port lands near parity, not ahead.** Assembling those parts over
ModernBERT's 22 layers gives about 30 ms against the shipping Core ML lane's
25.5 ms. Reach for this path for control over shapes and packaging, not for an
expected speedup.

## Measuring honestly

`run_cached_with_stats` is meant to return `hw_execution_time_ns`, real
nanoseconds on the Neural Engine excluding XPC and dispatch. It returns 0 on our
hardware: the bindings set the statistics mask on the request after compilation
and it appears to belong on the model before load. Until that is fixed, timing
is wall clock and includes dispatch, so measure ratios between arms in one run
rather than trusting an absolute figure.

`run_cached` reuses the request object and saves the dispatch overhead, but the
SAME `TensorData` objects must be passed every call — contents may change,
identities may not. `run_cached_direct` goes through
`_ANEClient.doEvaluateDirectWithModel` and skips the ANE daemon entirely.

Gate accuracy before accepting any speedup. A faster kernel that returns
different vectors is not a result.

## Cost of this path

These are undocumented private APIs found by runtime introspection. They can
break at any macOS release, and an app using them is rejected from the App
Store. That is acceptable for a locally installed daemon and for research; it is
a decision to revisit before anything here becomes the only way a lane serves.
