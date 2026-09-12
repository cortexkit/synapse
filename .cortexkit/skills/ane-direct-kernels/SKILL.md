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

Cloned locally at `~/Work/OSS/siliconswarm-at-ensue-plugin` (MIT).

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
as a constant. They were measured on other people's hardware and at least one
(the fusion depth limit) is reported differently on different chips.

## The tradeoff that matters

Fusing layers cuts the per-dispatch tax, but a fused graph's weights must still
fit under the 16 MB SRAM threshold or the whole dispatch drops to DRAM
bandwidth — a ~300x difference. So there is an optimum fusion depth rather than
"fuse as much as possible", and it moves with the model's per-layer weight size.

Work it out before writing kernels: divide the model's fp16 weight bytes by its
layer count. GTE ModernBERT is 149M parameters, about 300 MB in fp16 across 22
layers, so roughly 13.6 MB per layer — just under the threshold alone, and over
it if two layers are fused.

## Measuring honestly

`run_cached_with_stats` returns `hw_execution_time_ns`: real nanoseconds on the
Neural Engine, excluding XPC and dispatch. Use it to separate hardware time from
framing cost before optimizing either.

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
