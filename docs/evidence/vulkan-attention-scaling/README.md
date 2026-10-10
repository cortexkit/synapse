# Vulkan attention scaling: Qwen3 at 8192 tokens

On a rented RTX 5070 Ti (driver 610.57.04), the owned Vulkan worker matched
the fp32 references for Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B up to
513 tokens. Their 8192-token rows failed the worker's fixed 30 s request
timeout. One 512-token row took about 0.42 s. Those records, made at commit
`dd7412b0`, were filed as development evidence on the branch
`alfonso/task/bg_b173cdb59264cfff-prove-the-cuda-and-vulkan-catalog-lanes-on-a-ren`
(`docs/evidence/catalog-backends/`). This change replaces the Vulkan kernel
revision and all four Vulkan lane fingerprints, so every Vulkan record made at
`dd7412b0` is superseded and must be remade.

## Cause

The old attention kernel (op 6 in `shaders/common.glsl`) gave each
(query row, head) pair its own thread, and that thread walked every key alone:

- It re-read its 128-channel query from global memory for every key.
- It read K and V with lane-strided addresses: neighbouring threads were
  different heads, 128 floats apart. Each warp-wide load therefore touched up
  to 32 cache lines.
- It kept a `float acc[256]` that was indexed with a runtime bound, so the
  compiler spilled it to local memory. Every key read and wrote all 128
  accumulator values through that memory.
- No K/V row was reused across queries except by chance in cache.

Qwen3 pays this cost about five times over compared with gte-modernbert;
measured at 8192 tokens, the ratio is 5.8×. Every one of Qwen3's 28 layers is
global attention, compared with 8 of gte's 22. It has 16 heads of 128 channels,
against gte's 12 of 64. Causality halves the key count but does not close the
gap. The matrix multiplies, norms and RoPE are linear in length,
and the worker issues the same number of dispatches at every length.

## Fix

Attention now runs in its own pipeline, `shaders/attention.comp`, which
compiles the `ATTENTION` section of `common.glsl`:

- A workgroup of 64 threads owns a tile of consecutive query positions for one
  head: 16 positions for Qwen3's 128-channel heads, 32 for gte's 64-channel
  heads.
- Each 16- or 32-key tile of K and V is read from global memory once, with
  coalesced loads, into shared memory. It then serves every query in the tile.
- Each query's head is split across 2 to 8 threads. Each thread holds 32
  channels of the query and of the accumulator in registers, with fixed-bound
  unrolled loops.
- The softmax is still online. It is rescaled once per key tile instead of once
  per key, so there is still no sequence-squared buffer.
- Causal and local-window limits skip whole tiles that no query in the tile can
  see, then apply per key.

The kernel uses 24 KiB of shared memory. A separate pipeline keeps that
reservation from lowering the occupancy of every other kernel in the plain
shader. Matrix multiplies, norms and dispatch are unchanged.

## Growth curve (MoltenVK, Apple M5 Max, plain path)

The rows come from `cargo test --release -p synapse-worker-vulkan --features
moltenvk-diagnostic --test gpu_scaling -- --ignored --nocapture`, which times
one row taken from the `long-8192` fixture.

The per-stage split comes from a temporary host timer around each dispatch,
which was not committed. It appears in the logs as `TMP_VK_PROFILE`. The
worker waits for the queue to go idle after every dispatch, so each stage's
time is its own. `op6` is attention; `op1` is the linear projections, which
use the naive plain kernel here and the cooperative-matrix kernel on NVIDIA.

Logs: `moltenvk-before.log`, `moltenvk-after.log`.

Qwen3-Embedding-0.6B, seconds per row:

| tokens | before total | before attention | before linear | after total | after attention | after linear |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.383 | 0.375 | 0.913 | 1.019 | 0.050 | 0.880 |
| 1024 | 2.615 | 0.800 | 1.693 | 2.142 | 0.161 | 1.868 |
| 2048 | 5.574 | 2.549 | 2.918 | 3.752 | 0.432 | 3.189 |
| 4096 | 15.511 | 9.519 | 5.751 | 7.302 | 1.288 | 5.889 |
| 8192 | 51.595 | 38.959 | 12.284 | 16.781 | 4.493 | 12.059 |

gte-modernbert-base, seconds per row:

| tokens | before total | before attention | after total | after attention |
|---:|---:|---:|---:|---:|
| 512 | 0.321 | 0.065 | 0.251 | 0.013 |
| 1024 | 0.580 | 0.123 | 0.474 | 0.027 |
| 2048 | 1.373 | 0.451 | 0.958 | 0.077 |
| 4096 | 3.421 | 1.771 | 2.006 | 0.242 |
| 8192 | 9.931 | 6.747 | 4.024 | 0.774 |

Attention is still quadratic, as exact attention must be. Its cost now
roughly triples per doubling (0.05 → 0.16 → 0.43 → 1.29 → 4.49 s), but its
constant is 7.5× lower at 512 tokens and 8.7× lower at 8192. For Qwen3 at 8192,
attention was 76% of the row before the fix and is 27% after. The rest is the
plain linear kernel, which grows linearly. Whole-row growth from 512 to 8192
fell from 37× to 16×.

An alternative with two queries per thread and 8-key tiles measured slower here
(5.16 s of attention at 8192) and was not kept.

## Expected time on the RTX 5070 Ti

The rental's single-row Qwen3-Embedding times were 55.6 ms at 4 tokens,
100.3 ms at 128 and 418.3 ms at 512. Fitting `c + L·n + A·n²` with c = 54 ms
gives L ≈ 0.245 ms/token for the linear stages and A ≈ 9.1e-4 ms/token² for
attention, which was already about 57% of the 512-token row. That fit predicts
about 63 s for the old kernel at 8192, of which about 61 s is attention. That
is consistent with the observed timeout. The rental also bounds old attention
at 8192 to at least about 28 s.

Applying the measured 7.5–8.7× attention reduction gives:

- linear stages and dispatch: about 2.1 s;
- attention: about 3.2–8.2 s, from an old cost of 28–61 s;
- **total: about 5–10 s for one 8192-token Qwen3 row**, against the 30 s
  request timeout.

Scaling only by MoltenVK's new whole-row growth (16.5× from 512 to 8192) gives
0.42 s × 16.5 ≈ 6.9 s. That estimate is less reliable, because on MoltenVK the
naive linear kernel dominates the row.

## Numerics

Logs: `moltenvk-parity-before.log`, `moltenvk-parity-after.log` and
`old-vs-new-outputs.txt`. The parity test now runs every fixture length,
including the 127–513 boundary rows and the 8192-token row, each as its own
request.

| model | gate | old kernel | new kernel |
|---|---|---:|---:|
| gte-modernbert-base | batch min cosine | 0.999997848 | 0.999997855 |
| gte-modernbert-base | long-8192 cosine | 0.999999932 | 0.999999949 |
| gte-reranker-modernbert-base | batch max score error | 1.478e-3 | 1.289e-3 |
| gte-reranker-modernbert-base | long-8192 score error | 3.30e-5 | 6.09e-5 |
| qwen3-embedding-0.6b | batch min cosine | 0.999999780 | 0.999999755 |
| qwen3-embedding-0.6b | long-8192 cosine | 0.999999897 | 0.999999884 |
| qwen3-reranker-0.6b | batch max score error | 6.94e-6 | 5.62e-6 |
| qwen3-reranker-0.6b | long-8192 score error | 6.68e-6 | 1.32e-5 |

The old and new kernels agree to a cosine of at least 0.9999988 on every
output. The largest element difference is 3.5e-4, on a gte reranker batch
score. Both differences are summation-order effects in fp32. The linear
kernel rounds its inputs to fp16, so the effects grow a little over the
layers. Every gate still passes, with margins essentially unchanged.

## Regression controls

Each control was applied to the attention shader alone and restored before any
further run. The named test, `four_models_match_reference_vectors_scores_window_and_padding`,
went red each time:

- **Local window ignored** (`moltenvk-mutation-window.log`): gte-modernbert-base
  min cosine fell to 0.952105542.
- **Causal mask ignored** (`moltenvk-mutation-causal.log`): both gte models
  still passed, since they are bidirectional, and qwen3-embedding-0.6b min
  cosine fell to 0.100344844.

## Identity

`VULKAN_KERNEL_REVISION` is the SHA-256 of the SPIR-V set: plain, then
cooperative, then attention. Any shader edit moves it:

- Old: `7351dbef33cda19b4b20e8427ba71497580d9b401d023fdcc7db01456cfc9b4a`
- New: `7a5621d965123c2661ad6a795774d07b81f44a86db4d68167893ad4023e87730`

The revision feeds the numeric profile of every Vulkan lane. The four Vulkan
fingerprints in `crates/synapse-module/src/catalog/models.json` were re-minted
with the ignored test `mint_release_catalog_fingerprints`:

| lane | old | new |
|---|---|---|
| gte-modernbert-base-vulkan | `885776108d3f…` | `9d281846adc3…` |
| gte-reranker-modernbert-base-vulkan | `084f55877211…` | `160e6ae78d2b…` |
| qwen3-embedding-0.6b-vulkan | `62e7abe5dacb…` | `9a33500fe3db…` |
| qwen3-reranker-0.6b-vulkan | `e892cb51a90e…` | `be2a9be41f50…` |

The same mint reproduced all four CUDA pins byte for byte (`mint.txt`). With
the old revision temporarily restored in `synapse-core`, it reproduced all
eight old CUDA and Vulkan pins (`mint-old-revision.txt`), so the kernel
revision alone moved the Vulkan values. Only those four values changed in
`models.json`. No Metal or ANE pin changed.

The unit test `catalog_profiles_bind_identity_and_worker_package` in
`crates/synapse-module/src/lib.rs` pins each of the 16 manifest profiles'
fingerprint, computed with a fixture tokenizer. Its four `owned-vulkan` entries
moved to `c86eceac…`, `29a1a109…`, `07f11338…` and `47c91eb3…`. The other 12
values are unchanged. The kernel revision itself is pinned in
`crates/synapse-core/src/worker_engine_names.rs` and in
`crates/synapse-worker-vulkan/tests/fixtures/kernel-revision.txt`.

## Not yet proven

- No NVIDIA or AMD run has been made with this kernel. The 5–10 s estimate
  assumes that the reduction measured through MoltenVK carries over to the NVIDIA
  driver. In particular, it assumes the driver keeps the 32-channel query and
  accumulator slices in registers.
- The cooperative-matrix linear kernel's time at 8192 tokens has only been
  extrapolated from rows of 512 tokens or fewer.
- Lane certification at the new fingerprints must be redone on real hardware.
