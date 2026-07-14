# Vulkan wave 3: device-local weights on Ally RDNA3

## Verdict

**Device-local weight placement is correct hygiene, but it does not close the deep-family gap on this UMA GPU.** The implementation moved every immutable weight from memory type 1 on heap 0 (`HOST_VISIBLE | HOST_COHERENT`) to memory type 0 on heap 1 (`DEVICE_LOCAL`, not host visible), populated it through a staging copy, and kept activations on the existing host-visible path. Frozen parity was bit-for-bit unchanged at the reported metric precision for MiniLM, gte-modernbert, and Qwen3.

The performance hypothesis failed its predefined `1.3x` stop gate. Using the worst observed pass, gte-modernbert improved only **1.049x** (`3,888.6 -> 4,080.4 tok/s`) and Qwen3 improved **1.062x** (`984.3 -> 1,045.4 tok/s`). MiniLM was effectively flat within single-process variation (`43,820.3 -> 43,431.9 tok/s`, `0.991x`) under the attribution campaign's matching bucket policy. A post-change ModernBERT timestamp profile makes the negative result unambiguous: large-linear time fell only **3.49%**, while its share of GPU time rose from **81.50% to 81.64%**.

**No graduation.** ModernBERT and Qwen3 remain far below the wave-2 same-day llama.cpp-Vulkan results of `7,525.0` and `1,621.6 tok/s`. The stop gate fired before new same-day llama.cpp repeats, so those historical same-day cells are shown as the incumbent reference rather than mislabeled as wave-3 same-day measurements. The remaining gap is in large-GEMM execution efficiency, not Vulkan memory placement.

## Allocation change

The shader graph is unchanged. Cooperative-matrix selection, plain-PV fallback, descriptors, dispatch geometry, and barriers are identical to wave 2.

Immutable tensors now use this path:

1. Create a storage buffer with `STORAGE_BUFFER | TRANSFER_DST`.
2. Select a memory type containing `DEVICE_LOCAL` and explicitly reject every type containing `HOST_VISIBLE`.
3. Query `VK_EXT_memory_budget` before each immutable allocation and reject an allocation that would exceed the selected heap's current budget.
4. Create a coherent host-visible `TRANSFER_SRC` staging buffer, map and fill it, copy with `vkCmdCopyBuffer`, and publish the transfer write to compute-shader reads with a buffer memory barrier.
5. Retain the existing one-buffer-per-tensor layout. This is finer-grained than per-layer allocation and avoids a single giant allocation crossing the former roughly 512 MiB boundary.

The upload command pool and fence live with the Vulkan device and are reused at model load. Activations, intermediates, masks, and readback buffers still use the original host-visible coherent storage path. Immutable per-shape tables such as RoPE and attention-band constants also use the staged device-local path.

## Heap placement evidence

`vulkan_probe` now records physical heaps, memory types, and `VK_EXT_memory_budget` usage/budget. The Ally reported:

| Heap | Size | Flags | Budget at process start |
|---:|---:|---|---:|
| 0 | 4,242,145,280 B (3.95 GiB) | non-device-local | 4,030,038,016 B (3.75 GiB) |
| 1 | 8,484,290,560 B (7.90 GiB) | `DEVICE_LOCAL | MULTI_INSTANCE` | 8,060,076,032 B (7.51 GiB) |

| Pool | Before | After |
|---|---|---|
| immutable weights | type 1, heap 0, `HOST_VISIBLE | HOST_COHERENT` | type 0, heap 1, `DEVICE_LOCAL`, **not** `HOST_VISIBLE` |
| upload staging | none | type 1, heap 0, `HOST_VISIBLE | HOST_COHERENT` |
| activations/intermediates | type 1, heap 0, `HOST_VISIBLE | HOST_COHERENT` | unchanged |

The runtime's model-load reports prove the destination footprint, allocation granularity, and budget margin:

| Family | Immutable allocations | Device-local bytes | Share of 7.51 GiB budget | Upload time |
|---|---:|---:|---:|---:|
| MiniLM | 96 | 21,312,000 | 0.26% | 138.6 ms |
| gte-modernbert | 133 | 220,732,416 | 2.74% | 411.1 ms |
| Qwen3-Embedding-0.6B | 309 | 881,065,984 | 10.93% | 1,636.2 ms |

Qwen3 therefore crossed the old approximately 512 MiB cumulative-weight point while every tensor remained an independent allocation on the invisible device-local type. Heap budget did not force a split or fallback, and no weight allocation landed on the host-visible type.

## Frozen parity

All cells used the frozen ORT references and the same 400-row corpora as wave 2. The gate requested for this wave was cosine `>= 0.9999989`.

| Family | Mean cosine before | Mean cosine after | Top-10 overlap after | Gate |
|---|---:|---:|---:|---|
| MiniLM | 0.9999996779 | 0.9999996779 | 0.998750 | PASS |
| gte-modernbert | 0.9999990203 | 0.9999990203 | 0.997750 | PASS |
| Qwen3-Embedding-0.6B | 0.9999989402 | 0.9999989402 | 0.998250 | PASS |

This is expected for an allocation-only change, but it also certifies the staging copy and transfer-to-compute memory dependency. No parity movement occurred.

## Full-corpus throughput and stop gate

Every after cell was a fresh process, used policy-1 serving buckets, ran three passes, and had a 60-second idle interval before launch. The table reports all after passes and uses the worst pass for the ratio. ModernBERT and Qwen3 before values are the wave-2 unprofiled cooperative cells. MiniLM before is the attribution campaign's matching policy-1 three-pass cell because wave 2 documented its MiniLM control separately.

| Family / real tokens | Before pass tok/s | After pass 1 | After pass 2 | After pass 3 | Worst before -> after | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM / 66,783 | 43,820.3 / 44,453.4 / 44,410.7 | 44,670.3 | 43,431.9 | 43,434.7 | 43,820.3 -> 43,431.9 | 0.991x |
| gte-modernbert / 62,838 | 3,966.2 / 3,910.2 / 3,888.6 | 4,093.9 | 4,080.4 | 4,086.0 | 3,888.6 -> 4,080.4 | **1.049x** |
| Qwen3-Embedding-0.6B / 46,716 | 985.4 / 984.3 / 985.6 | 1,051.0 | 1,045.4 | 1,047.3 | 984.3 -> 1,045.4 | **1.062x** |

The predefined stop-and-ask condition was improvement below `1.3x`; ModernBERT triggered it immediately. The follow-up scope retained one Qwen3 cell because its 840 MiB immutable footprint could expose a working-set-dependent effect, plus one profiled ModernBERT cell. Worse-of-two fresh-process repeats and new llama.cpp cells were intentionally pruned rather than polishing a failed hypothesis. The one-process result is sufficient to reject a 1.3x placement effect: both deep families are near `1.05x`, and Qwen3's much larger footprint does not scale the gain.

For orientation only, not as wave-3 same-day cells:

| Family | Wave-3 worst tok/s | Wave-2 same-day llama.cpp-Vulkan | Owned / historical incumbent |
|---|---:|---:|---:|
| MiniLM | 43,431.9 | 34,295.7 | 1.266x |
| gte-modernbert | 4,080.4 | 7,525.0 | 0.542x |
| Qwen3-Embedding-0.6B | 1,045.4 | 1,621.6 | 0.645x |

## ModernBERT stage re-attribution

The timestamp method is unchanged from `VULKAN-ATTRIBUTION.md`: every dispatch records timestamps before dispatch, after dispatch, and after its existing compute barrier; the first execution of each shape is preload and excluded; means cover the three timed corpus passes.

| Stage | Before, ms/pass | Device-local, ms/pass | Change | After GPU share |
|---|---:|---:|---:|---:|
| QKV | 4,883.151 | 4,711.642 | -3.51% | 33.33% |
| Out | 851.382 | 832.734 | -2.19% | 5.89% |
| MLP up | 4,659.445 | 4,524.780 | -2.89% | 32.01% |
| MLP down | 1,562.432 | 1,470.543 | -5.88% | 10.40% |
| **four large linears** | **11,956.410** | **11,539.699** | **-3.49%** | **81.64%** |
| QK | 256.157 | 231.965 | -9.44% | 1.64% |
| softmax + mask | 642.335 | 617.139 | -3.92% | 4.37% |
| PV | 627.518 | 612.950 | -2.32% | 4.34% |
| pointwise | 604.588 | 566.719 | -6.26% | 4.01% |
| layout / transpose | 494.883 | 479.587 | -3.09% | 3.39% |
| barriers | 87.514 | 85.735 | -2.03% | 0.61% |
| **GPU span** | **14,670.028** | **14,134.421** | **-3.65%** | **100%** |

The large-linear share did **not** drop; it increased from `81.50%` to `81.64%` because all stage classes moved only a few percent. Device-local placement did not change the limiting mechanism. The remaining owned/llama gap must be sought inside the large-GEMM implementation: tile/dispatch geometry, cooperative-matrix coverage, subgroup utilization, occupancy, and data reuse are candidates for a separate shader wave. None was changed here.

### Depth cliff

ModernBERT has only 22 layers and did not exhibit the Qwen3 cliff in the original profile. Its mean large-linear time per layer was:

| ModernBERT layer band | Before | Device-local |
|---|---:|---:|
| layers 1-17 | 530.955 ms/pass | 516.038 ms/pass |
| layers 18-21 | 526.499 ms/pass | 515.385 ms/pass |
| late / early | 0.992x | 0.999x |

There is no new ModernBERT late-layer cliff. The original `1.32x` cliff was in Qwen3 layers 18-27. A post-change Qwen3 timestamp cell was not run after the stop gate narrowed the measurement scope, so this report does **not** claim that the cliff itself disappeared. What the full-corpus Qwen3 cell does establish is that placing all 840 MiB of immutable tensors on non-host-visible heap 1 yields only `1.062x`; the old cumulative 512 MiB boundary was not the primary cause of the end-to-end deficit.

## Graduation and next decision

Wave 3 is an honest negative performance result:

- **Keep the allocation code.** It enforces explicit immutable/device-local semantics, checks the real heap budget, records placement, preserves parity, and causes no material steady-state regression. It also removes accidental dependence on memory-type enumeration order.
- **Do not graduate the deep Vulkan families.** ModernBERT remains `0.542x` and Qwen3 `0.645x` of the prior same-day llama.cpp results.
- **Close the placement branch.** On this UMA APU, heap 0 and heap 1 ultimately share LPDDR5, and the AMD driver was evidently already serving the host-visible weight reads close to the device-local path.
- **Any later wave must be shader-only and separately justified.** The timestamp profile points to large-GEMM efficiency, not barriers, descriptors, layout, or memory placement.

## Verification and raw evidence

The Ally build passed strict clippy for both Vulkan binaries and the focused memory-policy test. The package's full `cargo test` run compiled the new backend and passed 20 tests, with one unrelated existing CPU hand-kernel tolerance test failing on the Ally (`cpu_backend::tests::hand_kernel_tracks_f16_reference_across_vector_tails`); the Vulkan memory-policy test was rerun alone and passed.

Fixture hashes, rig facts, heap data, full LaneResult JSON, load logs, and raw ModernBERT timestamp rows are in [`results/vulkan-wave3/`](results/vulkan-wave3/). The probe output is [`vulkan-probe.json`](results/vulkan-wave3/vulkan-probe.json), and the aggregation source is [`modernbert-profile.ndjson`](results/vulkan-wave3/modernbert-profile.ndjson).
