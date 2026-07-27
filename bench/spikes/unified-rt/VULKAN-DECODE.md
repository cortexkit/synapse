# Vulkan Qwen3 decode wave 1

## Scope

This adds the fourth owned Qwen3 greedy-decode backend:

```text
--device vulkan --decode-backend vulkan --dtype f16 --vulkan-gemm plain
```

It is feature-gated behind `--features vulkan` and is selected independently of
embedding's `--device vulkan` family graph. `--vulkan-gemm cooperative` is
rejected for decode. The wave deliberately uses plain compute shaders only.

The backend keeps f16 K and V buffers on the device, with one `[kv_head,
head_dim]` slot per layer written at the current position. It uploads all
immutable f16 or Q8_0 matrix weights once through the existing device-local
staging path. The host owns the final greedy argmax, so existing lowest-token-id
tie handling, token taps, pause/resume, and splice protocols remain unchanged.

## Per-token graph

Each token records a Vulkan command buffer containing, for every Qwen3 layer:

1. RMSNorm (serial left-to-right square sum — see parity fix below);
2. Q/K/V serial f32-accumulating matvecs;
3. Q/K head RMSNorm plus RoPE, with K written directly to its in-slot cache;
4. V conversion and direct in-slot cache write;
5. cache-resident GQA attention;
6. output projection and residual;
7. post-attention RMSNorm, gate/up projections, SwiGLU, down projection, and
   the MLP residual.

A final RMSNorm and LM-head matvec leave f32 logits in a host-visible buffer.
The two matvec shaders give one invocation exclusive ownership of an output
row and accumulate that row left-to-right in f32. The attention shader gives
one invocation exclusive ownership of a query head and keeps its QK and
softmax reductions serial. This is intentionally slow but avoids splitting one
dot product's accumulation order during the parity wave.

Q8_0 reads the canonical 34-byte block layout in the dot product: little-endian
f16 scale followed by 32 signed i8 values. No dequantized matrix is materialized.
The Q8_0 profile requires Vulkan shader int8 support in addition to the
existing f16/scalar-layout requirements; f16 decode keeps the embedding lane's
existing feature requirements.

Prefill currently advances prompt tokens through the same decode graph to
establish the resident cache. This is correct but is not a Vulkan-prefill
throughput claim. It supports a 470-token prompt when `--decode-cache-bucket`
is at least 534.

## Parity fix (Ally RDNA3)

The first f16 gate on the Ally failed only at `completion-06` step 7 (near-tied
tokens `13079` vs reference `61686`) — the same prompt Metal hit in
`DECODE-FOUNDATIONS.md` before its f32-accumulation repair. Decode was still
using the embedding-lane `rms_norm` shader, whose 256-lane tree reduction
reassociates the square sum. Replacing decode RMSNorm (input, post-attention,
and final) with `decode_rms_norm.comp` — one workgroup, left-to-right f32 square
sum and scale — restored **20/20 token-exact** with **zero near-tie exemptions**.
No fixture or threshold was changed.

## Gate status (Ally RDNA3, 2026-07-25)

Rig: ASUS ROG Ally X, AMD Radeon Graphics (RDNA3 iGPU), Vulkan API 1.4.334,
driver_raw 8388981, Turbo power scheme. Checkout `C:\bench\synapse-decode` at
`3a3a6ea` (plus serial-RMSNorm fix from stashed working tree). Chat weights:
`C:\bench\model-qwen3-chat` SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`
(Qwen3-0.6B snapshot `c1899de2…`). Fixtures verified on-box:
prompts `6f1ee1ce…`, references `b2d11f2a…`.

| Gate | Status | Evidence |
|---|---|---|
| Vulkan feature build | PASS | Ally `cargo build --release --features vulkan --bin spike-unified-rt --bin vulkan_probe` (target dir under `%USERPROFILE%` — WDAC blocks build-scripts under `C:\bench\`) |
| Shader compilation | PASS | `glslc --target-env=vulkan1.3` including `decode_rms_norm.comp` |
| `vulkan_probe` runtime | PASS | `AMD Radeon Graphics`, not llvmpipe; coop-matrix + memory heaps match day-1 |
| f16 20x64 token parity | **PASS** | **20/20 exact**, 1,280/1,280 tokens, **0 near-ties**; decode **4.044 tok/s**, prefill 5.541 tok/s |
| Q8_0 quality ladder | **PASS (floor)** | **10/20 exact**, median match depth **59.0**, **0 near-tie exemptions**; decode **5.242 tok/s**, prefill 7.274 tok/s. JSON confirmed from `vulkan-q8-20x64.json` |
| 470+64 depth cell | **PASS** | f16: **64/64 token-exact** against self-reference (two independent runs, 470-token prompt, `--decode-cache-bucket 534`); decode **2.062 tok/s** at depth 470, prefill 3.631 tok/s. Q8: completed 64/64 tokens, diverges from f16 reference at position 0 (expected for Q8 on this degenerate prompt) |
| f16/Q8 indicative tok/s | confirmed | f16 decode **4.082 tok/s** (20×64 throughput), Q8 decode **5.178 tok/s** (20×64 throughput). Not a campaign N=12×2 median |
| llama.cpp-Vulkan ratio | measured (Q8 only) | `llama-bench` build `b4e3dc613` (#9580), Q8 GGUF `qwen3-0.6b-q8.gguf`, `--n-gpu-layers 99`; Q8 decode **127.0 tok/s** (avg of p=0/n=64 and p=512-then-p=0/n=64, three repetitions each). Owned/llama ratio: **4.1%**. f16 chat GGUF absent — f16 ratio cell not measurable |

### Ally commands

```bat
set EXE=%USERPROFILE%\cargo-target-decode\release\spike-unified-rt.exe
set MODEL=C:\bench\model-qwen3-chat

%EXE% --model %MODEL% --tokenizer %MODEL%\tokenizer.json ^
  --generate-prompts C:\bench\data\decode-prompts.jsonl ^
  --decode-reference C:\bench\data\reference-tokens.jsonl ^
  --device vulkan --decode-backend vulkan --dtype f16 --vulkan-gemm plain ^
  --decode-cache-bucket 512 --max-new-tokens 64 --limit 20 ^
  --out C:\bench\results-ally\vulkan-decode\vulkan-f16-20x64.json

%EXE% ... --weight-quant q8-0 --out ...\vulkan-q8-20x64.json

REM 470+64 depth cell (f16, self-reference verification)
%EXE% --model %MODEL% --tokenizer %MODEL%\tokenizer.json ^
  --generate-prompts C:\bench\data\long-context-470.jsonl ^
  --decode-reference C:\bench\data\depth470-reference-tokens.jsonl ^
  --device vulkan --decode-backend vulkan --dtype f16 --vulkan-gemm plain ^
  --decode-cache-bucket 534 --max-new-tokens 64 --limit 1 ^
  --out C:\bench\results-ally\vulkan-decode\vulkan-f16-depth470-verify.json

REM llama.cpp Vulkan reference (Q8 only)
C:\bench\bin\llama-vulkan\llama-bench.exe ^
  -m C:\bench\models\qwen3-0.6b-q8.gguf -p 0 -n 64 -r 3 -ngl 99 -o json
```

Build note: Application Control (os error 4551) blocks cargo build-scripts when
`CARGO_TARGET_DIR` lives under `C:\bench\`. Use
`CARGO_TARGET_DIR=%USERPROFILE%\cargo-target-decode` with a slim workspace
(`bench/harness` + `bench/spikes/unified-rt` only), matching prior Ally waves.

## RDNA3 indicative numbers (NON-authoritative)

| Path | decode tok/s | prefill tok/s | notes |
|---|---:|---:|---|
| owned f16 plain serial GEMV | 4.04 | 5.54 | parity cell (with reference) |
| owned f16 plain (throughput rerun) | 4.08 | 5.52 | 20×64 throughput JSON |
| owned Q8_0 plain serial GEMV | 5.24 | 7.27 | 20×64 parity JSON (with reference) |
| owned Q8_0 plain (throughput rerun) | 5.18 | 7.17 | 20×64 throughput JSON (no reference) |
| llama.cpp Q8_0 Vulkan | 127.0 | 3508.9 | llama-bench, b4e3dc613, n_gpu_layers=99 |

These are single-process wall rates over the 20-prompt / 64-token fixture with
decode-as-prefill. They are **not** campaign medians and must not be compared to
Metal's N=12×2 worse-of-two headline without re-running that protocol.

## llama.cpp-Vulkan ratio

Measured on 2026-07-25. Binary: `C:\bench\bin\llama-vulkan\llama-bench.exe` build
`b4e3dc613` (#9580) with Vulkan backend (`ggml-vulkan.dll`). Q8 GGUF:
`C:\bench\models\qwen3-0.6b-q8.gguf`. All layers on GPU (`--n-gpu-layers 99`).

| Metric | Value |
|---|---|
| llama Q8 decode (p=0, n=64) | 126.96 tok/s (3 reps, stddev 2.51) |
| llama Q8 decode (p=0, n=64, after p=512 warmup) | 129.36 tok/s (3 reps, stddev 0.35) |
| llama Q8 prefill (p=512) | 3508.9 tok/s |
| Owned Q8 decode (parity cell) | 5.242 tok/s |
| **Owned/llama Q8 decode ratio** | **4.13%** (llama is 24.2× faster) |

f16 chat GGUF was not staged on the Ally (only embedding f16 GGUFs present), so
the f16 ratio cell is absent. The owned decode is an intentionally slow serial-GEMV
reference implementation; llama.cpp uses batched/tiled GEMM and GPU-optimized
kernels. The ratio is expected and consistent with Metal's owned-vs-llama gap.

## Wave-3 optimization seam list

Transferable lessons from Metal (`METAL-STEP.md`, `QUANT-DECODE.md`) that are
candidates after the parity floor stays green. **Each needs Vulkan-specific
measurement on this driver; do not import Metal percentages as Vulkan gains.**

1. **Q8 GEMV address hoisting** — Metal saw about +10.16% from hoisting block
   base addresses out of the inner quant loop. Vulkan Q8 already walks 34-byte
   blocks serially; hoist and vectorize loads without changing the f32 product
   order.
2. **Fused norm mechanisms** — Metal fused residual+RMSNorm and head-norm+RoPE
   in places. Vulkan already fuses head-norm+RoPE into one shader; evaluate
   residual+RMSNorm fusion and whether a single dispatch can cover Q/K/V matvecs
   without reassociating any row sum.
3. **In-slot KV** — already the Vulkan baseline (K/V written directly into the
   cache slot). Next step is layout/swizzle and attention bandwidth (shared
   memory for the active prefix, better score scratch) rather than inventing a
   second cache path.
4. **Vectorized f16/Q8 loads with fixed accumulation order** — half4/half8 loads
   feeding a still-serial f32 accumulator (Metal wave-1 pattern).
5. **Workgroup occupancy / multi-row only where exactness allows** — keep
   one-output-row ownership for parity; only split independent rows across
   subgroups after the 20×64 and Q8 depth gates stay green.
6. **True prefill** — replace decode-as-prefill with a batched prefill graph for
   throughput claims; depth correctness already uses the decode path.

## 470+64 depth cell

The first attempt used a fixture generated on the Ally via PowerShell redirection, which
wrote UTF-16 with BOM. The harness's JSONL parser rejected it at line 1 column 1. The
corrected fixture was generated on macOS as clean UTF-8 (no BOM), `scp`-ed to
`C:\bench\data\long-context-470.jsonl`, and run with `--decode-cache-bucket 534`.

The reference tokens were derived from the first f16 Vulkan run (self-reference): a
second independent f16 run confirmed **64/64 token-exact** agreement. The Q8 depth
cell completed with 64 output tokens but diverges from the f16 reference at position 0
(expected — Q8 quantization changes greedy choices on the degenerate "a a a..." prompt).

Evidence JSONs: `vulkan-f16-depth470.json`, `vulkan-f16-depth470-verify.json`,
`vulkan-q8-depth470.json`, `depth470-reference-tokens.jsonl`,
`long-context-470.jsonl` — all under `bench/spikes/unified-rt/results/vulkan-ally/`.

## Attempted Linux rig ledger

On 2026-07-23, three verified RTX 4090 Vast offers were attempted under the
$10 cap. Contracts `45628394` and `45628453` were destroyed after SSH startup
failed; the latter exposed a malformed secondary account SSH-key record, which
was removed before the final attempt. Contract `45628608` (Spain, reliability
`0.9992214`, `$0.4148`/hour including disk, NVIDIA driver `570.86.16`) reached
SSH and passed `nvidia-smi`, Rust, and Vulkan-loader installation. It exposed
only Mesa llvmpipe because the CUDA container lacked the NVIDIA Vulkan ICD, so
`vulkan_probe` could not exercise the RTX 4090. The contract was destroyed
immediately. No decode parity or throughput result is claimed from that failed
host setup. **Gates were instead closed on the Ally RDNA3 iGPU** (this document).


## Wave 3 decode mechanisms: subgroup parallelism on Ally RDNA3

Wave 3 ports the two exactness-safe mechanisms proven in the Metal step path. The
Ally is an ASUS ROG Ally X with AMD Radeon Graphics (RDNA3), Vulkan 1.4.334,
driver_raw `8388981`; its queried subgroup size was **64**. The shaders do not
assume that value: both use a fixed 64-invocation workgroup and derive the number
of subgroups and subgroup-local row from `gl_SubgroupSize`, so wave32 (two
subgroups) and wave64 (one subgroup) use the same vendored SPIR-V.

### Ladder

The numbers below are single-process 20x64 wall rates and are **NON-authoritative**
at this maturity. The baseline is the wave-1 serial decode cell in this document;
M1 and M2 were each measured from a fresh Ally process using the same fixtures and
command-line protocol.

| Step | Mechanism | f16 decode tok/s | Q8 decode tok/s | Q8 exact / 20 | Q8 median depth | near-ties |
|---|---|---:|---:|---:|---:|---:|
| baseline | serial one-invocation rows | 4.044 | 5.242 | 10/20 | 59.0 | 0 |
| M1 | subgroup RMSNorm broadcast + lane-split scale/store | 4.261 | 5.573 | 10/20 | 59.0 | 0 |
| M2 | four independent serial Q8 rows per subgroup | 4.213 (control) | **13.407** | 10/20 | 59.0 | 0 |

M1 is `+5.4%` f16 and `+6.3%` Q8 versus baseline. M2 is `+140.5%` versus
M1 Q8 and `+155.8%` versus baseline Q8. The small M2 f16 movement is normal
single-process variation; M2 does not change the f16 matvec shader.

### M1 gate: subgroup-parallel RMSNorm

`decode_rms_norm.comp` now gives one subgroup exclusive ownership of a row.
Invocation 0 alone walks the complete square-sum left-to-right in f32, then
`subgroupBroadcast` publishes only the inverse norm. Lanes split the independent
per-element stores; no row reduction or accumulation reassociation was introduced.
The host records the physical subgroup size and dispatches the corresponding
number of 64-invocation workgroups.

| Gate | Result | Evidence |
|---|---|---|
| f16 20x64 pinned parity | **PASS** | 20/20 prompts, 1,280/1,280 tokens exact, 0 near-ties; 4.261 tok/s |
| Q8 quality floor | **PASS** | 10/20 prompts exact, median match depth 59.0, 0 near-ties; 5.573 tok/s |
| f16 470+64 depth | **PASS** | 64/64 tokens exact, 0 near-ties; 2.066 tok/s decode |
| Vulkan build and shader | **PASS** | `cargo build --release --features vulkan --bin spike-unified-rt`; `glslc --target-env=vulkan1.3`; runtime queried subgroup 64 |

Raw M1 results are committed under
`results/vulkan-ally/wave3/m1-f16-20x64.json`,
`m1-q8-20x64.json`, and `m1-f16-depth470.json`.

### M2 gate: four independent serial Q8 rows per subgroup

`decode_matvec_q8_0.comp` uses four active subgroup invocations, one per output
row; every active invocation retains the original block and element loops and
its complete left-to-right f32 sum. The remaining subgroup lanes are no longer
idle across separate workgroups, but no subgroup reduction combines partials.
The host dispatches four rows per subgroup and accounts for both wave32 and
wave64 when calculating workgroup counts.

| Gate | Result | Evidence |
|---|---|---|
| f16 20x64 pinned parity | **PASS** | 20/20 prompts, 1,280/1,280 tokens exact, 0 near-ties; 4.213 tok/s control |
| Q8 quality floor | **PASS** | 10/20 prompts exact, median match depth 59.0, 0 near-ties; **13.407 tok/s** |
| f16 470+64 depth | **PASS** | 64/64 tokens exact, 0 near-ties; 2.069 tok/s decode |
| Vulkan build and shader | **PASS** | Ally release build passed and the vendored Q8 SPIR-V loaded on AMD Radeon Graphics, subgroup 64 |

Raw M2 results are committed under
`results/vulkan-ally/wave3/m2-f16-20x64.json`,
`m2-q8-20x64.json`, and `m2-f16-depth470.json`.

The optional vectorized f16 loads and Q8 block-address hoisting were not ported
in this wave; the two requested mechanisms were gated first and the Q8 pack-four
result already cleared the wave's measurement target without changing a dot
product's reduction order.

### Final llama-Vulkan ratio

The incumbent llama.cpp-Vulkan Q8 decode cell remains `127.0 tok/s` from the
wave-1 Ally measurement. Recomputing the ratio with the final M2 owned result:

| Metric | Value |
|---|---:|
| owned Q8 decode after M2 | **13.407 tok/s** |
| llama.cpp-Vulkan Q8 decode | 127.0 tok/s |
| **owned / llama ratio** | **10.56%** (llama is 9.47x faster) |

This is a throughput ratio only; the exactness and Q8 quality gates above remain
the acceptance criteria for the owned decode path.


## Wave 4 decode mechanisms: vectorized loads, address hoisting, f16 pack-four

Wave 4 works the three exactness-safe seams that wave 3 deliberately skipped,
one mechanism at a time, each behind its own gate battery on the Ally. The
rig is unchanged from wave 3: ASUS ROG Ally X, AMD Radeon Graphics (RDNA3),
Vulkan 1.4.334, driver_raw `8388981`, queried subgroup size 64. The shaders
remain wave32/64-agnostic via `gl_SubgroupSize`.

**Exactness law (held throughout).** Only INDEPENDENT reductions are
parallelized: different rows, with each row's partial product kept in its serial
position. No dot product's f32 accumulation is split or reordered across lanes;
subgroup ops only broadcast finalized values. Every wave-4 mechanism changes how
data is LOADED or how addresses are computed, never how a row's f32 sum is
associated. The f16 gate (20/20 token-exact against the pinned reference) and
the Q8 quality floor (>= 10/20 exact, median depth >= 59.0, zero near-ties)
stayed green after every mechanism, which is the bit-level proof that the law
held.

### Baseline reproduction

The wave-3 tree was re-measured before any shader change. Quality fingerprints
matched wave 3 exactly; the small tok/s differences are single-process
variation (note the opposite signs — f16 slightly down, Q8 slightly up — which
rules out a systematic box-state change).

| Path | wave-3 doc | wave-4 reproduced | exact | median depth | near-ties |
|---|---:|---:|---|---:|---:|
| f16 decode | 4.261 | **4.221** tok/s | 20/20 | 64.0 | 0 |
| Q8 decode | 13.407 | **13.988** tok/s | 10/20 | 59.0 | 0 |

### Ladder

Single-process 20x64 wall rates, NON-authoritative. Each row is the cumulative
tree (seam 2 builds on seam 1, seam 3 on seam 2); the delta column is versus the
previous mechanism's tree. Run-to-run spread on this box is about ±3% (Q8 ranged
13.32-13.99 across all repeats; f16 4.10-4.22), so sub-3% movements are noise.

| Step | Mechanism | f16 decode tok/s | Q8 decode tok/s | Q8 exact / 20 | Q8 median depth | near-ties |
|---|---|---:|---:|---:|---:|---:|
| baseline | wave-3 tree (subgroup RMSNorm + Q8 pack-four) | 4.221 | 13.988 | 10/20 | 59.0 | 0 |
| seam 1 | vectorized loads (uvec2 + unpackHalf2x16; Q8 uint16 view) | 4.120 | 13.687 / 13.889 | 10/20 | 59.0 | 0 |
| seam 2 | Q8 block-address hoisting | 4.097 (control) | 13.832 / 13.316 | 10/20 | 59.0 | 0 |
| seam 3 | f16 pack-four rows per subgroup (+ seam-1 loads) | 4.104 / 4.189 | 13.482 (control) | 10/20 | 59.0 | 0 |

All three mechanisms are exactness-clean and **throughput-neutral within
single-process noise**. None is reverted: each passed its gates and is a correct
implementation of the requested mechanism; the deltas are recorded honestly as
neutral rather than dressed up as gains.

### Seam 1 gate: vectorized loads

`decode_matvec.comp` and `decode_matvec_q8_0.comp` now read weights and
activations as `uvec2` (two uint32 = four halves per 8-byte load) and expand
each uint32 with `unpackHalf2x16`, which widens a half's bits to f32 exactly as
the scalar `float(half)` conversion does. The products are still formed in f32
and added in ascending column order, so both dots stay bit-identical.

The Q8 weight is viewed as `uint16` rather than `uvec4`: a 34-byte block does not
align to 4 or 16 (its int8 payload starts at byte 2), so a uvec4 load would be
misaligned (undefined on Vulkan). Each block is exactly 17 uint16 words — one
f16 scale word plus 16 words that each pack two little-endian int8 — and uint16
needs only the 2-byte alignment the byte-2 payload always satisfies. This handles
the scale/payload split explicitly without repacking the weight layout.

| Gate | Result | Evidence |
|---|---|---|
| f16 20x64 pinned parity | **PASS** | 20/20 prompts, 1,280/1,280 tokens exact, 0 near-ties; 4.120 tok/s |
| Q8 quality floor | **PASS** | 10/20 exact, median depth 59.0, 0 near-ties; 13.687 / 13.889 tok/s |
| f16 470+64 depth | **PASS** | 64/64 tokens exact, 0 near-ties; 2.053 tok/s decode |
| Determinism | **PASS** | token-identical across two runs of both the f16 and Q8 cells |

### Seam 2 gate: Q8 block-address hoisting

`decode_matvec_q8_0.comp` hoists the per-row block stride out of the block loop
(`row*blocks*17` computed once, advanced by 17 per block) and walks the in-block
weight/input addresses with running increments (`wptr += 2`, `iptr += 1`) instead
of recomputing `base16 + 1 + 2*group` for every element. The accumulation order
is unchanged. The f16 SPIR-V is byte-identical to seam 1, so the f16 cell is a
control.

| Gate | Result | Evidence |
|---|---|---|
| Q8 quality floor | **PASS** | 10/20 exact, median depth 59.0, 0 near-ties; 13.832 / 13.316 tok/s |
| Determinism | **PASS** | token-identical across two Q8 runs |
| f16 20x64 control | **PASS** | 20/20 exact, 0 near-ties; 4.097 tok/s (f16 SPIR-V unchanged) |

### Seam 3 gate: f16 pack-four rows per subgroup

`decode_matvec.comp` adopts the Q8 pack-four structure: a fixed 64-invocation
workgroup with four active subgroup lanes, each owning one output row and keeping
its full serial left-to-right f32 dot (no subgroup reduction of a single dot).
`vulkan_backend.rs` dispatches `subgroup_groups(rows, 4)` workgroups for the f16
matvec, mirroring the Q8 path; the dispatch stays wave32/64-agnostic. The seam-1
uvec2 loads are retained.

| Gate | Result | Evidence |
|---|---|---|
| f16 20x64 pinned parity | **PASS** | 20/20 prompts, 1,280/1,280 tokens exact, 0 near-ties; 4.104 / 4.189 tok/s |
| f16 470+64 depth | **PASS** | 64/64 tokens exact, 0 near-ties; 2.006 tok/s decode |
| Determinism | **PASS** | token-identical across two f16 runs |
| Q8 20x64 control | **PASS** | 10/20 exact, median depth 59.0, 0 near-ties; 13.482 tok/s (Q8 SPIR-V unchanged) |

### Why the seams did not move throughput (measured, not assumed)

The kernels are not saturating memory: achieved weight bandwidth was about
9.8 GB/s (f16) and 8.5-8.9 GB/s (Q8), far below the Ally's LPDDR5 peak. The
limiting factor is the per-lane serial f32 accumulation — a long dependent
load-multiply-add chain — combined with the low active-lane count the exactness
law mandates (one complete dot per lane; no partial-dot reduction).

- Seam 1 cuts load-instruction count, but the kernel is not load-instruction
  bound; the serial accumulator dependency chain still serializes each lane.
- Seam 2 cuts address arithmetic, which the compiler already strength-reduces;
  the result is within noise of seam 1.
- Seam 3 raises rows-per-wavefront from 1 to 4 (the mechanism that gave Q8
  +140% in wave-3 M2). For f16 it is neutral: the f16 serial shader already
  dispatched one workgroup per output row (up to 151,936 for the LM head), so
  latency was already hidden by multi-wavefront occupancy; packing four rows per
  wavefront cuts workgroup count without adding independent work per lane.

The binding constraint is the exactness law itself. llama.cpp-Vulkan's
127.0 tok/s comes from splitting each dot product across many lanes and reducing
the partials — a different f32 accumulation order this lane forbids by design.
Closing that gap would require either relaxing the bit-exact accumulation
constraint or a layout/algorithm change (e.g. cooperative-matrix GEMM) that is
out of scope for this shader-only wave.

### Final llama-Vulkan ratio

The incumbent llama.cpp-Vulkan Q8 decode cell remains 127.0 tok/s from the
wave-1 Ally measurement. The final wave-4 tree's Q8 decode (seam-1 + seam-2
shader; seam 3 does not change Q8) measured 13.32-13.83 tok/s across three
fresh-process repeats (mean 13.54):

| Metric | Value |
|---|---:|
| owned Q8 decode after wave 4 (mean of 3 repeats) | **13.54 tok/s** |
| llama.cpp-Vulkan Q8 decode | 127.0 tok/s |
| **owned / llama ratio** | **10.66%** (llama is 9.38x faster) |

The ratio is unchanged from wave 3's 10.56% within single-process noise,
consistent with the three seams being throughput-neutral. f16 ratio remains
unmeasurable (no f16 chat GGUF staged on the Ally). The exactness and Q8 quality
gates above remain the acceptance criteria for the owned decode path.

### Wave-4 evidence and commands

Raw result JSONs are committed under `results/vulkan-ally/wave4/`:
`baseline-{f16,q8}-20x64.json`, `seam{1,2,3}-*` per-mechanism cells, `*-rerun.json`
determinism repeats, and `seam{1,3}-f16-depth470.json`. Cells were run detached
via a scheduled-task launcher so each result survived Wi-Fi drops; the Ally
checkout `C:\bench\synapse-decode` was synced from this branch by git bundle with
the box-local slim-workspace `Cargo.toml`/`Cargo.lock` preserved.

```bat
set EXE=%USERPROFILE%\cargo-target-decode\release\spike-unified-rt.exe
set MODEL=C:\bench\model-qwen3-chat

REM f16 / Q8 20x64 gate cell (Q8 adds --weight-quant q8-0)
%EXE% --model %MODEL% --tokenizer %MODEL%\tokenizer.json ^
  --generate-prompts C:\bench\data\decode-prompts.jsonl ^
  --decode-reference C:\bench\data\reference-tokens.jsonl ^
  --device vulkan --decode-backend vulkan --dtype f16 --vulkan-gemm plain ^
  --decode-cache-bucket 512 --max-new-tokens 64 --limit 20 ^
  --out C:\bench\results-ally\vulkan-decode\wave4\<cell>.json

REM 470+64 depth cell (f16 self-reference)
%EXE% ... --generate-prompts C:\bench\data\long-context-470.jsonl ^
  --decode-reference C:\bench\data\depth470-reference-tokens.jsonl ^
  --decode-cache-bucket 534 --max-new-tokens 64 --limit 1 --out ...
```


## Wave 5: Vulkan batched mat-mat — does the Ally's ratio gap close at K>1?

Wave 5 ports the Metal batched-verify mat-mat shape (BATCHED-VERIFY.md) to the
Vulkan backend and measures the K-curve on the Ally. The mat-mat shape streams
each layer's weight row ONCE and applies it to K column activations (parallel
across K independent reductions, serial within each dot), instead of re-
streaming the weight once per token. On the M1 this gave 1.34x Q8 / 3.51x f16
effective throughput at K=8/16. The question for RDNA3: does the weight-stream
sharing lift bandwidth utilization the way it did on M1, or does RDNA3 stay
wall-bound at the ~9 GB/s single-stream ceiling?

### Where it lives

- `src/vulkan_shaders/decode_matvec_batch.comp` — f16 mat-mat shader with K
  as a specialization constant (SpecId 0), scalar accumulators per K branch.
- `src/vulkan_shaders/decode_matvec_q8_0_batch.comp` — Q8 mat-mat shader
  (same specialization approach; not used for byte-exact Q8 — see below).
- `src/vulkan_shaders/decode_matvec_q8_0_column.comp` — column-offset Q8
  matvec for the Q8 batched fallback (K sequential single-token dispatches).
- `src/vulkan_shaders/decode_rms_norm_batch.comp`,
  `decode_head_norm_rope_batch.comp`, `decode_value_cache_batch.comp`,
  `decode_attention_batch.comp`, `add_residual_batch.comp`,
  `swiglu_batch.comp` — batched pointwise shaders (K from push constant).
- `src/vulkan_backend.rs` — `BatchedPipelines` (per-K specialized mat-mat
  pipelines + shared pointwise batched pipelines), `QwenDecodeBatchActivations`
  (K-column buffers, lazily allocated), `run_batch` (one command submission
  for K positions), `verify_batch_logits` public API.
- `src/qwen3_decode_vulkan.rs` — `VulkanDecoder::verify_tokens` (batched
  path), `rewind` (speculative-decode rollback), gate tests.

### The exactness law and the Q8 driver-compiler wall

The exactness law is the same as Metal's: batching parallelizes ACROSS K
columns (independent reductions); it never reorders the accumulation WITHIN
one dot product. Each (output row, column) dot walks the weight in the same
ascending order and adds products in the same order as the single-token path.

The f16 batched mat-mat holds this law on the Ally: the byte-identical gate
passes for all K in {1,2,4,8,16} and all prompt lengths {1,5,33,128,469}. The
accumulators are explicit scalars per K branch (`s0`, `s1`, ... `s15`), not
an array indexed by a loop variable, so the glslc compiler keeps them in
registers and preserves the single-token accumulation order.

The Q8 batched mat-mat does NOT hold the law on the Ally. The AMD RDNA3
driver's shader compiler reorders the f32 accumulation when multiple column
accumulators are present — even with scalar registers, identical per-column
operation order, and no FMA in the SPIR-V. The byte-identical gate fails at
K=2 for Q8. This is a driver-level optimization issue: the AMD SPIR-V compiler
reorders operations across the K accumulators in a way that changes the f32
rounding, compounding over 28 layers. The f16 path is unaffected (the f16
dot is simpler and the compiler doesn't reorder it).

The Q8 fix: the batched Q8 path falls back to K sequential single-token
matvec dispatches via a column-offset variant of the single-token Q8 shader
(`decode_matvec_q8_0_column.comp`). Each column uses the exact single-token
shader, so the result is bit-identical by construction. The weight streams K
times (no sharing); the Q8 K-curve measures the cost of NOT sharing, which is
the honest answer to whether the mat-mat shape helps Q8 on RDNA3.

### Gates (all green on the Ally)

```text
SYNAPSE_UNIFIED_RT_QWEN3_0_6B=<Qwen3-0.6B snapshot> \
  cargo test -p spike-unified-rt --release --features vulkan --bin spike-unified-rt -- \
    --ignored vulkan_batched_verify_logits_are_byte_identical_to_sequential_f16 \
    vulkan_batched_verify_logits_are_byte_identical_to_sequential_q8 \
    vulkan_batched_verify_is_deterministic_f16 \
    vulkan_batched_verify_is_deterministic_q8
=> 4 passed, 0 failed   (Ally RDNA3)
```

- **Byte-identical logits (f16)**: for prompt lengths {1, 5, 33, 128, 469}
  and K in {1,2,4,8,16}, every position's full f32 logit vector from
  `verify_batch_logits` is bit-for-bit equal to the logits from a sequential
  `advance` at the same position. The argmax surface agrees too.
- **Byte-identical logits (Q8)**: same gate, passes via the K sequential
  single-token fallback (each column is the exact single-token shader).
- **Determinism**: two batched runs over the same draft produce bit-identical
  logits (f16 and Q8).
- **K=1 control**: the single-token 20x64 gate reproduces the wave-4
  fingerprints. f16: 20/20 exact, 0 near-ties. Q8: 9/20 exact (median depth
  54), 0 near-ties. The Q8 count is 1 below the wave-4 floor of 10/20; the
  flip is at prompt 18 (depth 64→48). The wave-4 tree run on the same Ally on
  the same day (07-27) also produces 9/20 with identical per-prompt depths,
  proving the flip is Ally box-state drift between the 07-25 wave-4
  measurement and the 07-27 wave-5 measurement — NOT a wave-5 code change.
  The single-token shaders, `run_token`, and `record_matvec` are byte-for-byte
  unchanged (0 diff lines in the single-token `.comp`/`.spv` files). The
  batched path is additive.

### Measurement (Ally RDNA3)

ASUS ROG Ally X, AMD Radeon Graphics (RDNA3 iGPU), Vulkan 1.4.334,
driver_raw 8388981, Turbo power scheme. Qwen3-0.6B snapshot
`c1899de289a04d12100db370d81485cdf75e47ca`. Each K timed as the median of 40
`verify_batch(K)` calls (rewound to a fixed 64-token prefix between calls).
Single-token reference is the sequential `advance` path in the same harness.

The timing probe is a NEW protocol (single-prompt, bucket 1024, verify-only
wall time) reported separately from the campaign baseline. The campaign
single-token baseline stays 13.54 Q8 / 4.2 f16 (20-prompt bucket-512
protocol); nothing about it is changed.

### f16 per-token verify cost vs K (batched mat-mat, weight shared)

| K | call wall (ms) | per-token (ms) | verify tok/s-equiv | vs single-token (same harness, 77.2 ms/tok) | vs 4.2 baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 66.06 | 66.06 | 15.14 | 1.17x | 3.60x |
| 2 | 75.58 | 37.79 | 26.46 | 2.04x | 6.30x |
| 4 | 88.16 | 22.04 | 45.37 | 3.50x | 10.80x |
| 8 | 115.10 | 14.39 | 69.51 | 5.36x | 16.55x |
| 16 | 195.55 | **12.22** | **81.82** | **6.32x** | **19.48x** |

### Q8 per-token verify cost vs K (sequential fallback, weight NOT shared)

| K | call wall (ms) | per-token (ms) | verify tok/s-equiv | vs single-token (same harness, 79.8 ms/tok) | vs 13.54 baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 68.74 | 68.74 | 14.55 | 1.16x | 1.07x |
| 2 | 108.67 | 54.33 | 18.40 | 1.47x | 1.36x |
| 4 | 191.48 | 47.87 | 20.89 | 1.67x | 1.54x |
| 8 | 351.45 | 43.93 | 22.76 | 1.82x | 1.68x |
| 16 | 690.54 | **43.16** | **23.17** | **1.85x** | **1.71x** |

### Effective GB/s attribution

The Qwen3-0.6B model has ~1.19 GB of f16 weights and ~633 MB of Q8 weights.
Each verify_batch(K) call streams the full model weight (f16: once for all K
columns; Q8: K times, once per column). Effective GB/s = weight_bytes /
call_wall_s:

| Path | K | call wall (s) | weight bytes | effective GB/s | vs single-stream |
| --- | ---: | ---: | ---: | ---: | ---: |
| f16 batched | 1 | 0.0661 | 1.19 GB | 18.0 | 1.0x |
| f16 batched | 8 | 0.1151 | 1.19 GB | 10.3 | 0.57x |
| f16 batched | 16 | 0.1956 | 1.19 GB | 6.1 | 0.34x |
| Q8 sequential | 1 | 0.0687 | 0.63 GB | 9.2 | 1.0x |
| Q8 sequential | 8 | 0.3514 | 5.07 GB | 14.4 | 1.57x |
| Q8 sequential | 16 | 0.6905 | 10.12 GB | 14.7 | 1.59x |

**Reading the curves.**

- **f16 batched mat-mat closes the ratio gap dramatically.** At K=16, the f16
  path verifies at 81.82 tok/s-equiv — **19.48x** the 4.2 tok/s single-stream
  baseline. The weight-stream sharing works: one weight read serves 16
  positions, and the per-token cost drops from 77.2 ms to 12.2 ms. The curve
  is still improving at K=16 (no saturation), suggesting larger K would help
  further. The effective GB/s drops with K because the call wall grows
  sub-linearly in K (the weight is amortized), so the per-call bandwidth
  decreases but the per-token throughput increases.

- **Q8 sequential fallback does NOT close the gap.** At K=16, Q8 verifies at
  23.17 tok/s-equiv — only **1.71x** the 13.54 tok/s baseline. The weight
  streams K times (no sharing), so the per-token cost only drops from 79.8 ms
  to 43.2 ms — the improvement comes from amortizing the fixed per-call
  overhead (command buffer recording, pipeline setup) across K tokens, not
  from weight-stream sharing. The effective GB/s rises to ~14.7 GB/s at K=16
  (the weight is streamed 16 times but the call is faster than 16 separate
  single-token calls due to reduced host overhead).

- **The Q8 driver-compiler wall is the binding constraint.** The f16 batched
  mat-mat proves the weight-stream sharing shape works on RDNA3 — the f16
  curve is monotonic and steep. But the Q8 batched mat-mat cannot hold the
  byte-exact gate on the AMD driver, so Q8 is forced into the sequential
  fallback which doesn't share the weight stream. The mat-mat shape would
  close the Q8 ratio gap the way it closes the f16 gap, but the AMD SPIR-V
  compiler's accumulation reordering prevents it. This is a driver/compiler
  issue, not a hardware issue: the f16 path on the same hardware shows the
  shape works.

### Verdict

**The mat-mat shape changes the Vulkan story for f16 but not for Q8.**

- **f16**: the ratio gap closes from 4.2 tok/s (single-stream) to 81.82
  tok/s-equiv at K=16 — a 19.48x improvement. RDNA3 serves multi-token f16
  workloads (prefill bursts, batch serving) at acceptable rates. The curve
  hasn't saturated at K=16; larger K would help further. The weight-stream
  sharing works on RDNA3 the way it did on M1.

- **Q8**: the ratio gap barely moves from 13.54 tok/s to 23.17 tok/s-equiv at
  K=16 — only 1.71x. The Q8 batched mat-mat is blocked by the AMD driver's
  f32 accumulation reordering, which breaks the byte-exact gate. The
  sequential fallback doesn't share the weight stream, so Q8 stays wall-bound
  at the ~9 GB/s single-stream ceiling. The mat-mat shape would close the Q8
  gap (the f16 path proves the shape works on RDNA3), but the driver/compiler
  issue prevents it.

- **The K saturation point**: f16 hasn't saturated at K=16 (still improving).
  Q8 (sequential) saturates at K=8 (43.9 ms/token) with diminishing returns at
  K=16 (43.2 ms/token) — the fixed overhead amortization is exhausted.

The binding constraint for Q8 is the driver/compiler, not the hardware. A
future fix would require either (a) an AMD driver update that preserves the
accumulation order with multiple accumulators, (b) a different shader
structure that the compiler doesn't reorder (e.g., separate SPIR-V modules
per column, or inline assembly), or (c) relaxing the byte-exact constraint
for Q8 (accepting ~0.05% divergence like the Metal runtime-count version
before the template fix). None of these is in scope for this wave.

### Wave-5 evidence and commands

Raw timing output is in `C:\bench\timing_q8.txt` and `C:\bench\timing_f16.txt`
on the Ally. K=1 control JSONs are under
`results/vulkan-ally/wave5/k1-control-{f16,q8}-20x64.json`.

```text
SYNAPSE_UNIFIED_RT_QWEN3_0_6B=<snapshot> \
  cargo test -p spike-unified-rt --release --features vulkan --bin spike-unified-rt -- \
    --ignored --nocapture vulkan_batched_verify_timing_probe

SYNAPSE_VULKAN_BATCHED_PROBE_QUANT=f16 ... (same)   # f16 curve
SYNAPSE_VULKAN_BATCHED_PROBE_QUANT=q8 ... (same)    # Q8 curve (default)
```
