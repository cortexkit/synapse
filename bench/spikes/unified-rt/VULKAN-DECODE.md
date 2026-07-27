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
