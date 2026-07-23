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

## Gate status (Ally RDNA3, 2026-07-23)

Rig: ASUS ROG Ally X, AMD Radeon Graphics (RDNA3 iGPU), Vulkan API 1.4.334,
driver_raw 8388981, Turbo power scheme. Checkout `C:\bench\synapse-decode` at
`2b1741e` plus the serial-RMSNorm fix. Chat weights:
`C:\bench\model-qwen3-chat` SHA-256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`
(Qwen3-0.6B snapshot `c1899de2…`). Fixtures verified on-box:
prompts `6f1ee1ce…`, references `b2d11f2a…`.

| Gate | Status | Evidence |
|---|---|---|
| Vulkan feature build | PASS | Ally `cargo build --release --features vulkan --bin spike-unified-rt --bin vulkan_probe` (target dir under `%USERPROFILE%` — WDAC blocks build-scripts under `C:\bench\`) |
| Shader compilation | PASS | `glslc --target-env=vulkan1.3` including `decode_rms_norm.comp` |
| `vulkan_probe` runtime | PASS | `AMD Radeon Graphics`, not llvmpipe; coop-matrix + memory heaps match day-1 |
| f16 20x64 token parity | **PASS** | **20/20 exact**, 1,280/1,280 tokens, **0 near-ties**; decode **4.044 tok/s**, prefill 5.541 tok/s |
| Q8_0 quality ladder | **PASS (floor)** | process exit 0; log line `exact Some(10)/20` meets `>= 10/20`; decode **5.2 tok/s**, prefill 7.3 tok/s. Full JSON (median depth / near-tie count) remained on the Ally after SSH dropped mid-wave — re-pull when the box is reachable to confirm median depth `>= 59.0` and zero near-ties from the result file |
| 470+64 depth cell | PENDING re-run | first attempt failed on a BOM/encoding issue in the long-prompt JSONL writer; Ally SSH became unreachable before the corrected fixture ran |
| f16/Q8 indicative tok/s | **NON-authoritative** | single fresh process each after parity: f16 decode **~4.1 tok/s**, Q8 decode **~5.2 tok/s** (same 20×64 protocol, no reference). Not a campaign N=12×2 median |
| llama.cpp-Vulkan ratio | PENDING | Ally has `C:\bench\bin\llama-vulkan\` (b9580-era) and `qwen3-0.6b-q8.gguf`; no f16 chat GGUF staged. SSH dropped before the reference cells finished |

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
```

Build note: Application Control (os error 4551) blocks cargo build-scripts when
`CARGO_TARGET_DIR` lives under `C:\bench\`. Use
`CARGO_TARGET_DIR=%USERPROFILE%\cargo-target-decode` with a slim workspace
(`bench/harness` + `bench/spikes/unified-rt` only), matching prior Ally waves.

## RDNA3 indicative numbers (NON-authoritative)

| Path | decode tok/s | prefill tok/s | notes |
|---|---:|---:|---|
| owned f16 plain serial GEMV | 4.04 | 5.54 | parity cell (with reference) |
| owned f16 plain (throughput rerun) | ~4.1 | ~5.5 | no reference |
| owned Q8_0 plain serial GEMV | ~5.2 | ~7.3 | parity + throughput reruns |

These are single-process wall rates over the 20-prompt / 64-token fixture with
decode-as-prefill. They are **not** campaign medians and must not be compared to
Metal's N=12×2 worse-of-two headline without re-running that protocol.

## llama.cpp-Vulkan ratio

Not measured in this session. Staged on the Ally from the embedding waves:

- binary: `C:\bench\bin\llama-vulkan\llama-cli.exe` / `llama-server.exe` (campaign
  build ~b9580)
- Q8 GGUF: `C:\bench\models\qwen3-0.6b-q8.gguf`
- f16 chat GGUF: **absent** (only embedding f16 GGUFs were present)

When SSH returns, run the same prompts/protocol and record owned/llama decode
tok/s with binary provenance. Until then the ratio row is intentionally blank.

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
