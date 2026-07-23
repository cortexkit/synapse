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

1. RMSNorm;
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
is at least 534; that depth cell still requires Linux-rig parity validation.

## Gate status

The implementation and SPIR-V were built locally, but this worktree has no
Vulkan loader or Linux 4090. No parity or throughput figure below is inferred
from a different backend.

| Gate | Status | Evidence |
|---|---|---|
| Vulkan feature build | PASS | `cargo build --features vulkan --bin spike-unified-rt --bin vulkan_probe` |
| Shader compilation | PASS | `glslc --target-env=vulkan1.3` compiled the five decode shaders |
| `vulkan_probe` runtime | BLOCKED locally | macOS could not load `libvulkan.dylib` |
| f16 20x64 token parity | PENDING Linux 4090 | frozen fixture checksums retained: prompts `6f1ee1ce…`, references `b2d11f2a…` |
| Q8_0 quality ladder | PENDING Linux 4090 | no threshold or fixture change |
| 470+64 depth parity | PENDING Linux 4090 | decode-prefill path supports the capacity; not measured |
| f16/Q8 indicative tok/s | PENDING Linux 4090 | parity wave, not estimated |

The required Linux commands are:

```sh
cargo build --release --features vulkan --manifest-path bench/spikes/unified-rt/Cargo.toml \
  --bin spike-unified-rt --bin vulkan_probe
./target/release/vulkan_probe
./target/release/spike-unified-rt \
  --model /models/Qwen3-0.6B --tokenizer /models/Qwen3-0.6B/tokenizer.json \
  --generate-prompts bench/campaign/decode-fixtures/decode-prompts.jsonl \
  --decode-reference bench/campaign/decode-fixtures/reference-tokens.jsonl \
  --device vulkan --decode-backend vulkan --dtype f16 --vulkan-gemm plain \
  --decode-cache-bucket 512 --max-new-tokens 64 --limit 20 --out /tmp/vulkan-f16.json
```

Repeat the command with `--weight-quant q8-0` for the standard quantized
profile. Run the 470-token depth prompt with a cache bucket of at least 534,
then record the one-run f16 and Q8 throughput as `NON-authoritative` after
parity is green. A Linux 4090 rig was not rented from this development
worktree, so there is no spend or destruction ledger entry to report.

## Wave 2 plan

1. Run the ordered parity gates on the target NVIDIA Vulkan driver, then repeat
   the same cells on Ally RDNA3 before performance work.
2. Add llama.cpp-Vulkan reference cells with the same prompts, cache bucket,
   and measurement protocol.
3. Profile decode stages. The embedding attribution points to deep GEMM and
   memory placement, but decode is GEMV/bandwidth dominated and must be
   measured separately.
4. Preserve one-output-row ownership while improving memory access: vectorize
   f16/Q8 loads, stage row slices, and tune workgroup occupancy without
   reassociating a row's f32 sum.
5. For Q8_0, keep the 34-byte block stride, fuse dequantization into GEMV, and
   test projection-pair fusion only after the standard Q8 depth gates stay
   green. These are the transferable lessons from `METAL-STEP.md` and
   `QUANT-DECODE.md`.


## Attempted Linux rig ledger

On 2026-07-23, three verified RTX 4090 Vast offers were attempted under the
$10 cap. Contracts `45628394` and `45628453` were destroyed after SSH startup
failed; the latter exposed a malformed secondary account SSH-key record, which
was removed before the final attempt. Contract `45628608` (Spain, reliability
`0.9992214`, `$0.4148`/hour including disk, NVIDIA driver `570.86.16`) reached
SSH and passed `nvidia-smi`, Rust, and Vulkan-loader installation. It exposed
only Mesa llvmpipe because the CUDA container lacked the NVIDIA Vulkan ICD, so
`vulkan_probe` could not exercise the RTX 4090. The contract was destroyed
immediately. No decode parity or throughput result is claimed from this failed
host setup.
