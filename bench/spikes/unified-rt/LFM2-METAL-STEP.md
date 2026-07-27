# LFM2 on the Metal step path — persona-plane decode port

**Status:** foundational increment landed and gated (scope A). The genuinely new
piece — the short-convolution decode-step kernel and its device-resident
conv-cache model — is implemented, wired into the build, and proven
**bit-identical and deterministic against the `lfm2.rs` CPU reference on real
LFM2-1.2B dimensions** (`hidden = 2048`, `conv_L_cache = 3`). End-to-end hybrid
orchestration, the Q8 path, the 20×64 sha256 fixtures, and the authoritative M1
timing are tracked as precise follow-ups at the bottom of this document.

**Measurement authority note:** timed cells belong on the locked M1
(`[bench-host]`, `[bench-user-home]/bench.lock`). This increment was built
and gated on an M5 Max; per project discipline M5 *timing* is advisory only.
The gate here is an *exactness* gate, not a timing gate: the conv kernel uses
per-dot serial f32 accumulation with Metal fast-math and FMA contraction
disabled, so its bit-identity vs the CPU reference is a structural property
(IEEE-754 f32 multiply/add are correctly rounded on both sides), demonstrated on
M5 and expected to hold on M1. The end-to-end gate (follow-up B) should still be
re-confirmed on M1 before any timing claim.

---

## 1. Why

LFM2-1.2B-class models are the persona/bridge tier (WERNI's dual-brain
fast-brain, voice bridge) and need fast warm local decode. The owned LFM2 Metal
decode currently runs on the deprecated MPSGraph path at a confirmed
`6.345058617091391` tok/s on the M1 — roughly 20–32× slower than llama.cpp's
`130.74` (F16) / `203.65` (Q8_0) tok/s on the same machine and model. Qwen3's
custom Metal step path took that family from MPSGraph-class throughput to
`149.40` tok/s Q8 (~72% of llama-cli) through direct kernels. This port brings
LFM2 onto the same step engine.

## 2. Kernel inventory — reused vs new

The LFM2 step path is deliberately additive: it reuses the Qwen3 campaign's
proven kernels for everything the two families share, and adds exactly one new
kernel for the LFM2-specific short-convolution layer. The Qwen3 kernels live in
`qwen3_decode_metal_step.metal` and are **not** duplicated or mutated here, so
the Qwen3 byte-identity fixtures stay undisturbed.

| Step stage | LFM2 needs | Source | Status |
|---|---|---|---|
| Input RMSNorm (`operator_norm`, `ffn_norm`, `final_norm`) | yes | Qwen3 `metal_step_rmsnorm` / `metal_step_residual_rmsnorm` | reused as-is (proven by Qwen3 campaign) |
| Q/K/V projection matvec (f16) | 6 attn layers | Qwen3 `metal_step_qkv_matvec` (f16 matvec) | reused as-is |
| Q/K/V projection matvec (Q8 pack-4) | 6 attn layers | Qwen3 pack-4-rows Q8 GEMV | reused as-is (col dims are ×4; see §6) |
| QK-norm + RoPE | 6 attn layers | Qwen3 `metal_step_qk_norm_rope` | reused as-is — **LFM2 applies per-head `q_layernorm`/`k_layernorm` before RoPE**, exactly the QK-norm the Qwen3 kernel implements (see §4 note) |
| GQA causal attention (decode, position-parallel) | 6 attn layers | Qwen3 `metal_step_attention` | reused as-is (32 q-heads / 8 kv-heads / head_dim 64) |
| Output projection + residual | all 16 layers | Qwen3 `metal_step_matvec_residual` | reused as-is |
| Gate/up SwiGLU + down projection | all 16 layers | Qwen3 `metal_step_gate_up_swiglu` + `metal_step_down_proj` | reused as-is (`intermediate = 12288`, ×4) |
| LM head + on-GPU argmax | yes (tied embeddings) | Qwen3 `metal_step_lm_head` + `metal_step_argmax_*` + `metal_step_embedding_gather` | reused as-is (`vocab = 65536`, ×4) |
| **Short-convolution decode step + rolling conv-cache** | **10 conv layers** | **NEW: `lfm2_conv_step` in `lfm2_decode_metal_step.metal`** | **new, proven bit-exact + deterministic here** |

**New code in this increment**
- `src/lfm2_decode_metal_step.metal` — the `lfm2_conv_step` kernel.
- `src/lfm2_decode_metal_step.m` — native Metal harness (device/queue/library,
  per-layer device-resident conv-cache buffers, step/read/write ABI).
- `src/lfm2_decode_metal_step.rs` — safe Rust driver + the exactness gates.
- `build.rs` — additive: compiles the new `.m` into the existing ObjC static lib
  and builds a **separate** `lfm2_decode_metal_step.metallib` (the Qwen3 compile
  line is untouched).
- `src/main.rs` — additive `#[cfg(target_os = "macos")] mod lfm2_decode_metal_step;`.

## 3. The short-convolution step kernel and conv-cache model

LFM2's ten convolution layers are causal depthwise short convolutions with a
per-layer rolling cache — the conv-cache analogue of the KV cache. The reference
semantics are taken **directly from `lfm2.rs::decode_conv`** (not from LFM2
papers):

1. `in_proj` maps `hidden → 3·hidden` and splits into `[x, gate, y]`;
   `product = x * y` (a reused matvec kernel produces these; they are the kernel
   inputs here).
2. **Advance the rolling cache** (shape `kernel_size × hidden`, row 0 = oldest):
   `state.copy_within(hidden.., 0)` then `state[(kernel_size-1)*hidden..] = product`.
3. **Depthwise causal conv at the newest position** — the CPU reference
   (`CpuProvider::depthwise_causal_conv1d` evaluated at the final sequence
   position) is, per channel `c`:
   `conv[c] = Σ_{tap=0..kernel_size-1} state[tap*hidden + c] * conv_weight[c*kernel_size + tap]`,
   accumulated **tap-ascending, serial f32**.
4. **Gate:** `out[c] = gate[c] * conv[c]`, fed to `out_proj` (a reused matvec).

`lfm2_conv_step` fuses steps 2–4 into one dispatch. Each Metal thread owns one
channel column of the cache, so the in-place advance and the convolution only
touch that column — there is no cross-thread dependency and the kernel is
deterministic for a fixed grid. The operands and reduction order match
`decode_conv` exactly, which is what makes the result bit-identical.

**Conv-cache model.** Each conv layer owns a `MTLResourceStorageModeShared`
buffer of `kernel_size * hidden` f32, allocated zero-initialised (matching
`Model::empty_decode_cache`) and kept resident for the life of the context. A
step advances it **in place on device** and reads back only the `hidden`-wide
gated output — exactly like the attention step reads back one context row. The
ABI exposes `cache_read`/`cache_write` so a future rewind/rollback can be added
without breaking it; rollback is **not** implemented (no speculative loop on
LFM2 yet) but is not structurally precluded.

### Note on QK-norm (brief vs code)

The task brief said "LFM2 uses no QK-norm". The code disagrees and is
authoritative for token-exactness: `lfm2.rs::full_attention`/`decode_attention`
apply `rms_norm_heads` with `mixer.q_norm` and `mixer.k_norm` (loaded from
`self_attn.q_layernorm` / `k_layernorm`) to Q and K **before** RoPE. This is the
same QK-norm-then-RoPE structure the Qwen3 `metal_step_qk_norm_rope` kernel
implements, so that kernel transfers to LFM2 attention as-is. Any end-to-end
work must follow the code (q/k norm present), not the brief's parenthetical.

## 4. Gate transcripts

Gates run on the M5 Max build host (`cargo test`, debug profile, Metal
developer tools via `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`).
They exercise the kernel at the production dimensions `hidden = 2048`,
`kernel_size = 3` with deterministic pseudo-random activations (no checkpoint
required — the kernel is independent of specific weight values).

```
$ cargo test -p spike-unified-rt lfm2_decode_metal_step
running 3 tests
test lfm2_decode_metal_step::tests::conv_cache_write_round_trips ... ok
test lfm2_decode_metal_step::tests::conv_step_kernel_is_deterministic ... ok
test lfm2_decode_metal_step::tests::conv_step_kernel_is_bit_exact_vs_cpu_reference ... ok
test result: ok. 3 passed; 0 failed; 0 ignored
```

- **`conv_step_kernel_is_bit_exact_vs_cpu_reference`** — 16 consecutive decode
  steps (well past the 3-row window) at `hidden = 2048`: every output element is
  compared by `f32::to_bits` against the `lfm2.rs` CPU path (`CpuProvider`
  `depthwise_causal_conv1d` + the verbatim `decode_conv` cache advance and gate),
  and the final device-resident cache is compared bit-for-bit against the CPU
  rolling state. **All bits identical.**
- **`conv_step_kernel_is_deterministic`** — two independent runs over an
  identical 12-step input stream produce byte-identical output streams and
  caches (the two-runs-identical requirement applied to the conv step).
- **`conv_cache_write_round_trips`** — seeding a window via `cache_write` and
  reading it back round-trips bit-exactly (the rewind primitive).

Regression guard: the existing CPU-reference test
`lfm2::tests::causal_conv_uses_unflipped_cross_correlation_taps` still passes,
and the Qwen3 metallib compile line is unchanged.

### The fast-math / FMA finding (important for all follow-up Metal work)

The first cut of the gate failed at step 1 (step 0 passed). Metal compiles with
fast-math **on** by default, which let the compiler contract
`accumulator += cache[..] * conv_weight[..]` into a fused multiply-add. An FMA
rounds `a*b + c` once; the CPU reference does a correctly-rounded multiply then a
correctly-rounded add — different bits whenever the accumulator is already
non-zero (i.e. from step 1 on; step 0's single non-zero term has a zero addend,
where FMA == plain multiply). The fix is to compile the LFM2 library IEEE-strict:

```
xcrun -sdk macosx metal -std=macos-metal2.3 -fno-fast-math -ffp-contract=off -c src/lfm2_decode_metal_step.metal ...
```

This is applied **only** to the LFM2 metallib in `build.rs`; the Qwen3 compile is
deliberately unchanged (its byte-identity gate is Metal-vs-Metal, so a common
fast-math setting cancels out there). Any follow-up that adds LFM2 matvec/norm
kernels to this library must keep these flags or bit-exactness vs the CPU
reference will silently break.

## 5. Throughput reference cells and persona-plane envelope

No owned-step tok/s number exists yet — the step engine is not assembled
end-to-end (follow-up B), so there is nothing honest to time. The cells below are
the **prior locked M1 measurements** from `LFM2-DECODE-BASELINES.md`
(2026-07-20, `[bench-host]`, Apple M1 Max, `[bench-user-home]/bench.lock`,
12 varied prompts × 2 repeats, 64-token cap, median of 24). They are the targets
the step path must beat, reproduced here for context — **not** freshly measured
by this task.

| LFM2-1.2B stack | Decode tok/s | ms/token (warm) | Cold load | Provenance |
|---|---:|---:|---:|---|
| Owned Metal **MPSGraph** f16 (current) | **6.345** | ~157.6 ms | 1.186 s | campaign baseline `6.345058617091391` |
| llama.cpp Metal F16 (b9580) | **130.74** | ~7.65 ms | 0.516 s | LFM2-DECODE-BASELINES.md |
| llama.cpp Metal Q8_0 (b9580) | **203.65** | ~4.91 ms | 0.515 s | LFM2-DECODE-BASELINES.md |
| Owned Metal **step** f16 | _follow-up B_ | _follow-up B_ | — | not yet assembled |
| Owned Metal **step** Q8 | _follow-up C_ | _follow-up C_ | — | not yet assembled |

**Persona-plane envelope (derived, indicative only).** The Qwen3 step engine hit
`149.40` tok/s Q8 ≈ 72% of llama-cli on Qwen3-0.6B. If the LFM2 port lands in the
same 70%-of-llama band, the persona fast-brain would decode at roughly
`~140–150` tok/s Q8 / `~90` tok/s f16 on the M1, i.e. **~5–7 ms/token warm Q8**
and **~11 ms/token warm f16** — a ~20–25× improvement over the 157 ms MPSGraph
baseline and comfortably inside a real-time voice/bridge budget. This is an
envelope estimate from the Qwen3 result and the llama reference cells, **not a
measurement**; the authoritative number comes from follow-up D on the M1.

## 6. Honest gap analysis

**Proven here**
- The short-convolution decode step (cache advance + depthwise causal conv +
  gate) is bit-identical to `lfm2.rs::decode_conv` and deterministic at the
  production dims, with a device-resident rolling cache that matches the CPU
  state after a multi-step run.
- The conv-cache model (resident rolling window, in-place advance, read/write
  hooks) is in place and does not preclude rewind.
- The build wiring is additive; the Qwen3 path is untouched and still compiles.

**Assumed / not yet proven (each is a follow-up)**
- *End-to-end token-exactness.* The reused Qwen3 kernels are proven for Qwen3's
  shapes; they have not yet been re-proven for LFM2's dims inside a full LFM2
  forward, nor has the conv step been chained with them. LFM2's matmul column
  dims are all multiples of 4 (`hidden 2048`, `intermediate 12288`, `head_dim 64`,
  `kv_width 512`, `q_width 2048`, `in_proj 6144`, `vocab 65536`), so the pack-4
  Q8 GEMV no-tail convention holds and no tail path should be needed — but that
  must be confirmed by the end-to-end gate, not assumed.
- *f16 vs f32 anchor.* This gate proves the conv kernel in f32 against the f32
  CPU reference (the cleanest correctness anchor). The full pipeline runs f16
  activations/weights; matching the CPU reference at f16 requires the reference
  to round at the same points (the existing f16 diagnostic is 17/20 token-exact
  with near-tie forks). The end-to-end gate must settle the f16 policy.
- *Q8 path.* Not exercised here; depends on the end-to-end orchestration.
- *20×64 sha256 fixtures.* Not cut; needs the assembled decoder + the pinned
  twenty-prompt oracle (`LFM2-BACKBONE.md`, `decode-prompts.jsonl`).
- *M1 timing.* Not measured; M5 timing is advisory only.

## 7. Follow-up list (prompt seed)

Each item names its entry points and what is proven vs assumed.

**B — End-to-end f16 hybrid step decoder + token-exactness gate.**
Assemble a `MetalStepDecoder`-equivalent for LFM2 that walks `layer_types`
(`Config::layer_types`, `full_attn_idxs = [2,5,8,10,12,14]`): for each layer run
`operator_norm` → mixer → residual → `ffn_norm` → SwiGLU FFN → residual, then
`final_norm` + tied LM head + argmax. Conv layers call `lfm2_conv_step`
(proven); attention layers reuse the Qwen3 kernels via a new LFM2 native context
(the Qwen3 `Qwen3MetalStepLayerParams` is attention-only and must be generalized
to a per-layer variant that also carries conv weights/cache handles — do **not**
mutate the Qwen3 context). Gate: greedy decode bit-identical to
`lfm2_decode.rs::Decoder` / `Model::decode_token` on the real checkpoint
(`SYNAPSE_UNIFIED_RT_LFM2_1_2B`-style env var → snapshot `933cee00…`), 20 prompts
× 64 tokens, plus a two-runs-identical determinism gate. *Proven:* conv step.
*Assumed:* reused kernels transfer to LFM2 dims; f16 rounding policy.

**C — Q8 path.** Wire `Weight::q8_0` for the LFM2 projections through the reused
pack-4 GEMV kernels; verify the no-tail ×4-column convention on LFM2 dims and add
a tail path only if the gate shows it is needed (do not pad weights). Gate
bit-identical to the Q8 CPU reference. *Assumed:* ×4 convention holds (all LFM2
column dims are ×4).

**D — Authoritative M1 timing.** On `[bench-host]` under
`[bench-user-home]/bench.lock` (mkdir; load < 8; AC), measure single-stream decode
tok/s f16 + Q8 with the same 64-token completion protocol as
`LFM2-DECODE-BASELINES.md` (12 varied prompts × 2 repeats, median of 24, fresh
process per sample). Vary prompt text per iteration to defeat llama slot caching
if a fresh llama reference cell is needed. Report against the `6.345` MPSGraph
baseline and the `130.74`/`203.65` llama cells. *Assumed:* nothing — this is the
measurement that retires the §5 envelope estimate.

**E — 20×64 sha256 fixture cut.** Generate the golden token + hidden fixture from
the assembled decoder (or the CPU oracle of `LFM2-BACKBONE.md`) and pin sha256s
in the campaign registration `lfm2-1.2b-f16-single-stream-decode`. Depends on B.

## 8. Reproduction

```sh
# Build (compiles the new .m and the separate lfm2_decode_metal_step.metallib)
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo build -p spike-unified-rt

# Exactness + determinism gates (no checkpoint needed; real dims, synthetic data)
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cargo test -p spike-unified-rt lfm2_decode_metal_step
```

Reference cells and protocol: `LFM2-DECODE-BASELINES.md`. Qwen3 step engine this
port reuses: `METAL-STEP.md`, `src/qwen3_decode_metal_step.{rs,m,metal}`. LFM2
reference semantics: `src/lfm2.rs` (`decode_conv`, `decode_attention`,
`empty_decode_cache`) and `src/lfm2_decode.rs`.
