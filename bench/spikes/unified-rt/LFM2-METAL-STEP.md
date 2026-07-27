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

**Real production weights.** A fourth, checkpoint-gated test
(`conv_step_matches_cpu_on_real_lfm2_1_2b_conv_weights`, `#[ignore]` + env var
like the Qwen3 real-model gates) loads the actual LFM2-1.2B snapshot, takes a
real conv layer's depthwise weights, and re-runs the bit-exact comparison against
the CPU reference. It passes, confirming the production weights flow through the
kernel identically (not just synthetic values at real dims):

```
$ SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    conv_step_matches_cpu_on_real -- --ignored
test lfm2_decode_metal_step::tests::conv_step_matches_cpu_on_real_lfm2_1_2b_conv_weights ... ok
test result: ok. 1 passed; 0 failed
```

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
- The same bit-identity holds for the **actual LFM2-1.2B conv weights** loaded
  from the production snapshot (checkpoint-gated test), not only synthetic values.
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


---

## 9. Stage B — f16 oracle settled, fixture pinned, reused kernels wired (engine assembly is the remaining increment)

Stage B's job is the end-to-end hybrid step decoder and its token-exactness gate.
Before any Metal orchestration is written, two questions must be settled because
they decide what "token-exact vs the CPU reference" even means for an f16 engine
and therefore how much new Metal the engine needs. This increment settles both
with evidence on the real checkpoint, pins the fixture the engine will match, and
wires the reused kernels into the LFM2 metallib under the required math
discipline. The hybrid engine assembly itself, the 20×64 Metal-vs-fixture gate,
and the M1 timing remain as the precise continuation specified at the bottom of
this section.

### 9.1 The two precision questions, settled

The `lfm2.rs` CPU reference — the token-exactness contract — runs **f32
activations** with the checkpoint weights loaded as **bf16→f32** (`load_with_quant`
ignores its precision argument; there is no f16 rounding on the CPU path). A Metal
step engine that reuses the Qwen3 kernels differs in two ways:

1. **Weights.** The reused kernels read IEEE **f16** weight bits
   (`encode_f16_bits` = `half::f16::from_f32`), not the bf16→f32 the CPU uses.
2. **Activations.** The Qwen3 step kernels keep inter-layer activations in **f16**
   (their scratch buffers are `uint16_t`), whereas the CPU reference keeps them
   f32.

Either difference could flip a greedy argmax and break token-exactness. The
stage-A note flagged this as the open "f16 vs f32 anchor … the end-to-end gate
must settle the f16 policy," and the prior owned f16 diagnostic was only 17/20
token-exact. Both are now measured directly.

**Probe.** A checkpoint-gated test
(`lfm2_decode_metal_step::tests::f16_weight_rounding_policy_probe`, `#[ignore]` +
`SYNAPSE_UNIFIED_RT_LFM2_1_2B` like the other real-model gates) runs the CPU
reference greedy decode three ways over the pinned twenty-prompt × 64-token set
(`decode-prompts.jsonl`), each with the deterministic one-thread platform gemm:

- **native** — the literal `Model::decode_token` contract (bf16→f32 weights, f32
  activations);
- **f16-weight** — every weight replaced by its `decode_f16_bits(encode_f16_bits)`
  round-trip, activations still f32;
- **f16-activation** — native weights, but the running activation vector rounded
  to f16 at every layer boundary (via a test-only
  `Model::decode_token_f16_activations` that mirrors `decode_embedding`).

```
$ SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    f16_weight_rounding_policy -- --ignored --nocapture
=== LFM2 f16 rounding-policy probe ===
prompts: 20, max_tokens: 64
native cpu decode: 58.1s, f16-weight cpu decode: 57.4s, f16-activation cpu decode: 56.3s
native fixture sha256:         49ee80e8ba5d4940854fdbcd044406f5f3af4d5f6d35456eb247cfd506bd307b
f16-weight fixture sha256:     49ee80e8ba5d4940854fdbcd044406f5f3af4d5f6d35456eb247cfd506bd307b
f16-activation fixture sha256: 49ee80e8ba5d4940854fdbcd044406f5f3af4d5f6d35456eb247cfd506bd307b
POLICY (weights):     f16-weight CPU reference is token-identical to the native CPU reference on 20/20 prompts
POLICY (activations): f16-activation CPU reference is token-identical to the native CPU reference on 20/20 prompts
test result: ok. 1 passed; 0 failed
```

(Timings are the M5 Max build host, debug profile, one-thread CPU decode —
advisory only; they are the cost of generating the fixture, not a decode
benchmark.)

**Finding.** Rounding the weights to f16 changes **zero** greedy tokens across
20×64, and rounding the activations to f16 at every layer boundary also changes
**zero** greedy tokens — all three references produce the byte-identical fixture
(same sha256). Consequences:

- The literal `lfm2.rs` CPU reference **is** a valid 20/20 oracle for the f16
  step engine. No separate f16-weight oracle is needed.
- The f16-activation Qwen3 kernels are valid for a token-exact LFM2 decode; an
  f32-activation rewrite is **not** required. This was the precision ambiguity
  the brief said to stop and ask about; it is resolved empirically, so no
  decision is outstanding.

**Residual caveat (honest).** The activation probe rounds at **layer boundaries**
(after each residual add), which is a lower bound on the rounding the real engine
does — the Qwen3 attention kernels also round **inside** the layer (Q/K/V and the
attention context are f16), and the conv path keeps its internals f32. Boundary
rounding being harmless is strong evidence but not a proof that the finer
in-kernel rounding flips no token; only assembling the engine and running the
Metal-vs-fixture gate certifies that. The gate's pinned target is the fixture
below, so the certification is mechanical once the engine exists.

### 9.2 Pinned fixture

The native CPU-reference greedy tokens for all twenty prompts × 64 tokens are cut
to `fixtures/lfm2-f16-step-reference.jsonl` (one `{"id","tokens"}` row per prompt,
stop-token truncation included — e.g. `completion-10` stops after 1 token,
`completion-12` after 22). The pinned digest the Metal step engine must reproduce
20/20 is:

```
49ee80e8ba5d4940854fdbcd044406f5f3af4d5f6d35456eb247cfd506bd307b
```

It is pinned in `PINNED_DECODE_FIXTURE_SHA256` and asserted (with the 20/20 policy)
by the probe on the full set. Regenerate the file with
`LFM2_F16_FIXTURE_OUT=fixtures/lfm2-f16-step-reference.jsonl`.

### 9.3 Reused kernels wired into the LFM2 metallib (IEEE-strict)

`build.rs` now compiles `qwen3_decode_metal_step.metal` a **second time** with
`-fno-fast-math -ffp-contract=off` and links that air alongside the LFM2 conv
kernel into `lfm2_decode_metal_step.metallib`. The LFM2 metallib therefore exports
both `lfm2_conv_step` and the full reused set (`metal_step_rmsnorm`,
`metal_step_qkv_matvec`, `metal_step_qk_norm_rope`, `metal_step_attention`,
`metal_step_matvec_residual`, `metal_step_residual_rmsnorm`,
`metal_step_gate_up_swiglu`, `metal_step_lm_head`, `metal_step_argmax_*`,
`metal_step_embedding_gather`), all under the IEEE-strict discipline the conv step
needs for bit-exactness vs the CPU reference. Verified the linked metallib exports
both kernel families and the stage-A conv gates still pass.

This is strictly additive: the Qwen3 source file is unmodified and the Qwen3
metallib compile line (default fast-math) is untouched.

### 9.4 Qwen3 unperturbed (gate 3, satisfied by diff scope)

No shared Qwen3 kernel or `.m`/`.rs` surface changed. `git diff --stat` against
the stage-B base touches only `build.rs` (the additive compile line above),
`fixtures/lfm2-f16-step-reference.jsonl` (new), `src/lfm2.rs` (a `#[cfg(test)]`
probe method), and `src/lfm2_decode_metal_step.rs` (tests). The three Qwen3 step
files are byte-identical to base:

```
$ for f in src/qwen3_decode_metal_step.{m,metal,rs}; do
    git diff --quiet <base> HEAD -- "bench/spikes/unified-rt/$f" && echo "UNCHANGED: $f"
  done
UNCHANGED: src/qwen3_decode_metal_step.m
UNCHANGED: src/qwen3_decode_metal_step.metal
UNCHANGED: src/qwen3_decode_metal_step.rs
```

Because nothing shared changed, the Qwen3 fixture battery and the 149.40 tok/s
baseline do not need re-running for this increment.

### 9.5 Continuation — assembling the hybrid engine (follow-up B, de-risked)

The two questions that made follow-up B open are settled; what remains is
mechanical orchestration plus the certification gate. Exact entry points:

- **Native context.** Write `lfm2_decode_metal_step.m`'s hybrid context by
  adapting `qwen3_decode_metal_step.m`'s `encode_*` dispatch helpers (they are
  dimension-parameterized and take LFM2's dims directly: `hidden 2048`,
  `query_heads 32`, `kv_heads 8`, `head_dim 64`, `intermediate 12288`,
  `vocab 65536`, `epsilon 1e-5`). Generalize the per-layer params to a conv/attn
  variant: attention layers carry q/k/v/o weights + q/k norms + KV-cache handles
  (reuse as-is); conv layers carry `in_proj`/`out_proj` weights + the depthwise
  `conv_weight` + a conv-cache handle (the stage-A `lfm2_conv_step`). Do **not**
  mutate the Qwen3 context.
- **Conv layer dispatch.** `operator_norm` → `in_proj` matvec (`hidden→3·hidden`,
  reuse `metal_step_matvec_residual` with `add_residual=0`) → split
  `product[c]=proj[c]*proj[2h+c]`, `gate[c]=proj[h+c]` (one small new kernel) →
  `lfm2_conv_step` (proven) → `out_proj` matvec + residual. Match
  `lfm2.rs::decode_conv` operand order exactly.
- **Attention layer dispatch.** `operator_norm` → `metal_step_qkv_matvec` →
  `metal_step_qk_norm_rope` (LFM2 **does** apply q/k layernorm before RoPE —
  `rope_theta = 1e6`, so generate the rope cos/sin tables with LFM2's theta, not
  Qwen3's) → `metal_step_attention` → `metal_step_matvec_residual` (o_proj).
- **Shared tail (all layers).** `residual_rmsnorm` (ffn_norm) →
  `gate_up_swiglu` → `matvec_residual` (down_proj); then `final_norm` →
  `lm_head` (tied embeddings) → `argmax`.
- **Gate.** Greedy-decode the twenty prompts × 64 tokens, hash the token rows with
  the same `fixture_sha256` ordering, and assert equality with
  `49ee80e8…307b` for 20/20 prompts; plus a two-runs-byte-identical determinism
  gate. This certifies the in-kernel rounding the §9.1 caveat leaves open.

**Proven here:** f16 weight rounding (20/20), f16 boundary-activation rounding
(20/20), the pinned fixture, and the IEEE-strict reused-kernel metallib.
**Assumed / to certify:** that the reused kernels compose correctly at LFM2 dims
inside a full forward, and that the finer in-kernel f16 rounding flips no greedy
token (expected yes from §9.1, certified only by the gate). **Not started:** the
Q8 path (follow-up C) and the authoritative M1 timing (follow-up D) — the latter
has nothing new to time until the engine is assembled.


---

## 10. Stage C — hybrid engine assembled and certified (M1 authority)

Stage C assembles the end-to-end hybrid step decoder specified in §9.5 and
certifies it on the locked M1. The engine is built, gated, and timed; the
genuinely new finding is that the f16 engine's single near-tie fork is
**GPU-architecture-dependent**, which reshapes the exactness gate into a
two-tier form (a machine-independent structural invariant plus an M1-only pinned
signature) and is documented in full below.

### 10.1 What was assembled

`src/lfm2_decode_metal_step.m` gains a hybrid decode-step context (separate from
the stage-A conv-only context and from the Qwen3 context, which is untouched)
that walks all 16 layers on device per §9.5:

- **Conv layer (×10):** `operator_norm` → `in_proj` matvec (`hidden→3·hidden`,
  reused `metal_step_matvec_residual`, `add_residual=0`) → `lfm2_conv_split`
  (new: `product[c]=proj[c]*proj[2h+c]`, `gate[c]=proj[h+c]`, widening f16→f32)
  → `lfm2_conv_step` (stage-A, f32, proven) → `lfm2_conv_f32_to_f16` (new:
  narrows the f32 conv output back to f16) → `out_proj` matvec + residual.
- **Attention layer (×6):** `operator_norm` → `metal_step_qkv_matvec` →
  `metal_step_qk_norm_rope` (q/k layernorm before RoPE, rope tables regenerated
  with LFM2's `rope_theta = 1e6`) → `metal_step_attention` → `o_proj` + residual.
- **Shared tail (all layers):** `residual_rmsnorm` (ffn_norm) →
  `gate_up_swiglu` → `down_proj` + residual; then `final_norm` → tied `lm_head`
  → on-GPU argmax.

Device-resident KV caches (attention) and rolling conv caches (conv), f16
weights, f32 conv internals. The Rust driver (`Lfm2HybridStepEngine`) extracts
the weights, generates per-position rope with LFM2's theta, prefills
token-by-token via the explicit-token verify path, then runs chained greedy
decode with on-GPU argmax. Two new small kernels were added to
`src/lfm2_decode_metal_step.metal` (`lfm2_conv_split`, `lfm2_conv_f32_to_f16`);
the build wiring is unchanged from stage B (the reused Qwen3 kernels are already
linked IEEE-strict into the LFM2 metallib). Diff scope is the three LFM2 step
files only, all additive — the Qwen3 step files are byte-identical to base, so
gate 3 (Qwen3 unperturbed) holds by diff scope and the Qwen3 fixture battery and
`149.40` baseline need no re-run.

### 10.2 The GPU-dependence finding (the substantive result)

The engine matches the f32 CPU-reference oracle to **~0.02–0.03 vocab-wide logit
precision** (measured `max|Δlogit|` across all 65536 logits at every position,
on both machines). Greedy tokens therefore agree with the oracle everywhere
except at near-ties whose CPU top-2 gap falls inside that error band, where the
f16 rounding tips the coin-flip. **Which** near-tie flips depends on the GPU:

| Machine | Role | Byte-exact | Fork | CPU top-2 at the fork | Gap | Engine sha |
|---|---|---|---|---|---:|---|
| M5 Max (build host) | advisory | 19/20 | completion-15 / step 17, engine 523 vs oracle 518 | (518, 523) | 0.000362 | `4356ac40…bd307c` |
| M1 Max (authority) | **primary** | 19/20 | completion-05 / step 8, engine 7693 vs oracle 1827 | (1827, 7693) | 0.007270 | `7e52432f…db688a7` |

The reused kernels' transcendentals — `exp` in the attention softmax and `rsqrt`
in RMSNorm — round differently on different Apple GPUs even compiled IEEE-strict
(`-fno-fast-math` stops reassociation and FMA contraction, not hardware
transcendental rounding). The fixture set contains several sub-band near-ties;
each GPU's rounding flips a different one. This is the same shape as the
documented Qwen3 f16 precedent (`METAL-STEP.md`: the `completion-06` near-tie
drifts on the M5 Metal compiler while the M1 is the fixture authority). The two
families differ only in fixture luck — Qwen3's M1 draw had no sub-band near-tie
(so M1 was 20/20 there); LFM2's draw has more than one, so each machine forks
one — **not** in engine quality. The ~0.03 vocab-wide agreement is the actual
quality statement.

**The oracle (the pinned CPU fixture, `49ee80e8…`) is machine-independent and is
left untouched.** Only the f16 engine's coin-flip resolution is machine-dependent,
and it is bounded by the band invariant below. Re-pinning the oracle to the
engine's own output was explicitly rejected (it would make the engine its own
reference).

### 10.3 The two-tier exactness gate

`hybrid_step_engine_matches_pinned_fixture_within_certified_near_tie` asserts:

1. **Structural invariant (every machine):** at most `MAX_CERTIFIED_FORKS = 2`
   prompts diverge, and each divergence is a **top-2 swap** (the engine's token
   is the oracle's runner-up) whose CPU top-2 logit gap is below
   `NEAR_TIE_BAND = 0.05`. The band is justified by the measured ~0.03 f16 error
   with margin. A real regression — a wrong token at a decisive gap, or many
   forks — cannot hide inside it. A length mismatch is never certified.
2. **Primary gate (M1 authority only, auto-detected by `hw.model MacBookPro18,2`
   with a `SYNAPSE_LFM2_STEP_AUTHORITY=m1` override):** the exact M1 fork
   signature is pinned (completion-05 / step 8 / engine 7693 vs oracle 1827); any
   deviation on the M1 fails.
3. **M5 advisory:** the structural invariant plus determinism, with the observed
   fork printed as a canary note (completion-15 / step 17). No cross-machine
   engine-sha pin — per-machine determinism is the regression guard.

Gate transcripts:

```
$ SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    hybrid_step_engine -- --ignored --nocapture        # M5 build host (advisory)
[metal] DIVERGENCE completion-15: first diff at step 17: engine 523 vs oracle 518
[metal] fork completion-15 step 17: CPU top-2 = (518, 523), gap 0.000362 (band 0.05)
[metal] advisory (non-M1): 1 fork(s) within band; M5 canary reference is completion-15 step 17
determinism: two runs byte-identical, sha 4356ac40ae5b1d30094899afcd2e8d9864570c601133bee5d30dcb1e0b60f30c
test result: ok. 2 passed; 0 failed

$ # locked M1 Max ([bench-host], [bench-user-home]/bench.lock, load 1.02)
[metal] DIVERGENCE completion-05: first diff at step 8: engine 7693 vs oracle 1827
[metal] fork completion-05 step 8: CPU top-2 = (1827, 7693), gap 0.007270 (band 0.05)
[metal] M1 AUTHORITY: pinned fork signature confirmed (completion-05 step 8, engine 7693 vs oracle 1827)
determinism: two runs byte-identical, sha 7e52432f7cea385e21298cf8f9cc4e5ec8ddb7098f960a79a8fd8436adb688a7
test result: ok. 2 passed; 0 failed
```

The stage-A conv gates still pass (regression clean). A per-position bisection
probe (`hybrid_step_localize_divergence`) feeds both the engine and the CPU
reference the same token stream and prints the per-position top-3 logits and
`max|Δlogit|`, which is how the fork was localized and certified as a near-tie.

### 10.4 M1 timed f16 cell (authority)

Locked M1 Max (`[bench-host]`, `MacBookPro18,2`), exclusive
`[bench-user-home]/bench.lock` held, AC, one-minute load 1.02. Release build
(`cargo test --release`), `hybrid_step_timing_probe`: prefill untimed, then one
chained 64-token greedy decode per prompt, median of 20 prompts × 2 repeats (40
samples) plus an uncounted warmup. The spread is tight (49.84–50.80, <2%), so the
within-process median is stable; the baselines' fresh-process protocol would not
move it materially.

| LFM2-1.2B stack | Decode tok/s | ms/token (warm) | vs owned MPSGraph | vs llama F16 | Provenance |
|---|---:|---:|---:|---:|---|
| **Owned Metal step f16 (this work)** | **50.09** | **19.96** | **7.9×** | **38.3%** | M1 authority, 40 samples, min 49.84 / max 50.80 |
| Owned Metal MPSGraph f16 (baseline) | 6.345 | ~157.6 | 1.0× | 4.9% | LFM2-DECODE-BASELINES.md `6.345058617091391` |
| llama.cpp Metal F16 (b9580) | 130.74 | ~7.65 | 20.6× | 100% | LFM2-DECODE-BASELINES.md |
| llama.cpp Metal Q8_0 (b9580) | 203.65 | ~4.91 | 32.1× | 156% | LFM2-DECODE-BASELINES.md |

### 10.5 Gap analysis vs llama

The f16 step engine clears the owned MPSGraph baseline by **7.9×** (157.6 →
19.96 ms/token) — a real-time-grade improvement for the persona fast-brain — but
lands at **~38% of llama.cpp F16**, below the ~70%-of-llama band the Qwen3 step
engine achieved (`149.40` Q8 ≈ 72% of llama-cli on Qwen3-0.6B) and therefore
below the §5 envelope estimate (~90 tok/s f16). Likely contributors:

- **The f32 conv path.** Ten of sixteen layers run the short-conv step in f32
  (the stage-A exactness contract) with two extra widening/narrowing kernels and
  a one-thread-per-channel serial reduction; llama's LFM2 kernel set is f16/fused
  throughout. This is the structural cost of keeping the conv internals f32 to
  stay bit-faithful to the CPU reference.
- **Model size / bandwidth.** LFM2-1.2B is ~2× Qwen3-0.6B; the decode step is
  memory-bandwidth-bound, so the same reused kernels move proportionally more
  weight per token.
- **No fusion yet.** The campaign steering seed (LFM2-DECODE-BASELINES.md) names
  fused residual+RMSNorm as the most transferable Qwen3 win; it is not applied
  here.

### 10.6 Follow-up seed

- **Q8 path (follow-up C, unchanged):** wire `Weight::q8_0` for the LFM2
  projections through the reused pack-4 GEMV kernels (all LFM2 column dims are
  ×4, so the no-tail convention should hold); gate bit-identical to the Q8 CPU
  reference, then re-time on the M1. Q8 is the lever most likely to close the
  llama gap (llama's own F16→Q8 jump is 130.74 → 203.65).
- **Fused residual+RMSNorm** and an **f16 (or warp-per-key) attention/conv
  redesign** are the structural levers for the f16 gap; both are new-kernel work,
  not mechanical transfers.
- **Reducing the f16 logit error band** (the attention softmax `exp`/`rsqrt` is
  the dominant source) would shrink the set of flippable near-ties; banked as a
  future seam, not this task — the band invariant already bounds it.

### 10.7 Reproduction

```sh
# Build (M5 build host or M1; compiles the .m and the LFM2 metallib)
DEVELOPER_DIR=/Applications/Xcode.app/Contents/developer cargo build -p spike-unified-rt

# Stage-A conv gates (no checkpoint)
DEVELOPER_DIR=... cargo test -p spike-unified-rt lfm2_decode_metal_step

# Two-tier exactness + determinism gates (checkpoint required; M1 = authority)
SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    hybrid_step_engine -- --ignored --nocapture

# M1 timed f16 cell (release)
SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test --release -p spike-unified-rt \
    hybrid_step_timing_probe -- --ignored --nocapture

# Per-position bisection probe (default completion-15; LFM2_PROBE_PROMPT overrides)
SYNAPSE_UNIFIED_RT_LFM2_1_2B=<snapshot> cargo test -p spike-unified-rt \
    hybrid_step_localize_divergence -- --ignored --nocapture
```
