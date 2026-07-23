# Metal custom decode step

Status: **wave 6 GPU-chained multi-token decode remains a token-exact banked correctness win below its 3% shipping bar; campaign 4b now ships three order-preserving Q8 winners at 103.6292 tok/s on the locked M1, with the Q8 baseline re-pinned accordingly**.
The default Qwen3 decode backend remains `mpsgraph`. Select the custom path with
`--decode-backend metal-step`. The chained span is `SYNAPSE_METAL_STEP_CHAIN_K`
(default 1 = the fully instrumented per-token path, byte-identical to the
pinned baseline; chaining is opt-in because its M1 win is sub-bar while faster
machines benefit substantially — the span is a per-machine serving decision).

## Scope and handoff

This path is limited to Qwen3 causal decode in `bench/spikes/unified-rt`.
Embedding/prefill remains on the existing MPSGraph graph. At the first decode
step, the MPSGraph context exports each layer's `[kv_heads, bucket, head_dim]`
key and value arrays as f16 bits. The custom context imports those arrays once
with a blit into private, persistent Metal buffers. Every later token uses only
our command buffer and Metal functions; no MPSGraph executable is called by the
step path.

The custom context keeps activations, logits, weights, and KV handles alive
across calls. Dense weights are uploaded once through a shared staging buffer
into private Metal buffers; the step path never reads model-owned host memory.
It submits one command buffer per token, waits once, and reads the shared logits
buffer for the existing CPU greedy argmax and hook loop. Cache inspection uses
the same layer/concatenated-KV interface as the MPSGraph path, so pause, splice,
token taps, and addressable weight regions remain host-side protocols rather
than backend-specific behavior.

## Wave 3 kernel attribution

The step has an opt-in attribution mode controlled by
`SYNAPSE_METAL_STEP_PROFILE=1`. In that mode each kernel invocation is encoded
in its own command buffer, waited to completion, and attributed from
`GPUStartTime`/`GPUEndTime`. This is deliberately a profiling mode, not a
throughput mode: the extra command-buffer synchronization makes its wall time
invalid for the headline cells, while the GPU spans identify where the normal
single-command-buffer step spends time.

The decode result JSON exposes aggregate seconds in
`decode_stages.kernel_gpu` (and `prefill_stages.kernel_gpu` remains zero for the
custom step). The fields are `rmsnorm_s`, `qkv_matvec_s`, `qk_norm_rope_s`,
`attention_s`, `o_proj_s`, `residual_rmsnorm_s`, `down_proj_s`,
`gate_up_swiglu_s`, and `lm_head_s`; `samples` counts command buffers with
valid GPU timestamps. Divide each field by `decode_stages.step_calls` and
multiply by 1,000 for the per-token attribution table. A full step has 226
profiled command buffers (eight layer kernels x 28 layers plus final norm and
LM head), so `samples` should be close to `226 * step_calls`.

Example, with a short single-prompt profile:

```text
SYNAPSE_METAL_STEP_PROFILE=1 \\
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \\
  target/release/spike-unified-rt ... --decode-backend metal-step --limit 1
```

Keep this profile separate from the locked-M1 timing cells. It is also suitable
for a synthetic 512-position prompt: compare the attention field at a short
prompt and at position 511 to make the context-growth curve explicit before
choosing an attention or matvec optimization.

## Kernel list

The build script compiles `src/qwen3_decode_metal_step.metal` with `xcrun metal`
and links the resulting `qwen3_decode_metal_step.metallib` beside the Cargo
executable. Runtime loading first resolves that executable-relative file, with
the build output as a development fallback; the timed step never compiles
Metal source.

The per-token command buffer encodes these kernels for each of the 28 layers:

1. `metal_step_rmsnorm` — pre-attention RMSNorm; one simdgroup owns the short vector and uses lane-strided reduction.
2. `metal_step_qkv_matvec` — one grid for Q, K, and V. V writes directly to
   the layer's `kv_heads * bucket * head_dim` cache slot.
3. `metal_step_qk_norm_rope` — Q/K head RMSNorm and Qwen rotary embedding;
   normalized K writes directly into the addressed KV cache slot.
4. `metal_step_attention` — stable two-pass softmax and P/V accumulation
   over the resident cache. One 32-wide simdgroup owns each query head: lanes
   split independent KV positions while keeping each QK dot serial, lane zero
   performs the reference-order softmax reductions, and all lanes cooperate
   over the independent value dimensions.
5. `metal_step_matvec_residual` — O projection.
6. `metal_step_residual_rmsnorm` — fused attention residual add and pre-MLP RMSNorm.
7. `metal_step_gate_up_swiglu` — fused gate/up matvec and SiLU product.
8. `metal_step_matvec_residual` — down projection plus MLP residual.
9. A final `metal_step_rmsnorm` and `metal_step_lm_head` produce f32 logits.

The same matvec kernels select either resident f16 weights or GGUF-compatible
Q8_0 blocks. F16 rows use one lane per output row, with two adjacent half4
loads unrolled while the f32 accumulator remains serial. Q8_0 uses the existing
34-byte layout: little-endian f16 scale followed by 32 signed int8 values. Q8_0
rows use one simdgroup per output row; eight lanes each issue an aligned char4
load for a disjoint slice of every block, and four block iterations are
unrolled. The Q8 lane grouping and reduction are intentionally subject to the
existing depth gate. Dequantization occurs inside the dot product and no
dequantized matrix is materialized.

## Correctness gate transcripts

The Xcode toolchain was explicitly selected after the CLT-only build failed:

```text
$ DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun -sdk macosx --find metal
/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.yr6fBk/Metal.xctoolchain/usr/bin/metal
$ DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo build -p spike-unified-rt --release --bin spike-unified-rt
Finished `release` profile [optimized]
$ stat target/release/qwen3_decode_metal_step.metallib
40914 bytes; no cargo:warning emitted
```

The ordered model and hook gates are green:

```text
Gate 1 f16:
  20/20 exact prompts, 1,280/1,280 tokens, 0 near-tie exemptions
Gate 2 Q8_0:
  15/20 exact prompts, median match depth 64.0, 0 near-tie exemptions
  quantized_weight_sha256=4c774c188ec089ac1e9b30e9797b364993d5c2445d3e479d5d1b94f6d10969d0
Gate 3 hooks:
  cargo test -p spike-unified-rt qwen3_decode — 8 passed, 0 failed
  token tap, pause/resume, splice, addressable regions, and deterministic ties passed
Gate 4 prefill handoff:
  all-MPSGraph and metal-step outputs equal for completion-01, 64/64 tokens
```

Wave 2 refresh (Xcode Metal toolchain, local M5):

```text
$ DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun -sdk macosx --find metal
/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.yr6fBk/Metal.xctoolchain/usr/bin/metal
$ DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer cargo build -p spike-unified-rt --locked --release --bin spike-unified-rt
Finished `release` profile [optimized]; no cargo:warning emitted
$ stat target/release/qwen3_decode_metal_step.metallib
48866 bytes; loaded executable-relative beside spike-unified-rt
Gate 1 f16: 20/20 exact prompts, 1,280/1,280 tokens, 0 near-tie exemptions
Gate 2 Q8_0: 14/20 exact prompts, median match depth 64.0, 0 near-tie exemptions
  quantized_weight_sha256=4c774c188ec089ac1e9b30e9797b364993d5c2445d3e479d5d1b94f6d10969d0
Gate 3 hooks: cargo test -p spike-unified-rt qwen3_decode — 7 passed, 0 failed
  token tap, pause/resume, splice, addressable regions, ties, constraints, and Q8 regions passed
Gate 4 prefill handoff: included in the 20-prompt metal-step f16 run; no mismatch
```

The parity failure was localized by dumping every layer's new cache slot: all
pre-existing slots were byte-identical, while the new slot differed only in the
K/V values produced by the broken state ping-pong. The down projection wrote into
the old current buffer, but the host loop then advanced `current` to the O
projection buffer, dropping the MLP output before layer 1. Removing that swap
made the current-state buffer persistent and produced the gate results above.

## Measurement table

Timed on `[bench-host]` (M1 Max), AC power, while holding and then promptly releasing `[bench-user-home]/bench.lock`. Each cell used 12 distinct prompts
from the fixed stride-seven schedule, repeated twice (24 fresh processes total).
The metallib was transferred beside the executable and loaded executable-relative.
The fresh competitor controls used the M1-native `llama-cli` at
`[bench-user-home]/bench-tools/llama-b9580/llama-cli`, built from llama.cpp tag `b9580`
(commit `b4e3dc613baa92a3884d4151e3d631395c81934a`) with Xcode/AppleClang 21,
CMake 4.4.0, and `GGML_METAL=ON`; the installed `llama-cli` SHA-256 is `02590612ba30c89133d656b7c1300028f345ec6c1cb879fb8f750a3626c02491`. The official Q8_0 GGUF came from
`Qwen/Qwen3-0.6B-GGUF` snapshot `23749fefcc72300e3a2ad315e1317431b06b590a`
(SHA-256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`).
The f16 GGUF was converted on the M1 from the cached
`Qwen/Qwen3-0.6B` snapshot `c1899de289a04d12100db370d81485cdf75e47ca` with
that tag's `convert_hf_to_gguf.py --outtype f16` (SHA-256
`c81c7c27b35225376a52387800c5eca0748a93b46db885a1dbad370a318f55bb`).

The llama command used `-n 64 -ngl 99 -c 512 --single-turn --temp 0
--top-k 1 --top-p 1 --seed 0 --simple-io --no-display-prompt`; its default
single sequence (`-np 1`) provided the single-stream cell. `--single-turn` was
needed because this Qwen GGUF carries a chat template and llama-cli otherwise
auto-enters its interactive loop. Admission was AC Power, no `Runner.Worker`,
no pre-existing bench lock, and 1-minute load averages from 1.09 to 1.21.

| Backend | Weights | Prompts / repeats | Decode tok/s | Encode / GPU / host per token | Status |
| --- | --- | --- | ---: | --- | --- |
| MPSGraph reference | f16 | 12 x 2 | 84.32 baseline | pending fresh breakdown | prior baseline |
| Metal step winners 5+6 | f16 | 12 x 2 | **48.6506 median** | timed on locked M1; 20/20 exact | campaign-2 confirmation |
| Metal step winners 5+6 | Q8_0 | 12 x 2 | **85.2589 median** | timed on locked M1; 13/20 exact; median depth 64.0 | campaign-2 control for campaign 4b |
| Metal step campaign 4b ABC | f16 | 12 x 2 | **51.2478 median** | locked M1; 20/20 exact; 470-token parity 64/64 | **shipped three-winner tree** |
| Metal step campaign 4b ABC | Q8_0 | 12 x 2 | **103.6292 median** | locked M1; 13/20 exact; median depth 64.0; zero near-ties; hooks green | **new Q8 baseline** |
| Metal step baseline | f16 | 12 x 2 | **5.8080 median** | 0.1025 ms / 171.8693 ms / 0.0146 ms | parity-first baseline |
| Metal step baseline | Q8_0 | 12 x 2 | **5.0405 median** | 0.1021 ms / 198.0879 ms / 0.0141 ms | parity-first baseline |
| Metal step wave 1 | f16 | 12 x 2 | **17.7992 median** | 0.1021 ms / 55.8754 ms / 0.0340 ms | 20/20 exact; AC |
| Metal step wave 1 | Q8_0 | 12 x 2 | **29.2656 median** | 0.1018 ms / 33.8624 ms / 0.0327 ms | 14/20 exact; median depth 64.0; AC |
| Metal step wave 2 | f16 | 12 x 2 | **32.3296 median** | 0.1022 ms / 30.6257 ms / 0.0324 ms | 20/20 exact; AC; +81.6% vs wave 1 |
| Metal step wave 2 | Q8_0 | 12 x 2 | **49.9470 median** | 0.1022 ms / 19.7155 ms / 0.0328 ms | 14/20 exact; median depth 64.0; AC; +70.6% vs wave 1 |
| Metal step wave 3 | f16 | 12 x 2 | **42.3634 median** | 0.1020 ms / 23.2997 ms / 0.0323 ms | 20/20 exact local gate; AC; +31.1% vs wave 2 |
| Metal step wave 3 | Q8_0 | 12 x 2 | **67.6920 median** | 0.1023 ms / 14.4657 ms / 0.0332 ms | 11/20 exact local gate; median depth 64.0; AC; +35.5% vs wave 2 |
| llama.cpp Metal | Q8_0 | 12 x 2 | **207.40 median (200.40–208.20)** | — | fresh `llama-cli` b9580; AC |

The fresh llama.cpp Q8_0 repeat medians were **207.45 tok/s** (spread
201.60–208.20) and **207.40 tok/s** (spread 200.40–207.80); the combined
24-sample median was **207.40 tok/s**, with a 200.40–208.20 per-cell range.
The owned-step host column excludes GPU command-buffer wait, logits readback,
and sampler time; median readback was 0.033 ms/token and median sampling was
0.158 ms/token. A same-machine f16 llama-cli row is recorded in
`DECODE-WAVE1.md`.

The fresh locked-M1 Q8 control was **74.2507 tok/s**; the round-1 winner
measured **81.3764 tok/s**, and winners 5+6 measured **85.2589 tok/s** under the
same 12-prompt, two-fresh-process, worse-of-two protocol. The campaign
projection from its 75.0938 control and reported +9.54%/+4.90% gains was
86.3 tok/s; the confirmed run was 85.2589 tok/s because this fresh control was
1.1% below the historical control, still inside the 2% control-drift gate.
Campaign 4b subsequently used that 85.2589 winners-5+6 tree as its pinned
control and shipped a separately gated ABC tree at 103.6292 tok/s; the newer
baseline is recorded in the campaign-4b section below.

## Campaign 2 winners 5+6

Campaign `[consult-id]` promoted two attention
micro-optimizations in order:

1. **Winner 5 — attention value-phase softmax reuse** (proposal
   `proposal_c84e5170d4a944db966218f9428c100c9c9fdf62fb9b18aff403c7a47f8ff8c6`,
   Claude, round 1). For Qwen3's `head_dim=128`, each lane owns four value
   dimensions. The specialized path computes the half-rounded softmax
   probability once per KV position and reuses it across four serial f32
   accumulators; non-128 dimensions keep the original fallback. The campaign
   measured **+9.54%** over its control; the fresh M1 confirmation was 81.3764
   tok/s versus 74.2507 tok/s for the control.
2. **Winner 6 — half4 QK vector loads with serial f32 accumulation** (proposal
   `proposal_c0900e5f37c1360274a8ed5a6b304374f2910d7d4f550480c9edb03cd0b57c70`,
   Kimi, round 3). The score phase loads four adjacent half values at once but
   performs the same four half-to-f32 products and ascending serial additions,
   preserving the exactness contract. The campaign measured **+4.90%** on the
   round-1 tree. The combined M1 tree passed f16 20/20, Q8 13/20 with median
   depth 64.0 and zero near-ties, all hooks, and 470-token/64-token long-context
   parity.

The attention-micro seam is now mined out for this campaign. Four exactness-
gated follow-ups are banked negatives rather than retained changes: query-head
threadgroup caching (+0.45%) and paired QK-position query reuse (-0.42%) did
not clear the promotion bar; pairwise KV-position value-loop unrolling
(+1.43%) remained below the 3% objective; and the softmax/occupancy sub-gates
(max reduction +0.75%, pairwise denominator exp scheduling +0.81%, and the
smaller attention threadgroup +1.20%) were likewise below promotion value.
These are follow-ups to revisit only with a new gate hypothesis, not claims of available throughput.

## Campaign 4b: three composition-honest winners

Campaign `[consult-id]` closed
`consecutive_dry_rounds` with three promoted winners. Its control was the
campaign-2 winners-5+6 tree at **85.2589 tok/s Q8_0**, and all three promotion
patches were pulled from the campaign store by proposal ID and applied in
promotion order. The M1 authority run used AC power, an exclusive
`[bench-user-home]/bench.lock`, no `Runner.Worker`, 12 stride-seven prompts, two fresh
processes per prompt, and the worse repeat per prompt.

### Promoted mechanisms and provenance

1. **R1a — order-preserving QK-norm half4 sum-of-squares** (Claude,
   proposal `proposal_ccfb167d9893ec22ebddf1948abb35e78b76669246d6b47130145e78575c8202`,
   patch digest `944bfd19e6f435676f46d11064a853c76a0b02067ad7fdaf53d79afd3754b981`).
   The query and key norm sums load adjacent four-value groups, cast each
   component to f32, and add the four products in the original ascending
   order. The campaign estimate was **+4.31%**, measured in parallel against
   the campaign control.
2. **R1b — parallel QK-norm output with serial-order reductions** (Sol,
   proposal `proposal_1241479e1962747279de00ffd0c70abca3b763d751a435ebf17356d766bace67`,
   patch digest `28bf79d7af6e0614caf9bfb4091b00dd08958884acff3e84aea73c752e8c286f`).
   One simdgroup owns each independent Q/K head; lane zero keeps the exact
   ascending f32 norm reduction while the other lanes split the independent
   normalize, RoPE, and cache writes. The campaign estimate was **+4.80%**,
   measured in parallel against the campaign control.
3. **R2 — Q8 block-address hoisting in GEMV inner loops** (Sol,
   proposal `proposal_fffabfdfe1867b0d660d2f79bca2ea842af5dfebc41568f68f97eeacc202114d`,
   patch digest `b1cc2024cff09e111c7bb39e9a390290b4321cbfc4c541016d4105e21d903c60`).
   Each Q8 row now computes its row base once and walks input and weight block
   pointers through the unrolled dot-product loop instead of regenerating the
   row/block address for every chunk. Its campaign estimate was **+10.16%**,
   measured against the promoted R1 tree.

R1a and R1b touch the same QK-norm seam, so their estimates are not multiplied
or treated as independent additive evidence. The measured AB tree composes both
mechanisms: R1b supplies the head-parallel dispatch and lane split, while R1a's
half4 sum-of-squares loop remains inside lane zero. R2 is then applied to that
AB tree, as the campaign protocol requires.

### Locked-M1 composition table

All measured trees passed the complete battery: f16 **20/20** exact against the
pinned oracle and MPSGraph, Q8 **13/20** exact with median match depth **64.0**
and zero near-ties, all five hook tests, and 470-token prefill parity **64/64**.
The Q8 medians below are the worse-of-two fold over 12 fresh processes; the f16
cells use the same 12 x 2 protocol and are recorded for comparison.

| Tree | f16 tok/s | Q8_0 tok/s | Q8 vs fresh control | Decision |
| --- | ---: | ---: | ---: | --- |
| CONTROL (master + winners 5+6) | 48.6392 | **85.2813** | —; +0.03% vs pinned 85.2589 | fresh control; drift inside 2% |
| A-only (R1a) | not measured | not measured | — | AB composed well above the single-winner branch threshold |
| B-only (R1b) | not measured | not measured | — | AB composed well above the single-winner branch threshold |
| AB (R1a + R1b) | **51.2484** | **93.5819** | **+9.73%** | passed; retained as the R2 base |
| ABC (AB + R2) | **51.2478** | **103.6292** | **+21.51%** | **shipped; new Q8 baseline** |
| llama.cpp Metal reference | — | **207.40** | ABC is **49.97%** of the fresh reference | b9580, AC, same 12 x 2 protocol |

AB's +9.73% gain is more than one percentage point above the better R1 single
estimate (+4.80%), so the protocol's A-only/B-only branch was not triggered.
R2 adds **+10.74%** over AB in the integrated tree. The fresh control was
**85.2813 tok/s**, below the 2% drift stop rail; the ABC result is therefore an
authority-machine measurement rather than a compound estimate.

The campaign's parity-failed R3 proposal — threadgroup-sharing Q8 activations
across the eight packed rows (proposal
`proposal_85983b7ef8c60b126a8e57b1eea435a38bb497edf624ae9efe8f6dd736473ebd`) —
is a banked negative: it changed the f16 parity stream and was not timed. The
R1 dispatch-shape proposal (proposal
`proposal_80e2cdb313d47112592ec5ee92cc2c6cd0f7cbc3e2496a989e5698e92624c4b2`)
was also banked negative: its full-simdgroup dispatch measured **85.9164**
against an **86.1638** control (**-0.29%**) and did not promote. These are
negative evidence, not omitted wins.

R2 opens the address-arithmetic seam for the next measured pass. The obvious
follow-up is LM-head stride hoisting: apply the same row-base/pointer-walk
analysis to the 151k-row LM head, with its own exactness and depth gates rather
than assuming the GEMV result transfers.

## Wave 6 GPU-chained multi-token decode

Wave 6 removes the per-token host round trip. Before this wave every decode token
cost one command buffer, one `waitUntilCompleted`, and a 604KB fp32 logits
readback (151,936 vocab x 4 bytes) before the host argmax could pick the next
input token. On the M1 that serialized wait plus readback is a first-order slice
of the ~11.7ms/token budget; the CUDA twin measured the equivalent readback
immaterial at 1.9ms/token, but Metal's serialized-per-token shape makes it
material here.

### Mechanism

1. **On-GPU argmax** (`metal_step_argmax_partial` + `metal_step_argmax_final` in
   `qwen3_decode_metal_step.metal`): a two-stage reduction over the 151,936 vocab
   logits writes one int32 token id. Each float logit is mapped to an unsigned
   radix key via the canonical IEEE-754 total-order transform (set the top bit of
   non-negatives, flip all bits of negatives), so a plain `>` reproduces the host
   sampler's `f32::total_cmp` order exactly, including the sign of zero. Ties
   resolve to the LOWEST token id, byte-identical to `logit_precedes` in
   `qwen3_decode.rs`. Stage one is one thread per ~4,096-entry slice scanning in
   ascending id order; stage two folds the partials with the same rule.
2. **On-GPU embedding gather** (`metal_step_embedding_gather`): a chained step
   reads its input token device-side from the previous step's argmax output and
   gathers that row of the resident f16 embedding table. Those bits are exactly
   what `encode_f16_bits(embedding(token))` produces on the per-token path, so the
   fed activation is identical.
3. **Chained encode** (`synapse_qwen3_metal_step_chain`): k full forward passes
   plus argmax are encoded into one command buffer. Position advances per step and
   the per-step rope block is selected by offset into a host-supplied k-position
   table (each block is byte-identical to what the per-token path computes for
   that position). Only the first step's input token is host-seeded; the rest are
   produced device-side.
4. **Readback every k tokens** (4*k bytes) replaces the per-token 604KB logits
   readback. Logits are only read back on the per-token path (k=1).
5. **Runtime k**: `SYNAPSE_METAL_STEP_CHAIN_K` (default 1, the fully
   instrumented per-token path). Setting k>1 opts plain generation into the
   chained path; constrained or hook-armed runs always use the per-token path.

### Hook contract (D-009)

k=1 is byte-for-byte the pre-wave per-token path — verified 0/20 token
differences against the pre-change tree on M5. `generate_chained` intentionally
carries no per-token top-logit tap and accepts no JSON constraint, pause, or
splice; those product invariants are served by the fully instrumented `generate`
at chain span 1, and any constrained run is routed there automatically. Stop
tokens are honoured by truncation: the fused submission may produce up to k-1
tokens past a stop, and the host truncates the returned stream at the first stop,
matching per-token generation. Two unit tests
(`chained_generation_matches_per_token_generation`,
`chained_generation_truncates_at_stop_token_like_per_token`) lock this on every
CI run without a GPU.

### The nine process deaths

This mechanism was proposed nine times across three campaigns and never survived
patch application — it had never been measured on Metal. Every prior attempt died
in the automated patch step (context-window or apply failures on the large
`.m`/`.metal`/`.rs` edit set), not on its merits. This wave was built directly
with full file access, which is why it exists as a coherent, gated tree rather
than a tenth banked patch failure. The externally validated precedent is BaseRT
(arXiv:2607.00501): 321 tok/s on Qwen3-0.6B Q8 on an M4 Pro — less bandwidth than
our M1 Max — using exactly this GPU-chained design.

### Local M5 correctness gates (green)

Built with the Xcode Metal toolchain
(`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`), Qwen3-0.6B snapshot
`c1899de289a04d12100db370d81485cdf75e47ca`, the campaign 20-prompt fixture, 64
new tokens, bucket 512.

- **Chaining changes no token** (the core invariant): k=1 vs k=4, k=8, and k=16
  are token-for-token identical across all 20 prompts, for both f16 (0/20 diffs at
  every k) and Q8_0 (0/20 diffs at every k). This is the machine-independent proof
  that the whole gate rests on; it does not depend on which Apple GPU runs it.
- **k=1 == pre-change master**: 0/20 token differences, so the per-token hook path
  is unchanged.
- **f16 fixture gate**: 19/20 exact at k=1 and at k=8, median match depth 64.0.
  The single miss is completion-06 at step 7, the documented M5 cross-machine
  near-tie (13079 at 16.744871 vs fixture 61686 at 16.741877, gap 0.003). The M1
  is the fixture authority and there it is the full 20/20 at both k=1 and k=8
  (below). No threshold was changed.
- **Q8_0 quality gate**: 13/20 exact, median match depth 64.0 at k=1, k=8, and
  k=16 — above the campaign floor (>= 10/20 exact, median depth >= 59.0), and
  unchanged by chaining.
- **Hooks**: `cargo test -p spike-unified-rt qwen3_decode` — 10 passed, 0 failed,
  including the 5 harness hook tests and the 2 new chained-invariant tests.

An earlier M5-local throughput probe was +29.7% (Q8 k=16 vs k=1) and predicted a
comfortable clear. It did not transfer: the M1 timing below is the authority, and
it clears only +2.58%. The M5 indicative is not repeated here to avoid implying a
result the authority machine did not confirm.

### Locked-M1 authority gates (green) and timing (measured, below bar)

Locked M1 Max, `[bench-host]`, exclusive `[bench-user-home]/bench.lock` held
then released, AC power 100% charged, no `Runner.Worker`, 1-minute load average
1.07--1.33. Built with the M1's own cargo (`[bench-user-home]/.cargo/bin/cargo`, full
Xcode default), Qwen3-0.6B snapshot `c1899de289a04d12100db370d81485cdf75e47ca`,
executable-relative 64,591-byte metallib beside the binary.

Correctness (fixture authority):

- **f16 fixture gate: 20/20 exact at k=1 AND k=8**, median match depth 64.0.
  completion-06 is exact here — it was the M5-only cross-machine near-tie.
- **Q8_0 quality: 13/20 exact, median match depth 64.0 at k=1, k=8, and k=16**,
  zero near-tie exemptions.
- **Chaining changes no token on the authority machine**: k=1 == k=8 (f16) and
  k=1 == k=8 == k=16 (Q8), token-for-token across all 20 prompts.
- **Long-context fixture**: the 470-token prompt at bucket 1024 produced 64/64
  tokens identical between metal-step (k=8) and the MPSGraph reference.
- **Hooks**: `cargo test qwen3_decode` on the M1 — 10 passed, 0 failed.

Timing (N=12 stride-seven prompts x 2 fresh processes, worse-of-two per prompt,
median across the 12):

| Weights | k=1 (fresh control) | k=4 | k=8 | k=16 | best-k vs control |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8_0 | **85.2192** | 86.1610 | 86.9541 | 87.4162 | **+2.58% at k=16** |
| f16 | **48.6168** | — | 49.1345 | 49.2656 | +1.33% at k=16 |

The fresh control reproduced the winners-5+6 baseline within drift: Q8 85.2192
tok/s vs the pinned 85.2589 (-0.05%), f16 48.6168 vs 48.6530 (-0.07%), both well
inside the 2% control-drift gate.

### Decision: banked correctness win, not shipped

Best-k Q8 is **+2.58%**, below the **+3%** shipping bar, so per the wave rules the
change is **not shipped as the throughput win**: the pinned Q8 baseline
(85.2589 tok/s), the `campaign-lab.jsonc` blocks, the harness `BASELINE_TOK_S`,
and every threshold are **left untouched**. The mechanism is nonetheless retained
in-tree as a correct, token-exact, fully gated implementation (k=1 preserves the
per-token hook path byte-for-byte), because it finally converts a mechanism that
died nine times in patch application into a measured M1 result.

Why the M1 clears only +2.58% while the physics motivated the wave: on the M1 the
per-token `waitUntilCompleted` plus 604KB logits readback that chaining removes is
a smaller fraction of the ~11.7ms Q8 token than the motivating estimate assumed.
Q8 decode is close to bandwidth-bound here (~54 GB/s effective active-weight rate
at 85 tok/s), so amortizing the host round trip over k tokens recovers only about
2.6%, not the double-digit figure the M4-Pro BaseRT precedent and the M5 probe
suggested. This is a real, honestly-measured ceiling for this mechanism on this
machine, not a gate failure — no tokens changed at any k.

## Cross-machine exactness note

Completion-06 f16 near-tie resolves differently on M5-current Metal stack as
of 2026-07-22 (both backends self-consistent, 13079 vs fixture 61686 at step
7); fixtures remain M1-homed; M5 local gates use the remaining 19 prompts +
mutual MPSGraph/step agreement until re-cut. The M1 control and both winner
trees passed the full 20/20 fixture gate; no threshold was changed.

## Campaign targets

Wave 1 keeps the MPSGraph backend as the default. On the locked M1, the
correctness-preserving attention P/V parallelism and f16 dot unrolling lifted
f16 from 5.8080 to 17.7992 tok/s; Q8_0 cooperative dequant matvec lifted Q8_0
from 5.0405 to 29.2656 tok/s, making Q8_0 faster than f16. GPU execution still
accounts for 55.8754 ms/token f16 and 33.8624 ms/token Q8_0, while encode/feed
is 0.1021/0.1018 ms/token, so dispatch consolidation is not yet the limiting
stage. The remaining gap to the 84.32 tok/s MPSGraph baseline is a kernel-level
profile problem, not host dispatch overhead.

Wave 2 keeps MPSGraph as the default and applies only order-preserving
parallelism. Attention lanes now split KV positions rather than splitting a
single QK reduction; f16 rows remain independent serial reductions; and Q8
keeps its one-simdgroup-per-row reduction while exposing four block iterations
per lane. On the locked M1, f16 reached 32.3296 tok/s and Q8_0 reached 49.9470
tok/s, with GPU execution falling to 30.6257 ms and 19.7155 ms per token.
Both cells beat their wave-1 references; the measured GPU stage remains well
above the roughly 10 ms dispatch-fusion threshold.


## Wave 1 progression log

The local M5 runs below were correctness/performance probes, not substitutes for
locked-M1 results. Every kernel probe was followed by the f16 20-prompt gate;
Q8_0 rows were reported only after a fresh Q8_0 quality gate.

| Change | f16 gate | Local probe | Locked-M1 result | Decision |
| --- | --- | --- | --- | --- |
| 32-wide QK reduction and fully cooperative attention prototype | 19/20, 0 near ties; completion-06 diverged at step 7 | 33.9162 tok/s | not timed | Reverted: reduction order changed logits |
| Lane-zero reference-order scores plus 32-wide P/V attention | 20/20, 0 near ties | 28.2297 tok/s first passing run | included in final | Kept: exact score order, parallel value dimensions |
| Cooperative f16 matvec with `simd_sum` | 19/20, 0 near ties; completion-06 diverged at step 25 | 68.6494 tok/s combined probe | not timed | Reverted: f16 parity is a hard gate |
| Four-product f16 dot unroll, then half4 loads | 20/20, 0 near ties after each probe | 43.6843 then 43.9977 tok/s | **17.7992 tok/s** | Kept: same f32 accumulation order, materially faster than baseline |
| Fuse f16 QK-norm+RoPE into QKV epilogue | 20/20, 0 near ties | 43.5677 tok/s | not timed | Reverted: command fusion was slower than the unfused half4 candidate |
| Q8_0 block-row cooperative matvec | 20/20 f16 gate | final Q8_0: 14/20 exact, median depth 64.0, 55.2564 tok/s | **29.2656 tok/s** after 14/20, median depth 64.0 gate | Kept: Q8_0 is now faster than f16 |
| Shared-to-private weight upload | 20/20 f16 gate; Q8_0 gate remained 14/20, median depth 64.0 | no material isolated delta on M5; residency verified | included in final | Kept: private residency removes repeated host-visible weight access |

The locked-M1 final f16 sweep used AC power, `[bench-user-home]/bench.lock`, 12
varied prompts selected by the house stride-seven schedule, two fresh-process
repeats, and 24 samples total. The median stage breakdown was 55.8754 ms GPU
execution, 0.1021 ms feed/encode, and 0.0340 ms logits readback per token.
The locked-M1 Q8_0 quality gate was 14/20 exact with median match depth 64.0
and zero near-tie exemptions; its 12 x 2 timed median was 29.2656 tok/s with
33.8624 ms GPU execution, 0.1018 ms feed/encode, and 0.0327 ms readback per
token. The local final f16 gate was 20/20 exact and the local final Q8_0 gate
was 14/20 exact with median depth 64.0.

## Wave 2 progression log

Every wave-2 kernel change passed the local f16 exactness gate before the
locked-M1 timing cells; the Q8 row was reported only after a fresh Q8 quality
gate. The M1 cells used AC power (100%, charged), `[bench-user-home]/bench.lock`, no
`Runner.Worker`, the fixed stride-seven 12-prompt schedule, two fresh-process
repeats, and the executable-relative 48,866-byte metallib.

| Change | f16 gate | Local probe | Locked-M1 result | Decision |
| --- | --- | --- | --- | --- |
| KV-position-parallel QK dots with lane-zero reference-order softmax | 20/20 exact, 0 near ties | 67.8324 tok/s gate run | **32.3296 tok/s**, 30.6257 ms GPU | Kept: 20/20 exact; independent scores removed the serial attention bottleneck |
| F16 row-parallel path with two-half4 dot unroll | 20/20 exact, 0 near ties | included above; 161.73 GB/s effective weight rate | included above | Kept: serial per-row accumulation preserved exactness |
| Q8 one-simdgroup-per-row four-block unroll | 20/20 f16 gate; Q8 14/20, depth 64.0 | 92.3741 tok/s quality run | **49.9470 tok/s**, 19.7155 ms GPU | Kept: Q8 quality floor passed; +70.6% vs wave 1 |

The wave-2 f16 cell is +81.6% over 17.7992 tok/s, and the Q8_0 cell is +70.6%
over 29.2656 tok/s. Neither crosses the 84.32 tok/s MPSGraph reference; the
remaining gap is GPU execution rather than dispatch (feed stayed 0.1022 ms/token).

A fresh llama.cpp control was not available on the locked host: neither
`llama-cli` nor `llama-server` was installed, and no source checkout or archived
MANIFEST was present under `[bench-user-home]/bench-tools`. The historical Q8_0
control remains 190.36--203.45 tok/s from `DECODE-WAVE1.md`; it is not relabeled
as a fresh wave-1 measurement.

## Wave 3 profiler attribution

The Xcode Metal build produced a 51,570-byte executable-relative metallib with
no `cargo:warning`. The opt-in profiler ran on the local M5 with one short
prompt (`completion-01`, two decode steps) and a synthetic 469-token prompt
(one decode step, position 469). Values below are GPU milliseconds per decode
token; they are not throughput timings because profiling serializes one command
buffer per kernel invocation.

| Kernel class | f16 short | f16 position 469 | Q8_0 short | Q8_0 position 469 |
| --- | ---: | ---: | ---: | ---: |
| RMSNorm | 1.020 | 1.017 | 1.012 | 1.024 |
| QKV matvec | 1.577 | 1.578 | 0.450 | 0.456 |
| QK norm + RoPE | 0.621 | 0.624 | 0.619 | 0.621 |
| Attention | 0.484 | **17.930** | 0.480 | **17.927** |
| O projection | 1.038 | 1.039 | 0.297 | 0.302 |
| Residual RMSNorm | 1.079 | 1.075 | 1.078 | 1.086 |
| Gate/up/SwiGLU | 1.379 | 1.397 | 0.642 | 0.640 |
| Down projection | 1.543 | 1.552 | 0.407 | 0.413 |
| LM head | 0.695 | 0.690 | 0.416 | 0.415 |

The short-context f16 kernel total fell from 13.2532 to 9.4342 ms/token after
half4 vector loads were added to both norm kernels; Q8 fell from 9.3482 to
5.4007 ms/token in the same local profile. At position 469, attention is the
wall: it grows from 0.484 to 17.930 ms/token for f16 and from 0.480 to 17.927
for Q8. The attempted probability-cache attention change was reverted because
it measured 18.197/18.301 ms at position 469. The synthetic run therefore
records the growth curve without claiming a regressed optimization.

## Wave 4 profiler attribution

Wave 4 retained the full-simdgroup RMSNorm change after the profiler showed that the
one-thread dispatch was the fixed cost. The profile below is from the local M5,
with one short prompt and two decode steps, plus a 470-token prefill and one step
at position 470. Values are GPU milliseconds per decode token; profiled command
buffers are serialized and are not throughput measurements.

| Kernel class | f16 short | f16 position 470 | Q8_0 short | Q8_0 position 470 |
| --- | ---: | ---: | ---: | ---: |
| RMSNorm | 0.490 | 0.489 | 0.485 | 0.478 |
| QKV matvec | 1.552 | 1.561 | 0.447 | 0.451 |
| QK norm + RoPE | 0.633 | 0.637 | 0.632 | 0.632 |
| Attention | 0.485 | **17.977** | 0.485 | **17.968** |
| O projection | 1.027 | 1.032 | 0.301 | 0.302 |
| Residual RMSNorm | 1.078 | 1.075 | 1.080 | 1.073 |
| Gate/up/SwiGLU | 1.361 | 1.345 | 0.639 | 0.645 |
| Down projection | 1.533 | 1.530 | 0.410 | 0.411 |
| LM head | 0.676 | 0.692 | 0.417 | 0.415 |

The retained kernel reduced the profiled short total from 9.4342 to 8.8331
ms/token for f16 and from 5.4007 to 4.8966 ms/token for Q8_0. The long-context
attention wall is unchanged, so this wave makes no unsupported claim about
solving the serving-depth bottleneck. A multi-simdgroup attention prototype was
measured at 23.851 ms (four simdgroups per head) and 34.448 ms (packed 1,024-
thread groups) at position 470; both were reverted as slower than the 17.930 ms
wave-3 reference.

## Wave 4 progression log

Every retained wave-4 change passed the local f16 20-prompt exactness gate. The
fresh Q8 quality gate passed before reporting its local result. Reduction-order
changes that did not pass the hard f16 gate were reverted rather than hidden.

| Change | f16 gate | Q8 gate | Local profile/probe | Decision |
| --- | --- | --- | --- | --- |
| Full-simdgroup pre-attention RMSNorm | 20/20 exact | 13/20 exact; median depth 64.0 | f16 1.020 -> 0.490 ms; Q8 1.012 -> 0.485 ms | Kept |
| Full-simdgroup QK norm + RoPE | 19/20; first mismatch completion-06 | not measured after rejection | 0.621 -> 0.261 ms | Reverted: hard f16 gate |
| Full-simdgroup residual RMSNorm | 19/20; first mismatch completion-06 | not measured after rejection | no retained result | Reverted: hard f16 gate |
| Four-simdgroup attention head | not gated after local regression | not gated | 17.930 -> 23.851 ms at position 470 | Reverted: slower |
| Packed 1,024-thread attention groups | not gated after local regression | not gated | 17.930 -> 34.448 ms at position 470 | Reverted: slower |

The retained local f16 run was 20/20 exact across 1,280 generated tokens. The
retained Q8 run was 13/20 exact with median match depth 64.0 and zero near-tie
exemptions. A separate long-context parity spot-check used the 470-token prompt,
a 1,024 cache bucket, and 64 generated tokens: MPSGraph and Metal step were
identical for 64/64 tokens. No locked-M1 throughput cell is claimed by this local
wave log; the existing locked-M1 rows above remain unchanged.

## Wave 3 progression log

Every retained wave-3 change passed the local f16 exactness gate and the fresh
Q8 depth gate before timing. The locked-M1 cells used AC power, an exclusive
`[bench-user-home]/bench.lock`, no active `Runner.Worker`, the fixed stride-seven
schedule (`completion-01,08,15,02,09,16,03,10,17,04,11,18`), 12 fresh processes
per weight mode, and the executable-relative 51,570-byte metallib.

| Change | f16 gate | Q8 gate | Local profile/probe | Locked-M1 result | Decision |
| --- | --- | --- | --- | --- | --- |
| Per-kernel GPU timestamp attribution | 20/20 exact | 11/20 exact; median depth 64.0 | table above | profile-only | Kept; opt-in and excluded from throughput cells |
| Q8 char4 block slices with four-block unroll | 20/20 exact | 11/20 exact; median depth 64.0 | retained candidate | included below | Kept; Q8 M1 +35.5% vs wave 2 |
| Half4 vector loads in RMSNorm and residual RMSNorm | 20/20 exact | 11/20 exact; median depth 64.0 | f16 total 13.2532 -> 9.4342 ms/token | included below | Kept; short-context norm bottleneck removed |
| F16 threadgroup size 256 -> 512 | 20/20 exact | 11/20 exact; median depth 64.0 | f16 profile regressed 9.4342 -> 9.5368 ms/token | not timed | Reverted: slower on M5 |
| Attention probability cache before P/V | 20/20 exact | 11/20 exact; median depth 64.0 | long attention regressed 17.930 -> 18.197 ms/token | not timed | Reverted: slower on M5 |
| Four Q8 rows per simdgroup with subgroup reduction | 20/20 exact | **0/20; median depth 1.0** | failed hard Q8 depth gate | not timed | Reverted immediately |

The locked-M1 wave-3 medians were 42.3634 tok/s f16 and 67.6920 tok/s Q8_0.
The Q8 goal of 84.32 tok/s was not reached. Effective active-weight
bandwidth was approximately 101.0 GB/s f16 (`2,384,199,680` bytes/token) and
42.8 GB/s Q8 (`633,495,552` bytes/token), below the M1 Max theoretical ~400
GB/s. The profiler identifies short-context norm/projection work and long-
context attention growth as the remaining measured headroom; no unsupported
bandwidth saturation claim is made.


## Wave 5 row-level memory-parallelism probe (banked negative)

Wave 5 tested the row-level memory-parallelism mechanism that CUDA campaign #14
winner 9 measured at **+13.06%** over its fresh `524.0289724489505` tok/s
control. The Metal candidate assigned adjacent output rows to each active lane
pair inside a simdgroup. Each pair loaded the activation half4 once, applied it
to both rows' Q8_0 weight streams, and kept one serial ascending f32 accumulator
per row. The same shape was applied to QKV, O projection, fused gate/up, down
projection, and the 151k-row LM head; f16 used the same row-pair layout where
its exactness gate remained green. No attention, norm, or MPSGraph code was
changed.

The local M5 gates were green under the cross-machine exactness rule: f16 was
19/20 against the M1-homed fixture with mutual MPSGraph/step agreement, Q8 was
11/20 exact with median depth 64.0 and zero near-tie exemptions, all 8 decode
hook tests passed, and the 470-token prefill plus 64-token long-context outputs
matched 64/64. The M1 authority gate was also green for the candidate: f16
20/20, Q8 11/20 with median depth 64.0 and zero near-tie exemptions. The M1
fresh control reproduced the expected cells before the candidate timing.

| Tree | f16 tok/s | Q8_0 tok/s | Protocol | Decision |
| --- | ---: | ---: | --- | --- |
| Fresh winners 5+6 control | **48.6530** | **85.2449** | 12 prompts x 2 fresh processes; worse repeat per prompt | fresh control |
| Wave 5 exact row-paired candidate | 36.9881 | 30.5319 | same locked-M1 protocol | **Reverted: banked negative** |

The candidate was **-64.19%** versus the fresh Q8 control (and -23.98% for
f16), so it missed the 3% shipping bar by a wide margin despite passing the
quality gates. The Metal source was reverted and neither the campaign baseline
nor `campaign-lab.jsonc` registration was changed. This is evidence that the
CUDA row axis does not transfer to this Metal step shape: preserving serial
per-row accumulation left too little useful intra-row parallelism for the
Metal scheduler, even though the candidate exposed more independent row
streams.
