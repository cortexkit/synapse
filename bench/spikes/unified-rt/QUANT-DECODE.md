# Owned CUDA quantized decode spike

## Status

Q8_0 is implemented as an additive CUDA decode path for LFM2 and Qwen3. The
existing fp32/f16 model tensors and provider paths remain present and are still
the default (`--weight-quant none`). The spike does **not** graduate yet: Qwen3
misses the bandwidth-efficiency goal and the required 100-prompt label fixture
is absent from the repository. Q4_K was therefore not started.

The runtime invocation is:

```text
--device cuda --dtype f32 --weight-quant q8-0
```

Activations, accumulation, RoPE/norm parameters, convolution state, and KV
cache remain fp32. Embedding lookup remains fp32. Every dense decode matrix,
including a tied or untied LM head, is Q8_0.

## Storage and loading choice

The spike requantizes the already-supported safetensors fp32 view at model
load. This avoids introducing a second model-name mapping and sharding parser
before kernel viability is known. It also makes a controlled fp32-versus-Q8_0
comparison use exactly the same source checkpoint.

Each matrix row is split into 32-element blocks. A block is exactly the GGUF
`block_q8_0` byte layout:

```text
little-endian f16 scale | 32 signed i8 quants
```

There is no Rust or CUDA struct padding: each block occupies 34 bytes. The scale
is `max(abs(block)) / 127`; values are rounded and clamped to `[-127, 127]`.
The loader rejects non-finite values and matrices whose row width is not block
aligned. Unit tests cover byte layout, zero-block canonicalization, round-trip
error, deterministic regions, and preservation of the source fp32 tensor.
Measured active-block SHA-256 digests were:

- LFM2-1.2B: `5874faabdce2567dcc0e7339e9547d79421ba312c71e3442c9cc3c4ed3cb47d0`
- Qwen3-0.6B: `4c774c188ec089ac1e9b30e9797b364993d5c2445d3e479d5d1b94f6d10969d0`

The result JSON records `quantized_weight_sha256`, computed over active matrix
blocks in stable model order. `WeightRegion` addresses and checksums select the
Q8_0 bytes when quantization is active, so taps/regions inspect complete
quantized blocks. Pause and splice still own only token/activation/KV state and
are unchanged.

Trade-off: the host keeps the source fp32 tensors, and CUDA keeps a dequantized
fp32 copy for prefill in addition to Q8_0 decode storage. This is intentionally
a viability implementation, not the final memory-minimum loader. A direct GGUF
loader can later map the same 34-byte blocks without changing the decode kernel
or fingerprint definition.

## Kernels

A CUDA block computes one output row. Eight warps traverse independent Q8_0
blocks; each lane consumes one quant, multiplies by the block's f16 scale and an
fp32 activation, and participates in warp/block fp32 reductions. The kernel
therefore reads each matrix block once and never materializes a dequantized
matrix during decode.

LFM2 selects the fused kernel only for Q8_0 matrices. The pre-existing fp32
cuBLASLt calls are untouched. Conv input/output projections, attention
projections, all MLP projections, and the LM head use fused Q8_0 matvec; the
small depthwise convolution remains fp32.

Qwen3 has a CUDA fp32 decode implementation alongside Q8_0 so both baselines
use the same fp32 activation/KV layout. Q/K/V/O, MLP, and LM-head matrices use
the same fused kernel. Qwen3 prefill currently advances through the prompt with
the decode kernel to establish the KV cache. This is correct but intentionally
not a prefill performance claim; replacing it with one-time dequantization plus
cuBLASLt is follow-up work and does not affect the separately timed decode
number. LFM2 prefill uses the one-time dequantized fp32 matrices and cuBLASLt.

`weight_bytes_per_token` is the sum of active dense storage plus fp32 norm and
small convolution parameters read once per token. `achieved_weight_gb_s` is
that byte count multiplied by measured generated tokens/s. It is effective
model-weight bandwidth, suitable for comparison with the RTX 4090's roughly
1000 GB/s physical ceiling; it does not claim to count cache-line replays.

## Gates and measurements

Rig requirement: one RTX 4090, reliability above 0.99, driver at least 570.
All timings are single-stream, greedy, and exclude prompt prefill from
`decode_tok_per_s`.

### Throughput and effective bandwidth

| Model | Weights | Decode tok/s | Active bytes/token | Effective GB/s | Versus fp32 |
|---|---:|---:|---:|---:|---:|
| LFM2-1.2B | fp32 | 178.35 | 4,681,362,432 | 834.9 (83.5% of 1000) | 1.00x |
| LFM2-1.2B | Q8_0 | 361.80 | 1,243,868,160 | 450.0 (45.0% of 1000) | **2.03x** |
| Qwen3-0.6B | fp32 | 239.77 | 2,384,199,680 | 571.7 (57.2% of 1000) | 1.00x |
| Qwen3-0.6B | Q8_0 | 592.87 | 633,495,552 | 375.6 (37.6% of 1000) | **2.47x** |

The historical LFM2 fp32 comparison point was 178.5 tok/s; the fresh same-rig
178.35 result reproduces it. Qwen3's 239.77 tok/s row is its first owned CUDA
fp32 decode baseline. LFM2 Q8_0 is useful but does not approach the 3.76x active
byte reduction. The direct-KV-cache winner, the stacked CUDA winners, the batched-QKV winner, winner
6's fused QK head-norm/RoPE launch, winner 7's decode graph, and the shipped
round-5 row-paired GEMV winner raise Qwen3 Q8_0 to 592.87 tok/s and 375.6 GB/s
(2.47x fp32). This is 1.14x the same-day llama.cpp reference while remaining
launch/dequant limited rather than bandwidth saturated.

### Quality ladder

Token exactness is not a quantization gate. Every result row reports
`match_depth`, the number of leading greedy tokens equal to its fp32 oracle.
The aggregate preserves exact-prompt count only as descriptive evidence.

| Model | Weights | 20-prompt match depth | Label-validity harness | Gate |
|---|---:|---:|---:|---:|
| LFM2-1.2B | Q8_0 | 13/20 exact; median depth 54.5 | unavailable; 15/15 schema-valid supplementary run | incomplete |
| Qwen3-0.6B | Q8_0 | 10/20 exact; median depth 59.0 | unavailable; 15/15 schema-valid supplementary run | incomplete |
| either | Q4_K | not attempted | not attempted | N/A |

Per-prompt Q8_0 match depths (`depth/generated`, `*` means the entire fp32
sequence matched) are:

| Prompt | LFM2 | Qwen3 |
|---|---:|---:|
| completion-01 | 64/64* | 64/64* |
| completion-02 | 35/64 | 64/64* |
| completion-03 | 45/64 | 17/64 |
| completion-04 | 64/64* | 54/64 |
| completion-05 | 0/9 | 64/64* |
| completion-06 | 64/64* | 64/64* |
| completion-07 | 64/64* | 10/64 |
| completion-08 | 64/64* | 24/64 |
| completion-09 | 32/32* | 8/64 |
| completion-10 | 1/1* | 64/64* |
| completion-11 | 64/64* | 64/64* |
| completion-12 | 22/22* | 36/64 |
| completion-13 | 48/64 | 64/64* |
| completion-14 | 36/36* | 64/64* |
| completion-15 | 8/64 | 48/64 |
| completion-16 | 61/61* | 2/64 |
| completion-17 | 4/64 | 64/64* |
| completion-18 | 61/64 | 35/64 |
| completion-19 | 64/64* | 64/64* |
| completion-20 | 64/64* | 22/64 |

The repository contains the 20 raw-completion prompts and a 15-prompt
constrained-JSON fixture, but not the referenced 100-prompt micro-LLM label
fixture. Both Q8_0 models completed all 15 schema-constrained prompts with valid
`{result: allow|deny, score: number}` output when given a 256-token cap. That is
useful supplementary evidence, but a run on substituted or repeated prompts
would not satisfy the 100-prompt label gate, so no such number is fabricated.

### Regression and instrumentation

- Q8_0 is opt-in; fp32 tensors are retained byte-for-byte and the fp32 native
  dispatch is unchanged.
- CPU-side runtime tests include tap-before-commit, pause/resume, splice,
  deterministic region bytes, quant layout, and source-fp32 preservation.
- Linux CUDA-feature compilation succeeded, and its test suite passed 47 tests
  with 4 hardware/model fixtures ignored.
- Region handles address the quant block buffer. Pause/splice do not serialize
  weights and therefore require no semantic change.

## Traps and follow-up

1. Q8_0's 34-byte block is not naturally 32-byte aligned. Kernels must index by
   34 bytes rather than reinterpret an array with compiler-dependent padding.
2. Quantizing a tied embedding separately for LM-head use is required;
   embeddings themselves remain fp32.
3. Comparing quantized cached decode to an fp32 full-reprefill path is not a
   token-exact invariant. Compare both against the fp32 oracle and report match
   depth.
4. The viability loader duplicates fp32 and Q8_0 storage. A production GGUF
   loader should mmap/hash the original tensor bytes and keep only a bounded
   prefill dequant workspace.
5. Q4_K should start only after Q8_0 quality, label validity, and bandwidth gates
   pass. Its super-block scales/mins and nibble layout require a separate kernel;
   silently treating it as symmetric Q4 would break GGUF interoperability.

## Rig and spend

The measurement rig was a dedicated Vast.ai RTX 4090 (24,564 MiB), offer
reliability 0.9952, driver 580.142, CUDA 12.6, and advertised 901.1 GB/s device
memory bandwidth. Its rate was $0.35556/hour. Two earlier qualifying rentals
were destroyed after their container images stalled during startup; their
combined charge was about $0.10. The measurement rental cost about $0.30, for
approximately **$0.40 total spend**, well below the $25 cap. All three
`owned-quant-decode` instances were destroyed after evidence capture.


## Campaign baseline: CUDA Q8_0 single-stream decode

The second campaign targets the owned Qwen3-0.6B Q8_0 CUDA decode path on an RTX
4090. After the shipped round-5 winner, the re-pinned throughput baseline is
**592.8694799258782 tok/s** for one stream and 64 new tokens (`375.6 GB/s`
effective weight bandwidth). This is 1.14x llama.cpp's `521.4 tok/s` competitor
reference; that reference was measured on this rig under the same-day protocol
earlier and remains a comparison rather than an acceptance gate. The rented rig
must be an RTX 4090 with reliability above `0.99` and driver `>=570`.

`bench/campaign/cuda-quant-harness.sh` is a self-contained controller. It embeds
and hashes the raw-completion fixtures (`decode-prompts.jsonl` SHA-256
`6f1ee1ce17fbc3ca34ebc316bc93d44db7c8840a6d4a05906b13bc0ef8901e60` and
`reference-tokens.jsonl` SHA-256
`b2d11f2aaf92cdce0fc906dc7ef0468308bce43bf5661b490f336cc1215b1ee9`) plus the
15-prompt constrained-JSON fixture (`56fee1844e5a8991c28b81e46018c42a0e811dc07233538048b32df9b11e5ed3`) and schema (`7b691bb9ce46f8ab3fcce415ba9d28129924fa8bd1a0b4d5475895eff7837394`).
Candidate staging, build, tests, and decode processes all go through ALF's copied
candidate runner; the result writer is controller-owned and reclaim-safe.

The quality gate is quantization-aware rather than token-exact: every prompt's
reported `match_depth` is recomputed against the pinned fp32 oracle. The frozen
campaign floor is at least `10/20` exact prompts and median match depth at least
`59.0`; near-tie exemptions are not accepted. The constrained fixture must also
produce `15/15` schema-valid `{result: "allow"|"deny", score: number}` objects.
The hook gate runs the CPU-side `cargo test -p spike-unified-rt` regression suite. Only
after both gates pass does the harness start `N=12` fresh single-stream processes
with varied prompts and report median `decode_tok_per_s` in the same result JSON
schema as the Metal campaign (`gate_passed`, `hooks_passed`, `median_tok_s`,
`samples`, `workspace_commit`, and `baseline_note`).

The rented-box preflight records `nvidia-smi` driver, P-state, SM clock, power,
and compute-process state in `scene.json` and in `baseline_note`; a foreign GPU
compute process rejects the run. Battery checks are not applicable. The
provisioning script additionally installs the candidate scheduler denials and
fenced sudo verbs for identity drops, process reaping, candidate-only
iptables/ip6tables egress denial, and ownership repair. It leaves the rig alive
for the overnight campaign and does not edit `campaign-lab.jsonc` or start
Athena.

### Campaign winner 1 confirmation

Campaign `[consult-id]` winner 1 writes the value
matvec and RoPE-normalized key directly to their per-layer KV-cache slots at the
current token offset. This removes two device-to-device `cudaMemcpyAsync`
launches for every layer and generated token. The campaign's controlled RTX 4090
measurement was 375.46 tok/s versus 363.74 tok/s for the control (`+3.22%`;
first pair `+2.93%`).

The direct-endpoint confirmation on 2026-07-19 rebuilt the CUDA feature and ran
the 20-prompt, 64-token Q8_0 fixture: 10/20 exact prompts, median match depth
64.0, and no near-tie exemptions. The constrained JSON fixture was also 15/15
schema-valid. The shared host's pre-measurement load average was `31.71 32.63
32.27` (higher than the usual roughly-20 ambient load), while
the exclusive RTX 4090 had no compute processes (driver 595.58.03, P8, 210 MHz,
24.05 W). Twelve fresh, varied-prompt processes generated 64 tokens each and
had a median of **376.824259765553 tok/s** (`+3.60%` versus the campaign control;
range 372.77–389.25 tok/s). This replaces the historical 343.8 tok/s registry
baseline; that non-paired comparison is `+9.61%` and is not the campaign delta.

The CPU-side shared decode hook suite also passed all five tap, pause/resume,
splice, addressable-region, and greedy-tie tests. These hooks are not
Metal-only; they exercise the shared Qwen3 decode state rather than a
CUDA-device-specific hook implementation.


### Campaign winners 2-4 confirmation

Campaign `[consult-id]` banked three stacked CUDA
winners, applied in round order:

1. Round 1 fused each residual add with the following RMSNorm boundary into one
   kernel: `+3.67%` versus the `376.824259765553` tok/s control.
2. Round 3 fused the gate and up Q8_0 GEMVs with SwiGLU into one MLP kernel per
   layer: `+9.0%` versus the round-1 tree.
3. Round 6 assigned one warp to each cached-key attention score dot product:
   `+6.1%` versus the round-3 tree, with campaign-final steady throughput
   `446.7` versus `420.9` tok/s for the control.

The composed tree was rebuilt on the direct-endpoint RTX 4090 and passed the
quantization-aware quality gate: `10/20` exact prompts, median match depth
`59.0`, `accepted_near_ties=0`, and constrained JSON `15/15` schema-valid. The
CPU-side suite passed `48` tests with `2` ignored, including all five required
decode hook tests. Two repeats of `N=12` fresh varied prompts, 64 new tokens,
reported medians of `445.7626019334018` and `446.6215646524347` tok/s; the
combined median across all 24 samples was `446.21439926190544` tok/s. That
combined median was the re-pinned baseline for winner 4, while the campaign-final
steady figure is the rounded `446.7` tok/s used for the controlled comparison;
that is `+18.5%` versus the `376.824259765553` registry baseline.

The shared host load average was `16.77 15.79 14.54` before the run and
`16.01 18.52 17.19` after it (`15.61 17.05 16.07` between repeats). The
exclusive RTX 4090 reported driver `595.58.03`, P8, 210 MHz, 21.95 W before
and 21.05 W after, with 1 MiB used of 24564 MiB and no compute processes in
both snapshots. At the campaign-final steady figure, effective weight
bandwidth was `446.7 tok/s * 633,495,552 bytes = 282,982,463,078 bytes/s`, or
approximately `283 GB/s`; winner 5 raises the re-pinned measurement to
`301.1 GB/s` while leaving model bytes unchanged. This remains launch-limited
territory rather than memory-bandwidth saturation. The winner-5 result is
`475.33 / 521.4 = 91.16%` of the llama.cpp reference, approximately `91%`.

### Campaign winner 5 confirmation

Campaign `[consult-id]` winner 5 batches the Q8_0
Q/K/V GEMVs for each layer into one decode launch. The kernel walks the query
rows, key rows, and value rows in one grid, reusing the same normalized input
while selecting each matrix's quantized row range. Q and raw K still feed the
existing head-norm/RoPE kernels, and V is written directly into winner 1's
current-token KV slot; the subsequent K head-norm/RoPE kernel still writes the
normalized key into that slot. The per-layer, per-token Q/K/V matvec launch
count therefore falls from three to one, a reduction of two launches (66.7%),
without changing the bytes moved or KV-cache layout.

On the exclusive RTX 4090, the composed tree passed the quality floors: `10/20`
exact prompts, median match depth `59.0`, `accepted_near_ties=0`, constrained
JSON `15/15`, and the CPU-side hook suite was green. Two independent `N=12`
repeats of 64-token, fresh varied-prompt decode reported medians of
`475.70490396191275` and `474.7860938299272` tok/s; the combined median across
all 24 samples was **`475.33051605283094` tok/s**, which was the winner-5
baseline before winner 6. With the unchanged `633,495,552` active bytes/token,
that was `301,119,767,649` bytes/s, or `301.1 GB/s` effective weight bandwidth,
and `+6.53%` versus the winner-4 baseline. The rig reported driver `595.58.03`,
P8, 210 MHz SM clock, and 22.83/23.04 W during the two preflights.

### Campaign winner 6 confirmation

Campaign `[consult-id]` promoted winner 6,
proposal `c10004d6e5d172718b3811a3d63a0f1f8a00570c9cb044c6630c11721ab5e202`,
which fuses each layer's query and key head-RMSNorm plus half-split RoPE into
one launch. A block selects a query head or KV head, performs the same norm and
rotation as the two former kernels, and writes the result to the selected
output. The K output remains the current token's in-slot KV pointer, so winner
1's direct K write is preserved; winner 5's V output remains a direct write to
its current-token value slot.

The launch arithmetic is explicit: before winner 6, Q head norm/RoPE plus K
head norm/RoPE cost `2` launches per layer/token; the fused kernel costs `1`,
saving one launch (50% for this stage). Including winner 5's batched Q/K/V
projection, the projection-plus-head stage is now `3 + 2 = 5` launches in the
pre-winner tree, `1 + 2 = 3` after winner 5, and `1 + 1 = 2` after winner 6:
three launches removed versus the original stage and one versus winner 5.

On the exclusive RTX 4090, two independent full harness runs both passed the
quality and hook gates: `10/20` exact prompts, median match depth `59.0`,
`accepted_near_ties=0`, constrained JSON `15/15`, and all five required hook
tests. The two `N=12` medians were `489.8380553784001` and
`490.27491117248616` tok/s; the combined median across all 24 samples was
**`490.14609753769247` tok/s** (`+3.12%` versus winner 5). With unchanged active
bytes, that is `310,505,372,620` bytes/s, or `310.5 GB/s` effective weight
bandwidth. The rig reported driver `595.58.03`, P8, 210 MHz SM clock, and
22.96/22.41 W during the two preflights.

### Campaign winner 7 confirmation

Campaign `[consult-id]` promoted winner 7,
proposal `823f4100342746a109378a2837c00637327d37edc24aad40cd711105b4969930`,
which captures the complete Qwen3 decode-token launch chain once and replays it
for each generated token. The graph is instantiated lazily on the first decode
step after model preparation; capture records the kernels without executing a
token, and later steps only feed, replay, and read back logits. The host still
feeds the embedding and a one-element
position buffer before replay; Q/K/V projection, in-slot KV writes, fused
QK-norm+RoPE, warp-per-key attention, fused residual+norm, fused MLP, and the
final projection all remain inside the graph. The position is device-indirect,
so changing token position does not require graph re-capture. Attention's shared
scratch is allocated for `capacity` scores at capture time, the worst case; each
replay reads `*position_pointer + 1` for its active sequence, avoiding a
position-dependent scratch resize.

Capture starts after the two per-token host-to-device feeds, while replay is
followed by the unchanged logits readback. CPU-side tap, pause/resume, and splice
hooks therefore remain between replays, and splice advances the same resident KV
cache rather than invalidating graph-owned pointers. Setting
`SYNAPSE_CUDA_GRAPH_VERIFY=1` runs one uncaptured step and one replay from the
same embedding/cache state and asserts byte equality for logits and the newly
written key/value slots; the rig check printed `captured_exact=true`.

A fresh clone under `/root` on the exclusive RTX 4090 passed the full battery
twice: each repeat had `10/20` exact prompts, median match depth `59.0`,
`accepted_near_ties=0`, constrained JSON `15/15`, and all five hook tests. The
`N=12` medians were `524.3733480637275` and `523.3550044127364` tok/s; the
combined median across all 24 samples was **`524.0289724489505` tok/s** (samples
`518.27`–`531.66`), `+6.91%` versus winner 6 and `332.0 GB/s` effective weight
bandwidth. This is `0.50%` above llama.cpp's `521.4 tok/s` reference, measured
on this same rig under the same-day protocol earlier; the reference is a vintage
competitor comparison, not a fresh winner-7 sample or an acceptance gate.

The remaining-structure guidance is unchanged: online-softmax attention has one
`parity_failed` result, so it is risky-but-open rather than closed. RoPE tables are
a banked double-negative and should remain closed as a mechanism; they are not a
reason to relax the online-softmax gate.


### Campaign #14 winners 8+9: overlapping round-5 mechanisms

Campaign `[consult-id]` promoted two round-5
proposals. The authoritative store provenance is winner 8,
`proposal_2f092e7671c295704f2ba115f494bce68dce0162044b480bf5302be1d9d51059`
(claude), followed by winner 9,
`proposal_50f2b03cbc1a73e1658ffa08b964cddaa296ba2d2332e5992d5c19af91165ec6`
(alibaba). The campaign prompt's winner labels named the proposal IDs in the
opposite order; the store's member, title, patch digest, and round queue are the
source of truth.

Winner 8's dual-row Q8_0 GEMV assigns two output rows to each Qwen3 decode block
and issues independent row loads to hide memory latency. Winner 9's row-paired
GEMV uses the same activation vector while computing two adjacent output rows in
the shared family matvec path. These are the same row-level memory-parallelism
axis, not independent optimizations: both round-5 estimates were measured in
parallel against the same `524.0289724489505` tok/s control, not as a stacked
sequence. The predicted compound result, `524.0289724489505 * 1.1284 * 1.1340 =
670.5504077079228` tok/s, therefore does not describe the shipped tree.

The standing RTX 4090 rerun used two independent `N=12` batteries per tree and
selected the worse median from each pair as the `N=24` result. Every tree passed
`10/20` exact prompts, median match depth `59.0`, `accepted_near_ties=0`,
constrained JSON `15/15`, all five hook tests, and
`SYNAPSE_CUDA_GRAPH_VERIFY=1` byte-exact replay:

| Tree | Patch set | Repeat medians (tok/s) | N=24 worse-of-two (tok/s) | Versus fresh control |
|---|---|---:|---:|---:|
| A-only | winner 8 / claude dual-row | 590.847875596791; 589.2987126160701 | **589.2987126160701** | +12.38% |
| B-only | winner 9 / alibaba row-paired | 594.5900806795532; 592.8694799258782 | **592.8694799258782** | +13.06% |
| A+B | both patches applied in promotion order | 590.6283085492648; 590.4832909403555 | **590.4832909403555** | +12.61% |

Winner 9's B-only tree is the highest passing configuration and is the one
shipped. Its `633,495,552` active bytes/token produce `375.5801784495971`
GB/s effective weight bandwidth and a `1.137x` ratio to the vintage llama.cpp
`521.4` tok/s reference. The A+B tree is recorded as a banked negative,
**superseded by an overlapping mechanism**, rather than as a loss; winner 8's
A-only result remains banked evidence for the same reason.
