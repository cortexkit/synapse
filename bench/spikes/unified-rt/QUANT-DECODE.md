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
| Qwen3-0.6B | Q8_0 | 343.79 | 633,495,552 | 217.8 (21.8% of 1000) | **1.43x** |

The historical LFM2 fp32 comparison point was 178.5 tok/s; the fresh same-rig
178.35 result reproduces it. Qwen3's 239.77 tok/s row is its first owned CUDA
fp32 decode baseline. LFM2 Q8_0 is useful but does not approach the 3.76x active
byte reduction. Qwen3 is launch/dequant limited: its effective bandwidth falls
to 217.8 GB/s, so the 1.43x gain is a negative bandwidth-efficiency result, not
evidence of a saturated compressed-weight path.

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
4090. The frozen throughput baseline is **343.8 tok/s** for one stream and 64
new tokens (`217.8 GB/s` effective weight bandwidth); llama.cpp's `521.4 tok/s`
comparison is a competitor reference, not an acceptance gate. The rented rig must
be an RTX 4090 with reliability above `0.99` and driver `>=570`.

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
campaign floor is at least `13/20` exact prompts and median match depth at least
`54.5`; near-tie exemptions are not accepted. The constrained fixture must also
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
