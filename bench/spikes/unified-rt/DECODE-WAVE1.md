# Qwen3 decode throughput wave 1

## Verdict

The owned Qwen3-0.6B f16 Metal decoder reaches **84.32 tok/s** on the locked M1 Max after the winner-4 GQA attention change. Two fresh N=12 confirmations measured 84.29885065485085 and 84.34879518203019 tok/s; their combined 24-sample median was 84.31679142003034 tok/s. The 20-prompt gate remained token-exact against Transformers CPU fp32: **20/20 prompts, 1,280/1,280 tokens, zero near-tie exemptions**.

The foundations report's 3.2 tok/s result used a debug binary on a contended M5, so it is context rather than a locked release baseline. Instrumented release controls on the M1 showed that the step graph was already cached, but executable preparation and a very slow package-miss first dispatch occurred inside the first timed generation call. Wave 1 now prepares both bucket executables at model load, uses O1 for this large decode graph, keeps KV updates on the GPU, and avoids most rejected-candidate work in the CPU top-k tap. Winner 2 keeps the lm-head logits matmul in f16 and casts only its result to fp32. Winner 3 also keeps QK^T and PV attention matmuls in f16 while retaining the scale, mask, and softmax island in fp32. Winner 4 is the first structural winner: rather than eliminating a cast, it removes the materialized `repeat_kv` KV-head expansion from the decode graph and broadcasts the KV heads inside the attention matmuls (GQA grouping), so attention moves less data.

This clears the `>=40 tok/s` bar with margin. The fresh same-machine llama.cpp Metal `llama-cli` controls measured **207.40 tok/s Q8_0** and **180.15 tok/s f16** (combined 24-sample medians), so the owned winner is 40.7% of the Q8_0 control and 46.8% of the f16 control. The historical 190.36–203.45 band was from the earlier `llama-server` Q8_0 control; it remains recorded as historical context, not as the fresh `llama-cli` result.

## Locked-M1 setup

| Item | Value |
|---|---|
| Host | `[bench-host]`, Apple M1 Max |
| Model | `Qwen/Qwen3-0.6B`, safetensors f16 storage |
| Cache | one stream, bucket 512 |
| Decode | greedy raw completion, 64 generated tokens |
| Graph | one full-bucket prefill executable and one query-length-1 step executable |
| Compilation | `MPSGraphOptimizationLevel1`, one serialized package per pass/bucket |
| Lock | `mkdir [bench-user-home]/bench.lock`; `/tmp/aft-measure.lock` absent; no `Runner.Worker` |

The release binary was built in `[bench-user-home]/ck-campaign/workspaces/mason-winner-2/target/release/spike-unified-rt`. The hardened campaign harness SHA-256 was `008d43490e9504bd420bee96823b950e387ea90a1a81f46e20e9890980741a9c`. Timed cells ran only while the benchmark lock was held. AC power was confirmed before admission with `pmset -g batt`: `Now drawing from 'AC Power'`, internal battery **98%**, charging.

The fresh llama.cpp controls used the M1-native `[bench-user-home]/bench-tools/llama-b9580/llama-cli`, built from llama.cpp tag `b9580` at commit `b4e3dc613baa92a3884d4151e3d631395c81934a` with Xcode/AppleClang 21, CMake 4.4.0, and `GGML_METAL=ON`; the installed `llama-cli` SHA-256 is `02590612ba30c89133d656b7c1300028f345ec6c1cb879fb8f750a3626c02491`; the companion libraries remain installed beside the binary. Fresh competitor admissions showed `AC Power`, internal battery **100%**, charged, with no active `Runner.Worker`. The Q8_0 model is the official `Qwen/Qwen3-0.6B-GGUF` snapshot `23749fefcc72300e3a2ad315e1317431b06b590a`, SHA-256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`. The f16 GGUF was converted on the M1 from the cached `Qwen/Qwen3-0.6B` snapshot `c1899de289a04d12100db370d81485cdf75e47ca` with that tag's `convert_hf_to_gguf.py --outtype f16`, SHA-256 `c81c7c27b35225376a52387800c5eca0748a93b46db885a1dbad370a318f55bb`. The install path is intentionally retained for later campaign lanes.

## Winner provenance

Campaign `[consult-id]` banked winner-2 patch (`8c6da25c9dd5431acc8e632d9cc675520e75ae5cf85b0931d613f867896d9740`). It applied cleanly to the current file, which already contains winner 1; no hand port was needed. Its mechanism is **lm-head logits fp32 cast round-trip removal**: the f16 `linear` result is cast to fp32 only after the matmul, rather than casting the selected activations and lm-head weights to fp32 before the matmul.

Campaign `[consult-id]` banked winner 3. Its attention change leaves QK^T and PV matmuls in f16, casts QK^T scores up before scale/mask/softmax, then casts probabilities down before PV. The campaign reported **78.31 tok/s** for the controlled single-prompt cell and preserved the 20-prompt token-exact gate.

Campaign `[consult-id]` banked winner 4. Its change is structural rather than a cast elimination: it drops the materialized `repeat_kv` expansion of the KV heads and instead reshapes Q to `[1, kv_heads, groups, seq, head_dim]` and K/V to `[1, kv_heads, 1, keys, head_dim]`, letting the attention matmuls broadcast the KV heads across each query-head group (GQA). The scale/mask/softmax island is unchanged (softmax moves to the trailing axis of the 5-D scores). On the locked M1 the campaign measured a steady candidate **84.00 tok/s** versus a **78.72 tok/s** control (+6.7%; first pair 84.35/78.90 = +6.9%) and preserved the token-exact gate. The change sits in the shared attention path after the step/prefill cache concat, so it applies to both the prefill and the step executable; no `repeat_kv` materialization remains in either.

The cumulative steady-decode arc is **40.55 -> 59.03 -> 73.81 -> 78.88 -> 84.32 tok/s**: the first number is the frozen pre-wave baseline, the second is winner 1's AC confirmation, the third is winner 2's AC confirmation, the fourth is the two-repeat winner-3 confirmation, and the fifth is the two-repeat winner-4 confirmation. The separately measured 78.31 tok/s figure is winner 3's controlled single-prompt campaign cell, not the campaign baseline. The prior campaign's battery winner was approximately **73.68 tok/s**; the 73.81 AC confirmation remained within 5%.

## Attribution before changing the path

The legacy controls remain reproducible with `SYNAPSE_QWEN3_DECODE_OPT_LEVEL=0` and `SYNAPSE_QWEN3_DECODE_LEGACY_READBACK=1`. Instrumentation separates graph preparation, host input preparation, feed construction, executable execution, logits readback, KV update, and sampling.

The first package-miss O0 process generated eight tokens at 4.33 tok/s. Its step executable was constructed once, not once per token. Execution dominated:

| First-use O0 stage | Measurement |
|---|---:|
| Graph build/compile/package preparation | 232.41 ms once, inside timed decode |
| Execute | 200.38 ms/token |
| Feed construction | 0.55 ms/token |
| Logits readback | 0.06 ms/token |
| CPU KV readback/update | 0.06 ms/token |
| Greedy top-5 | 0.54 ms/token |
| End-to-end | 230.88 ms/token, 4.33 tok/s |

A package-hit O0 control was already much faster: 39.12 tok/s over 64 tokens. Its execute phase was 22.68 ms/token; package loading still cost 91.44 ms inside generation. This explains the first-use release behavior and confirms that per-token lowering was not occurring. It does not retroactively turn the foundations report's debug M5 number into a comparable M1 release result.

The other suspects were ruled in or out directly:

- The step graph is incremental: query length is always one and attention reads the fixed resident KV bucket. It does not rerun the prefix transformer.
- The old path copied only current K/V outputs, about 112 KiB per token, rather than the full cache. At 0.06 ms/token on the warm control, that CPU copy was not the throughput bottleneck.
- Logits readback is the expected 151,936-float copy and cost about 0.06 ms/token.
- Graph execution, not readback or sampling, is the architectural floor.

## Changes

### Executables are ready before generation

`MetalDecoder::new` now prepares both prefill and step plans. Preparation remains cached per bucket and serialized per pass/bucket, but no graph build, package load, compile, or specialization occurs in `DecodeSession::generate`. Package roots include a canonical graph digest, and the graph revision is now v11 (bumped for the winner-4 GQA broadcast graph so older topologies are cache misses).

O1 is the default for decode. On this large graph, O1 removes the pathological first-dispatch behavior seen after a package miss. O0 remains an attribution control through `SYNAPSE_QWEN3_DECODE_OPT_LEVEL=0`.

### KV stays device-resident

The cache buffers use `MTLResourceStorageModePrivate`. Prefill K/V outputs are exported directly into the full cache buffers on a Metal command buffer. Each step exports its `[1, 8, 1, 128]` K/V outputs to small private staging buffers and blits each head into the addressed cache position. There is no per-token K/V readback or CPU memcpy.

Pause-time inspection still works: it performs an explicit, synchronized blit of the requested layer into a shared staging buffer only when `inspect_cache_layer` is called. The legacy CPU-readback control remains available through `SYNAPSE_QWEN3_DECODE_LEGACY_READBACK=1`.

### Top-k tap rejects losers cheaply

The token tap still receives the same sorted top-k logits before commitment. Once the top-k list is full, a candidate that cannot beat the current worst entry is rejected with one comparison instead of scanning all five entries. On the 20-prompt M1 gate, sampling fell from 0.588 to 0.173 ms/token without changing tie ordering or any generated token.

### LM-head logits stay native f16

The second campaign winner replaces the lm-head's fp32 cast-matmul-cast round trip with the shared f16 `linear` path. The logits are converted to fp32 only after the f16 matmul, preserving the existing sampler input while removing two large conversion operations from every decode step.

### QK^T and PV stay native f16

Winner 3 applies f16 only to the two attention matmuls. Scores are converted to fp32 before scale, causal-mask addition, and softmax; probabilities are converted back to f16 before the PV matmul. This retains the numerical island that feeds the existing fp32 sampler while removing conversion work from the attention GEMMs.

### GQA broadcasts KV inside the attention matmul

Winner 4 removes the `repeat_kv` helper that physically expanded the KV heads to the query-head count before the attention GEMMs. Q is reshaped to `[1, kv_heads, groups, sequence, head_dim]` and K/V to `[1, kv_heads, 1, keys, head_dim]`; the QK^T and PV matmuls then broadcast the size-one KV-group dimension across the `groups` query heads, and the 5-D context is reshaped back to `[1, query_heads, sequence, head_dim]`. The causal mask is reshaped to `[1, 1, 1, sequence, keys]` and softmax runs on the trailing keys axis, so the masking and normalization semantics are identical to the expanded path. Because the broadcast replaces a materialized expansion, attention moves less data; this is a bandwidth reduction rather than the cast elimination of winners 2 and 3. The change sits in the shared attention path after the step/prefill cache concat, so it applies to both the prefill and the step executable, and no `repeat_kv` materialization remains in either.

### Clean package-reload bisect

On the locked M1 running macOS 26.5.2 (build 25F84) immediately after an operator reboot, a clean-cache baseline O1 package prepared in 4.6786909103393555 s and reloaded in 0.1319180727005005 s. A clean-cache full winner-3 O1 package prepared in 5.966507077217102 s and reloaded in 0.13382792472839355 s. Both reloads completed under the 60-second hard timeout. The earlier timeouts came after hard-killed experiments and were machine-state contamination, not a reproducible package-deserializer defect. The M5 Max serialized-winner reload pass is consistent with this clean-M1 result. QK-only and PV-only variants were not needed because the full winner reload passed.

A step-only eager comparison also passed reload, but its preparation time was 2.7091280221939087 s versus 0.13382792472839355 s for the fully serialized winner, a 2.575300097465515 s cold-load penalty with no steady-decode benefit. The final implementation therefore restores serialization for both decode passes. The package-root graph digest remains a hash of the model family and graph-builder revision, so future graph changes necessarily create a cache miss instead of reusing a package built for another topology.

The production package infrastructure in `crates/synapse-engine-owned/src/mpsgraph_runtime.m` continues to use one package per shape. No special decode serialization guard is required from this episode; production should retain the graph-digest discipline when it adopts the same topology.

## Confirmed AC locked-M1 winner data

The table preserves the winner-2 AC control that established the former 73.81 tok/s baseline and records the clean post-reboot winner-3 confirmation.

| Cell | Result |
|---|---:|
| winner-2 AC admission | `AC Power`, internal battery 98%, charging |
| winner-3 AC admission | `AC Power`, internal battery 100%, charged |
| N=12 varied-prompt median | **73.8137 tok/s** |
| N=12 exact prompts | **12/12** |
| winner-2 20-prompt correctness gate decode rate | 72.5642 tok/s |
| winner-3 20-prompt correctness gate | **20/20**, 78.26889942577073 tok/s |
| winner-3 N=12 repeat 1 median | **78.87651018718387 tok/s** |
| winner-3 N=12 repeat 2 median | **78.87032081725204 tok/s** |
| winner-3 combined 24-sample median | **78.87547100540786 tok/s** |
| winner-4 AC admission | `AC Power`, internal battery 100%, charged |
| winner-4 20-prompt correctness gate | **20/20**, 1,280/1,280 tokens, 82.78542872036225 tok/s |
| winner-4 N=12 repeat 1 median | **84.29885065485085 tok/s** |
| winner-4 N=12 repeat 2 median | **84.34879518203019 tok/s** |
| winner-4 combined 24-sample median | **84.31679142003034 tok/s** |

The winner-2 AC median is within 5% of the campaign's approximate battery winner (**73.68 tok/s**) and is **0.18% higher**. The winner-3 20-prompt aggregate rate is lower than its steady N=12 result because it includes all 1,280 tokens in one correctness process. Package preparation remained outside `decode_wall_s`. The winner-4 20-prompt aggregate rate (82.79 tok/s) is likewise below its steady N=12 median (84.32 tok/s) for the same reason. One transient winner-4 N=12 cell measured 51.6 tok/s while the 1-minute loadavg was still settling from the preceding release build; it remained token-exact and the median is robust to it (repeat 2 ran all-clean at loadavg ~1.7-1.8).

The package-miss O1 prime reached 26.5 tok/s while compiling both packages at model load; its generation loop itself contained no compilation. Subsequent package loads took about 185-194 ms total for both executables.

## Correctness and intervention gates

Pinned oracle command:

```sh
MODEL=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
uv run --python 3.12 \
  --with 'transformers==4.51.0' --with 'torch==2.13.0' --with 'accelerate==1.14.0' \
  bench/spikes/unified-rt/reference_qwen3_decode.py \
  --model "$MODEL" --prompts bench/spikes/unified-rt/decode-prompts.jsonl \
  --out target/qwen3-reference-20x64.jsonl --max-new-tokens 64 --top-k-logits 5
```

Candidate result on the locked M1 through the hardened campaign gate:

- exact prompts: **20/20**;
- exact generated tokens: **1,280/1,280**;
- accepted near ties: **0**;
- token-tap rows: **1,280**;
- cache path: `device-resident-blit`;
- optimization level: `1`;
- correctness-gate decode rate: **72.5642 tok/s** on AC power.

Hook tests:

```text
qwen3_decode::tests::token_stream_tap_observes_before_commit_without_changing_tokens — passed
qwen3_decode::tests::paused_state_resumes_to_uninterrupted_tokens — passed
qwen3_decode::tests::splice_matches_prefilling_the_concatenated_sequence — passed
qwen3_decode::tests::addressable_weight_regions_are_byte_identical_across_loads — passed
qwen3_decode::tests::greedy_argmax_uses_lowest_token_id_for_exact_ties — passed
```

The token tap, pause/resume state, forced splice, addressable weights, and on-demand cache inspection remain intact. O1 and the GPU cache path produced the same 64-token first-prompt sequence as the O0 CPU-update control before the full oracle gate was run.

Regression gates on AC power also passed: `cargo test -p spike-unified-rt` reported **54 passed, 4 ignored, 0 failed**, and the pinned constrained-decode fixtures reported **15/15 valid prompts and 647 generated tokens** with `--verify-decode-cache`. The unconstrained 20-prompt gate remained byte-for-byte token exact.

## llama.cpp Metal comparison

The refresh closes both comparison gaps. On the locked M1, the M1-native `llama-cli`
used 12 prompts from the fixed stride-seven schedule, repeated twice with a
fresh process per prompt. Prompt text changed on every iteration; generation
was greedy (`--temp 0 --top-k 1 --top-p 1`), single stream, `-n 64`, `-ngl 99`,
`-c 512`, and `--single-turn`. Each repeat acquired and promptly released
`[bench-user-home]/bench.lock` after AC-power and no-`Runner.Worker` admission.
The values below are llama-cli's generation-rate timings, not process wall time.

| Runtime | Storage | Repeat 1 median (spread) | Repeat 2 median (spread) | Combined 24-sample median (range) |
|---|---|---:|---:|---:|
| owned MPSGraph winner 4 | f16 | 84.32 tok/s N=12 confirmation | 82.79 tok/s 20-prompt gate | — |
| llama.cpp Metal (`llama-cli` b9580) | Q8_0 | **207.45 tok/s (201.60–208.20)** | **207.40 tok/s (200.40–207.80)** | **207.40 tok/s (200.40–208.20)** |
| llama.cpp Metal (`llama-cli` b9580) | f16 | **180.30 tok/s (176.60–180.50)** | **180.05 tok/s (177.50–181.90)** | **180.15 tok/s (176.60–181.90)** |
| owned / llama.cpp | f16 vs f16 | — | — | **46.8%** |
| owned / llama.cpp | f16 vs Q8_0 | — | — | **40.7%** |

The earlier `llama-server` Q8_0 control remains useful historical context at
190.36 and 203.45 tok/s, but it is not substituted for either fresh CLI row.
The f16 cell is now a like-for-like storage-precision reference for the owned
f16 decoder, and the installed CLI/model paths are retained for future campaign
lanes.

## Next levers

1. **Replace the large MPSGraph step with a lower-dispatch Metal path.** Graph execution remains the dominant measured cost; the native f16 linear removed a major conversion round trip, but a lower-dispatch path is still the next architectural lever.
2. **Persist dynamic feed wrappers and small input buffers.** Feed construction costs about 0.58 ms/token, but this is secondary to execution.
3. **Device-side top-k with a small tap readback.** Full logits readback plus CPU top-5 costs about 0.23 ms/token. Any device sampler must still expose immutable top-k values before commitment.
4. **Write step K/V directly into addressed cache slices.** The current GPU export-plus-blit path costs about 0.21 ms/token. A custom kernel or supported aliasing output could remove staging without weakening inspection/splice semantics.
5. **Use the fresh llama-cli controls as the comparison baseline.** The f16-to-f16 cell is now measured at 180.15 tok/s, while the Q8_0 control is 207.40 tok/s; neither should be mixed with the historical server-only row.

The measured ceiling is now architectural rather than hidden graph rebuilding, fp32 linear conversion, or CPU/GPU cache traffic. Further large gains require reducing MPSGraph execution cost or replacing the step graph, not tuning the controller hooks away.
