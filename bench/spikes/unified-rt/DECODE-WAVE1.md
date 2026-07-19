# Qwen3 decode throughput wave 1

## Verdict

The owned Qwen3-0.6B f16 Metal decoder reaches **78.31 tok/s** for one 64-token stream on the locked M1 Max after the winner-3 attention change. The campaign's final 20-prompt gate remains token-exact against Transformers CPU fp32: **20/20 prompts, 1,280/1,280 tokens, zero near-tie exemptions**.

The foundations report's 3.2 tok/s result used a debug binary on a contended M5, so it is context rather than a locked release baseline. Instrumented release controls on the M1 showed that the step graph was already cached, but executable preparation and a very slow package-miss first dispatch occurred inside the first timed generation call. Wave 1 now prepares both bucket executables at model load, uses O1 for this large decode graph, keeps KV updates on the GPU, and avoids most rejected-candidate work in the CPU top-k tap. Winner 2 keeps the lm-head logits matmul in f16 and casts only its result to fp32. Winner 3 also keeps QK^T and PV attention matmuls in f16 while retaining the scale, mask, and softmax island in fp32.

This clears the `>=40 tok/s` bar with margin. It does not approach llama.cpp: the same-day llama.cpp Metal control produced 190.36 and 203.45 tok/s, so the winner is 31.0% and 29.0% of that control. The available control was `llama-server` with the official Q8_0 GGUF, not the requested `llama-cli` f16 cell; see the comparison caveat below.

## Locked-M1 setup

| Item | Value |
|---|---|
| Host | `[bench-host]`, Apple M1 Max |
| Model | `Qwen/Qwen3-0.6B`, safetensors f16 storage |
| Cache | one stream, bucket 512 |
| Decode | greedy raw completion, 64 generated tokens |
| Graph | one full-bucket prefill executable and one query-length-1 step executable |
| Compilation | `MPSGraphOptimizationLevel1`, two eager decode executables per bucket; decode-package serialization disabled |
| Lock | `mkdir [bench-user-home]/bench.lock`; `/tmp/aft-measure.lock` absent; no `Runner.Worker` |

The release binary was built in `[bench-user-home]/ck-campaign/workspaces/mason-winner-2/target/release/spike-unified-rt`. The hardened campaign harness SHA-256 was `008d43490e9504bd420bee96823b950e387ea90a1a81f46e20e9890980741a9c`. Timed cells ran only while the benchmark lock was held. AC power was confirmed before admission with `pmset -g batt`: `Now drawing from 'AC Power'`, internal battery **98%**, charging.

## Winner provenance

Campaign `[consult-id]` banked winner-2 patch (`8c6da25c9dd5431acc8e632d9cc675520e75ae5cf85b0931d613f867896d9740`). It applied cleanly to the current file, which already contains winner 1; no hand port was needed. Its mechanism is **lm-head logits fp32 cast round-trip removal**: the f16 `linear` result is cast to fp32 only after the matmul, rather than casting the selected activations and lm-head weights to fp32 before the matmul.

Campaign `[consult-id]` banked winner 3. Its attention change leaves QK^T and PV matmuls in f16, casts QK^T scores up before scale/mask/softmax, then casts probabilities down before PV. The campaign reported **78.31 tok/s** for the controlled single-prompt cell and preserved the 20-prompt token-exact gate.

The cumulative decode arc is **40.55 -> 59.03 -> 73.81 -> 78.31 tok/s**: the first number is the frozen pre-wave baseline, the second is winner 1's AC confirmation, the third is winner 2's AC confirmation, and the fourth is campaign `[consult-id]` winner 3. The prior campaign's battery winner was approximately **73.68 tok/s**; the 73.81 AC confirmation remained within 5%.

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

`MetalDecoder::new` now prepares both prefill and step plans. Preparation remains cached per bucket, but no graph build, package load, compile, or specialization occurs in `DecodeSession::generate`. Both winner-3 decode plans deliberately compile eagerly at load because their serialized packages can wedge during deserialization. Package roots include a canonical graph digest, and the graph revision is now v10.

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

### Serialized decode-package guard

On the locked M1 running macOS 26.5.2 (build 25F84), freshly built O1 packages from both the baseline fp32 attention graph and the full winner-3 f16 QK^T/PV graph blocked the next process during MPSGraph package deserialization and required a hard kill after 60 seconds. The failure therefore is not stale cache identity or unique to the winner topology: it is a Qwen3 decode-package deserializer failure on this M1 configuration. The M5 Max running the same OS build successfully reloaded the serialized winner package, so the failure is hardware-specific or another M1 runtime distinction. QK-only and PV-only variants are no longer needed to decide the integration guard.

`MetalExecutionConfig::decode_package_path` returns no package path for either Qwen3 decode pass. The Objective-C builder therefore compiles both plans eagerly at model load and never reads or writes decode packages. This moves decode compilation to cold load without changing steady decode execution. The package-root graph digest is a hash of the model family and graph-builder revision, so a future graph change necessarily creates a cache miss instead of reusing a package built for another topology.

The affected production package infrastructure is `crates/synapse-engine-owned/src/mpsgraph_runtime.m`, which also reads and writes one package per shape. The engine-owned Qwen3 graph in `crates/synapse-engine-owned/src/qwen3_mpsgraph.m` does not yet share this decode-step topology. If production adopts this f16 QK^T/PV decode graph, it must disable its decode-package I/O and retain the graph-digest discipline before enabling serialization.

## After: AC locked-M1 winner data

The table preserves the winner-2 AC control that established the former 73.81 tok/s baseline. Winner 3's measured locked-M1 confirmation and cold-load comparison are recorded with the campaign evidence; its serialized-decode reload safeguard is described above.

| Cell | Result |
|---|---:|
| AC admission | `AC Power`, internal battery 98%, charging |
| N=12 varied-prompt median | **73.8137 tok/s** |
| N=12 exact prompts | **12/12** |
| 20-prompt correctness gate decode rate | 72.5642 tok/s |

The AC median is within 5% of the campaign's approximate battery winner (**73.68 tok/s**), and is **0.18% higher**. The 20-prompt gate's aggregate rate is lower because it includes all 1,280 tokens in one correctness process; it is not the steady single-stream number used for the throughput result. Package preparation remained outside `decode_wall_s`.

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

## Same-day llama.cpp comparison

The M1 had llama.cpp's Metal-enabled `llama-server` and the official `Qwen3-0.6B-Q8_0.gguf`, but no `llama-cli` binary and no f16 generation GGUF. The official repository URL for `Qwen3-0.6B-F16.gguf` returned 404, so a same-precision cell could not be fabricated. The server used the same llama.cpp Metal backend and a five-token raw prompt, with `n_predict=64`, temperature zero, and no prompt cache request.

| Runtime | Storage | Run 1 | Run 2 |
|---|---|---:|---:|
| owned MPSGraph winner 3 | f16 | 78.31 tok/s campaign cell | 20/20 token-exact gate |
| llama.cpp Metal (`llama-server`) | Q8_0 | 190.36 tok/s | 203.45 tok/s |
| owned / llama.cpp | mixed precision | 31.0% | 29.0% |

This is a useful same-day backend ceiling but not the requested f16-to-f16 `llama-cli` ratio. A future certification should add an f16 GGUF and `llama-cli` to the locked image rather than relabeling the available Q8 server result.

## Next levers

1. **Replace the large MPSGraph step with a lower-dispatch Metal path.** Graph execution remains the dominant measured cost; the native f16 linear removed a major conversion round trip, but a lower-dispatch path is still the next architectural lever.
2. **Persist dynamic feed wrappers and small input buffers.** Feed construction costs about 0.58 ms/token, but this is secondary to execution.
3. **Device-side top-k with a small tap readback.** Full logits readback plus CPU top-5 costs about 0.23 ms/token. Any device sampler must still expose immutable top-k values before commitment.
4. **Write step K/V directly into addressed cache slices.** The current GPU export-plus-blit path costs about 0.21 ms/token. A custom kernel or supported aliasing output could remove staging without weakening inspection/splice semantics.
5. **Re-run a true f16 llama-cli cell.** Install the matching binary and f16 GGUF on the M1 before using the comparison as a graduation ratio.

The measured ceiling is now architectural rather than hidden graph rebuilding, fp32 linear conversion, or CPU/GPU cache traffic. Further large gains require reducing MPSGraph execution cost or replacing the step graph, not tuning the controller hooks away.
