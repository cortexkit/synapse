# LFM2 small-model Metal decode baselines

**Measurement date:** 2026-07-20
**Rig:** locked `<bench-host>` / Apple M1 Max (`MacBookPro18,2`, 10
logical CPUs, 64 GiB), macOS 26.5.2 build `25F84`. All timed cells ran on AC
power at 100% charged, under `$SYNAPSE_BENCH_ROOT/bench.lock`; no campaign, llama, or
rsync process was present. The one-minute load average was recorded for every
sample and stayed below the `2.5` admission threshold during timed cells.

## Protocol

Owned rows use the release `spike-unified-rt` binary, Metal MPSGraph, f16
resident weights, explicit execution, current `GRAPH_REVISION=11`, and a fresh
bucket-512 package-cache root per model. Each timed sample is a fresh process;
there are 12 varied raw-completion prompts, greedy sampling, a 64-token cap, and
two repeats. The displayed decode value is the median of all 24 per-prompt
rates. Natural EOS is retained rather than padded. The 512-token prefill cell
uses one exact 512-token prompt, `max_new_tokens=1`, and bucket 1024 so the
prompt itself fits.

LFM2-350M and LFM2-700M use a ten-prompt CPU fp32 Transformers oracle for the
family smoke gate. LFM2-1.2B uses the pinned twenty-prompt oracle from
`LFM2-BACKBONE.md`. The CPU oracle was generated with Transformers `5.12.1`,
Torch `2.12.0`, fp32 arithmetic, greedy generation, and 64 new tokens. The
LFM2.5 quick checks use the first ten rows of the same prompt file.

llama.cpp uses the official LiquidAI GGUF files, build `9580`
(commit `b4e3dc613`), the relocated M1 `llama-server` wrapper, one slot,
`--no-cont-batching`, Flash Attention on, context 1024, batch 1024, and
`cache_prompt=false`. It was a raw `/completion` server, not a chat wrapper.
Each cell had one uncounted warmup request and then the same 12 prompts for two
repeats. llama.cpp rows report eval/decode rate only; its server's prompt timer
is not used as an owned prefill comparison.

## Owned-Metal and llama.cpp rows

`—` means the row was intentionally not timed. `n/r` means no comparable
measurement was produced. The owned f16 quality result is shown separately from
the timed rate because f16 arithmetic can fork at rounded logit ties even when
the f32 family gate is exact.

| Model | Stack / dtype | Decode tok/s | Prefill tok/s | Cold load | Correctness / note |
|---|---|---:|---:|---:|---|
| LFM2-350M | owned Metal f16 | **10.45** | **10.18** | **0.417 s** | 10/10 exact, hidden min cosine 0.999953 |
| LFM2-350M | llama.cpp Metal F16 | **304.05** | — | **0.523 s** | official GGUF |
| LFM2-350M | llama.cpp Metal Q8_0 | **355.31** | — | **0.516 s** | official GGUF |
| LFM2-700M | owned Metal f16 | — | — | — | dropped after f16 gate: 9/10 exact; hidden cosine 0.999889 |
| LFM2-700M | llama.cpp Metal F16 | **186.19** | — | **0.515 s** | official GGUF |
| LFM2-700M | llama.cpp Metal Q8_0 | **257.54** | — | **0.515 s** | official GGUF |
| LFM2-1.2B | owned Metal f16 | **6.35** | **6.36** | **1.186 s** | campaign baseline; f16 diagnostic 17/20 exact, cache 18/20 |
| LFM2-1.2B | llama.cpp Metal F16 | **130.74** | — | **0.516 s** | official GGUF |
| LFM2-1.2B | llama.cpp Metal Q8_0 | **203.65** | — | **0.515 s** | official GGUF |

The owned f16 medians are the combined 24-sample medians. Per-repeat medians
were 10.439 and 10.461 tok/s for 350M, and 6.345 and 6.345 tok/s for 1.2B.
The confirmed 1.2B campaign baseline is therefore `6.345058617091391` tok/s.
The 512-token prefill cells measured 10.184 tok/s (350M) and 6.364 tok/s
(1.2B); the varied-prompt decode runs' prefill fields are not substituted for
those values.

The 700M f32 family gate itself passed 10/10 exact with hidden cosine 1.0 and
cached-vs-reprefill 10/10. Its f16 gate exposed a rounded tie on
`completion-03` and a hidden-state cosine below the 0.9999 threshold, so it was
not promoted into a misleading timed owned-f16 row. This is a precision gate
failure, not a family-detection or config-dimension failure; no loader change
was made.

The 1.2B f32 gate passed 20/20 prompts, 1,280/1,280 tokens, hidden minimum
cosine 1.0, and cache verification 20/20. The f16 diagnostic retained the
known near-tie behavior: 17/20 token-exact, hidden minimum cosine 0.99999087,
and 18/20 cached-vs-full-reprefill. The harness consequently runs the exact f32
fixture gate and measures the registered f16 objective separately.

## LFM2.5 family-scope check

The current LFM2.5 configs report `model_type: "lfm2"`, use the same 16-layer
hybrid conv/full-attention layout, and the 1.2B-Instruct checkpoint loaded
without a runtime code change. The 1.2B-Instruct ten-prompt f32 gate passed
10/10, hidden cosine 1.0, and cache verification 10/10. Its f16 quick gate was
9/10 exact with hidden minimum cosine 0.99997069; a two-repeat ten-prompt owned
f16 smoke median was **6.33 tok/s** (20 samples), with one tie-driven outlier
sequence. The existing campaign remains pinned to LFM2-1.2B because its
fixtures are token-exact and already stable; LFM2.5-1.2B-Instruct should only
replace it after a new instruct-format fixture set is cut and its f16 quality
policy is decided.

| Model | Stack / dtype | Decode tok/s | Prefill tok/s | Cold load | Correctness / disposition |
|---|---|---:|---:|---:|---|
| LFM2.5-350M | owned Metal f16 | — | — | — | skipped: config omits `rope_theta`; loader rejected it before inference |
| LFM2.5-350M | llama.cpp Metal F16 | **304.21** | — | **0.518 s** | official GGUF; owned loader incompatibility noted above |
| LFM2.5-350M | llama.cpp Metal Q8_0 | **360.83** | — | **1.075 s** | official GGUF |
| LFM2.5-1.2B-Instruct | owned Metal f16 | **6.33** | 5.7* | **1.645 s** | ten-prompt quick smoke; f32 gate exact, f16 9/10 |
| LFM2.5-1.2B-Instruct | llama.cpp Metal F16 | **132.58** | — | **1.532 s** | official GGUF |
| LFM2.5-1.2B-Instruct | llama.cpp Metal Q8_0 | **205.74** | — | **1.027 s** | official GGUF |

`*` The LFM2.5-1.2B-Instruct prefill value is the ten-prompt quick-gate
observation, not the exact 512-token prefill cell used for the locked LFM2
campaign baseline.

The 350M failure is specifically `missing field rope_theta`, not a simple
model-type detection mismatch. Per scope, it was not patched. The LFM2.5 llama
cells were measured with the same build and server protocol: cold loads were
0.518 s / 1.075 s for 350M F16/Q8 and 1.532 s / 1.027 s for 1.2B-Instruct
F16/Q8. The 350M owned row remains skipped, but its official GGUF comparison is
still useful because llama.cpp accepts the published metadata.

## Snapshot and artifact provenance

Snapshot identifiers are Hub commit SHAs. Weight and GGUF hashes are the
SHA-256 values represented by the cache blob names.

| Artifact | Snapshot / repository revision | safetensors or GGUF SHA-256 |
|---|---|---|
| `LiquidAI/LFM2-350M` | `b3afba27815ee83a64b76162cef4d8a4780d6ca7` | `387638dc889ff1a1395c3c2ab9605211e4c7e16f2d375361dd4e423b909a254e` |
| `LiquidAI/LFM2-700M` | `e6b5a31428dc6874081a7b187eeaa307d0d6fc82` | `a7b52669217ecb538740187f41ce2a5802afa1c8d81d8d153b847bbd21d1bdda` |
| `LiquidAI/LFM2-1.2B` | `933cee00d754fb3bfe06c644c0cb95453f2d8bb2` | `60fef6ef4481c533ce7427793bed50200b55b3c68d0d00c52bc56f207a9acecd` |
| `LiquidAI/LFM2-350M-GGUF` | `8fdc9d526b7ed346b19257551b05816c7912ecc2` | F16 `379ffdcbf08147c0313f6f1ce7ff558a2bc935eda633f4b46c52347032419c42`; Q8 `b7bfeab6495a1ae3ae78811c1297df9f301b35261ff9580d42fb30dc4dc9034b` |
| `LiquidAI/LFM2-700M-GGUF` | `43e05b4efd464155b3807bde379942bb43d8ee3c` | F16 `e51cf86c25c2c96e157a3882ae3acd6462cac8cab40cf1c72e830cdbefc3456b`; Q8 `0967d902ed270d07cb374a1972bb5da4338d2a303b565cf4eb7948b958e751ab` |
| `LiquidAI/LFM2-1.2B-GGUF` | `5399e76c648f4eb8c053feb1ab747277dea5bf8b` | F16 `0ddedfb8c5f7f73e77f19678bbc0f6ba2554d0534dd0feea65ea5bca2907d5f2`; Q8 `0d9ec100a0f33048168d1d5b9fb6403f4836adcbbe9c3f2ab7794c96ffee3c3b` |
| `LiquidAI/LFM2.5-350M` | `b9d6e4e2d75f440b12a2b4d731c808004ecbbd89` | `1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e` |
| `LiquidAI/LFM2.5-1.2B-Instruct` | `868df74dd56ff8a0c2ac5dbf281690c2dbebe4c9` | `1ba63d9adb03ae43581db0e136e4416febe0441aff7296397bd455fb6017f73a` |
| `LiquidAI/LFM2.5-350M-GGUF` | `bb7ee58b243e4cede04187e323e760b04f8a0091` | F16 `18e2f6b61045bed8c3e8575c01fa7898b2df5ae884964322dcf49e5d12f6eb79`; Q8 `be036a757295e550098b85e13f6af2735d0fa73b41e1156a40c7d8e8e32a5766` |
| `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | `047e06635fbe71469926b35ea414537245218200` | F16 `1e1d8a5ca01c0f1ee51a6fd729c80efd626f54812a1241358bea20824fea790d`; Q8 `f6b981dcb86917fa463f78a362320bd5e2dc45445df147287eedb85e5a30d26a` |

## Campaign steering seed

The most transferable Qwen3 mechanism is **fused residual plus norm: yes**. LFM2
has the same repeated normalized-mixer/MLP residual structure, so a fused
residual-and-RMSNorm kernel can remove intermediate traffic around both the ten
short-convolution blocks and the six attention blocks, provided the convolution
state update remains an explicit ordering boundary. **Warp-per-key attention
applies only to the six full-attention layers**, indices `[2, 5, 8, 10, 12,
14]`; the ten convolution layers have no attention keys and should not pay for
that specialization. **GQA broadcast is N/A on the current Metal path**: the
architecture is 32 query heads over 8 KV heads, but `lfm2.rs::causal_gqa`
selects `kv_head = query_head / groups` and copies each selected KV stream into
per-head scratch before provider matmuls. There is no graph-level `repeat_kv`
tensor for the Qwen3 broadcast winner to eliminate; changing this would be a
new LFM2-specific Metal attention design, not a mechanical transfer.

The five existing hook tests are mechanically shared through the
`qwen3_decode` controller, and the harness still runs them. They currently
instantiate Qwen3 fixtures and are therefore **Qwen3-bound rather than a true
LFM2 family-parametric quality gate**. LFM2's own decode tests cover cached
hybrid-layer continuation and incremental/full hidden-state parity; a future
fixture cut should migrate the five intervention tests to a family-neutral
fixture before treating `hooks_passed` as an LFM2 certification claim.

## Reproduction pointers

```sh
MODEL=$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2-1.2B/snapshots/933cee00d754fb3bfe06c644c0cb95453f2d8bb2
uv run --python 3.12 \
  --with 'transformers==5.12.1' --with 'torch==2.12.0' --with 'accelerate==1.14.0' \
  bench/spikes/unified-rt/reference_lfm2.py \
  --model "$MODEL" --prompts bench/spikes/unified-rt/decode-prompts.jsonl \
  --tokens-out target/lfm2-reference-tokens.jsonl \
  --hidden-out target/lfm2-reference-hidden.jsonl --max-new-tokens 64

cargo build --release -p spike-unified-rt
```

The campaign registration is
`lfm2-1.2b-f16-single-stream-decode`; its controller embeds the fixture hashes,
checks the model content digest
`afd99d6cc2a5a6ff6c57ceca2d03f1f73d58d31f3528eadca3035f4164a2009d`, runs the
f32 exactness gate, executes the shared hook suite as a diagnostic, and measures
f16 samples with the confirmed `6.345058617091391` tok/s baseline.
