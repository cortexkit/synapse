# LFM2 and LFM2-Audio on the owned CUDA backend

## Scope and result

This port adds an fp32 resident CUDA path for the hybrid LFM2 causal backbone and
its convolution/KV decode cache. It also routes every learned dense operation in
the LFM2-Audio FastConformer and projector through a persistent CUDA cuBLASLt
context. The existing shared Rust implementation still owns audio DSP,
depthwise convolution, normalization, relative-score assembly, softmax, and
pointwise activations, matching the division used by the Metal correctness path.

The pinned checkpoints were:

- `LiquidAI/LFM2-1.2B@933cee00d754fb3bfe06c644c0cb95453f2d8bb2`,
  weight SHA-256
  `60fef6ef4481c533ce7427793bed50200b55b3c68d0d00c52bc56f207a9acecd`;
- `LiquidAI/LFM2-Audio-1.5B@c798aad30dc3cd72e72970beab51326b8443bd94`,
  weight SHA-256
  `d0cae5b6a1cbc308472535d6fa310fe446bb9ea601934a14db2366040a9fa129`.

All required fp32 gates passed. CUDA f16 was not attempted: the new resident
state path intentionally establishes the fp32 exactness baseline first and does
not yet contain a mixed-storage implementation.

## Rig and method

The measurement rig was vast.ai contract `45249270`: one RTX 4090 24 GiB (SM
8.9), NVIDIA driver `595.58.03`, CUDA toolkit `12.6` (`nvcc 12.6.85`), and 32
effective AMD EPYC 7B13 cores. The driver exceeded the required 570 floor,
`nvidia-smi pmon` showed no other compute process before the work, and both
checkpoint hashes were verified on-box before inference. The instance rate,
including the 160 GB disk allocation, was `$0.437037/h`.

The runtime was built release-mode with `--features cuda`. LFM2 measurements use
strict fp32 storage, fp32 cuBLASLt operands/accumulation, custom fp32
normalization, RoPE, short-convolution, attention, and cache-update kernels, and
uncaptured launches (`--cuda-graphs false`). The N=12 throughput row uses twelve
varied rows from `decode-prompts.jsonl`, each capped at 64 generated tokens. An
untimed full-hidden comparison for the same shape populated persistent weights
and shape plans before each timed prefill, so the row is a warm serving-rate
measurement rather than cold model upload.

## Correctness gates

| Gate | Result |
|---|---|
| LFM2 final hidden states, CUDA fp32 vs pinned Transformers fp32 | Passed every position on 20/20 prompts; minimum reported cosine `1.00000000` (threshold `0.9999`). |
| LFM2 greedy decode, CUDA fp32 | **20/20 token-exact**, up to 64 generated tokens. |
| CUDA cached decode vs full re-prefill | **20/20 token-exact**. Full re-prefill does not mutate the resident decode cache. |
| LFM2-Audio mel frontend | **20/20**; global max absolute error `0.00017464161`, below `1e-3`. |
| LFM2-Audio FastConformer + projector, CUDA dense path | Passed every frame on **20/20** clips; global minimum cosine `0.999999999967199`, above `0.9999`. |
| LFM2-Audio greedy ASR, CUDA fp32 | **20/20 token-exact**, including terminal token 7. |
| Fresh audio-reference sanity gate, CPU/faer fp32 | **20/20 token-exact**; encoder minimum cosine `0.9999999999547785`. |
| Existing CUDA unit suite | 43 passed, 0 failed, 4 fixture-dependent tests ignored by the unfiltered run. |
| Qwen3 embedding persistent-feed regression | Passed `qwen3_cuda_static_feeds_survive_multiple_calls`. |
| ModernBERT persistent-feed regression | Passed `modernbert_cuda_static_feeds_survive_multiple_calls`. |
| MiniLM CUDA family graph smoke | Passed an eight-row, 928-real-token batch; captured output matched uncaptured output exactly. |

The text gate regenerated references with Python 3.12.12, Transformers 5.12.1,
and the `torch==2.12.0` distribution. That CUDA wheel reports the local build
suffix `2.12.0+cu130`; `reference_lfm2.py --allow-version-drift` was needed only
because its string equality check does not strip that suffix. The model revision,
Transformers version, base torch distribution version, checkpoint dtype, and
fp32 oracle arithmetic were otherwise unchanged. Reference token and hidden
JSONL SHA-256 values were
`521baaa6834bdfbe8297723c7ac8c24a8e718b1310ac29196436e8bb0a0c35c8` and
`85c133007be1053d0a15c5a7842982f3a3297854ed564b0c2e4bef6f33175687`.

## Throughput

These are correctness-executable wall rates, not a continuous-batching serving
claim. Historical CPU and Metal values are copied from `LFM2-BACKBONE.md`; that
document identifies an M5 Max development host, so no M1-specific number is
available to substitute.

| Backend | dtype | Prefill tok/s | Single-stream decode tok/s | Source |
|---|---:|---:|---:|---|
| **CUDA RTX 4090** | fp32 | **1,710.6** | **178.49** | Warm N=12 varied prompts, 64-token cap |
| CUDA RTX 4090 | f16 | not implemented | not implemented | Deferred until a mixed-storage state/cache path has its own exactness policy |
| Metal MPSGraph | fp32 | ~6.2 | ~5.8 | Historical correctness observation |
| Metal MPSGraph | f16 | ~6.5 | ~5.7 | Historical diagnostic; only 17/20 token-exact |
| CPU/Accelerate | fp32 | ~15.8 | ~14.0 | Historical correctness observation |

The independent 20-prompt CUDA correctness run measured 1,725.2 prefill tok/s
and 178.38 decode tok/s. For the 20 audio clips, CUDA spent 1.861 s in the
frontend/encoder and 1.631 s in LFM2 prefill/decode. The fresh CPU/faer sanity
run took 3.550 s and 64.409 s respectively. Historical Metal fp32 totals were
2.97 s and 149.81 s on the previous fixture generation.

## Regenerated audio fixture provenance

The original `target/` fixtures were intentionally not tracked, so this run
regenerated the designed corpus before touching the CUDA gate. The generator
has no random-number path or seed. Its complete generation coordinates were
`prepare_lfm2_audio_clips.py`, voice `Samantha`, rate `180`, the script's ordered
20-sentence constant, and the host `say` plus `afconvert` implementation. Thus
the generator seed is **not applicable (no RNG)**. The manifest SHA-256 was
`011a868c532fad721342268f6f92a4b877fc394b5f72f2ba4e848d359e89aac8`.

The oracle used Python 3.12.12 and the exact
`lfm2-audio-reference-requirements.txt` pins: liquid-audio 1.2.0 at
`84cdb243859aaa53db660bc3f4718b54133336bd`, torch 2.13.0, torchaudio 2.11.0,
Transformers 5.13.1, Accelerate 1.14.0, librosa 0.11.0, NumPy 2.4.6, and
soundfile 0.14.0. Source revision and checkpoint hash checks ran without skip
flags. The resulting reference JSONL SHA-256 was
`cae1bd9d1d80e24fdf36a83532790da7821fae0476027352b3a1d69e7fe39b36`.
The shipped CPU path then passed 20/20 against those bytes before the CUDA path
used them.

| Clip | WAV SHA-256 |
|---|---|
| `say-01` | `904c039735a6eec4186bed49369408638421937fb02104729afafea8c41b266d` |
| `say-02` | `f8a01750735c9694bbf2a7ccaf9cfed421dc1c4e97e48e1ff61b410b94a1bb6d` |
| `say-03` | `56790c487cf62facee883bad1bbb2a73a13ecd18b77005858436fb738a0fad61` |
| `say-04` | `dd429518c95e8a99ae60246e9cc89eda487af3cd614d493c982769138394012e` |
| `say-05` | `20e2533a9fb31e8a90b27f6adf742cd0c5cf26fde9eb0a3c2ecf93c9eae10c97` |
| `say-06` | `f12614680b99c535af92c34371d8866b4d492194c521be96578605942dd77f83` |
| `say-07` | `b39631c7e6b5b8983acfe8c71d6a06466f22f8a6eae8f590bc851899d273ff68` |
| `say-08` | `b87c419b2f675c870c394deabe17deba13d8e9ce784176b335f3f4d96cc43ba6` |
| `say-09` | `9547f3a87142eded44f1901fbe32713399cc1e36afbabe40b2eac0a4126f7762` |
| `say-10` | `4322a8228f3c4fe518b20b8ac5e6656bbab6e1ff5efd395eb4ede8d4d4d9f300` |
| `say-11` | `9a1cc7072682bf7756fdc7ea20192f47fa19c1c33aa83d67b0986c71cfc1f456` |
| `say-12` | `1c996b179931e37e990f226da7d47d440838f1e7c7e57148b9359c1f5233c23b` |
| `say-13` | `3582505664b64564853f4ea14ad61bff2daf573d5c8e1035ebf4a8228fa1334e` |
| `say-14` | `fb8d9208361727d41a81229ec89894c0fad9c327083f0dd59fc0c52f747e7bd4` |
| `say-15` | `140e649c08768fc6315f1ba899ba8fbc938c40b39c6849d54af44c319a0eafed` |
| `say-16` | `58df2367b784c355b4c1ba97d518d36f755893a760953ba0343652ca69e8f798` |
| `say-17` | `694b02f5b67d8b4c045f4c5cfc58cecb90bfc2206378e1b1c32d530e6c9ddd05` |
| `say-18` | `0c5279d0ee948bb69a8af16fe6b1135da168307bb9bb3f99801d6cef4305a6d5` |
| `say-19` | `ada31b95b940363aa5171d3f8e193679e633226e118c31c76ec1bf0b5958292b` |
| `say-20` | `c9d7ebbc33b303ab4473f5c38174786896d65c24d2ab53f7834de8be9198ae1e` |

## Implementation and traps

1. **Prefill must populate both state families.** The resident prefill executes
   the full sequence once, saves the final three `B*x` values for every short
   convolution layer, and copies normalized/RoPE-applied K plus V into each
   attention layer's append-only cache. Decode validates the expected absolute
   position before modifying either state family.
2. **Full re-prefill must not touch decode state.** Hidden/token reference checks
   use the same resident weights and kernels but separate full-sequence
   workspaces. This made the cached-vs-reprefill gate meaningful instead of
   accidentally resetting the cache under test.
3. **GQA uses query-head geometry.** The causal and decode attention kernels
   launch one block per query head (and per query position for prefill), map each
   query head to `query_head / (query_heads / kv_heads)`, and never materialize a
   repeated KV tensor.
4. **Nonblocking-stream weight uploads need explicit ordering.** The first audio
   implementation copied a newly encountered static RHS with a synchronous
   default-stream `cudaMemcpy` and immediately consumed it on a nonblocking
   inference stream. Pageable host staging allowed the streams to race; one
   4096-to-512 pre-encoder result visibly changed and ASR diverged. Static RHS
   uploads now use `cudaMemcpyAsync` on the inference stream. LFM2's bulk
   default-stream model upload uses `cudaDeviceSynchronize` before the first
   nonblocking launch. This was the only first-token corruption found.
5. **The host cache remains a control mirror on CUDA.** Rust owns capacity and
   absolute-position checks exposed through the shared decode interface; large
   convolution/KV values stay resident and are not copied back per token.
6. **No negative branches were retried.** The port does not use direct BSH
   views, fused QKV, or algorithm enumeration. It retains row-major cuBLASLt
   plans and 256-byte-aligned CUDA allocations, and the GQA kernels are sized
   from query width rather than hidden-width assumptions.

CUDA graph capture and f16 storage remain follow-up work. When an LFM2 workload
is selected, the CLI reports that graph capture is unavailable and constructs
an uncaptured CUDA provider rather than labeling ordinary launches as captured.

## Reproduction

```bash
MODEL=/path/to/933cee00d754fb3bfe06c644c0cb95453f2d8bb2
AUDIO_MODEL=/path/to/c798aad30dc3cd72e72970beab51326b8443bd94

cargo build --release -p spike-unified-rt --features cuda

# Text, hidden-state, token, and cache gate.
target/release/spike-unified-rt \
  --model "$MODEL" --tokenizer "$MODEL/tokenizer.json" \
  --generate-prompts bench/spikes/unified-rt/decode-prompts.jsonl \
  --decode-reference target/lfm2-reference-tokens.jsonl \
  --decode-hidden-reference target/lfm2-reference-hidden.jsonl \
  --max-new-tokens 64 --decode-cache-bucket 512 --decode-top-k 5 \
  --verify-decode-cache --device cuda --dtype f32 --cuda-graphs false \
  --out target/lfm2-owned-cuda-gate.json

# Audio gate, after the documented CPU sanity run against the same reference.
target/release/spike-unified-rt \
  --model "$AUDIO_MODEL" --tokenizer "$AUDIO_MODEL/tokenizer.json" \
  --asr-audio target/lfm2-audio-inputs.jsonl \
  --asr-reference target/lfm2-audio-reference.jsonl \
  --max-new-tokens 64 --decode-cache-bucket 512 \
  --device cuda --dtype f32 --cuda-graphs false \
  --out target/lfm2-audio-owned-cuda.json
```

## Spend

The contract was destroyed immediately after final verification; the post-destroy
instance list was empty. The final Vast account-ledger delta was `$0.2176`.

- Runtime: `0.5161 h` (about 31.0 minutes)
- Total charged compute, disk, and transfer spend: **`$0.2176`**
