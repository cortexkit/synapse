# LFM2 hybrid backbone

## Scope and pinned reference

This wave adds text-only LFM2 causal decode to the owned runtime. Audio towers,
feature extraction, and ASR integration are deliberately out of scope.

The validation snapshot is
[`LiquidAI/LFM2-1.2B@933cee00d754fb3bfe06c644c0cb95453f2d8bb2`](https://huggingface.co/LiquidAI/LFM2-1.2B/tree/933cee00d754fb3bfe06c644c0cb95453f2d8bb2).
Its `model.safetensors` LFS SHA-256 is
`60fef6ef4481c533ce7427793bed50200b55b3c68d0d00c52bc56f207a9acecd`.
The source comparison used Transformers `5.12.1`; the inspected upstream source
was
[`huggingface/transformers@70b1ee26bf72e70ab1f5102490c5ab93b8a80e7c`](https://github.com/huggingface/transformers/tree/70b1ee26bf72e70ab1f5102490c5ab93b8a80e7c).
`reference_lfm2.py` also pins PyTorch `2.12.0` and rejects an unpinned Hub
revision unless explicitly overridden.

## Owned architecture

Detection reads `model_type` from `config.json` and dispatches `lfm2` to a new
`ModelFamily`. The family owns tokenizer policy (BOS 1, EOS 7), config and tensor
validation, and tied embeddings. Its decoder uses the existing `DecodeModel`
seam, so token taps and pause/resume operate exactly as they do for Qwen3.

The pinned checkpoint resolves to:

- 16 layers, hidden size 2048, vocabulary 65,536, final and pre-block RMSNorm
  epsilon `1e-5`;
- 10 short-convolution layers and 6 full-attention layers in the order
  `C,C,A,C,C,A,C,C,A,C,A,C,A,C,A,C`;
- 32 query heads and 8 KV heads, head dimension 64, scale `1/sqrt(64)`, shared
  per-head-dimension Q/K RMSNorm gains, and half-split (non-interleaved) RoPE
  with theta 1,000,000;
- SwiGLU MLPs with actual tensor width 8192 and no biases;
- tied `embed_tokens.weight` output projection.

Attention keeps an append-only GQA KV cache. Q/K normalization is applied after
projection and head reshape and before RoPE. The implementation accepts either
a modern `layer_types` list or the legacy `full_attn_idxs` representation, but
requires one of them and verifies every selected mixer against its tensor
shapes.

### Short convolution and cache semantics

For normalized input `h_t`, the mixer computes:

```text
[B_t, C_t, x_t] = W_in h_t
u_t = B_t * x_t
z_t[c] = k[c,0] u_{t-2}[c] + k[c,1] u_{t-1}[c] + k[c,2] u_t[c]
out_t = W_out (C_t * z_t)
```

There is no activation in this path. Taps are used in stored order; this is
cross-correlation, not a flipped mathematical convolution.

Each convolution layer owns a `[3, hidden]` rolling state. Prefill saves
`[u_{T-3}, u_{T-2}, u_{T-1}]`, zero-padding on the left. Before an incremental
step the state is shifted and `u_t` is appended, after which all three samples
are dotted with the three taps. Keeping three samples rather than only the two
historical samples is intentional and matches Transformers'
`LinearAttentionLayer` cache.

CPU/Accelerate uses the provider's causal depthwise loop. Metal forms explicit
`[rows, channels, kernel]` causal windows and sends one MPSGraph reduction to
the native bridge. Explicit windows were chosen over MPS convolution padding:
the kernel is fixed at three, this makes tap orientation and left padding
unambiguous, and the same primitive covers prefill and the rolling-state decode
check. Metal f16 static RHS values are cached by the original weight identity;
this avoids repeatedly converting BF16-derived f32 weights and prevents
allocator-address collisions between temporary f16 buffers.

## Config versus checkpoint reality

The raw snapshot has several legacy surprises that the loader handles and
checks:

1. There is no serialized `layer_types`; `full_attn_idxs = [2,5,8,10,12,14]`
   defines the layout. Newer configs may provide `layer_types` directly.
2. `block_ff_dim` says 12,288, but the constructor's two-thirds rounding makes
   the actual MLP width 8192. The owned runtime derives this value from every
   `w1/w2/w3` tensor and rejects inconsistent layers.
3. No tie flag is serialized. LFM2's default is tied and the checkpoint omits
   `lm_head.weight`; the loader verifies both facts and reuses embeddings.
4. `max_position_embeddings` says 128,000 while the model card advertises
   32,768. The runtime validates the architectural value but makes no quality
   claim beyond the model card range.
5. Casting the checkpoint to fp32 gives fp32 arithmetic over serialized BF16
   weights; it cannot recreate pre-quantization fp32 weights.

## Correctness gates

Validation ran on an Apple M5 Max development host (128 GiB, macOS 26.5.2), not
the occupied M1 or a locked benchmark rig. Prompts were the 20 rows in
`decode-prompts.jsonl`, direct tokenizer completions with one BOS, greedy
sampling, EOS 7, and a maximum of 64 new tokens.

| Gate | Result |
| --- | --- |
| Final hidden states, CPU/Accelerate fp32 | Passed all positions on 20 prompts; minimum cosine reported `1.00000000` (threshold `0.9999`). |
| Final hidden states, Metal MPSGraph fp32 | Passed all positions on 20 prompts; minimum cosine reported `1.00000000`. |
| Greedy tokens, CPU/Accelerate fp32 | **20/20 token-exact**, 64-token cap. |
| Greedy tokens, Metal MPSGraph fp32 | **20/20 token-exact**, 64-token cap. |
| Cached decode versus full re-prefill | **20/20 token-exact** on CPU/Accelerate, up to 64 generated tokens per prompt. A synthetic unit test also checks every prefix state. |
| Metal MPSGraph f16 | 17/20 prompts token-exact. Minimum prefill hidden-state cosine was `0.99998592`. First forks: `completion-04` step 0 (`535` vs `1334`, both rounded logit `13.78125`), `completion-05` step 8, and `completion-15` step 22. This is reported rather than certified token-exact, per the f16 gate policy. |

The development-only debug runs indicated:

- CPU/Accelerate fp32: about 15.8 prefill tok/s and 14.0 decode tok/s for the
  hidden-state plus token run; the separate cache gate observed 19.8 and 17.7
  tok/s.
- Metal MPSGraph fp32: about 6.2 prefill tok/s and 5.8 decode tok/s.
- Metal MPSGraph f16: about 6.5 prefill tok/s and 5.7 decode tok/s.

These include a scalar orchestration path and are correctness-run observations,
not performance claims.

### Reproduction

```bash
MODEL="$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2-1.2B/snapshots/933cee00d754fb3bfe06c644c0cb95453f2d8bb2"
python3 bench/spikes/unified-rt/reference_lfm2.py \
  --model "$MODEL" \
  --prompts bench/spikes/unified-rt/decode-prompts.jsonl \
  --tokens-out target/lfm2-reference-tokens.jsonl \
  --hidden-out target/lfm2-reference-hidden.jsonl \
  --max-new-tokens 64

cargo run -p spike-unified-rt --bin spike-unified-rt -- \
  --model "$MODEL" --tokenizer "$MODEL/tokenizer.json" \
  --generate-prompts bench/spikes/unified-rt/decode-prompts.jsonl \
  --decode-reference target/lfm2-reference-tokens.jsonl \
  --decode-hidden-reference target/lfm2-reference-hidden.jsonl \
  --max-new-tokens 64 --decode-cache-bucket 512 --decode-top-k 5 \
  --verify-decode-cache --device cpu --dtype f32 \
  --out target/lfm2-owned.json
```

Run the owned command with `--device metal --dtype f32 --execution lazy` for the
Metal fp32 gate, or `--dtype f16` for the diagnostic f16 variant.

## Traps for the ASR wave

- Convolution state is model state. Beam rollback, speculative decode, prompt
  lookup, and arbitrary cache rewinds are invalid unless both KV and every
  convolution layer state are restored together.
- Transformers' slow cached convolution branch is correct for one-token decode,
  but a multi-token cached chunk broadcasts its final convolution value. Chunked
  continuation must use sequential one-token updates or the full prefill path.
- Padding affects convolution history. Batched ASR should mask padding before
  projection and use left padding so the newest cache samples are real tokens.
- Audio and text positions must agree on the absolute RoPE position supplied to
  attention layers; convolution layers have no independent position counter.
- Do not infer the text MLP width from `block_ff_dim`, assume a serialized layer
  list, or look for a separate LM head. Derive these from tensors and the legacy
  compatibility fields as this loader does.
- The family intentionally implements text tokens only. Reusing it for
  LFM2-Audio requires an explicit modality/feature seam rather than teaching the
  tokenizer path about waveforms.
