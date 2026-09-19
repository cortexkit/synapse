# Study: TypeSafe Jev and Laya — non-autoregressive typed decision models (2026-09-19)

Sources: https://typesafe.ai/blog/introducing-system-one-models-and-jev (2026-09-15, closed weights, API only), github.com/NandhaKishorM/laya at 9ac2115 (Apache-2.0, cloned to `~/Work/OSS/laya`), huggingface.co/convaiinnovations/laya at c5d78730 (Apache-2.0). Every architectural claim below is cited to laya source; Jev's internals are unpublished and only its published claims are quoted.

## What the class is

A "System One model" answers **typed questions** over a state in one encoder forward pass: no token generation, so nothing to parse and no schema violation is possible. Three question types (laya `common.py:33-46`, mirrored by Jev's API): `choice` (categorical over named options with descriptions), `score` (ordinal rubric levels, answer is the expectation over levels), `noul` (calibrated P(true) for a statement). Every answer carries calibrated probabilities; the training objective is a strictly proper scoring rule under RL ("RLCD"), so the confidence is meant to be usable as a gate rather than decoration. Jev claims 70-500 ms end to end and a cardinality up to 255 options; laya measures 33 ms per question and 7.2 ms/question batched on a T4.

This is the "smart if-statement" tier: classify, route, score, guard, extract-by-choice. Our stack has no lane for it. Today every such decision in the fleet is either a decode with grammar (`microllm.oneshot` + JSON schema, ~100 ms+ for a 2B student) or a hosted LLM call. The Athena classify student we trained (Qwen3.5-2B, 92% class exact) is this task shape solved with a generator.

## Laya's architecture, from source

`laya/common.py:89-126` (`DecisionModel`):

1. **Encoder**: bidirectional ModernBERT. The English checkpoint is `answerdotai/ModernBERT-large` — hidden 1024, 28 layers, 16 heads, FFN 2624, local attention 128 with a global layer every 3, vocab 50368, RoPE (`encoder/config.json` on the Hub). 421 M parameters total, encoder weights shipped in f16 (`model.safetensors`, 804 MiB).
2. **Type embedding** added to every position: `h = last_hidden_state + type_emb[qtype]`, `type_emb` is `[3, 1024]`.
3. **Decision head**: two plain `nn.TransformerEncoderLayer`s over the whole sequence (pre-norm, 16 heads, FFN 4096, default ReLU activation, key-padding mask). Standard MHA with a fused `in_proj [3072, 1024]` and `out_proj`. No RoPE, no windowing — a vanilla transformer on top of the encoder's output.
4. **Gather** the head's hidden state at each option's `[MASK]` marker position; **scorer** `LayerNorm → Linear(1024,1024) → GELU → Linear(1024,1)` gives one logit per option; unused option slots are masked to −1e4.
5. **Calibration**: logits are divided by a temperature chosen by `(qtype, option-count bucket)` from `rl_agent_config.json` (`temperature_by_options`, e.g. `choice:2` 1.906, `choice:11+` 0.101), then softmax. `score` reports the expectation Σ i·pᵢ; `noul` reports p[1].
6. **Act head** (the "confidence" output): features `[top1, top1−top2, normalised entropy, k/255]` from the *untempered* softmax, concatenated with the head's CLS state, through `Linear(1028,256) → GELU → Linear(256,2)`; `act_probability = softmax[0]`. Trained against `act_costs` (escalate = 0.5, wrong act = 3.0), so it is the learned "should software act on this or escalate" decision.

**Input layout** (`common.py:49-86`, `build_sequence`): one sequence *per question*, `[CLS] "<type> question: <instructions>" [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]`, with a 192-token head budget (each option capped at 48 tokens, redistributed if the head overflows), the state filling the remainder up to `max_len = 512`, truncated right by default. Questions over one state are batched into one forward (`agent.py:254-277`), which is what "all questions in a single forward pass" means: N questions = N rows sharing the state text, not one row answering N questions.

Published accuracy on its own eval families ranges 0.44 (sentiment) to 0.99 (intent) with ECE mostly under 0.06 (`eval/results.md` on the Hub); the README's Jev comparison is on shared public datasets and is the author's, not independent.

## Why it maps onto the owned lane almost directly

The encoder is the family we already serve. `crates/synapse-engine-owned/src/modernbert.rs:21-55` parses every dimension from `config.json` (hidden, layers, heads, FFN, local window, global interval, rope thetas, bias flags) and the Metal graph is built from them — nothing is baked to base's 768/22/12, and the bucket ladder covers 512. The rerank path already has the exact seam the decision head needs: `forward_rerank` (`modernbert.rs:422-460`) runs `initial_hidden` on the host, the Metal block forward in place over `[batch, seq, hidden]`, then `classify_cpu` on the host over the token-level hidden states. Laya's head is a different function at that same seam.

What is new:
- **Loader**: strip the `encoder.` prefix into the existing ModernBERT loader (check `attention_bias`/`mlp_bias`/`norm_bias = false` for large are honoured — the config parser has the fields); load `type_emb`, `head.*`, `scorer.*`, `act_head.*`, `temperature`, and the `temperature_by_options` map from `rl_agent_config.json` into a `DecisionHead`.
- **Sequence builder**, module-side (the module owns the tokenizer): a byte-exact port of `build_sequence` including its truncation arithmetic, proven against Python-produced ids on fixed inputs.
- **Head forward**: two vanilla transformer layers + gather + scorer + act head. CPU first (≈14 GFLOP per 512-token question, tens of ms on Accelerate), Metal only if the measurement says the head dominates.
- **A new task class** on the wire — not embed, rerank, or generate. The response is typed per question: `choice` with per-option probabilities, `score` with expectation and legend, `noul` with P(true), each with `confidence` and `act_probability`. Certification needs its own probe corpus (states × questions with reference probabilities) and a gate on probability agreement, not cosine.

What is *not* portable: Jev. Closed weights, API only, no artifact. Laya is the open-weight instance of the class and the only thing we can run.

## Why it matters for us

1. **Every fleet "decide" call that is not a generation.** Athena classify, MC historian trigger decisions, WERNI's mode-2 escalate-or-reply, CEREB's confidence gate on a grounded click, the gather sufficiency judge, guardrails. Each is a `choice`/`score`/`noul` over a state. At ~30 ms per question on Metal — or on the silent tier, since it is a 512-token ModernBERT encoder and we have that on ANE — the cost model changes: these become free enough to call per turn.
2. **Fine-tunable on our own data.** Laya ships a training recipe (README §Fine-Tuning; `proper_reward`, `td_lambda_targets` in `common.py`). Our Athena classify dataset (999 synthetic gold + 233 real rows, labelled by Opus 4.8) is already in typed-decision shape. A head trained on it would answer in one encoder pass instead of a 2B decode.
3. **Calibration is the product**, not accuracy. A 0.95-accurate model that says when it is in the 5% can be automated; the act head is that signal trained as an RL policy with explicit costs. Our own confidence gating today is heuristic.

## Spike

`.cortexkit/alfonso/prompts/laya-owned-lane-spike.md`: load the English checkpoint through the owned engine, port the sequence builder and the head, prove parity against the Python reference on a fixed battery (probabilities, argmax, act probability), and measure encoder-vs-head time at 512 on this machine. Stop rules: parity first, no Metal head until the CPU head passes, no fine-tuning.
