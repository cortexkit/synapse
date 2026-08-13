# VLM grounder acceptance test

A reusable battery for judging any vision-language model proposed as a UI grounder
or screen-perception tier. Grown out of the LFM2.5-VL-3B and MiniCPM-V 4.6
evaluations (docs/eval-lfm25-vl-3b.md, docs/eval-minicpm-v46.md); designed with
CEREB (the action-plane consumer) so the metrics match what an action plane can
actually accept.

## Why a dedicated test

Blog benchmarks (ScreenSpot, RefCOCO) measure grounding quality in isolation.
An action plane needs two different things, and the 2026-08 runs proved they
separate cleanly at 3B:

- **Reading** a surface (extract text, confirm a dialog's content): both 3B
  candidates scored ~70% exact-text. Plausible local capability.
- **Pointing** at a surface (emit a coordinate a plane can click): both
  candidates collapsed (LFM 0/5 valid coordinate calls; MiniCPM 1/10 constrained
  hits with 72-989px misses).

A misread is wrong information a caller can sanity-check. A mispoint is a
silent click on the wrong control. They are different capabilities with
different failure consequences; never report them as one score.

## The metrics — never averaged

1. **Convertibility** (unconstrained): fraction of grounding/click outputs
   mechanically convertible into a coordinate an action plane would accept —
   parseable numbers, inside image bounds, a single point or unambiguous box
   center. This measures how much work constraint machinery would have to do.
2. **Hit rate** (unconstrained): crop/coordinate verification puts the output
   on the named target.
3. **Constrained hit rate**: the same targets decoded under a coordinate
   grammar (GBNF), scored ONLY on hits — well-formedness is guaranteed by
   construction there, so parsing success is vacuous.
4. **Miss geometry** (constrained arm): per-miss distance from returned point
   to true target center, in the model-input coordinate space. This is the
   discriminator: misses clustered near targets = model knows but emits badly
   (grammar rescues it); misses scattered hundreds of px away = model lacks
   spatial knowledge (grammar converts visible failure into invisible wrong
   clicks — MORE dangerous with the machinery than without).

## Protocol pins

- Native-app shots: window-scoped `screencapture -l <windowid>` (no focus
  steal); the staged set with ground-truth manifest lives at
  bench/results/vl-eval-shots/ (gitignored; restage per MANIFEST.md if absent).
  Privacy rule: no Mail/Messages/Discord/user-browser-session captures.
- Web shots: HEADLESS Chrome only, isolated profile
  (`--headless=new --user-data-dir=$(mktemp -d) --screenshot=...`). Never an
  interactive browser.
- Coordinate space: ground truth and crop checks MUST be computed in the space
  the model's coordinates are defined in (read the model card's normalization
  contract); the report states the space and scale factor explicitly. A crop
  check in the wrong space scores correct models wrong and wrong models right.
- Serving trap (Qwen-derived families, incl. MiniCPM-V): the chat template
  opens `<think>` by default and a GBNF grammar then yields EMPTY output —
  pass `chat_template_kwargs: {"enable_thinking": false}` for any constrained
  arm. A grammar silently producing empty output reads as "the model cannot do
  it" and is actually a template bug.
- Failures stay in the report uncurated; a small honest battery beats a large
  curated one.

## Measured baselines (2026-08-13, stock llama.cpp b9580, M5 Max)

| Model | Read (exact text) | Unconstrained convertibility | Constrained hit | Verdict |
|---|---:|---:|---:|---|
| LFM2.5-VL-3B | 70% | not measured (0/5 valid click calls) | not run | No — unsafe for action |
| MiniCPM-V 4.6 | 70% | 0/10 | 1/10 (misses 72-989px) | No — grammar makes it dangerous |

Any third candidate is judged against constrained hit rate and miss distance,
not a leaderboard number.
