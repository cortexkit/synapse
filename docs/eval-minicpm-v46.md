# MiniCPM-V 4.6 capability evaluation

> **Verdict: no — MiniCPM-V 4.6 does not pass CEREB's v2 grounder-fallback convertibility bar on this stock llama.cpp path.** Build 9580 served the official Q8 model and answered an image smoke test, but it emitted **0/10 mechanically convertible** unconstrained grounding results and **0/5 convertible** unconstrained vision-click calls. Constraining syntax produced points, but only **1/10** grounding targets landed on target and **0/5** constrained click calls hit; it turns visible non-answers into silent wrong clicks. This is more dangerous, not safer, than the unconstrained failure. The fallback tier needs a different **output shape** (constrained decoding plus an independently validated coordinate representation/structured output), not merely a better model.

This is the same small, uncurated four-battery capability run as `eval-lfm25-vl-3b.md`, extended with the requested convertibility and constrained-decoding arms. Raw screenshots, exact prompts/responses, grammars, crops/target centers, server logs, and strips are in ignored `bench/results/minicpm-v46-eval/`.

## Setup and coordinate space

| Item | Measured setup |
|---|---|
| Model | `openbmb/MiniCPM-V-4.6-gguf`, resolved revision `78e02f066e9819a60573b78a4275df8a0c27f698` |
| Weights | best published quality `MiniCPM-V-4_6-Q8_0.gguf` (811,591,616 bytes, SHA-256 `5cc8be0b…d7422e4`) + `mmproj-model-f16.gguf` (1,108,746,944 bytes, SHA-256 `ca931d86…8462de`) |
| Server | pinned `/opt/zerobrew/bin/llama-server`, build 9580 (`b4e3dc613`), `--mmproj -ngl 99 -c 4096`; contingency 10405 was not needed |
| Host | M5 Max, 128 GiB RAM, macOS 26.6.1 |

**Coordinate space.** OpenBMB documents `<box>x1 y1 x2 y2</box>` and `<point>x y</point>` as 0–1000 coordinates normalized against the **supplied image** (`pixel = coordinate / 1000 × supplied width or height`). llama.cpp may resize/tile internally for vision tokens, but it exposes no resized action canvas and the model's documented output contract maps directly back to the original screenshot. Accordingly, all ground truth, crop checks, and miss distances in this report are in original supplied-image space, with scale factor **1.0** from the reported coordinate space after the documented normalization; they are not measured in an unobservable vision-tensor grid. For example, a point `208 192` on the 2400×1600 dashboard maps to (499, 307) px.

`/health` returned OK, and an image smoke on the staged TextEdit screen returned `Status: READY` (with extra reasoning text). The pinned server therefore loads this architecture; there was no reason to switch builds. Q8_0 is the only published Q8-class weight; the projector is F16.

## 1. Real screen understanding

Six staged, window-scoped native captures (Finder, Activity Monitor, TextEdit, Magic Context Dashboard, OrbStack, Zed) and four isolated-headless Chrome captures (llama.cpp repository, pull list, PR, Hugging Face model page) were each asked a description, state, and exact-text question. Scores are against what is visible in the images; failures and truncations remain counted.

| Category | MiniCPM correct | MiniCPM partial | MiniCPM wrong | MiniCPM accuracy | LFM correct | LFM accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Description/application/site | 1/10 | 5 | 4 | 10% | 6/10 | 60% |
| State/selection | 5/10 | 1 | 4 | 50% | 9/10 | 90% |
| Exact text | 7/10 | 0 | 3 | 70% | 7/10 | 70% |
| **All** | **13/30** | **6** | **11** | **43%** | **22/30** | **73%** |

Examples deliberately retained: it called OrbStack “Docker,” named Zed from a file rather than the editor, returned blanks for several questions, read `8-bit` instead of `Q8_0`, and called `minicpmv4.6.md` “llama.cpp.” It correctly read the TextEdit status and temperature, the dashboard `150000`, the GitHub repository, and the HF model size.

## 2. Element grounding and convertibility

The ten targets used OpenBMB's documented box prompt. A result is **convertible** only if it can mechanically become one action point: numeric, within the 0–1000 range, and either one point or an unambiguous positive box center. **Hit** is separate: programmatic coordinate/crop verification must put that point/crop on the named target. LFM did not measure convertibility, so its cells are explicitly not measured.

| Arm | Convertible | Hit / crop contains target | LFM hit | LFM convertible |
|---|---:|---:|---:|---:|
| MiniCPM unconstrained documented `<box>` | **0/10 (0%)** | **0/10 (0%)** | 8/10 (80%) | not measured |
| MiniCPM constrained `<point>x y</point>` | grammar-shaped outputs 10/10; not an accuracy measure | **1/10 (10%)** | — | — |

Unconstrained outputs were empty, prose plus malformed numeric fragments, or `</think>`; none were documented boxes. Thus no crop can honestly be credited. The constrained arm used llama-server's accepted GBNF request field with `chat_template_kwargs.enable_thinking=false`; without disabling Qwen's `<think>` turn, the grammar produces empty responses. This is an execution detail, not a success metric: parsing is guaranteed/induced by the grammar and is not reported as accuracy.

### Constrained-arm miss geometry

Distances below are from the returned constrained point to the independently selected target center in the coordinate space stated above. The sole hit is `Form tab` (22.9 px). The other nine are materially displaced rather than near-miss formatting failures.

| Target | Returned normalized point | Distance to true center |
|---|---:|---:|
| docs folder | 273, 322 | 296.8 px |
| Reads in value | 379, 497 | 988.8 px |
| READY status | 340, 198 | 317.4 px |
| Containers navigation item | 41, 322 | 287.1 px |
| model_routing key | 347, 333 | 799.7 px |
| Pull requests tab | 307, 150 | 120.4 px |
| New pull request button | 854, 247 | 72.7 px |
| Q8_0 label | 833, 359 | 271.2 px |
| MiniCPM-V 4.6 heading | 340, 239 | 248.6 px |

This geometry is the dangerous signature requested by CEREB: low unconstrained convertibility was **not** latent spatial knowledge waiting for a parser. Under constraint, the model confidently chooses locations far from targets. Grammar machinery therefore converts visible failure into invisible wrong clicks.

## 3. Frame-strip transition reading

Three sequential headless GitHub strips were composed from the public llama.cpp repository → pulls → PR flow (and cyclic order variants). A correct answer had to name every material page transition, not merely say frames changed.

| Strip | MiniCPM score | Observation | LFM score |
|---|---|---|---|
| Repo → pulls → PR | wrong | Repeated only “Frame 1 to Frame 2,” naming no repository/pulls/PR transition. | partial |
| Pulls → PR → repo | wrong | Repeated frame numbers, omitted both page changes. | partial |
| PR → repo → pulls | wrong | Repeated frame numbers, omitted both page changes. | wrong |
| **Fully correct transitions** | **0/3 (0%)** | — | **0/3 (0%)** |

The decisive proxy battery remains a negative result.

## 4. Function-calling smoke and click convertibility

The text arm supplied the same five schemas and required `[tool_name(arg="value")]`. The five vision requests used the first five grounding targets. Convertibility remains separate from call validity/hit.

| Mode | Valid/plausible | Unconstrained convertible | Unconstrained hit | Constrained hit | LFM validity | LFM convertible |
|---|---:|---:|---:|---:|---:|---:|
| Text-only (10) | 0/10 | n/a | n/a | n/a | 6/10 | not measured |
| Vision + text click (5) | 0/5 | **0/5 (0%)** | **0/5 (0%)** | **0/5 (0%)** | 0/5 | not measured |

Every unconstrained text call was empty. Four unconstrained vision outputs were empty; one began prose about the Containers item, so none formed a numeric click. The constrained click grammar emitted five bracket-shaped calls, but that is intentionally not counted as validity: `2182` was out of range and the four in-range calls all missed their targets. Their miss distances were docs 470.2 px, Reads in 7755.3 px (out of range before conversion), READY 196.3 px, Form 498.8 px, and Containers 221.3 px. This independently reproduces the grounding conclusion: constrained shape does not make MiniCPM a safe clicker.

## Performance cells

This was a capability run under contention, not certification. Load average was `16.35 10.76 8.61` before the battery and `26.97 15.29 10.58` afterward. The image smoke was 0.751 s prompt processing plus 0.288 s decoding (228.8 server-reported tok/s, 66 tokens); no clean repeat series or peak-RSS sample was collected, so no throughput comparison is claimed.

## Fit assessment and limitations

**CEREB grounder fallback tier — no.** MiniCPM's 0% unconstrained grounding/click convertibility fails the direct-action prerequisite. More importantly, constrained points hit only 10% of ten targets and constrained clicks hit 0% of five: a grammar makes the result parse but not spatially correct. If both convertibility and constrained hit are low, as here, the model is **more dangerous with constraint machinery than without** because it hides the failure as an executable click. A fallback needs a different output shape with independently validated structured coordinates (and likely a model with demonstrated spatial grounding), not a parser wrapped around this response.

**Screen-recording perception tier — no.** It described isolated facts at 43% correct but scored 0/3 on material transitions, the same decisive failure mode as LFM.

Limitations: this is a small ten-surface/ten-target/three-strip sample; target-center crop verification is coarse and uses supplied-image coordinates; the candidate model was not used as a second crop judge; and contention was high. These limitations cannot turn malformed/empty unconstrained outputs into mechanically actionable coordinates, nor turn the recorded constrained misses into hits.
