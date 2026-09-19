# Laya on the owned Metal lane — spike

## Finding

**Yes.** The English `convaiinnovations/laya` checkpoint runs with the existing owned ModernBERT graph and a host f32 decision head. All 24 question rows pass the pre-registered probability/argmax gate. On this machine under ambient load, a **padded 512-token single question costs 125.87 ms median**: about 40.67 ms in the Metal block and 84.10 ms in the CPU head. The CPU head exceeds the 40% threshold (67.0% paired median share). **No Metal head was implemented.** This is a feasibility measurement, not certification or a serving feature.

## Artifacts and reproducibility

- Reference source: Laya `9ac2115df190748bdfae6a7b1f6e57081f1b7527`, Apache-2.0. Architecture/sequence logic is ported from `laya/common.py`; temperature selection and answer semantics follow `laya/agent.py`. Preset questions come from `laya/presets.py`.
- Hub revision: `c5d78730f3493e4fe16d61507ef4b78eef7318cf`, English root checkpoint only; neither alternative subfolder is used.
- `model.safetensors` SHA-256: **`891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`**.
- The checkpoint and derived encoder-only package live under `$HF_HOME` (defaulting to the conventional user Hugging Face cache). Neither weights nor virtualenv nor tensor dumps are tracked.
- `battery.json` holds all inputs, exact token IDs/markers, raw decision logits, temperatures, probabilities, confidence, act probability, raw act logits, and score expectations for both references. `battery.sha256` pins it. Floating outputs are retained unrounded; the Agent's presentation API rounds to four decimals.
- `owned-f16.json`, `parity-f16.json`, `tensor-drift.json`, and `timings.json` retain results and every measured sample. `diagnostics.py` compares active-token encoder/head tensors dumped by both implementations. Dumps are reproducible, ignored scratch data.
- Python 3.14.2, torch 2.14.0, transformers 5.17.0; installed reference dependencies are pinned in `requirements.txt`. CPU reference uses four torch threads. The f16 reference casts **only the encoder** to f16, promoting its last hidden state before the f32 head. This worked on CPU.
- Hardware: Apple M5 Max, 128 GiB unified memory, macOS 27.0 arm64. Rust release build, Apple Accelerate SGEMM for CPU head, owned Metal f16 blocks. No explicit Accelerate thread override.

From this spike directory, with `LAYA_SOURCE` pointing to the source checkout at the revision above:

```sh
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
uv venv --seed .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e "$LAYA_SOURCE"
# The installed huggingface-cli was a deprecation stub and exited 1.
# Its supported replacement downloaded the exact pinned files successfully:
.venv/bin/hf download --revision c5d78730 convaiinnovations/laya \
  model.safetensors rl_agent_config.json encoder/config.json \
  tokenizer/tokenizer.json tokenizer/tokenizer_config.json tokenizer/special_tokens_map.json
.venv/bin/python reference.py
shasum -a 256 -c battery.sha256
export DEVELOPER_DIR="$(xcode-select -p)" # must select the full Xcode developer directory
cargo test
cargo build --release
.venv/bin/python measure.py
.venv/bin/python compare.py
.venv/bin/python diagnostics.py
```

The crate has its own `[workspace]`; the root workspace manifest was not changed. Cargo resolved/downloaded dependencies and compiled the independent lockfile. A root test invocation refreshed unrelated sibling-dependency entries in the root lockfile; that incidental diff was discarded, not included.

## Loader and hidden-state seam

The loader accepted the unmodified encoder config: hidden 1024, 28 layers, 16 attention heads, intermediate 2624, local window 128, global every third layer, and all three bias flags false. No parser refusal and no large-model weight/shape fix was required. `reference.py` strips `encoder.` from tensor names into a derived safetensors package beneath `$HF_HOME/laya-owned/<revision>` and copies `encoder/config.json` verbatim. The production loader consumes that package. The config's nested `rope_parameters` is ignored by this parser, but its values (160000 global / 10000 local) equal the parser defaults; `norm_eps` is explicitly present and equals `layer_norm_eps`. This finding is specific to this checkpoint, not a claim of general config compatibility.

**Final norm is already applied.** `ModernBertModel::forward_cpu` ends with `final_norm`; `MetalContext::forward` supplies `model.final_norm` to the existing MPS forward and copies the resulting token states back. The head must not normalize the encoder output again beyond its own learned norms.

The original seam was private, not callable from a path-dependent crate. Two production touches were explicitly authorized during the spike:

1. **Hidden-state adapter:** `HiddenStates` and `OwnedMetalEmbedEngine::encode_hidden` in `src/lib.rs`, an internal default-refusal method on the still-private runtime trait, and ModernBERT's implementation in `src/modernbert.rs`. It pads and builds the attention mask as rerank does, calls the existing `forward`/`block_forward` path, and returns unpooled token states. It does not require a classification head. No runtime/kernel trait was made public, no wire/task type was added, and embed/rerank computations were not changed. Shapes are bounded by loaded bucket capacity. The focused integration test in `tests/gte_full_context.rs` runs against the existing real GTE snapshot: `encode_hidden_cls_normalized_matches_gte_embed` selects CLS and L2-normalizes, comparing with the engine's embed result to 1e-5. GTE uses **CLS, not mean pooling**; the reviewer confirmed that correction to the initially requested test.
2. **Opt-in timing:** the existing `SYNAPSE_EMBED_PROFILE` emission format gains `modernbert_block_forward`, timed immediately around `context.forward`. The flag is cached once at that call site; disabled profiling takes no timestamp and allocates no log message. No graph/signature changes. The enclosing encode wall is measured in the spike, not in production.

Full engine regression: `DEVELOPER_DIR` selecting Xcode, `cargo test -p synapse-engine-owned`: **56 passed, 0 failed, 12 ignored** (existing ignored tests remain ignored), including both GTE integration tests. The focused equivalence test uses a required local GTE fixture, rather than silently treating a missing fixture as success.

## Sequence builder: 24 / 24 identical

Exactly 24 cases, one question row each:

- Eight preset cases spanning email, triage, guard, moderation, and router.
- Eight owned shapes: four Athena-style six-way work-mode choices, two evidence-sufficiency nouls, two urgency scores.
- Eight adversarial cases: 512-token truncation (state far longer than 512), head budget overflow, twelve choices (`choice:11+`), unicode, empty/None option descriptions, nested JSON state, conversation turns, and literal `[MASK]` in state **and** instructions/options.

`cargo test` gives **2 passed**: Python-compatible JSON spacing/unicode, and `battery_ids_and_markers_are_byte_exact` (24/24). No mismatch/fix round was needed. This spike ports the requested default right-truncation path; it does not expose optional left truncation or option permutation. Dictionary insertion order and Python JSON separators are preserved. The head-overflow row is 200 tokens after redistribution; the long-state row is exactly 512; twelve-choice has twelve markers.

A controlled `NON-VACUITY BREAK` disabled literal-mask replacement. Only `tests::battery_ids_and_markers_are_byte_exact` failed, on `adversarial-mask`; the JSON spacing test still passed. The staged original was restored and both tests passed again.

## Decision head and parity

CPU f32 implementation uses packed Q/K/V in that order, attention scale `1/sqrt(64)`, key-padding masking, two pre-norm transformer layers, **ReLU** FFNs (verified: Laya does not override PyTorch's default activation in `common.py`), and ignored evaluation dropout. The scorer is LayerNorm → Linear → exact-erf GELU → Linear. The act features use **untempered** softmax; the pooled state is CLS **after** the head layers. Reported probabilities use the option bucket override and fall back to the qtype temperature. Unused option slots would have −1e4 logits and exactly zero f32 softmax contribution for this checkpoint; the per-row CPU head evaluates just its k real options. Score is Σ i·pᵢ. Noul confidence follows Agent's `max(P(true), 1-P(true))`; other types use normalized entropy confidence.

Distribution is over row-wise maximum absolute probability differences, absolute act-probability differences, and score expectation differences on score rows; p95 uses NumPy's interpolated percentile. All values below compare owned f16 encoder + f32 CPU head:

| Reference | Argmax | Max Δp | p95 Δp | Max Δact | p95 Δact | Max Δscore | p95 Δscore |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python CPU fp32 | 24/24 | 0.003312737 | 0.000979459 | 0 | 0 | 0.000531852 | 0.000510826 |
| Python CPU f16 encoder | 24/24 | 0.005145371 | 0.001059306 | 0 | 0 | 0.001016855 | 0.000901875 |

**Pre-registered gate PASS:** max Δp ≤ 0.02 and no argmax disagreements on reference gaps ≥ 0.05. In fact there are **no argmax disagreements at all**, so the disagreement/gap table is empty; every row's gap is retained in `parity-f16.json`. Python's own f16-encoder-vs-fp32 max Δp is 0.002181709 (p95 0.001676685). Tolerances were not changed.

Important limitation: all 24 act probabilities round computationally to **1.0**, even before presentation rounding. Thus zero Δact is not meaningful evidence of act-head calibration. Raw act logits are included so saturation is visible (first owned row approximately `[4114.72, -3364.74]`). Max raw act-logit drift is 5.974 versus fp32, 4.507 versus f16-encoder. Optional dump comparison shows the largest active-token encoder absolute difference 22.465 and head difference 433.692 on the long-state row, but their RMSEs there are 0.05494 and 5.18275 respectively; output probabilities still pass. No layer-localization repair was required by the gate. Certification must include unsaturated act decisions; this battery alone cannot validate that capability.

A second controlled `NON-VACUITY BREAK` replaced one owned probability vector with a wrong one-hot vector. `pre_registered_fp32_probability_and_argmax_gate` failed: Δp 0.999797, argmax 23/24, disagreeing row gap 0.997165. Restoring the staged outputs restored the passing gate. This tests the report's actual file-reading comparator, not a proxy computation.

## Measurements

Each bucket has **3 warmups + 20 measured iterations**, retaining all samples in `timings.json`. The single question contains 83 real tokens and is padded to the stated bucket. The batch uses eight different preset questions over the same email state, padded to 512, one Metal encoder batch followed by per-row CPU heads. Tokenization/model loading/graph compilation are outside timed steady-state calls. The head timing includes host allocations, gather/scorer/act computation and assembling output values, not just matrix kernels. Profiling is enabled during measurement.

Ambient load before: `17.86 12.98 14.60`; after: `17.75 13.56 14.75` (1/5/15-minute load averages; both captured at 14:42 local time). No load isolation was attempted.

| Shape | Encode wall ms | Host prologue + wrapper ms | Metal block ms | CPU head ms | Total ms | Head share | Head / block |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 × 128 | 18.392 | 0.263 | 17.968 | 28.157 | 51.378 | 58.9% | 1.46× |
| 1 × 256 | 33.320 | 0.630 | 32.731 | 61.661 | 97.705 | 64.3% | 1.84× |
| 1 × 512 | 41.813 | 1.126 | 40.667 | 84.098 | **125.868** | **67.0%** | 2.08× |
| 8 × 512 | 292.751 | 7.753 | 285.077 | 825.141 | **1127.337** | **73.5%** | 2.85× |

Each column is its own median. Total, share, and ratio are computed per iteration before taking the median, so displayed component medians need not add exactly. The batch cost is about 140.9 ms/question here, not a throughput win for this deliberately sequential CPU-head implementation.

The host column is enclosing encode wall minus measured block wall: chiefly token lookup + embedding norm, but also adapter padding, locking, dispatch, and profiling emission overhead. It is **not** a pure isolated embedding kernel timing. The block timer includes its existing host mask/parameter/buffer preparation and transfer as well as Metal execution; it is wall time, not a GPU timestamp. At 512 the host prologue is small compared with both the block and CPU head. The head crosses 40%, making a later optimized head worth investigation, but this spike implements none.

## What a wire op would need

A dedicated typed-decision task rather than an embed/rerank alias: state plus named questions, `choice` with ordered criteria and per-option probabilities/selected label; `score` with rubric legend, probabilities and expectation; `noul` with P(true). All types need confidence and act_probability, usage, and immutable model/tokenizer/calibration identity. A certification probe corpus should pin states/questions, token IDs/markers and reference probability vectors across option buckets, truncation, unicode and structured states. This 24-case battery is a candidate seed, not sufficient certification: add batched/padded parity, calibration-sensitive cases with **non-saturated act logits**, more near-ties, and representative deployment distributions. No wire op, certification registration, fine-tuning, or evidence under `docs/evidence/` was added.

## Verification notes

- Python CPU fp32 and f16 reference runs: all 24 rows each completed; Python modules pass `py_compile`.
- Independent `cargo test`: 2 passed. Release build and all-row parity comparator: passed.
- Production engine suite: 56 passed, 12 ignored, no failures. Full Xcode developer directory selected.
- Battery checksum verified; publication banlist and whitespace diff checks run before commit.
- AFT diagnostics initially timed out, then returned fresh but incomplete producer coverage (no authoritative Rust reports for the spike). Cargo compilation/tests are the authoritative checks here.
- The first release build hit a ten-minute timeout during dependency compilation under high load; resuming with four build jobs succeeded. No runtime timeout or parity failure occurred.
