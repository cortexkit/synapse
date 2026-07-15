# LFM2 on the Apple Neural Engine

**Measured 2026-07-15. Primary model:** [`LiquidAI/LFM2-350M` at
`b3afba2`](https://huggingface.co/LiquidAI/LFM2-350M/tree/b3afba27815ee83a64b76162cef4d8a4780d6ca7),
Apache-2.0. The checkpoint SHA-256 is
`387638dc889ff1a1395c3c2ab9605211e4c7e16f2d375361dd4e423b909a254e`.

## Result in one paragraph

LFM2's operators place exceptionally well: both fixed buckets put **728/730
(99.726%) dispatchable operations on ANE**, including **102/102 convolution
operations**. On an M1 Max the seq128 package runs at 16.60 requests/s and
1.92 W on the ANE+CPU+GPU compute rails, while median GPU power is 0 W. That
is the desired quiet-tier power profile. It is **not serving-viable with the
pinned Core ML 8.3 fp16 recipe**, however: minimum prompt cosine against
Transformers CPU fp32 is only 0.92473 at seq128 and 0.92644 at seq256, far
below the 0.999 gate. `torch.export` is exact and PyTorch fp16 stays above
0.999, so the loss occurs in Core ML's fp16 lowering of the repeated gated
short-conv/SwiGLU computation. Stateful decode exports and converts all 22
conv/KV buffers, but its first ANE prediction fails with Apple Neural Engine
error 8/status `0x1d`; the same package executes on `CPU_ONLY`. **VERDICT:
LFM2-on-ANE is placement- and power-feasible but numerically not viable on
this toolchain, and stateful decode is blocked at ANE runtime. Do not
schedule LFM2-Audio encoder conversion as the next quiet-tier investment
until the fp16 drift or stateful-runtime boundary changes.**

## Scope and architecture

The conversion follows `bench/spikes/unified-rt/src/lfm2.rs` and
`LFM2-BACKBONE.md`, then checks the custom graph against the Transformers
implementation before Core ML conversion. The 350M checkpoint has:

- 16 layers: 10 gated depthwise causal short-conv blocks and 6 GQA blocks;
- hidden width 1024, 16 query heads, 8 KV heads, head width 64;
- tied 65,536-token embeddings;
- short-conv kernel/cache length 3; and
- **actual FF width 4,608**, read from `feed_forward.w1.weight`. The config's
  `block_ff_dim` is not trusted.

`LiquidAI/LFM2.5-1.2B-Instruct` was not converted after the primary 350M
model missed the parity gate in both buckets. Scaling the same repeated block
math cannot repair this lowering error, and the stateful failure occurs in
ANE execution rather than a size-specific model loader. This preserves the
ordered stretch-target budget instead of producing a larger known-invalid
package.

## Porting recipe

The implementation applies the checklist from the ANE book's
[Modern Inference](https://alvaro-videla.com/ane-book/00-why-ane.html),
[ANE Laws](https://alvaro-videla.com/ane-book/01-ane-laws.html), and
[Porting Recipe](https://alvaro-videla.com/ane-book/02-porting-recipe.html),
plus the proven `ane-minilm` rules:

1. Use `torch.export`, never TorchScript trace.
2. Keep fixed batch-1 seq128 and seq256 packages, with left-padded inputs and
   an explicit attention mask.
3. Reshape activations to channels-first `[1, d, T, 1]`; represent every
   linear Q/K/V/O, short-conv gate, and SwiGLU projection as a real 1x1
   `Conv2d`. Keep the short-conv as grouped depthwise `Conv2d` with kernel
   `(3, 1)` and left causal padding.
4. Convert fp16 weights/compute with `CPU_AND_NE` and macOS 15 minimum
   deployment. No int8/W8A8 path was attempted.
5. Capture Transformers CPU-fp32 and PyTorch-fp16 goldens before performance
   measurement. Mask pad positions when computing parity.
6. Inspect every `MLComputePlan` operation. Any convolution outside ANE would
   invalidate this graph and require a rebuild.
7. Benchmark only after recording the failed golden. Speed below is retained
   as a hardware diagnostic, not presented as valid-model serving speed.

Pinned conversion environment: Python 3.11.15, PyTorch 2.5.1,
Transformers 4.51.3, coremltools 8.3.0, NumPy 2.3.2, macOS 26.5.2 arm64.
Reference generation used a separate PyTorch 2.12.0 / Transformers 5.12.1 /
NumPy 2.4.6 environment because 4.51.3 predates `model_type=lfm2`; it loaded
the same pinned local checkpoint with `trust_remote_code=False`, eager
attention, CPU, and fp32 compute. The
custom eager graph is the cross-environment bridge and matches that reference
at effectively 1.0 cosine.

## Phase A: prefill

The Core ML output is the full `[1, T, 1024]` final hidden state. Twenty fixed
prompts from `../unified-rt/decode-prompts.jsonl` contain 229 non-pad tokens in
either bucket.

### Gate 1: placement

Placement was dumped on the timed M1 Max with `MLComputePlan`. Constants are
not dispatchable and are excluded from the denominator.

| bucket | dispatchable | ANE | CPU | GPU | ANE share | conv on ANE |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 730 | 728 | 2 | 0 | **99.726%** | **102/102 (100%)** |
| 256 | 730 | 728 | 2 | 0 | **99.726%** | **102/102 (100%)** |

The only falloffs are the integer token `ios18.gather` and one `ios18.cast`,
both CPU-only. No gated-conv, RoPE, softmax, GQA matmul, normalization, SiLU,
or SwiGLU operation falls off ANE.

| ANE operator class | count | ANE operator class | count |
|---|---:|---|---:|
| `ios18.conv` | 102 | `ios18.mul` | 224 |
| `ios18.add` | 101 | `ios16.reduce_mean` | 45 |
| `ios18.rsqrt` | 45 | `ios18.matmul` | 12 |
| `ios18.silu` | 16 | `ios18.softmax` | 6 |
| `ios18.reshape` | 50 | `ios18.transpose` | 56 |
| `ios18.slice_by_index` | 24 | `pad` | 10 |
| `split` | 10 | `tile` | 12 |
| `ios18.concat` | 12 | remaining shape/arithmetic ops | 4 |

The compact committed evidence is in `results/lfm2-350m-seq{128,256}.json`;
the runner emits every operation, its preferred device, and its supported
devices.

### Gate 2: parity

Cosines flatten only active token hidden vectors for each prompt. The gate is
the minimum of 20 prompt cosines.

| bucket | Transformers fp32 ↔ custom eager min | eager ↔ export min | Transformers fp32 ↔ Core ML min / mean | min token cosine | max abs | gate |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 0.99999999998 | 1.00000000000 | **0.924733 / 0.968376** | 0.827662 | 18.8195 | **FAIL** |
| 256 | 0.99999999998 | 1.00000000000 | **0.926441 / 0.968276** | 0.812091 | 18.8195 | **FAIL** |

A PyTorch fp16 MPS golden on the same seq128 rows has minimum/mean prompt
cosine **0.999987/0.999996**, minimum token cosine 0.999944, and max absolute
difference 0.2540 against fp32. A second CPU-fp16 capture has minimum prompt
cosine 0.999975. Checkpoint quantization and ordinary fp16 execution therefore
pass; the catastrophic result is specific to Core ML's lowering/runtime.

Failure localization on seq128:

- after the first short-conv layer, pure Core ML fp16 is already 0.99851 vs
  eager, while export is exact;
- isolating the first SwiGLU SiLU yields cosine 0.90897 in pure Core ML fp16;
- keeping only SiLU fp32 makes that isolated op 0.999999, but the full model
  still reaches only 0.986873 minimum prompt cosine (0.993709 mean);
- the remaining drift accumulates through fp16 1x1/depthwise convolutions,
  gated multiplies, and residual additions.

The `--silu-fp32` converter switch preserves this diagnostic, but it is not a
viability workaround: it still misses 0.999 and violates the intended pure
fp16 recipe. No int8 or exotic rewrite was attempted.

### Gate 3: speed and power

Timed host: Apple M1 Max, 64 GiB, macOS 26.5.2 (25F84). Runs used the bench
lock, `CPU_AND_NE`, two warmups, ten passes over 20 prompts, and macmon at
100 ms. Because fixed buckets execute pad slots, both fixed-bucket throughput
and useful active-token throughput are shown.

| bucket | p50 / p95 request | req/s | fixed-bucket tok/s | active tok/s | ANE W | CPU W | GPU W mean (p50) | compute-rail W |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 60.25 / 60.57 ms | 16.60 | 2,125.42 | 190.13 | 1.631 | 0.275 | 0.0149 (**0.000**) | **1.921** |
| 256 | 173.90 / 174.23 ms | 5.75 | 1,473.25 | 65.89 | 1.413 | 0.245 | 0.0132 (**0.000**) | **1.671** |

GPU is idle in the meaningful sense: median GPU power is zero and mean GPU
power is 13-15 mW, versus 1.4-1.6 W on ANE. Seq128 had 77 in-window samples;
seq256 had 220. macmon's `all_power` is the ANE+CPU+GPU compute-rail sum; RAM
and whole-system estimates are recorded separately by the raw capture.

A current owned-runtime Metal comparison used the **same 350M checkpoint,
M1 Max, and 20 prompts**. The owned Rust MPSGraph f16 path processes exact
lengths, builds the conv/KV decode caches, evaluates the tied LM head, and then
generates one token; Phase A Core ML executes a padded seq128 graph and returns
final hidden states without the LM head. The comparison is therefore direct in
model/input/hardware, but Core ML does less serving work per prompt.

| lane | prefill active tok/s | decode tok/s | ANE W | CPU W | GPU W | compute-rail W |
|---|---:|---:|---:|---:|---:|---:|
| Core ML seq128 | **190.13** | n/a | 1.631 | 0.275 | **0.0149** | **1.921** |
| owned Metal f16 | 11.55 | 11.68 | 0.000 | 1.396 | 0.981 | 2.377 |

On this diagnostic, Core ML hidden-state prefill is 16.5x the owned path's
serving prefill token rate and reduces compute-rail power by 19%, while leaving
the GPU effectively untouched. The result demonstrates the hardware upside;
it does not override Core ML's failed parity gate. The older repository
baselines remain useful context: `decision-1-runtime.md` records
LFM2.5-230M q8_0 at 30,278 combined tok/s and 1,171 decode tok/s on
llama-server Metal, while `LFM2-BACKBONE.md` records the owned 1.2B f16 M5
development floor at 6.5 prefill tok/s and 3.17 decode tok/s.

### Package, conversion, compilation, and load

| bucket | `.mlpackage` | `.mlmodelc` | local conversion | M1 compile | M1 first cold load | later process load |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 709,635,099 B (676.76 MiB) | 709,343,734 B | 12.94 s | 0.168 s | 5.66 s | 0.086 s |
| 256 | 709,766,171 B (676.89 MiB) | 709,474,806 B | 13.20 s | 0.131 s | 7.30 s | 0.093 s |

`MLModel.compileModel` mainly materializes the compiled bundle; first
`MLModel` load is where the large ANE artifact cost becomes visible.

## Phase B: stateful decode

`LFM2StatefulDecode` is a real token-step graph, not a toy state probe:

- inputs: token `[1,1]`, absolute position `[1]`, valid cache length `[1]`;
- output: tied-embedding logits `[1,65536]`;
- ten mutable conv states `[1,1024,3,1]`; and
- six key plus six value states `[1,8,512,64]`, implemented as a fixed rolling
  window. Total mutable buffers: 22.

`torch.export` captures all 22 buffer mutations in 0.76 s. coremltools 8.3
accepts them and creates an 843,844,763-byte (804.75 MiB) stateful package in
12.76 s. Thus neither export nor conversion is the failure boundary.

The package executes its first token on `CPU_ONLY` in 30.6 ms and returns the
expected `[1,65536]` shape. On `CPU_AND_NE`, the first prediction fails before
returning logits:

```text
com.apple.appleneuralengine Code=8
ANEProgramProcessRequestDirect() Failed with status=0x1d
statusType=0x9: Program Inference error
```

This is the Phase B boundary. Since ANE cannot execute token one, decode tok/s,
power, and 20-prompt token exactness are **not measurable** rather than zero;
the report leaves those fields null. CPU execution proves that the MLState
interface and package are structurally executable, but CPU fallback is not
the always-on prize.

## Reproduction

```bash
cd bench/spikes/ane-lfm2
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
python3.11 -m venv .reference-venv
.reference-venv/bin/pip install -r requirements-reference.txt

# Generate fp32 goldens in the separate LFM-aware reference environment.
.reference-venv/bin/python lfm2_fp32_reference.py --model "$LFM2_350M_SNAPSHOT" \
  --prompts ../unified-rt/decode-prompts.jsonl --seq-len 128 \
  --out artifacts/lfm2-350m-seq128-fp32.npz

# Conversion refuses failed parity unless explicitly asked to preserve a
# diagnostic package.
.venv/bin/python convert_lfm2_to_coreml.py --model "$LFM2_350M_SNAPSHOT" \
  --seq-len 128 --reference-npz artifacts/lfm2-350m-seq128-fp32.npz \
  --out artifacts/lfm2-350m-seq128.mlpackage \
  --report-json artifacts/lfm2-350m-seq128-conversion.json \
  --allow-parity-failure

python reference_to_jsonl.py \
  --reference-npz artifacts/lfm2-350m-seq128-fp32.npz \
  --out artifacts/prompts-seq128.jsonl
./build_runner.sh
.build/ane-lfm2 compile --model artifacts/lfm2-350m-seq128.mlpackage \
  --out artifacts/lfm2-350m-seq128.mlmodelc --stats artifacts/compile.json
.build/ane-lfm2 placement --model artifacts/lfm2-350m-seq128.mlmodelc \
  --out artifacts/placement.json
.build/ane-lfm2 run --model artifacts/lfm2-350m-seq128.mlmodelc \
  --input artifacts/prompts-seq128.jsonl --stats artifacts/run.json

# Expected exit 2 records the exact ANE stateful-runtime failure and CPU probe.
.venv/bin/python attempt_stateful_decode.py --model "$LFM2_350M_SNAPSHOT" \
  --window 512 --out artifacts/lfm2-350m-decode-state512.mlpackage \
  --report-json artifacts/lfm2-350m-decode-state512.json
```

Large checkpoints, references, Core ML packages, compiled bundles, and raw
macmon/per-operation captures stay under ignored `artifacts/`; compact
measurement records are committed under `results/`.
