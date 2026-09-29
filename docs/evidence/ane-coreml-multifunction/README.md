# One multi-function Core ML package for the ANE embedding buckets

**Question.** Can one multi-function Core ML package holding the 128, 256 and 512
buckets of `gte-modernbert-base-ane-fp16` replace the three single-bucket
packages we ship, with the same vectors, the same Neural Engine placement and
similar speed, at a much smaller size?

**Answer: yes on size, vectors, placement and warm latency; it costs about
2x on warm load.** `ct.utils.save_multifunction` deduplicated the weights. The
compiled three-function bundle is 301.6 MB, against 897.8 MB for the three
compiled single-bucket bundles (-66.4%). Every function's vectors are
byte-identical to the installed production bundle's on every row. Every
function's Core ML compute plan matches production exactly: 846 of 851
dispatchable operations prefer the Neural Engine, a share of 0.9941245593419507.
Warm latency is within 0.4% of production in every bucket. The cost: each
function takes about 70-90 ms to load warm instead of 30-50 ms, and the process
footprint with all three loaded is about 12 MB larger. Cold first loads are the
same as single-bucket packages; they are not cheaper. A 1024 function added from
the same model places the same way, is byte-identical to a single 1024 package,
and scores min cosine 0.99912 against fp32 PyTorch on the repeated-prose rows,
with no overflow seen. It adds 7.3 MB to the compiled bundle.

Scope: this is an evidence-only spike. No production code, Swift worker or model
catalog entry changed.

## Setup

- macOS 27.0 (26A428), arm64. Python 3.12.12, coremltools 8.3.0, torch 2.5.1,
  transformers 4.48.0: the pins in `bench/spikes/ane-minilm/requirements.txt`.
- Source model: `Alibaba-NLP/gte-modernbert-base`, local Hugging Face snapshot
  `e7f32e3c00f91d699e8c43b53106206bcc72bb22`.
- Scripts: `bench/spikes/ane-multifunction/`. `build.py` builds the packages,
  `build_probe.sh` builds the Swift Core ML probe (`probe.swift`), and `run.py`
  runs the stages below and writes `EVIDENCE.json` in this directory. Packages,
  compiled bundles, vectors and raw reports stay out of git, under
  `~/.local/share/cortexkit/synapse/ane-multifunction/`.
- Every probe process that touches the Neural Engine first waits until
  `campaign_rig_claim` in the prefrontal store (opened `?mode=ro`) is empty and
  the 1-minute load average is at or below 16. The latency probe repeats that
  check before every timed run. No campaign held the rig at any point. The load
  stage waited 50 s at the start for the load average to drop from 24.5.

## 1. Single-bucket packages and their match to production

`build.py` imports `bench/spikes/ane-minilm/convert_modernbert_to_coreml.py` and
calls its own `load_wrapper`, `smoke_inputs`, `convert_and_verify` and
`ensure_metadata`. That is torch.export, fp16, `CPU_AND_NE`, a macOS 14 deployment
target, CLS pooling and L2 normalization in the graph, and an output named
`embedding`. The production smoke-parity gate passed for 128, 256, 512 and 1024.

The installed production bundles were located by their materialization records
and copied to fresh stable paths as the baseline:

| bucket | installed source digest | installed materialized digest |
|---|---|---|
| 128 | `sha256:8c3bf4b2a50634ec4a3eb54e986b769c84d0d11946c281ecd88d24827afd7d80` | `sha256:78c82761765daaf252b3c10e76ab57fb82245d7eb8b885593f3e82acff4d16cf` |
| 256 | `sha256:a8626c487794f879b88c73bf9c8fe6f7864e3cc252a1db0d79bb5e096210b9fd` | `sha256:553dad5320920690cc7a570a890a22fd27e6ea40285d11fe4350748bfecc676f` |
| 512 | `sha256:9df6b44617b49e08a068602efc3fddcd7dadc0e46791929698b524c96d544f72` | `sha256:60e12dcd58ea78f55adaab9b0eace83e53afa9fdc8394c03fae14c8c93b85eff` |

The rebuilt bundles' `weights/weight.bin` is byte-identical (same SHA-256) to
production for all three buckets. The compiled `model.mil` differs only in its
`buildInfo` line: production was compiled with coremlc 3520.5.1, this machine
has coremlc 3600.25.2. The rest of the file is identical. Both bundles gave
byte-identical vectors on every row (section 4). The rebuilt singles and the
installed production bundles are therefore the same model; production is the
baseline in every comparison below.

## 2. Multi-function package: did coremltools share the weights?

The multi-function packages were assembled from the three (and four) single
`.mlpackage`s with `MultiFunctionDescriptor.add_function(path, "main", "seqN")`
and `save_multifunction`, default function `seq128`.

| artifact | `.mlpackage` bytes | `weight.bin` bytes | compiled `.mlmodelc` bytes |
|---|---:|---:|---:|
| single seq128 | 298,728,690 | 298,109,248 | 298,564,424 |
| single seq256 | 298,843,378 | 298,174,784 | 298,941,640 |
| single seq512 | 299,171,058 | 298,305,856 | 300,285,896 |
| **three singles, total** | **896,743,126** | **894,589,888** | **897,791,960** |
| **multi-function seq128/256/512** | **299,754,550** | **298,502,976** | **301,595,442** |
| single seq1024 | 300,219,637 | 298,568,000 | 305,334,814 |
| four singles, total | 1,196,962,763 | 1,193,157,888 | 1,203,126,774 |
| multi-function seq128/256/512/1024 | 301,629,965 | 299,027,520 | 308,850,112 |

- The weights are shared. The three-function `weight.bin` is 393,728 bytes
  larger than the seq128 package's alone. That residue is the per-bucket
  constants, which differ by sequence length. The compiled bundle is 66.4%
  smaller than three singles (74.3% smaller than four singles with 1024).
- What remains per function is mostly the text `model.mil`: 3.09 MB for three
  functions, 9.82 MB for four. The 1024 function adds 0.52 MB of weights and
  about 6.7 MB of program text.
- Format change: `save_multifunction` raises the specification version to 9
  (iOS 18 / macOS 15), whatever the inputs. The compiled program is
  `program(1.3)` with `ios18` opset functions; the production singles are
  `ios17` with a macOS 14 target. A shipped multi-function package therefore
  needs macOS 15 or later.
- `save_multifunction` took 6.4 s for three functions and 9.3 s for four.
  `MLModel.compileModel` took 217 ms for the
  three-function package and 86-95 ms per single (323 ms with four functions).

## 3. Placement (`MLComputePlan`, `cpuAndNeuralEngine`)

The bundles were compiled once to stable paths and never loaded from the
temporary compile output. Each function was planned with
`MLModelConfiguration.functionName` set, and that function's operations were read
from the plan. The share uses the production worker's formula: operations
preferring the Neural Engine, divided by operations with a known preferred
device.

| model | Neural Engine | CPU | unknown | share |
|---|---:|---:|---:|---:|
| production seq128 / seq256 / seq512 | 846 | 5 | 1,042 | 0.9941245593419507 |
| multi-function seq128 / seq256 / seq512 | 846 | 5 | 1,042 | 0.9941245593419507 |
| single seq1024 | 846 | 5 | 1,042 | 0.9941245593419507 |
| four-function seq128 / 256 / 512 / 1024 | 846 | 5 | 1,042 | 0.9941245593419507 |

The CPU operations are the same five in every model and function: one `cast`,
two `expand_dims`, one `gather` and one `tile`, the input embedding and mask
setup. Only their opset prefix differs (`ios18.` against `ios17.`). The share
equals the production worker's reported `placement_share=0.9941245593419507`.
This is preferred placement, not a runtime dispatch trace. Warm latency matching
production (section 6) is consistent with the same runtime placement.

## 4. Vectors

Rows: the eight rows of `bench/spikes/ane-direct-probe/rows.jsonl`, padded the
way the production worker pads (token 0, mask 0).

- The four short rows (12, 12, 38 and 21 tokens) are used as declared.
- The four near-limit rows are declared only at 512, 1024 and 2048. Each is the
  2048 row's content prefix wrapped in CLS/SEP, with lengths at fixed fractions
  of the shape (7/8, 15/16, 500/512, shape minus 1). The declared 512 and 1024
  rows were checked to equal that construction. The same construction gives
  112/120/125/127 tokens at 128 and 224/240/250/255 at 256.

| comparison | bucket | rows | byte-identical | max abs diff | min cosine |
|---|---|---:|---|---:|---:|
| multi-function vs production | 128 | 8 | all | 0 | 1.0 |
| multi-function vs production | 256 | 8 | all | 0 | 1.0 |
| multi-function vs production | 512 | 8 | all | 0 | 1.0 |
| four-function vs production | 128 / 256 / 512 | 8 each | all | 0 | 1.0 |
| four-function seq1024 vs single seq1024 | 1024 | 8 | all | 0 | 1.0 |
| production vs fp32 PyTorch | 128 | 8 | no | 0.00893 | 0.998899 |
| production vs fp32 PyTorch | 256 | 8 | no | 0.00928 | 0.998999 |
| production vs fp32 PyTorch | 512 | 8 | no | 0.0108 | 0.999171 |
| single / four-function seq1024 vs fp32 PyTorch | 1024 | 8 | no | 0.00994 | 0.999122 |

The fp32 PyTorch rows are the converter's eager wrapper on the same padded
inputs. They show the vectors under comparison are real embeddings, not a
degenerate output that would compare equal trivially. Core ML returned `float32`
vectors from every model; float equality therefore means byte equality.

## 5. Load time

Cold: each bundle was copied to a path Core ML had never loaded, and each
function's first load happened in its own fresh process. Warm: five more fresh
processes per function on the same path. Three trials alternated which variant
went first. The load average stayed between 7.8 and 15.1 throughout.

| model | cold first load, ms (3 trials) | warm load median, ms (15) | first predict after warm load, ms |
|---|---|---:|---:|
| production seq128 | 2565, 2523, 2665 | 40.1 | 5.6 |
| production seq256 | 3719, 3661, 3597 | 44.0 | 11.4 |
| production seq512 | 3995, 4032, 4008 | 49.6 | 28.0 |
| multi-function seq128 | 2647, 2618, 2663 | 85.9 | 5.6 |
| multi-function seq256 | 3701, 3844, 3745 | 86.6 | 11.0 |
| multi-function seq512 | 4275, 4226, 6087 | 91.8 | 28.4 |

- Cold: each function pays its own Neural Engine specialization. Loading seq128
  first did not make seq256 or seq512 cheaper afterwards, so the cold total for
  three functions (~10.6 s) matches three singles (~10.3 s). The 6,087 ms
  outlier is a single sample.
- Warm: every function load costs roughly twice a single-bucket load, and the
  cost is flat across buckets. That fits each load parsing the whole
  multi-function program, but this spike did not isolate the cause.
- In one process, loading all three models warm took a median 325 ms for the
  three functions against 160 ms for three production packages (6 samples each,
  memory stage).

## 6. Warm latency

One process loaded both packages per bucket, warmed each three times, then ran
30 interleaved iterations. The order within each bucket pair flipped every
iteration. One run embeds all eight rows of the bucket, one prediction per row
at batch 1. The 1-minute load average was recorded before and after every run.
It peaked at 7.69, and no run waited on the gate. (An earlier 25-iteration pass
gave the same medians within 1 ms. One of its runs ended with the load average
at 17.1, so it was superseded; its numbers are not reported.)

| bucket | production median, ms (p10-p90) | multi-function median, ms (p10-p90) | delta |
|---|---|---|---:|
| 128 | 28.11 (27.69-30.96) | 28.11 (27.66-31.29) | 0.0% |
| 256 | 72.37 (72.04-84.62) | 72.29 (72.10-85.28) | -0.1% |
| 512 | 202.94 (202.61-248.37) | 203.21 (202.88-242.22) | +0.1% |
| 1024 (single vs four-function) | 586.33 (586.12-668.18) | 586.97 (586.66-699.82) | +0.1% |

## 7. Memory

Each run was a fresh process that loaded all three models, ran one prediction on
each, and recorded `task_vm_info` (six runs per variant, alternating).

| variant | footprint after load (median) | footprint after predict (median) | resident after predict (median) |
|---|---:|---:|---:|
| three production packages | 40.2 MB | 42.5 MB | 54.8 MB |
| one multi-function package, three functions | 52.1 MB | 54.4 MB | 66.7 MB |

The process footprint is about 12 MB larger with the multi-function package.
Neither number contains the 298 MB of weights: the Neural Engine's model buffers
are not charged to the loading process. Process memory therefore cannot show
whether the Neural Engine holds the shared weights once or three times. The
driver also sampled system-wide wired memory (`vm_stat`) while the models were
held loaded. The deltas swung from -5.0 GB to +7.3 GB on this shared machine,
pure noise, so the Neural Engine-side memory question is left open.

## 8. The 1024 function

Added from the same model and conversion settings, as a fourth function. Its
placement is identical to the other buckets (share 0.9941245593419507, same five
CPU operations). Its vectors are byte-identical to a single-bucket 1024 package.
Against fp32 PyTorch it scores min cosine 0.999122 (max abs 0.00994). That is on
rows of 12 to 1023 real tokens, including four repeated-prose rows of 896, 960,
1000 and 1023 tokens. No non-finite values and no fp16 overflow appeared on these
rows; nothing was changed to prevent any. Warm latency is 586.97 ms per
eight-row run, against 586.33 ms for the single package. Two limits on this
check: these are eight rows of ordinary prose, and the conversion-time smoke gate
only exercises short inputs. Longer lengths (2048 and up) were not tried.

## What this does not settle

- Production adoption would need the Swift worker to load one bundle once per
  bucket with `functionName`, a macOS 15 floor, and a new packaging and digest
  scheme. None of that was touched.
- Whether the Neural Engine keeps one copy or several of the shared weights at
  run time (section 7).
- The source of the roughly 2x warm-load cost per function (section 5).
