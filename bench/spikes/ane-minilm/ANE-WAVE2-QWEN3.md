# ANE quiet-tier wave 2: Qwen3-Embedding-0.6B

## Verdict

Qwen3-Embedding-0.6B **does run quietly on the ANE**: all fixed buckets place
99.90% of dispatchable `MLComputePlan` operations on the locked M1 Neural
Engine, consume 4.02-4.50 W combined CPU+GPU+ANE power, and leave the GPU
idle. At seq128 it delivers 43.64 docs/s at 0.10845 J/doc; seq256 and seq512
are 19.32 docs/s / 0.23758 J/doc and 7.72 docs/s / 0.52750 J/doc respectively.

It is **not a drop-in replacement** for the frozen ORT fp32 Qwen vector space.
Every bucket misses both fingerprint gates: mean cosine is about 0.999947
against fp32, below 0.9999, and top-10 overlap is 0.987-0.990, below 0.995.
As with Wave 1 gte fp16, these ANE fp16 vectors require a separately versioned
index; they must not be mixed with the existing ORT/Metal index.

Product conclusion: the quiet ladder tops out at gte for a drop-in existing
index and for lowest energy (gte seq128 was 0.02623 J/doc). A 600M-class
encoder is nevertheless **real quiet serving** when a dedicated Qwen ANE
index is acceptable: Qwen seq128 stays under 5 W with effectively no GPU
residency and uses about one sixth of the locked-M1 Metal energy per document.
The 1.11 GiB-per-bucket package cost, the non-substitutable fingerprint, and
the 4.46% real-token loss at seq128 make it a deliberate serving tier rather
than the default quiet path.

## Toolchain, model policy, and conversion

Packages were built on Apple M5 Max, macOS 26.5.2 (`25F84`), with Python
3.12.12 and these exact pins:

- torch 2.5.1
- coremltools 8.3.0
- transformers 4.51.3
- tokenizers 0.21.0
- safetensors 0.5.3
- numpy 2.3.2

`transformers` 4.51.3 is Qwen3 support; Wave 1's 4.48 pin does not recognize
the `qwen3` model type. `requirements-qwen3.txt` keeps the proven torch 2.5.1
and coremltools 8.3 pair while documenting that required Qwen change.
coremltools emits its expected warning that torch 2.5.1 is newer than its
validated 2.5.0, but conversion and executed M1 parity completed.

The source is `Qwen/Qwen3-Embedding-0.6B` revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`; `model.safetensors` SHA-256 is
`0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`.
Every package is batch 1, fixed seq128/256/512, fp16,
`CPU_AND_NE`, and unquantized. No dynamic shapes, int8, W8A8, or TorchScript
trace package was used.

`convert_qwen3_to_coreml.py` uses **`torch.export` only**. It follows the
ANE-friendly 1x1 Conv2d projection layout from the Qwen-like PR #169 pattern,
but retains official Qwen behavior rather than reusing that PR's bidirectional
mask and mean pooling:

- causal plus key-padding attention mask;
- 28 causal decoder layers, GQA 16 query / 8 key-value heads;
- per-head Q/K RMSNorm before RoPE with theta 1,000,000;
- SwiGLU MLP and pre-RMSNorm residual blocks;
- left-padded fixed inputs with model-config terminal EOS `151643` in the
  final position; last-token pooling and L2 normalization occur in Core ML.

Left padding removes a dynamic last-token gather while preserving causal Qwen
results; the terminal EOS is the model config's token, not the tokenizer
config's distinct `<|im_end|>` token. The two non-ANE dispatches reported
below are Core ML's input gather and cast, not a pooling fallback.

```sh
cd bench/spikes/ane-minilm
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-qwen3.txt
./build_runner.sh

for bucket in 128 256 512; do
  .venv/bin/python convert_qwen3_to_coreml.py --seq-len "$bucket" \
    --out "/tmp/ane-wave2/models/qwen3-seq${bucket}.mlpackage" \
    --report-json "/tmp/ane-wave2/reports/qwen3-seq${bucket}.json"
done
```

### Export and conversion smoke parity

The custom Qwen wrapper exactly matched eager Hugging Face Qwen before export;
the table gives the mean cosine of two real texts after Core ML execution.
The Core ML value is intentionally a conversion smoke threshold, not the
stricter 400-row vector-space substitution gate.

| Bucket | eager wrapper vs HF max abs | eager vs exported max abs | Core ML max abs | Core ML mean cosine | Result |
|---:|---:|---:|---:|---:|---|
| 128 | 0 | 0 | 0.00156868 | 0.99991751 | conversion pass |
| 256 | 0 | 0 | 0.00181272 | 0.99990946 | conversion pass |
| 512 | 7.45e-09 | 0 | 0.00145611 | 0.99991894 | conversion pass |

The conversion smoke is safely above the 0.999 conversion-bug stop threshold.
The stable 400-row fingerprint below shows that its smaller fp16 difference is
a genuine new vector space, not a broken export.

## Locked-M1 protocol

Host: `[bench-host]`, MacBookPro18,2, Apple M1 Max, 64 GiB, macOS
26.5.2 (`25F84`). The canonical corpus is the 400-row Qwen text corpus,
SHA-256 `5a9bfdc8c069657aa46cbb45bef91bc1a0ddc72602bfb96b189af31ba55f630c`.
The frozen ORT fp32 vector reference SHA-256 is
`cacee1f64d12704ea94cded9861f6aef903a018800b2e0a1ec67589c33c7cf46`.

Every accepted M1 timed cell acquired `[bench-user-home]/bench.lock`, rejected an
active `Runner.Worker`, started macmon at a requested 100 ms interval, waited
for its first sample and two more seconds, then started sudo-primed
powermetrics at 100 ms. A trap killed both samplers and released the lock.
The password was supplied only to `sudo -S -v` in the operator-approved SSH
session; it is not stored in a script, result, or this repository.

Power windows start at runner invocation and end at runner exit, so their
J/doc includes the fresh-process package load, warmup, inference, and small
process overhead. This is deliberately conservative for a 1.2 GB model.
macmon's exact timestamp-filtered window is the primary column; powermetrics
is an independent privileged cross-check. Raw artifacts remain under
`[bench-host-alias]:~/bench-tools/ane-wave2/results/`; compact committed evidence is in
`results/wave2-qwen3-summary.json`.

## Placement proof

`MLComputePlan` was loaded from every compiled package with `CPU_AND_NE`.
Constants have no runtime dispatch and are excluded from the dispatchable
denominator.

| Bucket | Dispatchable ops | ANE | CPU | ANE share | CPU operations |
|---:|---:|---:|---:|---:|---|
| 128 | 2,093 | 2,091 | 2 | 99.904% | `ios17.gather`, `ios17.cast` |
| 256 | 2,093 | 2,091 | 2 | 99.904% | `ios17.gather`, `ios17.cast` |
| 512 | 2,093 | 2,091 | 2 | 99.904% | `ios17.gather`, `ios17.cast` |

This clears the 80% placement stop gate by a wide margin. The plan contains
3,081 `const` nodes reported as unknown; they are not execution dispatches.

## Parity and fingerprint

`prep_qwen3_tokenized_jsonl.py` generated bucket-specific, left-padded input
JSONL with the terminal model EOS. The frozen fp32 corpus vectors apply
unchanged to seq256 and seq512 because their 46,716 active tokens match the
canonical corpus. Seq128 has 44,630 active tokens after truncation, so the
shortened rows were re-executed using
`qwen3_fp32_reference.py` and the same EOS/pooling policy rather than being
compared to a differently tokenized frozen vector. The canonical parity
harness evaluated all 400 rows as rank queries with `k=10` and stride 4.

| Bucket | Active tokens | Mean cosine | Minimum cosine | top-10 overlap (100 queries) | Required | Verdict |
|---:|---:|---:|---:|---:|---|---|
| 128 | 44,630 | 0.99994617 | 0.99978153 | 0.987 | cosine >= 0.9999; overlap >= 0.995 | **FAIL** |
| 256 | 46,716 | 0.99994718 | — | 0.990 | cosine >= 0.9999; overlap >= 0.995 | **FAIL** |
| 512 | 46,716 | 0.99994684 | — | 0.990 | cosine >= 0.9999; overlap >= 0.995 | **FAIL** |

The fp16 Core ML ANE result is therefore a separate Qwen fingerprint in every
bucket. It may serve a newly built, entirely Qwen-ANE-fp16 index, but cannot
replace or incrementally update the frozen ORT fp32 / owned-Metal Qwen index.

## Locked-M1 throughput, latency, power, and energy

The runner used `MLArrayBatchProvider` batch size 8 over the 400 prepared
batch-1 rows. Mean latency is `infer_wall_s / 400`, not a p50 claim.

| Bucket | docs/s | mean ms/doc | real tok/s | CPU W | GPU W | ANE W | combined W | J/doc (macmon) | J/doc (powermetrics) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 43.64 | 22.91 | 4,869.38 | 0.406 | 0.015 | 4.080 | 4.501 | 0.10845 | 0.10779 |
| 256 | 19.32 | 51.75 | 2,256.86 | 0.404 | 0.015 | 4.065 | 4.484 | 0.23758 | 0.23568 |
| 512 | 7.72 | 129.56 | 901.46 | 0.281 | 0.015 | 3.726 | 4.022 | 0.52750 | 0.52155 |

powermetrics combined-power divergence from macmon is 0.62%, 0.80%, and 1.14%
for seq128/256/512 respectively, well below the 10% cross-check flag. The
GPU stayed quiet: mean effective GPU use was 0.378-0.420%, maximum use was
3.09-3.58%, and maximum GPU power was 0.117-0.171 W. This is real ANE
execution rather than a `CPU_AND_NE` configuration that fell onto Metal.

For context, the locked-M1 Metal Qwen lane in `GRADUATION-PROBE.md` achieved
about 7,000 real tok/s at 42.66 W GPU. ANE seq128 reaches 4,869 real tok/s at
4.50 W combined and 0.108 J/doc; it trades throughput for a materially lower
energy and thermal envelope. The comparison is directional because the Metal
lane uses its own variable exact-shape batching, while this experiment uses
fixed static buckets.

## Package, load, RSS, and residency costs

Every fp16 package is about 1.11 GiB on disk. `first load after compile` is a
one-row first dispatch immediately after `MLModel.compileModel`; the timed
`fresh-process load` comes from the 400-row power cell after the package root
had already been established. RSS is the runner process only; Core ML's shared
mapped or accelerator-resident weights are not attributed reliably to that
process RSS.

| Bucket | `.mlpackage` size | first load after compile | timed fresh-process load | peak runner RSS | 400-doc infer wall |
|---:|---:|---:|---:|---:|---:|
| 128 | 1,165,324 KiB | 12.64 s | 0.191 s | 79.58 MiB | 9.165 s |
| 256 | 1,165,484 KiB | 18.13 s | 0.222 s | 82.66 MiB | 20.700 s |
| 512 | 1,165,996 KiB | 30.69 s | 0.312 s | 90.06 MiB | 51.823 s |

A planned `128 -> 256 -> 512 -> 128` alternating-load sequence was correctly
aborted twice when `Runner.Worker` was active, rather than contaminating the
locked baseline. The observed timed cells did not force a reload, but this
wave does **not** claim an explicit four-package ANE-residency eviction bound.
Serving should retain the selected bucket and treat a cross-bucket switch as a
potential cold-path operation until that dedicated experiment can run under
an idle lock.

## Contended-M5 secondary datapoint

This is not a graduation measurement. The local Apple M5 Max was actively
contended (mean GPU effective use 16.83%, peak 50.71%), had no privileged
powermetrics capture, and includes a 28.83 s cold model load. It is useful
only to show the newer host's observed shape, not to establish quiet behavior.

| Host / bucket | docs/s | mean ms/doc | real tok/s | cold load | infer wall | CPU W | GPU W | ANE W | combined W | J/doc | peak runner RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| contended M5 / 256 | 37.14 | 26.93 | 4,337.42 | 28.83 s | 10.770 s | 14.733 | 1.703 | 2.023 | 18.459 | 1.86909 | 785.14 MiB |

Its mean cosine versus the frozen ORT fp32 seq256 vectors was 0.99994674, matching
the M1 fp16 fingerprint. Its high power and GPU use are ambient contention,
not evidence against the clean locked-M1 quiet result.

## Follow-up

1. Keep Qwen ANE fp16 behind an explicit fingerprint/version boundary and do
   not mix it with existing Qwen fp32 or owned-Metal embeddings.
2. Evaluate retrieval quality and truncation loss before selecting seq128 as a
   product policy; it removes 2,086 of the corpus's 46,716 active tokens.
3. Run the alternating-package residency test only when the locked M1 is idle;
   report a real cross-bucket swap bound before claiming concurrent bucket
   residency.
4. Preserve fp16 weights. The evidence does not justify encoder int8/W8A8
   experimentation, which is already known to collapse encoder fidelity.
