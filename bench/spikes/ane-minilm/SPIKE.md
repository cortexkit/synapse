# Spike: MiniLM on the Apple Neural Engine

## Status

**Result: quiet tier is real on M1-generation ANE for MiniLM, with honest caveats.**

On the dedicated **M1 Max** bench box (`MacBookPro18,2`, macOS `26.5.2`), the fixed-bucket Core ML path for `all-MiniLM-L6-v2`:

- kept **94.8% of dispatchable ML Program ops on the Neural Engine** for both `seq=256` and `seq=512`,
- kept the **GPU effectively idle** during inference (`0.7-1.1 mW` average GPU power; `2-6 mW` max; `0.95-1.55%` max GPU active residency),
- cleared the requested parity bar comfortably on the real 1,000-chunk corpus subset,
- and ran in the **~2.8-3.5 W combined CPU+GPU+ANE** band measured by `powermetrics`.

That is not the original ~2 W total-system hypothesis, but it is still dramatically quieter than the existing GPU lanes cited in the decision context, and it is a clean positive for an ANE-served encoder tier on M1-class hardware.

## Measurement setup

- Bench host: **M1 Max** over SSH (`<bench-host>`), machine hash `MacBookPro18,2`.
- Corpus: first **1,000** rows of `corpus/aft-chunks.jsonl`, field `embed_text`.
- Corpus integrity check: local and M1 copies matched exactly.
  - SHA-256: `1c36b742095318fe2fe5d0bd221974c94de9ce834e1dbee509eae2480cbe1479`
- Model source for conversion: `sentence-transformers/all-MiniLM-L6-v2` from local HF cache.
- Reference tokenizer / ONNX snapshot: `Qdrant/all-MiniLM-L6-v2-onnx` local cache.
- Final Core ML packages: fp16, fixed buckets `256` and `512`, `CPU_AND_NE`.
- Runner batching: `--batch-size 8`.
- Power tool: `sudo powermetrics -i 500 -s ane_power,gpu_power,cpu_power`.
- Energy calculation: **average Combined Power (CPU + GPU + ANE)** over the full embed invocation window (`cold_load_s + infer_wall_s`), divided by 1,000 docs.

## Important implementation note: trace probe vs export final path

I did try the straightforward TorchScript trace path first.

What happened in local smoke before the M1 timed run:

- `seq=256`: trace conversion was inconsistent across experiments.
- `seq=512`: trace conversion could succeed, but the trace-built model missed parity badly (**~0.359 mean cosine** on a local smoke subset).

Because of that, the committed conversion script's `--frontend auto` mode now:

1. probes the trace path for evidence,
2. records the outcome in a report JSON,
3. but writes the final package with **`torch.export`**.

No Conv2d reshape surgery was needed for this MiniLM proof because the plain export-backed Core ML model already placed well on the ANE.

## 1) Placement gate (MLComputePlan on the M1)

`dispatchable_device_share` is the meaningful residency number here; the large `unknown` bucket is Core ML `const` nodes with no runtime dispatch.

| Bucket | Dispatchable ops | NE ops | CPU ops | NE share | CPU fallback ops |
|---|---:|---:|---:|---:|---|
| 256 | 154 | 146 | 8 | **94.8%** | `cast`×2, `gather`×1, `add`×2, `layer_norm`×1, `expand_dims`×2 |
| 512 | 154 | 146 | 8 | **94.8%** | same |

### Placement verdict

The encoder blocks are genuinely landing on the Neural Engine. The spike does **not** fail its ANE claim.

## 2) Parity gate on the real 1,000-chunk set

Parity was computed against fp32 ORT reference vectors produced from the **same pretokenized JSONL** that was later rsynced to and consumed on the M1 run.

| Bucket | Mean cosine vs ORT fp32 | Rank overlap (k=10, stride=4) | Gate |
|---|---:|---:|---|
| 256 | **0.9999841970** | **0.994** | PASS |
| 512 | **0.9999841258** | **0.992** | PASS |

Both buckets clear the requested `>= 0.99` cosine and `>= 0.95` rank-overlap gates by a wide margin.

## 3) Timed M1 runs

These are the timed 1,000-doc embed runs on the M1 box.

| Bucket | Cold load (s) | Infer wall for 1,000 docs (s) | Docs/s | Output dim |
|---|---:|---:|---:|---:|
| 256 | 0.0677 | **2.6643** | **375.33** | 384 |
| 512 | 2.0765 | **9.0603** | **110.37** | 384 |

### Throughput read

- `256` is the useful fast path on this M1 box.
- `512` still works and stays on-ANE, but the cold-load penalty is materially larger and steady-state throughput is ~3.4x lower.

## 4) Power / J per doc (powermetrics)

### Average power during the full embed invocation

| Bucket | Avg CPU mW | Avg GPU mW | Avg ANE mW | Avg combined mW | Samples |
|---|---:|---:|---:|---:|---:|
| 256 | 518.3 | 0.7 | **2949.7** | **3469.0** | 3 |
| 512 | 675.5 | 1.1 | **2162.4** | **2838.9** | 20 |

### Energy per doc

| Bucket | Full runner wall (cold + infer, s) | Combined J/doc |
|---|---:|---:|
| 256 | 2.7320 | **0.00948 J/doc** |
| 512 | 11.1367 | **0.03162 J/doc** |

### GPU-idle evidence

| Bucket | Avg GPU power | Max GPU power | Max GPU active residency |
|---|---:|---:|---:|
| 256 | 0.7 mW | 2 mW | 0.95% |
| 512 | 1.1 mW | 6 mW | 1.55% |

Interpretation: the GPU stayed essentially idle while the ANE carried the workload.

## 5) Honest caveats

1. **This is an ANE-positive result, but not a 2 W total-system result on this M1 Max.**
   - The combined CPU+GPU+ANE domain sat around **2.84-3.47 W** here.
   - That is still quiet-tier material relative to our GPU lanes, just not as low as the initial hypothesis.

2. **`seq=512` is usable but much less attractive than `seq=256`.**
   - Same parity.
   - Same MLComputePlan residency share.
   - Significantly worse throughput and J/doc.

3. **`torch.export` is the safe baseline.**
   - The local smoke found a trace-built `seq=512` model that converted but failed parity.
   - The spike therefore standardizes on export-backed packages for both buckets.

4. **The power numbers are subsystem-domain numbers, not whole-machine wall-plug numbers.**
   - They come directly from the requested `powermetrics` counters: CPU, GPU, and ANE.

## Verdict

**Yes — the quiet tier is real on M1-generation ANE for MiniLM.**

The supported read from this spike is:

- fixed-bucket (`256` / `512`) Core ML encoder serving is viable,
- parity against ORT fp32 is excellent,
- MLComputePlan shows the model is mostly on the Neural Engine,
- the GPU remains effectively idle,
- and the measured power band is low enough to justify an ANE-backed "quiet" tier.

For the next **ModernBERT-class** follow-up, the main changes are:

1. keep the same **fixed-bucket** deployment shape discipline,
2. start from the **export-backed** conversion path, not raw trace,
3. verify **MLComputePlan** before doing any surgery,
4. only reach for the Linear→Conv2d / 4D-layout tricks if placement regresses,
5. expect `512`-class or longer buckets to make cold-load and J/doc materially more important.

If the product question is "does Synapse have a credible ANE quiet tier on M1-class hardware?", this spike's answer is **yes**.
