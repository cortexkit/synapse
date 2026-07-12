# ANE quiet-tier wave 1

## Verdict

The ModernBERT encoder graphs dispatch strongly to the M1 Neural Engine, but the fp16 embedding fingerprint does **not** clear the production substitution gate. `gte-modernbert-base` reaches 97.7-98.1% top-10 overlap and 0.99932-0.99933 mean cosine against ORT fp32, below the required 0.995 and 0.9999 gates. These vectors are a distinct vector space and must not be mixed with the ORT/Metal fp32 space.

The reranker clears its scoring gates in every fixed bucket. Its useful quiet configuration is seq128: a repeated 1x50 request measures 313.82 ms p50 / 320.28 ms p95 on the locked M1, inside the approximately 0.5 s AFT budget. Seq256 and seq512 miss the latency budget. MiniLM conversion, parity, and placement also reproduce on the current toolchain.

## Toolchain and conversion

All packages were produced on Apple M5 Max, macOS 26.5.2 (`25F84`), Python 3.12.12, with exact pins:

- torch 2.5.1
- coremltools 8.3.0
- transformers 4.48.0
- tokenizers 0.21.0
- numpy 2.3.2

coremltools 8.3 prints that torch 2.5.1 is newer than its tested torch 2.5.0. This is the same torch/coremltools pin pair used by the MiniLM spike; conversion and executed parity succeeded. Every package records these versions and macOS in Core ML user metadata and its conversion report.

Only `torch.export` is used. TorchScript tracing is disabled in both converters because the earlier trace-built MiniLM package converted but produced catastrophic parity. Models are fixed batch 1, fixed sequence 128/256/512, fp16, `CPU_AND_NE`, without quantization. The embedder performs CLS selection and L2 normalization inside Core ML. The reranker performs masked mean plus the checkpoint's dense/GELU/norm/classifier head inside Core ML and returns one raw logit.

Executed conversion pattern:

```sh
uv venv --python 3.12 bench/spikes/ane-minilm/.venv
uv pip install --python bench/spikes/ane-minilm/.venv/bin/python -r bench/spikes/ane-minilm/requirements.txt
for kind in embedder reranker; do
  for bucket in 128 256 512; do
    python bench/spikes/ane-minilm/convert_modernbert_to_coreml.py \
      --kind "$kind" --seq-len "$bucket" \
      --out "/tmp/ane-wave1/models/${kind}-seq${bucket}.mlpackage" \
      --report-json "/tmp/ane-wave1/reports/${kind}-seq${bucket}.json"
  done
done
```

### Export and conversion smoke parity

The smoke uses two real texts for embedding and the two query/document pairs from `unified-rt/fixtures/rerank-reference.json` for reranking.

| Model | Bucket | eager vs exported max abs | Core ML parity | Result |
|---|---:|---:|---:|---|
| gte-modernbert-base | 128 | 0 | cosine 0.99987185 | conversion pass |
| gte-modernbert-base | 256 | 0 | cosine 0.99987185 | conversion pass |
| gte-modernbert-base | 512 | 0 | cosine 0.99987185 | conversion pass |
| gte-reranker-modernbert-base | 128 | 0 | Pearson 1.0; max abs 0.03428 | conversion pass |
| gte-reranker-modernbert-base | 256 | 0 | Pearson 1.0; max abs 0.03428 | conversion pass |
| gte-reranker-modernbert-base | 512 | 0 | Pearson 1.0; max abs 0.03428 | conversion pass |

The smoke cosine is above the 0.999 conversion-bug stop threshold but already below the 0.9999 production fingerprint gate. The 400-row result below confirms that this is a stable fp16 fingerprint, not an export error.

## Locked M1 protocol

Host: `MacBookPro18,2`, Apple M1 Max, macOS 26.5.2 (`25F84`). Each measured process acquired `[bench-user-home]/bench.lock`, rejected an active `Runner.Worker`, started macmon, waited for its first sample and then another two seconds, and used a trap to kill macmon and release the lock. The executed protocol was:

```sh
until mkdir [bench-user-home]/bench.lock 2>/dev/null; do sleep 30; done
if pgrep -f Runner.Worker >/dev/null; then
  rmdir [bench-user-home]/bench.lock
  exit 75
fi
pid=""
trap '[[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true; rmdir [bench-user-home]/bench.lock 2>/dev/null || true' EXIT INT TERM HUP
[bench-user-home]/bench-tools/bin/macmon -i 100 pipe > "$MACMON_JSONL" 2> "$MACMON_LOG" &
pid=$!
for _ in {1..100}; do [[ -s "$MACMON_JSONL" ]] && break; sleep 0.1; done
sleep 2
python3 -c 'import time; print(time.time())' > "$START_EPOCH"
"$RUNNER" run ...
python3 -c 'import time; print(time.time())' > "$END_EPOCH"
kill "$pid" 2>/dev/null || true
wait "$pid" 2>/dev/null || true
pid=""
rmdir [bench-user-home]/bench.lock
trap - EXIT INT TERM HUP
```

Raw macmon windows and outputs remain at `[bench-host-alias]:~/bench-tools/ane-wave1/results/`. `results/wave1-summary.json` contains the filtered power summaries. Power is macmon's CPU, GPU, and ANE domains; combined energy uses their sum over runner cold-load plus inference time.

## Placement proof

MLComputePlan was loaded from every compiled package with `CPU_AND_NE`. Unknown constants are excluded from dispatchable counts. Full dumps are committed in `results/`.

| Model | Bucket | Dispatchable ops | ANE | CPU | ANE share |
|---|---:|---:|---:|---:|---:|
| embedder | 128 | 851 | 842 | 9 | 98.94% |
| embedder | 256 | 851 | 842 | 9 | 98.94% |
| embedder | 512 | 851 | 842 | 9 | 98.94% |
| reranker | 128 | 858 | 837 | 21 | 97.55% |
| reranker | 256 | 858 | 837 | 21 | 97.55% |
| reranker | 512 | 858 | 837 | 21 | 97.55% |

## Embedder parity and fingerprint

The first 400 `corpus/aft-chunks.jsonl` rows were pretokenized independently for each bucket. ORT fp32 used the cached checkpoint ONNX model, CLS pooling, and L2 normalization:

```sh
python prep_tokenized_jsonl.py --tokenizer "$GTE/tokenizer.json" \
  --bucket "$bucket" --input ../../../corpus/aft-chunks.jsonl \
  --text-field embed_text --limit 400 --output "/tmp/ane-wave1/embed-400-b${bucket}.jsonl"
python ort_reference.py --model "$GTE/onnx/model.onnx" \
  --input "/tmp/ane-wave1/embed-400-b${bucket}.jsonl" \
  --output "/tmp/ane-wave1/embed-ort-b${bucket}.jsonl" --pooling cls --batch-size 32
```

At seq512, 399 rows had byte-identical active tokens to seq256; their deterministic ORT vectors were reused. The sole longer row (`line:000329`, 257 tokens) was executed separately through ORT at seq512. The assembled reference therefore contains 400 executed, bucket-correct ORT results without substituting a differently tokenized row.

| Bucket | Mean cosine | Minimum cosine | top-10 overlap (100 stride-4 queries) | Gate |
|---:|---:|---:|---:|---|
| 128 | 0.99933036 | 0.99746465 | 0.977 | **FAIL** |
| 256 | 0.99932386 | 0.99786987 | 0.981 | **FAIL** |
| 512 | 0.99932768 | 0.99787492 | 0.981 | **FAIL** |

Required: cosine >= 0.9999 and overlap >= 0.995. ANE fp16 is therefore a **distinct vector space for all three buckets**. It may seed a new index built entirely with that fingerprint, but it cannot replace ORT/Metal vectors in an existing index.

## Reranker scoring

References were generated with Transformers on the same tokenizer policy and the matching maximum length for each bucket. The input is the committed real cosqa 1x50 fixture.

```sh
python bench/eval-coir/reference_rerank.py \
  --requests /tmp/ane-wave1/rerank-1x50-request.jsonl \
  --out "/tmp/ane-wave1/rerank-transformers-b${bucket}.jsonl" \
  --model "$RERANKER" --max-length "$bucket" --batch-size 8 --device mps
```

| Bucket | Pearson | tie-aware top-1 | Max logit abs error | Gate |
|---:|---:|---:|---:|---|
| 128 | 0.99996625 | 1.00 | 0.02314 | pass |
| 256 | 0.99997557 | 1.00 | 0.01778 | pass |
| 512 | 0.99997481 | 1.00 | 0.01338 | pass |

Required: Pearson >= 0.999 and tie-aware top-1 >= 0.98.

## M1 throughput, power, and energy

### Embedding

| Bucket | docs/s | real tok/s | CPU W | GPU W | ANE W | combined W | J/doc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 166.19 | 19,973.86 | 0.478 | 0.017 | 3.677 | 4.171 | 0.02623 |
| 256 | 64.07 | 10,064.62 | 0.363 | 0.014 | 3.569 | 3.946 | 0.06269 |
| 512 | 24.22 | 3,804.24 | 0.274 | 0.014 | 3.385 | 3.673 | 0.15311 |

Maximum GPU power was 0.106-0.129 W and maximum sampled GPU use was 2.78-3.34%; the GPU remained effectively idle while MLComputePlan placed the encoder on ANE.

### Reranking

| Bucket | pairs/s | real tok/s | CPU W | GPU W | ANE W | combined W | J/pair |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 163.72 | 11,912.03 | 0.660 | 0.000 | 2.617 | 3.276 | 0.02641 |
| 256 | 63.77 | 4,846.87 | 0.519 | 0.003 | 3.001 | 3.524 | 0.06345 |
| 512 | 23.76 | 1,805.98 | 0.268 | 0.013 | 2.920 | 3.200 | 0.14505 |

Short 50-pair windows contain 3, 6, and 14 macmon samples respectively; the repeated latency run provides the stable seq512 power window (263 samples, 3.695 W combined, 0.013 W GPU average).

## Reranker 1x50 latency verdict

Twenty repeated requests were measured in one loaded runner process per bucket, with batch size 8 and request boundaries kept separate.

| Bucket | p50 | p95 | Metal M1 comparison | AFT ~0.5 s verdict |
|---:|---:|---:|---:|---|
| 128 | 313.82 ms | 320.28 ms | 242.43 ms | **pass** |
| 256 | 796.55 ms | 801.75 ms | 242.43 ms | fail |
| 512 | 2,076.34 ms | 2,077.26 ms | 242.43 ms | fail |

Quiet reranking is viable only when a 128-token pair policy is acceptable. It is about 1.30x the locked-M1 fp32 Metal latency but stays under budget while using approximately 3.3 W rather than the Metal run's approximately 36-37 W GPU domain. Longer quiet buckets are not viable for the current latency budget.

## Contended M5 datapoint

These rows are labeled **contended M5**, not graduation measurements. Cold compilation/load dominated both power windows, and unrelated GPU activity is visible; they are useful only as a newer-generation datapoint.

| Model / bucket | throughput | cold load | infer wall | CPU W | GPU W | ANE W | combined W |
|---|---:|---:|---:|---:|---:|---:|---:|
| embedder / 256, 400 docs | 111.40 docs/s | 14.13 s | 3.591 s | 8.729 | 1.097 | 1.277 | 11.104 |
| reranker / 256, 50 pairs | 104.60 pairs/s | 16.31 s | 0.478 s | 6.380 | 1.308 | 0.110 | 7.799 |

## Comparison with locked-M1 Metal

The existing gte fp32 Metal lane reports approximately 23.2k tok/s at approximately 40 W GPU. ANE seq128 reaches 20.0k real tok/s at 4.17 W combined CPU+GPU+ANE and 0.026 J/doc, with GPU nearly idle. That is the quiet-tier energy story, but it comes with a separate fp16 embedding fingerprint. Seq256 and seq512 trade progressively more energy for fixed-bucket padding and are not throughput competitors.

For reranking, Metal's 242 ms 1x50 latency is faster, while ANE seq128's 314 ms p50 is quiet and remains within budget. ANE seq256/512 are too slow despite low power.

## MiniLM current-environment re-validation

The existing converter was rerun with trace disabled on the same macOS and pins, then compiled and executed on the M1 against the frozen 1,000-row references.

| Bucket | Mean cosine | top-10 overlap | ANE dispatch share | docs/s | Result |
|---:|---:|---:|---:|---:|---|
| 256 | 0.9999841970 | 0.994 | 94.805% | 377.23 | reproduced |
| 512 | 0.9999841258 | 0.992 | 94.805% | 110.27 | reproduced |

This confirms that current macOS 26.5.2, torch 2.5.1, coremltools 8.3.0, and transformers 4.48.0 preserve the July-8 conversion fingerprint and placement.

## Next steps

1. Treat gte ANE fp16 embeddings as a separately versioned index fingerprint; do not substitute them into ORT/Metal stores.
2. Evaluate whether the seq128 truncation policy preserves reranking quality on the full 50x20 gate corpus, since latency viability depends on it.
3. Investigate bucketing below 128 and request coalescing for reranking, without dynamic Core ML shapes.
4. Keep quantization out until an encoder-specific method demonstrates parity; naive W8A8 remains rejected.
5. Qwen3-0.6B through the PR #169 pattern remains future quiet-ladder work, not part of this wave.
