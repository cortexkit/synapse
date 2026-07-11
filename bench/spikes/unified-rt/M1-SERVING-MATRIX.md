# Locked-M1 serving matrix

## Verdict

The current explicit-O0 package path passes the 400-row ORT gates for all three model families in both fp32 and f16. On fresh package-cache HIT processes, f16 is a small win for MiniLM (`1.021x`) and Qwen3-Embedding-0.6B (`1.029x`), while gte-modernbert is faster in fp32 (`1.151x` fp32/f16). The serving recommendation is therefore f16 for MiniLM and Qwen3, and fp32 for gte-modernbert.

This wave used one inference pass per fresh process because the current main binary has no `--passes` option. Every MISS and HIT state was repeated in a second fresh process. These measurements include first-dispatch overhead and are comparable across cells in this matrix, but they are not in-process `first`/`warm`/`steady` measurements and must not be compared directly with the f16-evidence steady-state rows. The earlier executable probe found first-pass throughput at about 90% of in-process steady state for this graph class. Adding `--passes` to the main harness and recording true in-process warm and steady passes remains an open item after the concurrent source refactor.

## Host, binary, and inputs

| Item | Value |
|---|---|
| Host | `[bench-host]`, Apple M1 Max, 64 GiB |
| macOS | 26.5.2, build `25F84` |
| Xcode | 26.6, build `17F113` |
| Source revision | `2cb03a29d0016683351fa0c5d12501e23c43963f` (`origin/master` when built) |
| Binary SHA-256 | `312cb468b617898cd82e756901b266a1ba0d27eec061538328e8abae2432fad4` |
| Attention budget | 4,000,000 units; length-sorted batching |
| MiniLM corpus | first 400 rows of `minilm-corpus-1000.jsonl`; 69,596 tokens |
| gte-modernbert corpus | `modernbert-corpus-400.jsonl`; 62,838 tokens |
| Qwen3 corpus | `qwen3-corpus-400.jsonl`; 46,716 tokens |

The requested M1 checkout at `[bench-user-home]/synapse` did not exist. `[bench-user-home]/Work/synapse` existed only as a non-Git source staging directory, so it could not be pulled. The measured binary was instead built from the current local `origin/master` revision shown above and copied to `[bench-user-home]/bench-tools/unified-rt-serving/bin/spike-unified-rt`; no spike compilation occurred on the M1.

Every process acquired `[bench-user-home]/bench.lock`, checked that `pgrep -f Runner.Worker` was empty, and released the lock on exit. The first MiniLM process observed an active CI worker, released the lock without measuring, waited five minutes, reacquired it, rechecked the worker, and then ran. No runner process was stopped or changed.

## Measurement method and timing labels

For each family and dtype, `MISS run1` and `MISS run2` each deleted that cell's package directory before starting. `HIT run1` and `HIT run2` were fresh processes that reused the packages produced by `MISS run2`. Thus `run1` and `run2` are process-level repeatability labels, not in-process warmup labels.

`Load/prepare` below is the harness's `cold_load_s`: checkpoint loading plus its one-row warmup, including compile or package load for that warmup shape. `Inference` is the single corpus pass. The current harness discovers all remaining corpus shapes synchronously inside that timed pass, so MISS inference includes compilation and serialization and HIT inference includes package loading. This behavior is identical across matrix cells. The table reports those phases separately rather than hiding load time in whole-process wall time.

The MiniLM bridge additionally emits per-shape executable timing. Its six shapes, including the warmup shape, totaled:

| dtype | cache | run1 prepare | run2 prepare | specialization | serialization |
|---|---|---:|---:|---:|---:|
| fp32 | MISS/compile | 0.101568 s | 0.096879 s | 0.000136 / 0.000134 s | 0.017031 / 0.016565 s |
| fp32 | HIT/load | 0.017468 s | 0.017533 s | 0 / 0 s | 0 / 0 s |
| f16 | MISS/compile | 0.105581 s | 0.105968 s | 0.000127 / 0.000129 s | 0.017725 / 0.017480 s |
| f16 | HIT/load | 0.017256 s | 0.017303 s | 0 / 0 s | 0 / 0 s |

The ModernBERT and Qwen3 bridges do not expose their internal per-shape preparation counters. Their separately reported `Load/prepare` and `Inference` walls therefore remain the available phase boundary. All result JSON is retained under `results/m1-serving-matrix/`; combined process logs remain on the M1 at `[bench-user-home]/bench-tools/unified-rt-serving/results/` because this measurement-only wave permits committed result JSON but not log files.

## Full matrix

All throughput rows are the only corpus pass in a fresh process and therefore pay first-dispatch overhead.

| Family | dtype | cache | repeat | Load/prepare | Inference | tok/s | process wall |
|---|---|---|---:|---:|---:|---:|---:|
| MiniLM | fp32 | MISS | 1 | 0.131817 s | 0.622867 s | 111,734.9 | 1.10 s |
| MiniLM | fp32 | MISS | 2 | 0.116974 s | 0.599416 s | 116,106.4 | 1.05 s |
| MiniLM | fp32 | HIT | 1 | 0.100587 s | 0.523129 s | 133,038.0 | 0.96 s |
| MiniLM | fp32 | HIT | 2 | 0.102584 s | 0.522093 s | 133,302.0 | 0.96 s |
| MiniLM | f16 | MISS | 1 | 0.117495 s | 0.603255 s | 115,367.4 | 1.06 s |
| MiniLM | f16 | MISS | 2 | 0.119152 s | 0.599173 s | 116,153.5 | 1.06 s |
| MiniLM | f16 | HIT | 1 | 0.102791 s | 0.513656 s | 135,491.6 | 0.97 s |
| MiniLM | f16 | HIT | 2 | 0.103709 s | 0.509786 s | 136,520.0 | 0.95 s |
| gte-modernbert | fp32 | MISS | 1 | 0.413781 s | 3.066712 s | 20,490.4 | 4.18 s |
| gte-modernbert | fp32 | MISS | 2 | 0.401914 s | 3.066447 s | 20,492.1 | 4.17 s |
| gte-modernbert | fp32 | HIT | 1 | 0.352364 s | 2.879384 s | 21,823.4 | 3.92 s |
| gte-modernbert | fp32 | HIT | 2 | 0.355190 s | 2.887743 s | 21,760.2 | 3.93 s |
| gte-modernbert | f16 | MISS | 1 | 1.596008 s | 4.498975 s | 13,967.2 | 6.82 s |
| gte-modernbert | f16 | MISS | 2 | 0.442279 s | 3.610338 s | 17,405.0 | 4.75 s |
| gte-modernbert | f16 | HIT | 1 | 0.370654 s | 3.316397 s | 18,947.7 | 4.39 s |
| gte-modernbert | f16 | HIT | 2 | 0.366859 s | 3.321749 s | 18,917.1 | 4.40 s |
| Qwen3-Embedding-0.6B | fp32 | MISS | 1 | 0.961923 s | 7.322743 s | 6,379.6 | 9.26 s |
| Qwen3-Embedding-0.6B | fp32 | MISS | 2 | 0.897931 s | 7.316895 s | 6,384.7 | 9.20 s |
| Qwen3-Embedding-0.6B | fp32 | HIT | 1 | 0.830736 s | 7.114157 s | 6,566.6 | 8.92 s |
| Qwen3-Embedding-0.6B | fp32 | HIT | 2 | 0.831486 s | 7.115749 s | 6,565.2 | 8.92 s |
| Qwen3-Embedding-0.6B | f16 | MISS | 1 | 2.626810 s | 8.022152 s | 5,823.4 | 11.64 s |
| Qwen3-Embedding-0.6B | f16 | MISS | 2 | 1.070807 s | 7.228954 s | 6,462.3 | 9.28 s |
| Qwen3-Embedding-0.6B | f16 | HIT | 1 | 0.967332 s | 6.927691 s | 6,743.4 | 8.87 s |
| Qwen3-Embedding-0.6B | f16 | HIT | 2 | 0.967245 s | 6.907846 s | 6,762.7 | 8.86 s |

The first f16 MISS was substantially slower than the second for ModernBERT and Qwen3. Both are retained rather than discarded: clearing the package directory does not clear MPSGraph's OS-level process/compiler caches, and the repeat shows that package-cache MISS alone does not fully define machine-cold compilation. HIT repeatability was tight for every cell and is the serving comparison used below.

## Parity gates and MiniLM re-certification

Every one of the 24 processes evaluated both gates against its family-specific ORT fp32 reference. Values were invariant across MISS/HIT and both repeats at the shown precision.

| Family | dtype | mean cosine | top-10 overlap | cosine `>= 0.9999` | overlap `>= 0.995` |
|---|---|---:|---:|---|---|
| MiniLM | fp32 | 0.999999999999 | 1.000000 | PASS | PASS |
| MiniLM | f16 | 0.999999318566 | 0.999250 | PASS | PASS |
| gte-modernbert | fp32 | 0.999999999997 | 1.000000 | PASS | PASS |
| gte-modernbert | f16 | 0.999999038662 | 0.998750 | PASS | PASS |
| Qwen3-Embedding-0.6B | fp32 | 0.999999999993 | 1.000000 | PASS | PASS |
| Qwen3-Embedding-0.6B | f16 | 0.999998803394 | 0.998500 | PASS | PASS |

**MiniLM is re-certified for both fp32 and f16.** The oracle was the frozen evidence pack's original M1 file, not a locally regenerated reference:

- canonical reference: `[bench-user-home]/bench-tools/unified-rt-metal/ort-minilm-1000-vectors.jsonl`
- reference SHA-256: `7589eea5148562f6141c864d3357bab5dceb6881055afcf93b80efbdcae7d24d`
- canonical corpus: `[bench-user-home]/bench-tools/unified-rt-metal/corpus-1000.jsonl`
- corpus SHA-256: `b7c8424f5b6bc5df61d96146a03642671789c1d41cbe37e82864117330996a10`
- local working copies: `/tmp/ort-minilm-1000-vectors-official.jsonl` and `/tmp/minilm-corpus-1000-official.jsonl`

`bench/data/*.jsonl` is ignored by the repository, so the 7.7 MiB reference and 508 KiB corpus were not committed there. They were copied unchanged to `[bench-user-home]/bench-tools/unified-rt-serving/data/` for this run; hashes matched before measurement.

The regenerated Qdrant MiniLM ONNX file that produced the rejected local oracle identifies itself as PyTorch 2.1.2, ONNX IR 6, default-domain opset 11, with 638 nodes and no optimization metadata. The original ONNX bytes used to make the frozen reference were not retained beside the vectors, so an opset or optimizer-level difference cannot be proven from metadata alone. The unchanged fp32 path's earlier cosine of `0.99920633` against the regenerated vectors is the decisive evidence that they came from a numerically different export; it is not an fp16 regression.

## HIT serving comparison and M5 deltas

The two fresh-process package-HIT rows are averaged here. This is the graduation-probe serving number: model preparation is excluded from `tok/s`, while first dispatch and per-shape package loads during the corpus pass remain included.

| Family | dtype | M1 HIT mean tok/s | f16/fp32 | M5 contended tok/s | M1 delta vs M5 |
|---|---|---:|---:|---:|---:|
| MiniLM | fp32 | 133,170.0 | — | 71,950.2 | +85.1% |
| MiniLM | f16 | 136,005.8 | 1.021x | 86,957.7 | +56.4% |
| gte-modernbert | fp32 | 21,791.8 | — | 23,183.6 | -6.0% |
| gte-modernbert | f16 | 18,932.4 | 0.869x | 17,063.7 | +11.0% |
| Qwen3-Embedding-0.6B | fp32 | 6,565.9 | — | 6,551.5 | +0.2% |
| Qwen3-Embedding-0.6B | f16 | 6,753.1 | 1.029x | 6,813.0 | -0.9% |

The expected large locked-M1 advantage over the contended M5 appears for MiniLM and, more modestly, ModernBERT f16. It does not appear for ModernBERT fp32 or either Qwen3 dtype: those results are effectively architecture/compute-bound enough that the newer M5 offsets the recorded contention. The M5 MiniLM fp32 row was also a MISS/compile run, so its delta is directional rather than a cache-matched hardware comparison.

## Graduation-probe readout

| Family | recommendation | probe number today | llama-Metal M1 baseline | ratio to baseline |
|---|---|---:|---:|---:|
| MiniLM | f16 | 136,005.8 tok/s | 51,800 tok/s | 2.63x |
| gte-modernbert | fp32 | 21,791.8 tok/s | not measured | — |
| Qwen3-Embedding-0.6B | f16 | 6,753.1 tok/s | 3,400 tok/s | 1.99x |

MiniLM f16 wins by 2.1% over fp32 while passing both gates, so the certification probe would record 136.0k tok/s, 2.63x the llama-Metal M1 baseline. gte-modernbert f16 is 13.1% slower than fp32, so it should graduate in fp32 at 21.8k tok/s until its f16 topology improves; no llama-M1 baseline exists for this family. Qwen3 f16 wins by 2.9% and would record 6.75k tok/s, 1.99x its 3.4k llama-Metal M1 baseline.

## Commands executed

Local build and staging used:

```sh
export PATH="/opt/homebrew/bin:$HOME/.cargo/bin:$PATH"
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cargo build --release --manifest-path bench/spikes/unified-rt/Cargo.toml

ssh [bench-host-alias] 'mkdir -p [bench-user-home]/bench-tools/unified-rt-serving/bin \
  [bench-user-home]/bench-tools/unified-rt-serving/data \
  [bench-user-home]/bench-tools/unified-rt-serving/results \
  [bench-user-home]/bench-tools/unified-rt-serving/packages'
scp target/release/spike-unified-rt \
  [bench-host-alias]:[bench-user-home]/bench-tools/unified-rt-serving/bin/spike-unified-rt
scp /tmp/modernbert-corpus-400.jsonl /tmp/modernbert-ort-400-vectors.jsonl \
  /tmp/qwen3-corpus-400.jsonl /tmp/qwen3-ort-400-vectors.jsonl \
  [bench-host-alias]:[bench-user-home]/bench-tools/unified-rt-serving/data/
ssh [bench-host-alias] 'cp [bench-user-home]/bench-tools/unified-rt-metal/corpus-1000.jsonl \
  [bench-user-home]/bench-tools/unified-rt-serving/data/minilm-corpus-1000.jsonl; \
  cp [bench-user-home]/bench-tools/unified-rt-metal/ort-minilm-1000-vectors.jsonl \
  [bench-user-home]/bench-tools/unified-rt-serving/data/ort-minilm-1000-vectors.jsonl'
scp bench/spikes/unified-rt/run-m1-serving-matrix.sh \
  [bench-host-alias]:[bench-user-home]/bench-tools/unified-rt-serving/run-matrix.sh
ssh [bench-host-alias] '[bench-user-home]/bench-tools/unified-rt-serving/run-matrix.sh'
```

The first matrix invocation completed MiniLM, then found that the M1 ModernBERT snapshot had only ONNX weights. The exact safetensors blob used by the local gate was staged, hash-checked, and the remaining cells resumed without rerunning MiniLM:

```sh
scp ~/.cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots/\
e7f32e3c00f91d699e8c43b53106206bcc72bb22/model.safetensors \
  [bench-host-alias]:[bench-user-home]/.cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/\
snapshots/e7f32e3c00f91d699e8c43b53106206bcc72bb22/model.safetensors
ssh [bench-host-alias] 'SKIP_MINILM=1 [bench-user-home]/bench-tools/unified-rt-serving/run-matrix.sh'
```

The committed script contains the exact lock/retry guard, cache deletion, model snapshots, corpora, references, output names, and command arguments. Each M1-side raw log also begins with its fully expanded command. The measured command shape was:

```sh
/usr/bin/time -p "$BIN" \
  --model "$MODEL" --tokenizer "$MODEL/tokenizer.json" \
  --corpus "$CORPUS" --reference "$REFERENCE" --limit 400 \
  --out "$RESULT.json" --dtype "$DTYPE" --device metal \
  --package-cache "$PACKAGE_DIR" --model-label "$LABEL"
```

The 24 raw result JSON files and generated machine-readable summary are in `bench/spikes/unified-rt/results/m1-serving-matrix/`; timing logs remain at the canonical M1 path above.
