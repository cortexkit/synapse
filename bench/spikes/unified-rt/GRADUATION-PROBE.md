# Locked-M1 same-harness graduation probe

## Verdict

The owned Metal runtime graduates for all three measured Mac embedding fingerprints on the locked M1 Max. MiniLM f16 is **3.290x llama-server Metal and 1.037x MLX Python** at steady state. Qwen3-Embedding-0.6B f16 is **1.686x llama-server Metal and 1.117x MLX Python**. gte-modernbert fp32 is **1.728x its newly measured llama-server Metal cell**; no gte MLX implementation was available, so that family clears the direct serving/product comparator but retains an `MLX N/A` evidence cell.

These are certified same-hardware, same-corpus, same-timing-boundary ratios, with no directional qualification. MiniLM and Qwen3 clear both D-009 bars: the owned path beats direct llama-server, a stricter same-model serving comparator than an LMStudio wrapper, and it beats the current MLX Python incumbent. gte clears the available direct comparator; its cutover is supported, while a future gte MLX lane could strengthen rather than gate that evidence.

## Certified A/B table

Every throughput uses the canonical real-token numerator. `Load/boot` shows fresh-process repeat 1 / repeat 2. Owned throughput is pass 3 (`steady`) of the separate three-pass package-HIT process. Incumbent throughput is repeat 2 (`steady`) after repeat 1 populated framework caches; both repeat values are retained below. Power is system-wide mean GPU power over the same steady inference window.

| Family | Engine | dtype / weights | Load/boot r1 / r2 | Steady tok/s | GPU W (samples) | Mean cosine vs ORT | top-10 overlap | Standard gate |
|---|---|---|---:|---:|---:|---:|---:|---|
| MiniLM | owned runtime Metal | f16 | 0.108 / 0.107 s | **140,842** | 25.42 (3) | 0.99999932 | 0.99925 | PASS / PASS |
| MiniLM | llama-server Metal | f16 GGUF | 0.269 / 0.266 s | 42,815 | 10.07 (11) | 0.99998670 | 0.99825 | PASS / PASS |
| MiniLM | MLX Python | bf16 | 6.712 / 0.892 s | 135,800 | 26.53 (4) | 0.99996179 | 0.99450 | PASS / FAIL |
| Qwen3-Embedding-0.6B | owned runtime Metal | f16 | 0.974 / 0.973 s | **7,000** | 42.66 (43) | 0.99999880 | 0.99850 | PASS / PASS |
| Qwen3-Embedding-0.6B | llama-server Metal | f16 GGUF | 1.053 / 0.525 s | 4,153 | 36.38 (73) | 0.99999966 | 0.99925 | PASS / PASS |
| Qwen3-Embedding-0.6B | MLX Python | bf16 | 1.595 / 1.329 s | 6,266 | 46.41 (47) | 0.99989228 | 0.98825 | FAIL / FAIL |
| gte-modernbert | owned runtime Metal | fp32 | 0.365 / 0.370 s | **23,175** | 42.42 (17) | 1.00000000 | 1.00000 | PASS / PASS |
| gte-modernbert | llama-server Metal | f16 GGUF, CLS | 0.536 / 0.260 s | 13,408 | 27.53 (31) | 0.99999871 | 0.99875 | PASS / PASS |

The standard gate is mean cosine `>= 0.9999` and mean top-10 overlap `>= 0.995`, using every one of the 400 rows as a rank query. As expected, a different MLX vector space need not satisfy the owned-runtime fingerprint contract: MiniLM MLX misses overlap by 0.00050, and Qwen3 MLX misses both thresholds. The actual f16 GGUF spaces were closer to ORT than anticipated and passed both thresholds; they remain separate fingerprints despite passing this numerical gate.

Sub-second MiniLM power windows contain only 3-11 samples and are lower-confidence than their throughput timings. Power is included as a serving characteristic, not used to choose the certified speed ratios.

## Repeat evidence and certified ratios

| Family | Engine | Timed tok/s r1 / r2 | Read |
|---|---|---:|---|
| MiniLM | owned fresh package-HIT | 123,379 / 122,052 | Fresh-process first-dispatch rows; not the steady numerator |
| MiniLM | owned three-pass HIT | 140,842 (pass 3) | Certified steady numerator |
| MiniLM | llama-server | 48,292 / 42,815 | Canonical-token corrected; repeat 2 is steady |
| MiniLM | MLX Python | 77,737 / 135,800 | Repeat 1 includes first framework-shape compilation; repeat 2 is steady |
| Qwen3 | owned fresh package-HIT | 6,754 / 6,761 | Fresh-process first-dispatch rows; not the steady numerator |
| Qwen3 | owned three-pass HIT | 7,000 (pass 3) | Certified steady numerator |
| Qwen3 | llama-server | 4,081 / 4,153 | Repeat 2 is steady |
| Qwen3 | MLX Python | 6,161 / 6,266 | Repeat 2 is steady |
| gte-modernbert | owned fresh package-HIT | 21,770 / 21,773 | Fresh-process first-dispatch rows; not the steady numerator |
| gte-modernbert | owned three-pass HIT | 23,175 (pass 3) | Certified steady numerator |
| gte-modernbert | llama-server | 13,152 / 13,408 | Repeat 2 is steady |

| Family | Owned / llama-server | Owned / MLX Python |
|---|---:|---:|
| MiniLM | **3.290x** | **1.037x** |
| Qwen3-Embedding-0.6B | **1.686x** | **1.117x** |
| gte-modernbert | **1.728x** | N/A |

### Workload verdicts

- **MiniLM:** graduate. The owned f16 fingerprint clears the direct-serving product bar by 3.290x and the current MLX engineering bar by 3.7%, while passing the frozen ORT gates. The MLX first/steady split is retained because its first fresh process compiled corpus shapes after the one-row warmup; silently averaging that compile into steady throughput would not compare steady engines.
- **Qwen3-Embedding-0.6B:** graduate. The owned f16 fingerprint clears llama-server by 1.686x and MLX bf16 by 1.117x, with exact token agreement and owned-vector parity passing both gates.
- **gte-modernbert:** graduate against the available Mac serving incumbent. Owned fp32 is 1.728x llama-server f16 with exact token agreement and stronger ORT parity. There is no gte MLX cell, so this result does not claim a measured owned/MLX ratio.

## Fairness and token reconciliation

The accepted inputs are 400 rows for every engine. MiniLM uses the first 400 records of the canonical 1,000-row corpus; that text-only slice is byte-identical to the Qwen3 400-row corpus.

| Family | Corpus SHA-256 | Canonical real tokens | owned reported | llama reported | MLX reported |
|---|---|---:|---:|---:|---:|
| MiniLM | `5a9bfdc8…630c` | 66,783 | 66,783 | 69,596 | 66,783 |
| Qwen3 | `5a9bfdc8…630c` | 46,716 | 46,716 | 46,716 | 46,716 |
| gte-modernbert | `b4ff00f6…a8` | 62,838 | 62,838 | 62,838 | N/A |

The initial campaign-era `lane-llama` MiniLM run was rejected because it processed all 1,000 rows and exposed the known padded-token counter. The lane was rebuilt from source revision `c0df85400434ba5f253bf7b3fb41fc803b06f13b`, and the accepted run used the explicit 400-row slice. Its native counter still reported 69,596 tokens, 4.21% above canonical. Investigation reproduced the exact difference with the lane tokenizer at max length 512:

- encoded lengths: 69,596;
- sum of attention-mask real tokens: 66,783;
- mask-zero padding: 2,813.

llama-server receives decoded raw text, so those mask-zero client-side padding positions are not inferred and do not affect vectors or wall time. The published llama MiniLM throughput therefore uses `66,783 / infer_wall_s`, not the lane's inflated native rate. All other cells agreed exactly; the machine-readable summary preserves both reported and canonical counts.

llama-server used the lane's greedy batched `/v1/embeddings` path, one excluded warmup request, and naturally varying corpus text on every request, avoiding slot replay. Pooling was mean for MiniLM, last-token for Qwen3, and CLS for gte-modernbert; embedding normalization was L2.

## Host, artifacts, and environment

| Item | Value |
|---|---|
| Host | `[bench-host]`, Apple M1 Max, 64 GiB |
| macOS / Xcode | 26.5.2 (`25F84`) / 26.6 (`17F113`) |
| Source revision | `c0df85400434ba5f253bf7b3fb41fc803b06f13b` |
| Owned binary SHA-256 | `3b92806c…e2b3` |
| lane-llama SHA-256 | `15a233ce…a9b6` |
| llama-server | version 9580 (`b4e3dc613`), SHA-256 `77ec4104…d0e` |
| MLX environment | Python 3.12.13; mlx 0.32.0; mlx-embeddings 0.1.0; transformers 5.12.1; tokenizers 0.22.2 |
| Power | macmon 0.7.2, 100 ms requested interval, system-wide GPU metrics |

The campaign MLX virtualenv had been evicted. With approval, it was rebuilt using `~/.local/bin/uv` and the pinned `mlx-embeddings==0.1.0` requirement. These are current-version incumbents rather than the older campaign environment. That is conservative for graduation: an improved current MLX result raises the bar faced by the owned runtime. Full package, binary, model, corpus, and host provenance is retained in `results/graduation-probe/environment.json`.

Every inference process acquired `[bench-user-home]/bench.lock`, verified that `pgrep -f Runner.Worker` was empty, and released the lock. Before accepted measurements, an orphan from the already completed bucket campaign was found: outer PID 64588 had command `/bin/zsh [bench-user-home]/bench-tools/unified-rt-serving/run-bucket-matrix.sh` and start time `2026-07-12 05:52:50 CEST`; its retry child and macmon wrote only under `unified-rt-serving/results/m1-bucket-matrix/`. After ownership approval, those orphan processes and their `2026-07-12 10:52:55 CEST` stale lock were stopped. No Actions Runner service or `Runner.Worker` process was touched, and accepted graduation measurements began afterward.

## Timing and power boundaries

- **Owned runtime:** an untimed fresh package root was primed once per family. Two fresh package-HIT processes recorded load and first-dispatch rows. A third package-HIT process ran `--passes 3`; pass 3 is the steady throughput and power window. `--shapes exact` keeps bucket policy v1 out of the engine comparison.
- **llama-server:** `cold_load_s` is child boot, model load, health readiness, and one excluded warmup request. `infer_wall_s` contains only batched corpus HTTP inference. The result's `self_peak_rss_bytes` sampling and server shutdown occur outside that inference wall.
- **MLX Python:** `cold_load_s` is model/tokenizer load plus one excluded warmup batch. `infer_wall_s` contains only the sorted, batched corpus calls. Repeat 1/2 labels are first/steady because MLX framework compilation survives the process boundary.
- **Power:** macmon starts before model load. The summarizer takes a window with the result's final inference duration ending at the last sample with at least 1 W GPU power and 5% effective usage. Thus load/boot power is excluded from the reported inference watts even though its raw samples are retained.

## Commands executed

Local builds and staging:

```sh
export PATH="/opt/homebrew/bin:$HOME/.cargo/bin:$PATH"
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cargo build --release --manifest-path bench/spikes/unified-rt/Cargo.toml
cargo build --release --manifest-path bench/lanes/llama/Cargo.toml
scp target/release/spike-unified-rt [bench-host-alias]:[bench-user-home]/bench-tools/graduation-probe/bin/spike-unified-rt
scp target/release/lane-llama [bench-host-alias]:[bench-user-home]/bench-tools/graduation-probe/bin/lane-llama
scp bench/lanes/mlx-minilm/main.py bench/lanes/mlx-minilm/requirements.txt [bench-host-alias]:/tmp/
ssh [bench-host-alias] '[bench-user-home]/.local/bin/uv venv --python 3.12 [bench-user-home]/bench-tools/graduation-probe/venvs/mlx-embeddings'
ssh [bench-host-alias] '[bench-user-home]/.local/bin/uv pip install --python [bench-user-home]/bench-tools/graduation-probe/venvs/mlx-embeddings/bin/python -r [bench-user-home]/bench-tools/graduation-probe/scripts/mlx-requirements.txt'
scp bench/spikes/unified-rt/run-m1-graduation-probe.sh [bench-host-alias]:[bench-user-home]/bench-tools/graduation-probe/scripts/run-probe.sh
ssh [bench-host-alias] 'nohup zsh [bench-user-home]/bench-tools/graduation-probe/scripts/run-probe.sh >[bench-user-home]/bench-tools/graduation-probe/results/probe-run.log 2>&1 </dev/null &'
```

The committed runner contains the exact expanded model, tokenizer, corpus, reference, cache, pooling, and output paths. Its measured command forms were:

```sh
# Owned package-HIT repeats; --passes 3 for the separate steady process.
[bench-user-home]/bench-tools/graduation-probe/bin/spike-unified-rt \
  --model "$MODEL" --tokenizer "$MODEL/tokenizer.json" \
  --corpus "$CORPUS" --reference "$REFERENCE" --limit 400 \
  --out "$RESULT" --dtype "$DTYPE" --device metal \
  --package-cache "$PACKAGE_DIR" --shapes exact --passes "$PASSES" \
  --model-label "$LABEL"

# llama-server fresh-process repeat; POOLING=mean|last|cls by family.
[bench-user-home]/bench-tools/graduation-probe/bin/lane-llama embed \
  --server-binary [bench-user-home]/bench-tools/bin/llama-server-wrap.sh \
  --model "$GGUF" --tokenizer "$TOKENIZER" --corpus "$CORPUS" \
  --out "$RESULT" --vectors-out "$VECTORS" --reference "$REFERENCE" \
  --pooling "$POOLING" --model-label "$LABEL"

# Current MLX Python fresh-process repeat.
[bench-user-home]/bench-tools/graduation-probe/venvs/mlx-embeddings/bin/python \
  [bench-user-home]/bench-tools/graduation-probe/scripts/mlx-embed.py \
  --model "$MODEL" --corpus "$CORPUS" --limit 400 \
  --out "$RESULT" --vectors-out "$VECTORS" --model-label "$LABEL"

# Standard parity gate, every row as a query.
[bench-user-home]/bench-tools/graduation-probe/bin/synapse-bench parity \
  --reference "$REFERENCE" --candidate "$VECTORS" --k 10 --stride 1
```

Each raw M1 log begins with the fully shell-escaped expanded command. Raw command logs, vector JSONL, macmon JSONL, and epoch markers remain at `[bench-user-home]/bench-tools/graduation-probe/results/raw/`. Accepted lane results, parity reports, power windows, environment provenance, and the derived A/B summary are committed under `results/graduation-probe/`.
