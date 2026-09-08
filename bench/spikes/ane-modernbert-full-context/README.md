# GTE ModernBERT 8192-token ANE feasibility spike

This standalone spike tests fixed-shape, full-context `Alibaba-NLP/gte-modernbert-base` embedding on Apple Neural Engine hardware. It is export and measurement evidence only: it does not change the production Swift worker, routing, fleet configuration, or product fallback policy.

## Preserved model semantics

`modernbert_tiled.py` loads the checkpoint's unmodified safetensors and tokenizer, preserves all 22 layers, the global-every-third-layer schedule, 128-token local windows (`abs(query-key) <= 64`), global/local RoPE bases, and normalized CLS pooling. Global query tiles always use the complete key/value sequence. Local tiles include the full radius halo across tile boundaries. This is query tiling, not document chunking, truncation, or chunk-and-pool.

The export path splits both heads and query positions. With sequence length `S` and query tile `Q`, a global attention score output is bounded to one `S × Q` head instead of `heads × S × S`. Local tiles use at most `Q + 128` key positions and construct only tile-local masks. The final partial tile is retained. Mathematical equality does not imply floating-point byte equality, so the harness reports max/mean error and cosine similarity rather than claiming byte identity.

The independent reference uses online softmax over bounded query and key blocks. It sees every permitted key/value without allocating an 8192 × 8192 dense baseline.

## Set up the current measured toolchain

From this directory:

```bash
uv venv --python 3.12 ../../../.venv
uv pip install --python ../../../.venv/bin/python -r requirements.txt
./build_placement.sh
```

The direct dependency versions in `requirements.txt` are the versions actually installed for this M5/macOS 27 spike on 2026-09-08; they are not copied from the older Wave 1 environment. Every report also records the runtime package versions and a canonical toolchain digest.

## CPU and static checks first

```bash
PY=../../../.venv/bin/python
MODEL=~/.cache/huggingface/hub/models--Alibaba-NLP--gte-modernbert-base/snapshots/e7f32e3c00f91d699e8c43b53106206bcc72bb22

$PY -m unittest -v test_modernbert_tiled.py
$PY spike.py --model "$MODEL" cpu-check --report /tmp/modernbert-full-context/cpu-check.json
```

The CPU check compares the query-tiled implementation to both Hugging Face eager attention and the independent streaming full-context implementation on real tokenizer inputs, including multilingual text and a final partial query tile.

## Guarded staged run

Do not start hardware conversion or prediction while another matched-ANE measurement is active. Obtain an explicit measurement slot first, then record that authorization:

```bash
$PY run_stages.py \
  --model "$MODEL" \
  --artifacts /tmp/modernbert-full-context \
  --through 8192 \
  --query-tile 256 \
  --key-tile 256 \
  --measurement-slot-authorized "<who granted the non-overlapping slot and when>"
```

Stages run strictly in order: 1024, 2048, 4096, then 8192. Each stage is a sequence of separate process trees: input preparation, bounded full-context reference, export/conversion/save with `skip_model_load=True`, package reload/prediction, and `MLComputePlan` placement inspection. A Core ML minimum cosine below the fixed `0.999` gate is a failed stage and stops the sequence; smaller stages are de-risking evidence, not a proposed production fallback.

The driver samples owned-child RSS plus system free and wired memory. Defaults require 32 GiB available before starting and terminate only the experiment's process group if available memory drops below 16 GiB, wired memory exceeds 50% of physical memory, owned RSS exceeds 32 GiB, or a child runs longer than 60 minutes. Raw free pages are still captured but are not a default abort condition because macOS aggressively converts free pages to reclaimable caches; `--abort-free-gib` can set an operator-chosen floor. macOS RSS/ulimit controls do not reliably account for wired ANE allocations. These observations and aborts reduce risk but cannot guarantee that the workstation will avoid memory pressure; lower-risk scheduling and operator observation remain necessary.

## Diagnose a 1024 parity failure

`diagnose_1024.py` emits sampled checkpoints at raw embeddings, embedding normalization, every attention residual, every layer output, and final normalization for positions 0, 63, 64, 255, 256, and 1023. Its `reference`, `export`, and `reload` subcommands keep conversion/save separate from model reload. Run them through `run_guarded` (as demonstrated in the committed 1024 evidence) and compare both `cpu-and-ne` and `cpu-only`; do not continue longer stages until the failed gate is understood. The reference report also compares full-model pooling against Hugging Face on the exact stage input and independently reconstructs 1024 masks and RoPE.

## Durable evidence

Target-only packages, vectors, and verbose logs stay outside git. Compact JSON reports include:

- model/config/tokenizer, input, reference, output, package, and toolchain SHA-256 digests;
- actual active-token and unique-token counts for diverse tokenizer-generated rows;
- export, conversion, save, separate load, first-predict, and warm-predict timing;
- full-context reference parity and repeated-prediction determinism;
- exported-graph and MIL tensor-shape checks that reject sequence-square attention outputs;
- system free/wired memory and owned-child peak RSS around every process;
- complete operator/device counts, expensive-op placement (`matmul`/`einsum`, `softmax`, convolution, normalization, reductions), non-ANE operations, and exact errors.

`CPU_AND_NE` excludes GPU but does not by itself prove ANE residency. Stage summaries distinguish mixed/CPU attention placement from successful CPU+NE prediction whose attention operators are preferred on ANE. The latter combines runtime success with `MLComputePlan`; the plan is still static preferred-placement evidence, not a runtime dispatch trace.

## Prior-art boundary

At `smpanaro/ModernBERT-AppleNeuralEngine` commit `d0268940c0af4ffe2bca0f9eb3842fd05984215c`, `split_einsum_attn` tiles queries while retaining full keys/values, but its default query chunk size is 8192 and floor division drops a non-multiple tail. This spike uses ceiling-style half-open ranges, an explicit 256-token default, and tile-local masks.

The failure discussed in `john-rocky/CoreML-LLM` PR 169 was an un-tiled, fixed-8192, full-bidirectional Perplexity `pplx-embed` Qwen3-0.6B encoder on an M4 Max running macOS 26. It failed inference with `ANEProgramProcessRequestDirect status=0x15`. That is useful evidence about that model and graph; it is not proof that tiled ModernBERT on M5 is impossible.
