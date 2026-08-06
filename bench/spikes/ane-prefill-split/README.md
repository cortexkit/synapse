# ANE prefill / Metal decode split

This spike tests a stateless Core ML Qwen3-0.6B prefill whose normal outputs are
last-position logits plus all 28 post-RoPE K tensors and all 28 V tensors. The
host packs those tensors into the Metal step engine's f16 cache layout and the
existing device-resident step engine continues greedy decode.

The explicit outputs avoid `MLState`; the stateful ANE path remains unsupported.
The W128 model is the semantically valid arm. W32x4 deliberately applies the
same stateless W32 graph to four independent chunks, packs their tensors at
positions 0...127, and is retained as a measured control. It cannot propagate
attention context between chunks, so correctness—not only latency—must be
reported for that arm.

## Files

- `convert_qwen3_prefill.py`: torch.export-only fixed-shape converter. It reuses
  the proven ANE-friendly Qwen3 layer definitions from `../ane-spec-decode/`
  and emits `logits`, `key_00`, `value_00`, ..., `key_27`, `value_27`.
- `ane_prefill_runner.swift`: Core ML compile/run/placement runner. It separately
  times `MLModel.prediction`, explicit K/V copy plus cache-layout packing, logits
  copy, host argmax/top-2 selection, and artifact serialization.
- `src/main.rs`: Metal handoff harness. It times 16-token batched GPU prefill,
  the 512-token-bucket f16 cache upload, and 64-token greedy continuation.
- `measure_prefill_split.py`: fixed-128-token 20-prompt battery, lock checks,
  macmon power capture, and compact result aggregation.
- `ANE-PREFILL-SPLIT.md`: locked-M1 evidence and verdict.

`MLModel.prediction` is an opaque Core ML boundary and may itself include
framework-owned output materialization. `kv_copy_layout_ms` starts only after
that call returns and measures the additional explicit host copy into the exact
Metal layout. Artifact file writes are reported separately and excluded from
the handoff decision because the intended integration is in-process.

## Build and convert

```bash
cd bench/spikes/ane-prefill-split
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer ./build_runner.sh

MODEL=/path/to/Qwen3-0.6B
for window in 32 128; do
  .venv/bin/python convert_qwen3_prefill.py \
    --model "$MODEL" --window "$window" \
    --out "artifacts/qwen3-prefill-w${window}.mlpackage" \
    --report-json "artifacts/conversion-w${window}.json"
  .build/ane-prefill-runner compile \
    --model "artifacts/qwen3-prefill-w${window}.mlpackage" \
    --out "artifacts/qwen3-prefill-w${window}.mlmodelc" \
    --stats "artifacts/compile-w${window}.json"
done

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  cargo build --release --manifest-path Cargo.toml
```

## Locked-M1 measurement

The measurement command must run on `[bench-host]` only after its
`Runner.Worker` is absent. `--locked` atomically creates
`[bench-user-home]/bench.lock`, refuses an active worker or battery power, and removes
the lock in a `finally` block.

```bash
.venv/bin/python measure_prefill_split.py \
  --model "$MODEL" \
  --runner "$PWD/.build/ane-prefill-runner" \
  --harness "$PWD/target/release/ane-prefill-split-harness" \
  --models-dir "$PWD/artifacts" \
  --prompts ../unified-rt/decode-prompts.jsonl \
  --work-dir "$PWD/artifacts/locked-m1" \
  --out "$PWD/artifacts/locked-m1-result.json" \
  --locked
```

Large model packages, compiled bundles, binary cache dumps, logs, and raw power
samples remain under ignored `artifacts/`. Only the compact locked-M1 result is
committed under `results/`.
