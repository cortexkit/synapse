# ts-embed lane

TypeScript embedding benchmark lane for `Xenova/all-MiniLM-L6-v2`.

Install dependencies:

```bash
cd bench/lanes/ts-embed
bun install
```

Run the production Transformers.js path (default dtype, mapped to the shipped q8 profile):

```bash
cd bench/lanes/ts-embed
bun main.mjs \
  --engine transformersjs \
  --dtype default \
  --corpus ../../data/corpus-smoke.jsonl \
  --out ../../results/ts-transformersjs-default.json \
  --vectors-out ../../results/ts-transformersjs-default-vectors.jsonl \
  --model-label "Xenova/all-MiniLM-L6-v2@transformersjs-default"
```

Run the Transformers.js fp32 path:

```bash
cd bench/lanes/ts-embed
bun main.mjs \
  --engine transformersjs \
  --dtype fp32 \
  --corpus ../../data/corpus-smoke.jsonl \
  --out ../../results/ts-transformersjs-fp32.json \
  --vectors-out ../../results/ts-transformersjs-fp32-vectors.jsonl \
  --model-label "Xenova/all-MiniLM-L6-v2@transformersjs-fp32"
```

Run raw `onnxruntime-node` against the local Qdrant MiniLM snapshot:

```bash
cd bench/lanes/ts-embed
bun main.mjs \
  --engine ort-node \
  --corpus ../../data/corpus-smoke.jsonl \
  --out ../../results/ts-ort-node.json \
  --vectors-out ../../results/ts-ort-node-vectors.jsonl \
  --model-label "Qdrant/all-MiniLM-L6-v2-onnx@ort-node-fp32"
```

`onnxruntime-node` ran under Bun in this environment. If a local Bun build fails to load the native binding, rerun the same command with `node main.mjs`.

`--dtype default` maps to `q8` so the benchmark matches the shipped MiniLM production profile; use `--dtype fp32` for the unquantized comparison.
