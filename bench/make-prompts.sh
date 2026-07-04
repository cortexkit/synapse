#!/usr/bin/env bash
# Deterministic workload-B prompt set: every 40th chunk from corpus-v1, 100 total.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'EOF'
import json

prompts = []
with open("bench/data/corpus-v1.jsonl") as f:
    chunks = [json.loads(l) for l in f]

instruction = ("Classify the primary purpose of this code chunk. "
               "Answer with exactly one word from: config, test, logic, io, types, docs.\n\n")

for i, chunk in enumerate(chunks[::40][:100]):
    text = chunk["text"]
    # Rough 300-token cap: ~4 chars/token heuristic is fine for prompt building;
    # lanes report exact token counts from the real tokenizer.
    prompts.append({"id": f"p{i:03d}", "prompt": instruction + text[:1200]})

with open("bench/data/microllm-prompts-v1.jsonl", "w") as f:
    for p in prompts:
        f.write(json.dumps(p) + "\n")

print(f"wrote {len(prompts)} prompts")
EOF
