"""Generate independent direct-ANE padding expectations from pinned token fixtures.

This script imports no worker code. Run from any directory inside the worktree.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
manifest = json.loads((ROOT / "bench/parity/models.json").read_text())
cases = []
for slug, model in manifest["models"].items():
    path = ROOT / f"bench/parity/fixtures/{slug}/{slug}.ref-v1.transformers-5.16.1.seed-0.json"
    fixture = json.loads(path.read_text())
    pad = model["architecture"]["params"]["pad_token_id"]
    selected = [next(c for c in fixture["cases"] if c["id"] == key) for key in ["short-0", "boundary-129"]]
    selected.append({"id": "real-pad-id-is-not-padding", "input_ids": [pad, pad, pad]})
    for source in selected:
        ids = source["input_ids"]
        width = 128 if len(ids) <= 128 else 256
        cases.append({"model": slug, "source_case": source["id"], "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pad_id": pad, "input_ids": ids, "shape": width, "padded_ids": ids + [pad] * (width - len(ids)), "additive_mask": [0.0] * len(ids) + [-10000.0] * (width - len(ids))})
output = Path(__file__).parent / "fixtures/direct-ane-padding.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text('{\n  "schema": 1,\n  "lane": "ane-direct-worker",\n  "cases": [\n    ' + ',\n    '.join(json.dumps(case, separators=(",", ":")) for case in cases) + '\n  ]\n}\n')
print(f"Generated {len(cases)} independent padding cases: {output.relative_to(ROOT)}")
