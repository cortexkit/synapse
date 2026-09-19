"""Run the release binary and retain block timings alongside enclosing encoder wall."""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent
result = subprocess.run([str(ROOT / "target/release/laya-owned")], env={**os.environ, "SYNAPSE_EMBED_PROFILE": "1"}, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(result.stdout)
if result.returncode:
    print(result.stderr)
    raise SystemExit(result.returncode)
pattern = r"modernbert_block_forward batch=(\d+) seq=(\d+) forward_ms=([\d.]+)"
events = [(int(b), int(s), float(t)) for b, s, t in re.findall(pattern, result.stderr)]
# The binary first loads the encoder and evaluates all 24 reference cases.
# Keep only its final 92 block events: 3 warmups + 20 samples for each of 4 buckets.
measured = events[-4 * 23:]
assert len(measured) == 92
path = ROOT / "timings.json"
data = json.loads(path.read_text())
for i, bucket in enumerate(data["buckets"]):
    group = measured[i * 23:(i + 1) * 23]
    assert all((b, s) == (bucket["batch"], bucket["seq"]) for b, s, _ in group)
    block = [t for _, _, t in group[3:]]
    bucket["metal_block_ms"] = block
    bucket["host_prologue_and_wrapper_ms"] = [wall - gpu for wall, gpu in zip(bucket["encoder_wall_ms"], block)]
path.write_text(json.dumps(data, indent=2) + "\n")
