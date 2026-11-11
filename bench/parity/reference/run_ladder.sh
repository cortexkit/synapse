#!/usr/bin/env bash
# Run one independently returnable duration-ladder partition from the repo root.
set -euo pipefail
partition=${1:?usage: run_ladder.sh PARTITION}
case "$partition" in 0|1|2|3) ;; *) echo "partition must be 0..3" >&2; exit 2 ;; esac

# Keep the remote caller connected during long attention evaluations, which
# otherwise produce no output until a complete fixture is sealed.
while :; do echo "ladder-$partition alive $(date -u +%T)"; sleep 20; done &
heartbeat=$!
stop_heartbeat() {
  kill "$heartbeat" 2>/dev/null || true
  wait "$heartbeat" 2>/dev/null || true
}
trap stop_heartbeat EXIT

python3 -m venv bench/parity/target/parakeet-env
python=bench/parity/target/parakeet-env/bin/python
bench/parity/target/parakeet-env/bin/pip install --quiet torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
bench/parity/target/parakeet-env/bin/pip install --quiet -r bench/parity/reference/requirements-parakeet.txt
if [ "$partition" = 0 ]; then
  "$python" -m unittest discover -s bench/parity/reference -p test_parakeet_reference.py -v
fi
"$python" bench/parity/reference/generate_parakeet.py \
  --cache bench/parity/target/parakeet-cache \
  --output "bench/parity/target/parakeet-generated-$partition" \
  --group ladder --partition "$partition" --partitions 4 --workers 4
stop_heartbeat
trap - EXIT
"$python" - "$partition" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(f"bench/parity/target/parakeet-ladder-{sys.argv[1]}.tar")
size = path.stat().st_size
if size > 127 * 1024 * 1024:
    raise RuntimeError(f"archive {size} exceeds the reserved 127 MiB return allowance")
print(json.dumps({"path": str(path), "bytes": size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
for suffix in (".seals.json", ".timing.json"):
    evidence = path.with_suffix(suffix)
    print(json.dumps({"path": str(evidence), "bytes": evidence.stat().st_size,
                      "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}), flush=True)
PY
