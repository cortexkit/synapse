#!/usr/bin/env bash
# The train hook and workers use the same push-CI gate.
set -euo pipefail
cd "$(dirname "$0")/.."

# MOTOR's Linux image has Rust but no CMake. Only the native suites need it;
# keep quick guards cheap and do not change the runner's shared filesystem.
if [[ "${1:-}" = --worker && "$(uname -s)" = Linux ]]; then
  job="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["name"])' "$2")"
  gate_tools="$(mktemp -d "${TMPDIR:-/tmp}/synapse-gate-tools.XXXXXX")"
  # Keep warm immutable Cargo caches, but use a writable home if locked metadata
  # cannot resolve offline. Fetch uncached dependencies there, not in shared state.
  for manifest in Cargo.toml bench/parity/Cargo.toml; do
    if ! cargo metadata --locked --offline --format-version 1 --manifest-path "$manifest" \
        >"$gate_tools/metadata.json" 2>"$gate_tools/metadata.error"; then
      export CARGO_HOME="$gate_tools/cargo-home"
      export CARGO_NET_OFFLINE=false
      mkdir -p "$CARGO_HOME"
      echo "SETUP $job: locked metadata needs resolution; using writable temporary CARGO_HOME"
      break
    fi
  done
  if [[ "$job" != quick ]]; then
    if ! python3 - "$gate_tools" "$job" <<'PY'
import hashlib
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

start = time.monotonic()
root = Path(sys.argv[1])
url = ('https://files.pythonhosted.org/packages/59/e8/'
       '096984b89133681533650b9078c5ed1c5c9b534e869b5487f22d4de1935c/'
       'cmake-3.31.6-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')
with urllib.request.urlopen(url, timeout=120) as response:
    content = response.read()
assert hashlib.sha256(content).hexdigest() == '1c8b05df0602365da91ee6a3336fe57525b137706c4ab5675498f662ae1dbcec', 'CMake wheel digest mismatch'
archive = root / 'cmake.whl'
archive.write_bytes(content)
with zipfile.ZipFile(archive) as wheel:
    wheel.extractall(root)
for binary in (root / 'cmake/data/bin').iterdir():
    binary.chmod(0o755)
print(f'TIMING {sys.argv[2]} / CMake 3.31.6 bootstrap: {time.monotonic() - start:.2f}s', flush=True)
PY
    then
      echo "FAIL $job / CMake prerequisite" >&2
      exit 1
    fi
    export PATH="$gate_tools/cmake/data/bin:$PATH"
  fi
fi
# Windows runners may provide Python only as `python` or the `py` launcher.
for interpreter in python3 python "py -3"; do
  if $interpreter -c 'import sys; sys.exit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
    exec $interpreter scripts/gate.py "$@"
  fi
done
echo "FAIL gate prerequisite: no Python 3.11+ found as python3, python or py -3" >&2
exit 127
