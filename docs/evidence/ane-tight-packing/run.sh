#!/bin/sh
# Run one ignored test in the isolated Mac mini clone, never the serving Mac.
set -eu
root=${ANE_PACKING_ROOT:-"$HOME/mason-ane-tight-bg10a30"}
cd "$root/synapse/crates/synapse-worker-ane-direct"
python3 - <<'PY'
import subprocess, sys
busy = []
for line in subprocess.check_output(['ps', 'axo', 'pid,command'], text=True).splitlines()[1:]:
    pid, command = line.strip().split(None, 1)
    executable = command.split()[0].rsplit('/', 1)[-1]
    if executable.startswith(('ckdev-', 'ck-synapse', 'ck_synapse', 'cargo', 'rustc')) or executable == 'synapse':
        busy.append(line)
if busy:
    print('Another compiler or Neural Engine client is running; retry after it finishes:', *busy, sep='\n', file=sys.stderr)
    sys.exit(75)
PY
snapshot="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
# The build command prints the binary path; do not guess among old binaries.
bin=${ANE_PACKING_BINARY:?set ANE_PACKING_BINARY to the cargo test --no-run executable}
exec env -u TMPDIR \
  ANE_TEST_PACKAGES="$root/packages" \
  ANE_MULTIROW_INPUT="$HOME/.local/share/cortexkit/synapse/aft-headtohead/engram.jsonl" \
  ANE_MULTIROW_TOKENIZER="$snapshot/tokenizer.json" \
  ANE_MULTIROW_OUT="$1" \
  "$bin" --ignored --exact "$2" --nocapture --test-threads=1
