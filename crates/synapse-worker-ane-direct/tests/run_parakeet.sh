#!/usr/bin/env bash
# Build outside the exclusive hardware window; execute only at the owner's request.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
kit_dir="$PWD/target/scratch/ane-parakeet-kit"
mkdir -p "$kit_dir"
case "${1:-}" in
  build)
    xcrun --find metal
    cargo test --locked --release -p synapse-worker-ane-direct \
      --bin ck-synapse-worker-ane-direct --no-run --message-format=json > "$kit_dir/build.jsonl"
    python3 - "$kit_dir" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
executables = [row['executable'] for line in (root / 'build.jsonl').read_text().splitlines()
               if (row := json.loads(line)).get('reason') == 'compiler-artifact'
               and row.get('profile', {}).get('test') and row.get('executable')]
if len(executables) != 1:
    raise SystemExit(f'expected one test executable, found {executables}')
(root / 'executable').write_text(executables[0] + '\n')
print(executables[0])
PY
    executable="$(< "$kit_dir/executable")"
    codesign --force --sign - --identifier ckdev-synapse-ane-parakeet "$executable"
    ;;
  smoke|full|probe|probe-attention|probe-corrected)
    : "${ANE_PARAKEET_SNAPSHOT:?set to the pinned HF snapshot directory}"
    if [[ "$1" != probe* ]]; then
      : "${ANE_PARAKEET_MIRROR:?set to the content-addressed tensor mirror root}"
      : "${ANE_PARAKEET_WAVS:?set to the prepared WAV directory with inputs.json}"
    fi
    : "${ANE_PARAKEET_REPORT:?set to a JSON path outside target/ for retained evidence}"
    if [[ -n "${TMPDIR:-}" ]]; then
      echo 'unset TMPDIR before running the hardware kit' >&2
      exit 2
    fi
    export ANE_PARAKEET_CERTIFY=1
    export ANE_PARAKEET_PACKAGE="${ANE_PARAKEET_PACKAGE:-$kit_dir/parakeet.safetensors}"
    export ANE_PARAKEET_SMOKE=0
    export ANE_PARAKEET_PROBE=0
    export ANE_PARAKEET_PROBE_GROUP=all
    [[ "$1" != smoke ]] || export ANE_PARAKEET_SMOKE=1
    [[ "$1" != probe* ]] || export ANE_PARAKEET_PROBE=1
    [[ "$1" != probe-attention ]] || export ANE_PARAKEET_PROBE_GROUP=attention
    [[ "$1" != probe-corrected ]] || export ANE_PARAKEET_PROBE_GROUP=corrected
    executable="$(< "$kit_dir/executable")"
    exec "$executable" --exact parakeet::hardware::certify_parakeet --ignored --nocapture --test-threads=1
    ;;
  *) echo "usage: bash $0 {build|probe|probe-attention|probe-corrected|smoke|full}" >&2; exit 2 ;;
esac
