"""Upload only sidecar-verified candidate bytes, then remove stale release assets."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from check import extract, load, require


def gh(*args):
    return subprocess.check_output(['gh', 'release', *args]).decode()


def main(tag, source, directory):
    entries = load('bench/parity/release-assets.json')['assets']
    with tempfile.TemporaryDirectory() as temp:
        extract(directory, Path(temp), entries)
    notes = f'docs/evidence/certification/{source}/RELEASE-NOTES.md'
    exists = subprocess.run(['gh', 'release', 'view', tag], capture_output=True).returncode == 0
    if not exists:
        gh('create', tag, '--verify-tag', '--draft', '--prerelease', '--notes-file', notes)
    else:
        gh('edit', tag, '--draft=true')
    files = sorted(p for p in directory.iterdir() if p.is_file())
    require(bool(files), 'no verified bytes')
    gh('upload', tag, *(str(p) for p in files), '--clobber')
    uploaded = {p.name for p in files}
    for asset in json.loads(gh('view', tag, '--json', 'assets'))['assets']:
        if asset['name'] not in uploaded:
            gh('delete-asset', tag, asset['name'], '--yes')
    gh('edit', tag, '--draft=false', '--prerelease', '--notes-file', notes)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
