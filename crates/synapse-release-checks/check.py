"""No-build release gates, separate from the candidate's record validator."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile

ROWS = ('vulkan-windows-amd', 'vulkan-linux-amd', 'vulkan-linux-nvidia',
        'vulkan-windows-nvidia', 'cuda-linux-nvidia', 'cuda-windows-nvidia', 'metal-m5', 'ane-m5')
MODELS = ('gte-modernbert-base', 'gte-reranker-modernbert-base',
          'qwen3-embedding-0.6b', 'qwen3-reranker-0.6b')
BOUND = {'ck-synapse', 'ck-synapse-worker-cuda', 'ck-synapse-worker-vulkan', 'ck-synapse-worker-ane-direct'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args]).decode().strip()


def source(root):
    parents = git(root, 'rev-list', '--parents', '-n', '1', 'HEAD').split()
    require(len(parents) == 2, 'evidence commit must have exactly one parent')
    s = parents[1]
    for path in git(root, 'diff', '--name-only', '--no-renames', s, 'HEAD').splitlines():
        require(path.startswith('docs/evidence/'), 'non-evidence change: ' + path)
        if path.startswith('docs/evidence/certification/'):
            require(path.startswith('docs/evidence/certification/' + s + '/'), 'foreign certification change: ' + path)
    return s


def candidate_source(root):
    s = source(root)
    release = json.loads(subprocess.check_output(['gh', 'release', 'view', 'candidate-' + s, '--json', 'isDraft'], cwd=root))
    require(release['isDraft'] is True, 'candidate must be a draft release')
    return s


def extract(download, target, entries):
    names = {p.name for p in download.iterdir() if p.is_file()}
    expected = {e['asset'] for e in entries}
    require(names == expected | {n + '.sha256' for n in expected}, 'candidate inventory differs from assets/sidecars')
    for entry in entries:
        name = entry['asset']
        data = (download / name).read_bytes()
        sidecar = (download / (name + '.sha256')).read_text().strip()
        require(re.fullmatch(r'[0-9a-fA-F]{64} [ *]' + re.escape(name), sidecar) is not None, 'invalid sidecar: ' + name)
        require(sidecar[:64].lower() == digest(data), 'sidecar mismatch: ' + name)
        out = target / entry['os_arch']
        out.mkdir(parents=True, exist_ok=True)
        if not name.endswith('.zip'):
            continue
        with zipfile.ZipFile(download / name) as archive:
            members = archive.namelist()
            require(len(members) == len(set(members)), 'duplicate zip member')
            for member in members:
                require(not Path(member).is_absolute() and '..' not in Path(member).parts and '\\' not in member, 'unsafe zip member')
                require(not (out / member).exists(), 'overlapping zip member: ' + member)
            archive.extractall(out)
        for path in out.iterdir():
            if path.name.startswith('ck-'):
                path.chmod(0o755)


def inventory(entries, assets, records):
    require(len(entries) == len({e['asset'] for e in entries}), 'duplicate inventory entry')
    for e in entries:
        name = e.get('binary', '').removesuffix('.exe')
        if name in BOUND:
            os_arch = e['os_arch']
            expected = {'darwin-arm64': {'ck-synapse': {'metal-m5', 'ane-m5'}, 'ck-synapse-worker-ane-direct': {'ane-m5'}},
                        'linux-x64': {'ck-synapse': {'cuda-linux-nvidia', 'vulkan-linux-amd', 'vulkan-linux-nvidia'}, 'ck-synapse-worker-cuda': {'cuda-linux-nvidia'}, 'ck-synapse-worker-vulkan': {'vulkan-linux-amd', 'vulkan-linux-nvidia'}},
                        'windows-x64': {'ck-synapse': {'cuda-windows-nvidia', 'vulkan-windows-amd', 'vulkan-windows-nvidia'}, 'ck-synapse-worker-cuda': {'cuda-windows-nvidia'}, 'ck-synapse-worker-vulkan': {'vulkan-windows-amd', 'vulkan-windows-nvidia'}}}
            require(e['binding'] == 'certification' and set(e.get('rows', [])) == expected.get(os_arch, {}).get(name), 'lane binding rows differ')
            if name == 'ck-synapse-worker-cuda' and os_arch == 'windows-x64':
                require(e.get('runtime_files_from') == 'manifest.json', 'missing runtime manifest binding')
    for e in entries:
        binary = e.get('binary', '')
        if e['binding'] == 'exempt':
            require(binary.removesuffix('.exe') not in BOUND and bool(e.get('reason')), 'lane binary cannot be exempt')
            continue
        require(e['binding'] == 'certification' and e.get('rows'), 'invalid binding')
        file = assets / e['os_arch'] / binary
        sha = digest(file.read_bytes())
        for row in e['rows']:
            require(row in ROWS, 'unknown binding row')
            passed = [r for r in records if r['row_id'] == row and r['status'] == 'passed']
            require(passed and all(any(a['sha256'] == sha and Path(a['file']).name == binary for a in r['executed_artifacts']) for r in passed), 'missing extracted binary binding: ' + binary)
        if e.get('runtime_files_from'):
            manifest = load(file.parent / e['runtime_files_from'])
            require(manifest['worker_sha256'] == sha, 'worker manifest mismatch')
            dlls = {p.name.lower(): digest(p.read_bytes()) for p in file.parent.rglob('*') if p.suffix.lower() == '.dll'}
            listed = {a['file'].lower(): a['sha256'] for a in manifest['runtime_files']}
            require(len(listed) == len(manifest['runtime_files']) and dlls == listed and 'nvcuda.dll' not in dlls, 'DLL manifest differs from zip')
            for r in records:
                if r['row_id'] == 'cuda-windows-nvidia' and r['status'] == 'passed':
                    recorded = {Path(a['file']).name.lower(): a['sha256'] for a in r['executed_artifacts'] if a['file'].lower().endswith('.dll')}
                    require(recorded == dlls, 'record DLL set differs from manifest')


def median(series):
    require(len(series) == 23 and all(isinstance(n, (int, float)) and math.isfinite(n) and n > 0 for n in series), 'invalid raw series')
    s = sorted(series[3:])
    return (s[9] + s[10]) / 2


def benchmark(report, records):
    cells = report['cells']
    require({c['row_id'] for c in cells} == set(ROWS) and {c['model'] for c in cells} == set(MODELS), 'incomplete benchmark')
    require({(c['row_id'], c['model']) for c in cells} == {(r, m) for r in ROWS for m in MODELS}, 'missing benchmark combination')
    dropped = {(r['row_id'], r['model']) for r in records if r['status'] == 'dropped'}
    for c in cells:
        metrics = c['metrics']
        if (c['row_id'], c['model']) in dropped:
            require(c['status'] == 'dropped' and c.get('drop_cause') in ('parity', 'latency') and all(v is None for v in metrics.values()), 'numeric dropped cell')
            if c['drop_cause'] == 'latency':
                require(median(c['raw_series']['ane']) / median(c['raw_series']['metal']) > 3, 'invalid benchmark drop')
            continue
        status = c['status']
        require(status in ('measured', 'model_unsupported', 'hardware_unavailable'), 'invalid benchmark status')
        probe = c['probe']
        require((probe['exit_code'] == 0) == (status == 'measured'), 'probe/status mismatch')
        require(isinstance(probe['stdout'], str) and isinstance(probe['stderr'], str), 'missing raw probe')
        if status != 'measured':
            require(all(v is None for v in metrics.values()), 'metric in non-measured cell')
            require(c.get('cause') and c['cause'] in probe['stdout'] + probe['stderr'], 'unsupported cause not quoted')
        for b in c.get('embedding', []):
            require(b['batch_size'] <= c['inline_max_items'] or b.get('request_path') == 'job', 'missing above-inline request path')


def evidence(root, s, assets, entries):
    cert = root / 'docs/evidence/certification' / s
    records = [load(cert / row / (model + '.json')) for row in ROWS for model in MODELS]
    for path in cert.rglob('*.json'):
        obj = load(path)
        if 'source_commit' in obj:
            require(obj['source_commit'] == s, 'foreign record source')
    for r, (row, model) in zip(records, ((r, m) for r in ROWS for m in MODELS)):
        require(r['source_commit'] == s and r['row_id'] == row and r['model'] == model, 'record path mismatch')
    inventory(entries, assets, records)
    for model in MODELS:
        require(any(r['model'] == model and r['row_id'].startswith('cuda-') and r['status'] == 'passed' and r['machine'].get('driver_api') == 13020 for r in records), 'missing CUDA 13020 boundary: ' + model)
    if any(r['row_id'] == 'ane-m5' and r['status'] == 'passed' for r in records):
        stress = load(cert / 'ane-direct-stress.json')
        require(stress['leased_evict_count'] == 0 and stress['shape_not_admitted_count'] == 0 and stress['max_resident_per_model'] <= 4 and stress['max_resident_overall'] <= 8 and stress['request_count'] > 0 and stress['sample_count'] > 0, 'invalid ANE stress')
    notes = (cert / 'RELEASE-NOTES.md').read_text()
    for r in records:
        if r['status'] == 'dropped':
            require(r['row_id'] in notes and r['model'] in notes, 'missing dropped release note')
        if r['row_id'] == 'metal-m5':
            for path in (root / 'bench/parity/preload').glob(r['model'] + '-*.json'):
                preload = load(path)
                require(r['model'] in notes and r['fingerprint'] in notes and preload['expected_fingerprint'] in notes, 'missing Metal fingerprint pair')
    report_dir = root / 'docs/evidence/benchmark' / s
    benchmark(load(report_dir / 'report.json'), records)
    if any(r['row_id'] == 'ane-m5' and r['model'] == 'gte-modernbert-base' and r['status'] == 'passed' for r in records):
        t = load(report_dir / 'ane-transition.json')
        core, direct = median(t['coreml']), median(t['direct'])
        require(bool(t['session_id']) and len(t['machine_load']) == 23 and t['coreml_median'] == core and t['direct_median'] == direct and t['bar_met'] is (direct <= 1.05 * core), 'invalid ANE transition')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['source', 'validate', 'extract'])
    parser.add_argument('--downloads', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--os-arch')
    args = parser.parse_args()
    root = Path.cwd()
    if args.mode == 'source':
        print(candidate_source(root))
        return
    entries = load(root / 'bench/parity/release-assets.json')['assets']
    if args.os_arch:
        entries = [e for e in entries if e['os_arch'] == args.os_arch]
    extract(args.downloads, args.assets, entries)
    if args.mode == 'validate':
        s = source(root)
        evidence(root, s, args.assets, entries)
        subprocess.run([str((args.assets / 'linux-x64/ck-synapse').resolve()), 'certify', 'validate', '--assets', str(args.assets.resolve()), '.'], check=True)


if __name__ == '__main__':
    main()
