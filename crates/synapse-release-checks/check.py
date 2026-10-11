"""No-build release gates, separate from the candidate's record validator."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import zipfile

ROW_BACKENDS = {'vulkan-windows-amd': 'vulkan', 'vulkan-linux-amd': 'vulkan',
                'vulkan-linux-nvidia': 'vulkan', 'vulkan-windows-nvidia': 'vulkan',
                'cuda-linux-nvidia': 'cuda', 'cuda-windows-nvidia': 'cuda',
                'metal-m5': 'metal', 'ane-m5': 'ane'}
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


def catalog_cells(root):
    models = load(root / 'crates/synapse-module/src/catalog/models.json')['models']
    return [(row, model['id']) for row, backend in ROW_BACKENDS.items() for model in models
            if any(lane['backend'] == backend for lane in model['backends'])]


def cuda_models(cells):
    return {model for row, model in cells if ROW_BACKENDS[row] == 'cuda'}


def source(root):
    require(Path(git(root, 'rev-parse', '--show-toplevel')).resolve() == root.resolve(),
            'checkout must be the repository root')
    require(not git(root, 'status', '--porcelain'), 'dirty checkout')
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


def inventory(entries, assets, records, cells):
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
            require(row in ROW_BACKENDS, 'unknown binding row')
            for expected_row, model in cells:
                if expected_row == row:
                    passed = [r for r in records if r['row_id'] == row and r['model'] == model and r['status'] == 'passed']
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


def benchmark(report, expected_cells):
    cells = report['cells']
    require({(c['row_id'], c['model']) for c in cells} == set(expected_cells), 'benchmark differs from catalog cells')
    for c in cells:
        metrics = c['metrics']
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


def cuda_driver_floor(root, cells):
    models = cuda_models(cells)
    profiles = [p for p in load(root / 'bench/parity/models.json')['profiles'].values()
                 if p.get('lane') == 'owned-cuda' and p['model'] in models]
    require({p['model'] for p in profiles} == models, 'missing CUDA profiles')
    if not models:
        return None
    floors = [p['cuda_min_driver_api'] for p in profiles]
    require(all(type(floor) is int and floor > 0 for floor in floors) and len(set(floors)) == 1,
            'CUDA profiles disagree on driver floor')
    return floors[0]


def stress_evidence(stress):
    require('platform_uuid' not in stress.get('machine', {}), 'raw hardware UUID in ANE stress evidence')
    require(stress['request_count'] > 0 and stress['completed_count'] == stress['request_count'],
            'incomplete ANE stress requests')
    require(stress['request_errors'] == [], 'ANE stress request errors')
    require(stress['no_ane_resources']['count'] == 0, 'ANE stress exhausted resources')
    require(stress['reranker_pool_check']['finite'] == 'passed' and
            stress['reranker_pool_check']['byte_identical_repeats'] == 'passed',
            'ANE stress reranker pool failed')
    require(stress['leased_evict_count'] == 0 and stress['shape_not_admitted_count'] == 0 and
            0 <= stress['max_resident_per_model'] <= 4 and 0 <= stress['max_resident_total'] <= 8 and
            stress['sample_count'] > 0, 'invalid ANE stress')


def evidence(root, s, assets, entries):
    cert = root / 'docs/evidence/certification' / s
    cells = catalog_cells(root)
    paths = [cert / row / (model + '.json') for row, model in cells]
    records = [load(path) for path in paths]
    for path in cert.rglob('*.json'):
        obj = load(path)
        if 'source_commit' in obj:
            require(obj['source_commit'] == s, 'foreign record source')
        if path != cert / 'ane-direct-stress.json':
            require(path in paths, 'unexpected certification record: ' + str(path.relative_to(cert)))
    for r, (row, model) in zip(records, cells):
        require(r['source_commit'] == s and r['row_id'] == row and r['model'] == model, 'record path mismatch')
        require(r['status'] == 'passed', 'record must be passed')
        # Records store hardware UUIDs only as salted digests;
        # a raw value means the record came from an older or altered producer.
        machine = r.get('machine') or {}
        require('platform_uuid' not in machine and 'uuid' not in (machine.get('gpu') or {}),
                'raw hardware UUID in certification record')
    inventory(entries, assets, records, cells)
    driver_floor = cuda_driver_floor(root, cells)
    for model in cuda_models(cells):
        require(any(r['model'] == model and r['row_id'].startswith('cuda-') and r['status'] == 'passed' and r['machine'].get('driver_api') == driver_floor for r in records), f'missing CUDA {driver_floor} boundary: ' + model)
    if any(r['row_id'] == 'ane-m5' and r['status'] == 'passed' for r in records):
        stress = load(cert / 'ane-direct-stress.json')
        stress_evidence(stress)
    require((cert / 'RELEASE-NOTES.md').is_file(), 'missing release notes')
    report_dir = root / 'docs/evidence/benchmark' / s
    benchmark(load(report_dir / 'report.json'), cells)
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
    s = source(root) if args.mode == 'validate' else None
    entries = load(root / 'bench/parity/release-assets.json')['assets']
    if args.os_arch:
        entries = [e for e in entries if e['os_arch'] == args.os_arch]
    extract(args.downloads, args.assets, entries)
    if args.mode == 'validate':
        evidence(root, s, args.assets, entries)
        subprocess.run([str((args.assets / 'linux-x64/ck-synapse').resolve()), 'certify', 'validate', '--assets', str(args.assets.resolve()), '.'], check=True)


if __name__ == '__main__':
    main()
