import copy
import json
from pathlib import Path
from typing import Any
import subprocess
import tempfile
import unittest
import zipfile
from unittest.mock import patch
import check
import smoke
import promote


class ReleaseChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def rejects(self, function, *args):
        with self.assertRaises((ValueError, FileNotFoundError)):
            function(*args)

    def records(self) -> list[dict[str, Any]]:
        return [dict(row_id=row, model=model, status='passed', source_commit='a' * 40,
                     machine={'driver_api': 13020}, executed_artifacts=[], fingerprint='catalog')
                for row in check.ROWS for model in check.MODELS]

    def report(self) -> dict[str, Any]:
        return {'cells': [dict(row_id=row, model=model, status='measured',
                               metrics={'p50': 100}, probe={'exit_code': 0, 'stdout': '', 'stderr': ''},
                               inline_max_items=8, embedding=[{'batch_size': 128, 'request_path': 'job'}])
                          for row in check.ROWS for model in check.MODELS]}

    def test_benchmark_support_and_completeness(self):
        report = self.report()
        amd = report['cells'][0]
        amd.update(status='hardware_unavailable', metrics={'p50': None}, cause='TEI requires CUDA',
                   probe={'exit_code': 2, 'stdout': '', 'stderr': 'TEI requires CUDA'})
        check.benchmark(report, self.records())
        for edit in ('missing_row', 'missing_model', 'metric', 'probe', 'unsupported_success', 'inline'):
            bad = copy.deepcopy(report)
            if edit == 'missing_row':
                bad['cells'] = [c for c in bad['cells'] if c['row_id'] != check.ROWS[-1]]
            elif edit == 'missing_model':
                bad['cells'] = [c for c in bad['cells'] if c['model'] != check.MODELS[-1]]
            elif edit == 'metric':
                bad['cells'][0]['metrics']['p50'] = 1
            elif edit == 'probe':
                bad['cells'][1]['probe']['exit_code'] = 2
            elif edit == 'unsupported_success':
                bad['cells'][0].update(status='model_unsupported')
                bad['cells'][0]['probe']['exit_code'] = 0
            else:
                bad['cells'][1]['embedding'][0].pop('request_path')
            with self.subTest(edit=edit):
                self.rejects(check.benchmark, bad, self.records())

    def test_dropped_numeric_metric(self):
        records = self.records()
        records[-1]['status'] = 'dropped'
        report = self.report()
        report['cells'][-1].update(status='dropped', drop_cause='parity', metrics={'p50': None})
        check.benchmark(report, records)
        report['cells'][-1]['metrics']['p50'] = 10
        self.rejects(check.benchmark, report, records)

    def git(self, *args):
        return check.git(self.root, *args)

    def commit(self, message):
        self.git('add', '.')
        self.git('commit', '-qm', message)
        return self.git('rev-parse', 'HEAD')

    def test_source_parent_and_prior_evidence(self):
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.com')
        self.git('config', 'user.name', 'test')
        prior = self.root / 'docs/evidence/certification' / ('1' * 40)
        prior.mkdir(parents=True)
        (prior / 'old.json').write_text('{}')
        s = self.commit('source')
        cert = self.root / 'docs/evidence/certification' / s
        cert.mkdir()
        (cert / 'new.json').write_text('{}')
        self.commit('evidence')
        self.assertEqual(check.source(self.root), s)
        (prior / 'old.json').write_text('{"changed":true}')
        self.git('add', '.')
        self.git('commit', '--amend', '--no-edit', '-q')
        self.rejects(check.source, self.root)
        (self.root / 'code.rs').write_text('bad')
        self.git('add', '.')
        self.git('commit', '--amend', '--no-edit', '-q')
        self.rejects(check.source, self.root)
        tree = self.git('rev-parse', 'HEAD^{tree}')
        merged = self.git('commit-tree', tree, '-p', self.git('rev-parse', 'HEAD'), '-p', s, '-m', 'merge')
        self.git('reset', '--hard', merged)
        self.rejects(check.source, self.root)

    def fixture_inventory(self):
        entries = check.load(Path(__file__).resolve().parents[2] / 'bench/parity/release-assets.json')['assets']
        records = self.records()
        download = self.root / 'download'
        assets = self.root / 'assets'
        download.mkdir()
        for e in entries:
            p = download / e['asset']
            if p.suffix == '.zip':
                with zipfile.ZipFile(p, 'w') as z:
                    data = e['binary'].encode()
                    z.writestr(e['binary'], data)
                    if e.get('runtime_files_from'):
                        z.writestr('runtime.dll', b'runtime')
                        z.writestr('manifest.json', json.dumps({'worker_sha256': check.digest(data), 'runtime_files': [{'file': 'runtime.dll', 'sha256': check.digest(b'runtime')}]}))
                for r in records:
                    if r['row_id'] in e.get('rows', []):
                        r['executed_artifacts'].append({'file': e['os_arch'] + '/' + e['binary'], 'sha256': check.digest(data)})
                        if e.get('runtime_files_from'):
                            r['executed_artifacts'].append({'file': 'windows-x64/runtime.dll', 'sha256': check.digest(b'runtime')})
            else:
                p.write_text('{}')
            (download / (p.name + '.sha256')).write_text(check.digest(p.read_bytes()) + '  ' + p.name + '\n')
        check.extract(download, assets, entries)
        return entries, records, assets, download

    def test_legacy_and_new_inventory_pass(self):
        entries, records, assets, _ = self.fixture_inventory()
        check.inventory(entries, assets, records)

    def test_undeclared_or_absent_asset_and_sidecar(self):
        entries, records, assets, download = self.fixture_inventory()
        (download / 'new-worker.zip').write_bytes(b'new')
        self.rejects(check.extract, download, self.root / 'out', entries)
        (download / 'new-worker.zip').unlink()
        first = download / entries[0]['asset']
        first.unlink()
        self.rejects(check.extract, download, self.root / 'out', entries)

    def test_exempt_lane_binary_and_zip_digest_fail(self):
        entries, records, assets, download = self.fixture_inventory()
        bad = copy.deepcopy(entries)
        for e in bad:
            if e.get('binary') == 'ck-synapse-worker-vulkan':
                e.update(binding='exempt', reason='not certified')
        self.rejects(check.inventory, bad, assets, records)
        records[4]['executed_artifacts'][0]['sha256'] = check.digest((download / 'ck-synapse-linux-x64.zip').read_bytes())
        self.rejects(check.inventory, entries, assets, records)

    def test_missing_manifest_dll_or_record_dll_and_extra_dll_fail(self):
        entries, records, assets, _ = self.fixture_inventory()
        manifest_path = assets / 'windows-x64/manifest.json'
        original = manifest_path.read_text()
        m = json.loads(original)
        m['runtime_files'] = []
        manifest_path.write_text(json.dumps(m))
        self.rejects(check.inventory, entries, assets, records)
        manifest_path.write_text(original)
        r = next(r for r in records if r['row_id'] == 'cuda-windows-nvidia')
        r['executed_artifacts'] = [a for a in r['executed_artifacts'] if not a['file'].endswith('.dll')]
        self.rejects(check.inventory, entries, assets, records)
        r['executed_artifacts'].append({'file': 'runtime.dll', 'sha256': check.digest(b'runtime')})
        r['executed_artifacts'].append({'file': 'extra.dll', 'sha256': '0' * 64})
        self.rejects(check.inventory, entries, assets, records)

    def evidence_fixture(self):
        entries, records, assets, _ = self.fixture_inventory()
        cert = self.root / 'docs/evidence/certification' / ('a' * 40)
        cert.mkdir(parents=True)
        for r in records:
            path = cert / r['row_id'] / (r['model'] + '.json')
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(r))
        (cert / 'ane-direct-stress.json').write_text(json.dumps(dict(request_count=20, sample_count=40, max_resident_per_model=4, max_resident_overall=8, shape_not_admitted_count=0, leased_evict_count=0)))
        (cert / 'RELEASE-NOTES.md').write_text('No dropped cells')
        preload = self.root / 'bench/parity/preload'
        preload.mkdir(parents=True)
        report_dir = self.root / 'docs/evidence/benchmark' / ('a' * 40)
        report_dir.mkdir(parents=True)
        (report_dir / 'report.json').write_text(json.dumps(self.report()))
        transition = dict(session_id='same-session', coreml=[100] * 23, direct=[105] * 23, machine_load=[0.1] * 23, coreml_median=100, direct_median=105, bar_met=True)
        (report_dir / 'ane-transition.json').write_text(json.dumps(transition))
        return entries, records, assets, cert, report_dir

    def test_driver_boundary_and_foreign_source_fail(self):
        entries, records, assets, cert, _ = self.evidence_fixture()
        check.evidence(self.root, 'a' * 40, assets, entries)
        for row, api in [('cuda-linux-nvidia', 13040), ('cuda-windows-nvidia', 13080)]:
            r = next(r for r in records if r['row_id'] == row and r['model'] == check.MODELS[0])
            path = cert / row / (r['model'] + '.json')
            r['machine']['driver_api'] = api
            path.write_text(json.dumps(r))
        self.rejects(check.evidence, self.root, 'a' * 40, assets, entries)
        for r in records:
            r['machine']['driver_api'] = 13020
            (cert / r['row_id'] / (r['model'] + '.json')).write_text(json.dumps(r))
        (cert / 'foreign.json').write_text(json.dumps({'source_commit': 'b' * 40}))
        self.rejects(check.evidence, self.root, 'a' * 40, assets, entries)
        (cert / 'foreign.json').unlink()
        (cert / check.ROWS[0] / (check.MODELS[0] + '.json')).unlink()
        self.rejects(check.evidence, self.root, 'a' * 40, assets, entries)

    def test_stress_transition_and_release_notes_required(self):
        entries, _, assets, cert, report_dir = self.evidence_fixture()
        stress_path = cert / 'ane-direct-stress.json'
        stress = check.load(stress_path)
        stress['leased_evict_count'] = 1
        stress_path.write_text(json.dumps(stress))
        self.rejects(check.evidence, self.root, 'a' * 40, assets, entries)
        stress_path.unlink()
        self.rejects(check.evidence, self.root, 'a' * 40, assets, entries)
        stress['leased_evict_count'] = 0
        stress_path.write_text(json.dumps(stress))
        transition_path = report_dir / 'ane-transition.json'
        transition = check.load(transition_path)
        transition['direct'] = [106] * 23
        transition['direct_median'] = 106
        transition_path.write_text(json.dumps(transition))
        self.rejects(check.evidence, self.root, 'a' * 40, assets, entries)
        transition['bar_met'] = False
        transition_path.write_text(json.dumps(transition))
        check.evidence(self.root, 'a' * 40, assets, entries)
        transition_path.unlink()
        self.rejects(check.evidence, self.root, 'a' * 40, assets, entries)

    def test_feature_disabled_and_loader_abort_smokes_fail(self):
        binary = self.root / 'ck-synapse-worker-cuda'
        with patch.object(smoke.subprocess, 'run') as run:
            run.return_value.returncode = 127
            run.return_value.stdout = ''
            run.return_value.stderr = 'cuda_no_driver'
            self.rejects(smoke.run, binary, '--probe-floor', 2, 'cuda_no_driver')
        with patch.object(smoke.sys, 'platform', 'linux'), patch.object(smoke, 'run', return_value='ck-synapse-worker-cuda features=none manifest_digest=' + 'a' * 64):
            self.rejects(smoke.main, self.root)

    def test_promote_verifies_bytes_and_prunes_stale_assets(self):
        entries, _, _, download = self.fixture_inventory()
        calls = []
        def gh(*args):
            calls.append(args)
            if args[:3] == ('view', 'v1', '--json'):
                return json.dumps({'assets': [{'name': 'stale.zip'}, {'name': entries[0]['asset']}]})
            return ''
        with patch.object(promote, 'load', return_value={'assets': entries}), patch.object(promote, 'gh', side_effect=gh), patch.object(promote.subprocess, 'run') as run:
            run.return_value.returncode = 1
            promote.main('v1', 'a' * 40, download)
        self.assertIn(('delete-asset', 'v1', 'stale.zip', '--yes'), calls)
        self.assertEqual(calls[-1][:4], ('edit', 'v1', '--draft=false', '--prerelease'))
        upload = next(c for c in calls if c[0] == 'upload')
        self.assertIn(str(download / entries[0]['asset']), upload)
        (download / entries[0]['asset']).write_bytes(b'corrupt')
        with patch.object(promote, 'load', return_value={'assets': entries}), patch.object(promote, 'gh') as gh_mock:
            self.rejects(promote.main, 'v1', 'a' * 40, download)
            gh_mock.assert_not_called()

    def test_tag_workflow_has_no_build_or_bypass(self):
        workflow = (Path(__file__).resolve().parents[2] / '.github/workflows/release.yml').read_text()
        self.assertNotIn('cargo', workflow)
        self.assertIn('fetch-depth: 0', workflow)
        self.assertIn('needs: [validate, smoke]', workflow)
        self.assertIn('candidate-$S', workflow)
        self.assertIn('check.py source', workflow)
        for runner in ('ubuntu-24.04', 'macos-15', 'windows-2025'):
            self.assertIn(runner, workflow)

    def test_parent_without_candidate_release_fails(self):
        with patch.object(check, 'source', return_value='a' * 40), patch.object(check.subprocess, 'check_output', side_effect=subprocess.CalledProcessError(1, ['gh'])):
            with self.assertRaises(subprocess.CalledProcessError):
                check.candidate_source(self.root)
        with patch.object(check, 'source', return_value='a' * 40), patch.object(check.subprocess, 'check_output', return_value=b'{"isDraft":true}') as gh:
            self.assertEqual(check.candidate_source(self.root), 'a' * 40)
            self.assertIn('candidate-' + 'a' * 40, gh.call_args.args[0])

    def test_release_notes_name_dropped_and_metal_pairs(self):
        entries, records, assets, cert, _ = self.evidence_fixture()
        preload = self.root / 'bench/parity/preload' / (check.MODELS[0] + '-f16.json')
        preload.write_text(json.dumps({'expected_fingerprint': 'preload'}))
        self.rejects(check.evidence, self.root, 'a' * 40, assets, entries)
        notes = cert / 'RELEASE-NOTES.md'
        notes.write_text(check.MODELS[0] + ' catalog preload')
        check.evidence(self.root, 'a' * 40, assets, entries)
        r = records[-1]
        r['status'] = 'dropped'
        (cert / r['row_id'] / (r['model'] + '.json')).write_text(json.dumps(r))
        self.rejects(check.evidence, self.root, 'a' * 40, assets, entries)
        notes.write_text(notes.read_text() + '\n' + r['row_id'] + '/' + r['model'])
        report = check.load(self.root / 'docs/evidence/benchmark' / ('a' * 40) / 'report.json')
        report['cells'][-1].update(status='dropped', drop_cause='parity', metrics={'p50': None})
        (self.root / 'docs/evidence/benchmark' / ('a' * 40) / 'report.json').write_text(json.dumps(report))
        check.evidence(self.root, 'a' * 40, assets, entries)

    def test_transition_boundary(self):
        self.assertTrue(check.median([105] * 23) <= 1.05 * check.median([100] * 23))
        self.assertFalse(check.median([106] * 23) <= 1.05 * check.median([100] * 23))
        self.assertEqual(check.median([10000] * 3 + list(range(1, 21))), 10.5)


if __name__ == '__main__':
    unittest.main()
