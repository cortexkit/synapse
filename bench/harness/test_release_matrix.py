"""Producer tests use deterministic runtimes, never claim hardware measurements."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import competitor_adapter as competitor
import release_matrix as matrix
import synapse_adapter as synapse


class FakeAdapter:
    def __init__(self, config=None, evidence=None):
        self.requests = []
        self.config = config or {'precision': 'f16'}

    def call(self, request, allow_failure=False):
        self.requests.append(request)
        raw = {'exit_code': 0, 'stdout': '{}', 'stderr': ''}
        action = request['action']
        if action == 'probe':
            return raw, {**{k: request[k] for k in ('model', 'manifest_digest', 'input_digest', 'file_sha256')},
                         'version': 'test-1', 'settings': {'dtype': 'f16'}, 'precision': self.config['precision'],
                         'tokenization_differences': [], 'inline_max_items': 128, 'engine_batch_cap': {'max_items': 128}}
        if action == 'cold_load':
            return raw, {'elapsed_ms': 10, 'caches_cleared': True, 'cache_clear_evidence': {'command': ['test-clear'], 'exit_code': 0}}
        if action == 'parity':
            return raw, {k: request[k] for k in ('model', 'manifest_digest', 'input_digest')} | {'gates': {'cosine': True}}
        if action == 'prepare_transition':
            return raw, {'composed_length': 512, 'lane': request['lane']}
        value = self.config.get('latency', 100)
        return raw, {'elapsed_ms': value, 'request_path': request['request_path'],
                     'memory': {'kind': 'footprint', 'process_bytes': 1000}, 'machine_load': {'cpu_percent': 1}}


class ProducerTests(unittest.TestCase):
    def base(self, operation='embed', runtime='synapse'):
        return {'row_id': 'metal-m5', 'runtime': runtime, 'model': 'test', 'operation': operation,
                'manifest_digest': 'abc', 'input_digest': 'def', 'file_sha256': '123',
                'inputs': {'text': 'shared text', 'composed_length': 512, 'query': 'shared query',
                           'candidates': [f'doc-{i}' for i in range(100)], 'composed_lengths': [512] * 100}}

    def measure(self, operation='embed', runtime='synapse', precision='f16'):
        adapter = FakeAdapter({'precision': precision})
        cfg = {'precision': precision, 'command': ['test']}
        result = matrix.measure(adapter, self.base(operation, runtime), cfg,
                                {'inline': {'max_items': 64, 'max_tokens': 8192}}, 'footprint')
        return adapter, result

    def test_nearest_rank_and_median_discard_warmups(self):
        summary = matrix.summarize([900, 900, 900] + list(range(1, 21)))
        self.assertEqual(summary, {'p50': 10, 'p95': 19, 'median': 10.5, 'min': 1, 'max': 20})
        for bad in ([1] * 22, [1] * 22 + [float('nan')], [True] * 23):
            with self.assertRaises(ValueError):
                matrix.summarize(bad)

    def test_embed_batches_actual_admission_paths_and_caps(self):
        adapter, cell = self.measure()
        self.assertEqual([w['batch_size'] for w in cell['embedding']], [1, 8, 32, 128])
        self.assertEqual([w['request_path'] for w in cell['embedding']], ['inline', 'inline', 'job', 'job'])
        self.assertEqual(cell['embedding'][0]['engine_batch_cap'], {'max_items': 8, 'max_tokens': 3072})
        self.assertEqual(cell['metrics'], {'throughput': 10, 'p50': 100, 'p95': 100, 'peak_memory': 1000, 'cold_load': 10})
        self.assertEqual(len([r for r in adapter.requests if r['action'] == 'sample']), 92)
        self.assertEqual(len(cell['embedding'][3]['machine_load']), 23)
        self.assertEqual(len(cell['embedding'][3]['raw_series']), 23)

    def test_rerank_default_budget_split_and_single_candidate(self):
        adapter, cell = self.measure('rerank', precision='fp32')
        self.assertEqual([w['pool_size'] for w in cell['reranking']], [10, 100])
        self.assertEqual(cell['reranking'][1]['call_split'], [[0, 16], [16, 32], [32, 48], [48, 64], [64, 80], [80, 96], [96, 100]])
        singles = [r for r in adapter.requests if r.get('pool_size') == 1]
        self.assertEqual(len(singles), 23)
        self.assertEqual(cell['label'], 'precision-mismatch: fp32 lane vs f16 competitors')
        self.assertEqual(matrix.split_pool([2, 8192, 3], 100, 16384), [[0, 1], [1, 2], [2, 3]])
        with self.assertRaises(ValueError):
            matrix.split_pool([8193])

    def test_probe_failure_keeps_raw_and_null_metrics(self):
        class Unsupported:
            def call(self, request, allow_failure=False):
                self.request = request
                return {'exit_code': 2, 'stdout': '{"status":"model_unsupported","cause":"model rejected","version":"test-1","settings":{},"tokenization_differences":null}', 'stderr': 'model rejected\n'}, None
        cell = matrix.measure(Unsupported(), self.base(runtime='tei'), {'command': ['test']}, {'inline': {'max_items': 64}}, 'footprint')
        self.assertEqual(cell['status'], 'model_unsupported')
        self.assertEqual(cell['cause'], 'model rejected')
        self.assertEqual(cell['metrics'], dict.fromkeys(matrix.METRICS))
        self.assertEqual(cell['probe']['stderr'], 'model rejected\n')

    def test_transition_boundary_and_both_load_series(self):
        for direct, expected in ((105, True), (106, False)):
            cfg = {'transition': {'inputs': {'text': 'same', 'composed_length': 512},
                                  'ane-coreml-worker': {'latency': 100}, 'ane-direct-worker': {'latency': direct}}}
            with tempfile.TemporaryDirectory() as directory, patch.object(matrix, 'Adapter', FakeAdapter):
                result = matrix.transition(cfg, 'manifest', 'same-session', Path(directory))
                self.assertEqual(result['bar_met'], expected)
                self.assertEqual(result['coreml_median'], 100)
                self.assertEqual(result['direct_median'], direct)
                self.assertEqual(len(result['coreml']), 23)
                self.assertEqual(len(result['direct']), 23)
                self.assertEqual(set(result['machine_load'][0]), {'coreml', 'direct'})
                self.assertEqual(len(result['machine_load']), 23)
                self.assertTrue((Path(directory) / 'ane-transition.json').is_file())

    def matrix_fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for relative in ('bench/parity/models.json', 'crates/synapse-module/src/catalog/models.json',
                         'crates/synapse-release-checks/check.py'):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(matrix.ROOT / relative, destination)
        output = root / 'reports'
        self.write_reports(root, output)
        return root, output

    def write_reports(self, root, output):
        digest = hashlib.sha256((root / 'bench/parity/models.json').read_bytes()).hexdigest()
        expected = matrix.release_checks(root).catalog_cells(root)
        for row in dict.fromkeys(row for row, _ in expected):
            cells = []
            for cell_row, model in expected:
                if cell_row != row:
                    continue
                for runtime in ('synapse',) + matrix.COMPETITORS:
                    cell = {'row_id': row, 'model': model, 'runtime': runtime, 'status': 'measured',
                            'probe': {'exit_code': 0, 'stdout': '{}', 'stderr': ''}, 'metrics': {'p50': 10}}
                    if runtime == 'tei' and row.endswith('-amd'):
                        cell.update(status='hardware_unavailable', metrics={'p50': None}, cause='no AMD hardware',
                                    probe={'exit_code': 2, 'stdout': '', 'stderr': 'no AMD hardware'})
                    cells.append(cell)
            matrix.save(output / f'{row}.json', {'source_commit': 'source', 'manifest_digest': digest,
                                                'session_id': row, 'machine': row, 'cells': cells})

    def test_removing_backend_removes_assembled_cells_and_passes_tag_validator(self):
        root, output = self.matrix_fixture()
        before = matrix.assemble(root, 'source', output)
        before_cells = {(cell['row_id'], cell['model']) for cell in before['cells']}
        catalog_path = root / 'crates/synapse-module/src/catalog/models.json'
        catalog = matrix.load(catalog_path)
        model = 'gte-modernbert-base'
        entry = next(entry for entry in catalog['models'] if entry['id'] == model)
        entry['backends'] = [lane for lane in entry['backends'] if lane['backend'] != 'cuda']
        matrix.save(catalog_path, catalog)
        self.write_reports(root, output)
        result = matrix.assemble(root, 'source', output)
        removed = {('cuda-linux-nvidia', model), ('cuda-windows-nvidia', model)}
        self.assertTrue(removed <= before_cells)
        self.assertEqual({(cell['row_id'], cell['model']) for cell in result['cells']}, before_cells - removed)
        checks = matrix.release_checks(root)
        cells = checks.catalog_cells(root)
        checks.benchmark(result, cells)
        self.assertEqual(set(result['competitors']), set(matrix.COMPETITORS))
        for runtime, competitor_cells in result['competitors'].items():
            with self.subTest(runtime=runtime):
                self.assertEqual({(cell['row_id'], cell['model']) for cell in competitor_cells}, before_cells - removed)
                checks.benchmark({'cells': competitor_cells}, cells)
        bad = matrix.load(output / 'metal-m5.json')
        bad['cells'][0]['probe']['exit_code'] = 2
        matrix.save(output / 'metal-m5.json', bad)
        with self.assertRaisesRegex(ValueError, 'probe/status mismatch'):
            matrix.assemble(root, 'source', output)

    def test_assembled_matrix_refuses_dropped_owned_cell(self):
        root, output = self.matrix_fixture()
        report = matrix.load(output / 'ane-m5.json')
        owned = next(cell for cell in report['cells']
                     if cell['runtime'] == 'synapse' and cell['model'] == 'qwen3-embedding-0.6b')
        owned.update(status='dropped', metrics={'p50': None})
        matrix.save(output / 'ane-m5.json', report)
        with self.assertRaisesRegex(ValueError, 'invalid benchmark status'):
            matrix.assemble(root, 'source', output)

    def test_assembled_matrix_validates_each_competitor(self):
        root, output = self.matrix_fixture()
        for runtime in matrix.COMPETITORS:
            path = output / 'metal-m5.json'
            report = matrix.load(path)
            competitor_cell = next(cell for cell in report['cells'] if cell['runtime'] == runtime)
            original = competitor_cell['probe']['exit_code']
            competitor_cell['probe']['exit_code'] = 2
            matrix.save(path, report)
            with self.subTest(runtime=runtime), self.assertRaisesRegex(ValueError, 'probe/status mismatch'):
                matrix.assemble(root, 'source', output)
            competitor_cell['probe']['exit_code'] = original
            matrix.save(path, report)

    def test_manifest_digest_is_required(self):
        manifest = matrix.load(matrix.ROOT / 'bench/parity/models.json')
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(matrix, 'load', return_value={'models': {}, 'profiles': manifest['profiles']}):
                with self.assertRaisesRegex(ValueError, 'manifest digest mismatch'):
                    matrix.run_row({'row_id': 'metal-m5', 'source_commit': 'source', 'manifest_digest': 'wrong',
                                    'machine': {'memory_kind': 'footprint'}}, matrix.ROOT, Path(directory))

    def test_full_row_shared_files_and_precision_label(self):
        manifest = matrix.load(matrix.ROOT / 'bench/parity/models.json')
        digest = hashlib.sha256((matrix.ROOT / 'bench/parity/models.json').read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model_file = output / 'f16.safetensors'
            model_file.write_bytes(b'deterministic test file, not release weights')
            file_digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
            cfg = {'row_id': 'metal-m5', 'source_commit': 'source', 'manifest_digest': digest,
                   'machine': {'id': 'test-machine', 'memory_kind': 'footprint'}, 'models': {}}
            models = [model for row, model in matrix.release_checks(matrix.ROOT).catalog_cells(matrix.ROOT)
                      if row == 'metal-m5']
            for model in models:
                definition = manifest['models'][model]
                record = output / f'{model}-record.json'
                matrix.save(record, {'row_id': 'metal-m5', 'model': model, 'source_commit': 'source', 'status': 'passed'})
                adapters = {runtime: {'command': ['test'], 'precision': 'f16', 'file': str(model_file), 'file_sha256': file_digest}
                            for runtime in ('synapse',) + matrix.COMPETITORS}
                if model == 'gte-reranker-modernbert-base':
                    adapters['synapse']['precision'] = 'fp32'
                cfg['models'][model] = {'certification_record': str(record), 'f16_sha256': file_digest,
                                        'inputs': self.base(definition['operation'])['inputs'], 'adapters': adapters}
            with patch.object(matrix, 'Adapter', FakeAdapter):
                report = matrix.run_row(cfg, matrix.ROOT, output)
            self.assertEqual(len(report['cells']), len(models) * (1 + len(matrix.COMPETITORS)))
            self.assertEqual({cell['model'] for cell in report['cells']}, set(models))
            self.assertNotIn('qwen3-reranker-0.6b', models)
            self.assertEqual({c['input_digest'] for c in report['cells']}, {report['cells'][0]['input_digest']})
            self.assertEqual(sum(c.get('label') is not None for c in report['cells']), 1)
            self.assertTrue((output / 'metal-m5.json').is_file())
            record_path = Path(cfg['models']['gte-modernbert-base']['certification_record'])
            record = matrix.load(record_path)
            record['status'] = 'dropped'
            matrix.save(record_path, record)
            with patch.object(matrix, 'Adapter', FakeAdapter), self.assertRaisesRegex(ValueError, 'certification record must be passed'):
                matrix.run_row(cfg, matrix.ROOT, output)
            record['status'] = 'passed'
            matrix.save(record_path, record)
            cfg['models']['gte-modernbert-base']['f16_sha256'] = 'wrong'
            with patch.object(matrix, 'Adapter', FakeAdapter), self.assertRaisesRegex(ValueError, 'same f16 file'):
                matrix.run_row(cfg, matrix.ROOT, output)

    def test_adapter_commits_exact_subprocess_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, '-c', 'import sys; print("no hardware", file=sys.stderr); print("{}"); sys.exit(2)']
            raw, response = matrix.Adapter({'command': command}, directory).call({'action': 'probe'}, allow_failure=True)
            self.assertIsNone(response)
            self.assertEqual(raw['exit_code'], 2)
            self.assertEqual(raw['stdout'], '{}\n')
            self.assertEqual(raw['stderr'], 'no hardware\n')
            self.assertEqual(matrix.load(Path(directory) / '0001-probe.json'), raw)

    def test_all_competitor_payloads_use_same_inputs(self):
        request = self.base('rerank')
        for runtime in matrix.COMPETITORS:
            path, body = competitor.payload(runtime, request, texts=['same'])
            self.assertIn('same', body.get('input', body.get('inputs', [])))
            path, body = competitor.payload(runtime, request, candidates=['doc'])
            self.assertEqual(body['query'], 'shared query')
            self.assertEqual(body.get('documents', body.get('texts')), ['doc'])

    def test_http_adapter_executes_every_rerank_split(self):
        request = dict(self.base('rerank'), candidates=['a', 'b', 'c'], call_split=[[0, 2], [2, 3]])
        calls = []
        def post(endpoint, path, body, timeout):
            calls.append(body['documents'])
            return {'results': [{'relevance_score': 0.5} for _ in body['documents']]}
        with patch.object(competitor, 'post', post):
            competitor.invoke('llama.cpp', request, 'http://test', 5)
        self.assertEqual(calls, [['a', 'b'], ['c']])

    def test_owned_adapter_completes_job_pages_and_checks_path(self):
        request = dict(self.base(), texts=['a', 'b'], request_path='job')
        methods = []
        def run(argv, **kwargs):
            method, params = argv[1], json.loads(argv[2])
            methods.append((method, params))
            if method == 'embed.batch':
                value = {'job_id': 'j'}
            elif 'page' not in params:
                value = {'state': 'done', 'page_count': 2}
            else:
                value = {'vectors': [[0.1]]}
            return subprocess.CompletedProcess(argv, 0, json.dumps({'result': value}), '')
        cfg = {'rpc_command': ['rpc', '{method}', '{params}'], 'model_id': 'm', 'fingerprint': 'fp'}
        with patch.object(synapse.subprocess, 'run', run):
            _, path = synapse.invoke(cfg, request)
        self.assertEqual(path, 'job')
        self.assertEqual([params['page'] for _, params in methods if 'page' in params], [0, 1])
        self.assertEqual(methods[0][1]['required_fingerprint'], 'fp')


if __name__ == '__main__':
    unittest.main()
