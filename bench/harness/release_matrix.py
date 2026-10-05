#!/usr/bin/env python3
"""Release-candidate benchmark orchestration (standard library only)."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
ROWS = ('metal-m5', 'ane-m5', 'cuda-linux-nvidia', 'cuda-windows-nvidia',
        'vulkan-linux-amd', 'vulkan-windows-amd', 'vulkan-linux-nvidia',
        'vulkan-windows-nvidia')
COMPETITORS = ('llama.cpp', 'lm-studio', 'ollama', 'tei')
METRICS = ('throughput', 'p50', 'p95', 'peak_memory', 'cold_load')
LANES = {'metal': 'owned-metal', 'ane': 'ane-direct-worker',
         'cuda': 'owned-cuda', 'vulkan': 'owned-vulkan'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def summarize(series):
    require(len(series) == 23 and all(positive(v) for v in series), 'need 3 warmups and 20 positive samples')
    measured = sorted(series[3:])
    return {'p50': measured[9], 'p95': measured[18],
            'median': (measured[9] + measured[10]) / 2,
            'min': measured[0], 'max': measured[-1]}


def split_pool(lengths, max_items=64, token_budget=8192):
    require(max_items > 0 and token_budget > 0, 'invalid budget')
    require(all(0 < n <= min(8192, token_budget) for n in lengths), 'sequence exceeds budget')
    result, start = [], 0
    while start < len(lengths):
        end, used = start, 0
        while end < len(lengths) and end - start < max_items:
            n = lengths[end]
            if n > token_budget - used or (end > start and (n == 8192 or lengths[start] == 8192)):
                break
            used += n
            end += 1
        result.append([start, end])
        start = end
    return result


def machine_load():
    # Load averages are unavailable on Windows; retain that fact instead of inventing zero load.
    return {'timestamp_ns': time.time_ns(), 'load_average': list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
            'logical_cpus': os.cpu_count(), 'platform': platform.platform()}


class Adapter:
    """Keep the inference server running across samples to avoid timing server restarts."""
    def __init__(self, config, evidence):
        self.config, self.evidence, self.counter = config, Path(evidence), 0
        require(isinstance(config['command'], list) and config['command'], 'adapter command must be argv')

    def call(self, request, allow_failure=False):
        self.counter += 1
        result = subprocess.run(self.config['command'], input=json.dumps(request), text=True,
                                capture_output=True, timeout=self.config.get('timeout_seconds', 300), check=False)
        raw = {'command': self.config['command'], 'request': request, 'stdout': result.stdout,
               'stderr': result.stderr, 'exit_code': result.returncode}
        save(self.evidence / f'{self.counter:04d}-{request["action"]}.json', raw)
        if result.returncode:
            require(allow_failure, f'adapter failed: {raw}')
            return raw, None
        response = json.loads(result.stdout)
        require(isinstance(response, dict), 'adapter response must be an object')
        return raw, response


def series(adapter, request, memory_kind):
    samples, loads, memories = [], [], []
    for index in range(23):
        before = machine_load()
        _, response = adapter.call(dict(request, action='sample', sample_index=index, warmup=index < 3))
        require(positive(response['elapsed_ms']), 'missing request latency')
        if before['load_average'] is None:
            require(response.get('machine_load'), 'platform needs runtime CPU/device load telemetry')
        require(response['request_path'] == request['request_path'], 'adapter used wrong request path')
        memory = response['memory']
        require(memory['kind'] == memory_kind, 'wrong memory accounting for row')
        require(positive(memory['process_bytes']), 'missing process memory')
        if memory_kind == 'rss_plus_device':
            require(positive(memory['device_bytes']), 'missing discrete device allocation')
        total = memory['process_bytes'] + (memory['device_bytes'] if memory_kind == 'rss_plus_device' else 0)
        samples.append(response['elapsed_ms'])
        memories.append(total)
        loads.append({'before': before, 'after': machine_load(), 'runtime': response.get('machine_load')})
    return {'raw_series': samples, 'machine_load': loads, 'peak_memory': max(memories), **summarize(samples)}


def binding(response, base, config):
    for key in ('model', 'manifest_digest', 'input_digest', 'file_sha256'):
        require(response.get(key) == base[key], f'adapter {key} differs from shared inputs')
    require(response.get('version') and isinstance(response.get('settings'), dict), 'missing runtime provenance')
    require('tokenization_differences' in response, 'missing tokenization disclosure')
    require(response.get('precision') == config['precision'], 'adapter precision differs from configuration')


def measure(adapter, base, config, row_fixture, memory_kind):
    probe, response = adapter.call(dict(base, action='probe'), allow_failure=True)
    cell = {'row_id': base['row_id'], 'model': base['model'], 'runtime': base['runtime'],
            'probe': probe, 'metrics': dict.fromkeys(METRICS), 'inline_max_items': row_fixture['inline']['max_items'],
            'command': config['command'], 'manifest_digest': base['manifest_digest'],
            'input_digest': base['input_digest'], 'file_sha256': base['file_sha256']}
    if response is None:
        # Only model_unsupported or hardware_unavailable failures count as support evidence.
        failure = json.loads(probe['stdout'])
        require(failure.get('status') in ('model_unsupported', 'hardware_unavailable'), 'unclassified probe failure')
        cause = failure['cause']
        require(cause and cause in (probe['stdout'].splitlines() + probe['stderr'].splitlines()), 'cause must quote a complete probe line')
        require('version' in failure and isinstance(failure.get('settings'), dict)
                and 'tokenization_differences' in failure, 'missing failed-probe provenance')
        cell.update(status=failure['status'], cause=cause, version=failure['version'],
                    settings=failure['settings'], tokenization_differences=failure['tokenization_differences'])
        return cell
    binding(response, base, config)
    if base['runtime'] != 'synapse':
        cell['inline_max_items'] = response['inline_max_items']
        require(cell['inline_max_items'] >= 128, 'competitor adapter must support the largest batch')
    cell.update(status='measured', version=response['version'], settings=response['settings'],
                command=config['command'], tokenization_differences=response['tokenization_differences'])
    _, cold = adapter.call(dict(base, action='cold_load', clear_caches=True))
    require(cold.get('caches_cleared') is True and positive(cold['elapsed_ms'])
            and cold.get('cache_clear_evidence'), 'cold load needs cache-clear evidence')
    cell['cold_cache_evidence'] = cold
    workloads = []
    if base['operation'] == 'embed':
        for batch in (1, 8, 32, 128):
            path = 'inline' if batch <= cell['inline_max_items'] else 'job'
            if base['runtime'] == 'synapse' and batch * base['inputs']['composed_length'] > row_fixture['inline']['max_tokens']:
                path = 'job'
            request = dict(base, batch_size=batch, request_path=path, texts=[base['inputs']['text']] * batch)
            timing = series(adapter, request, memory_kind)
            workloads.append({'batch_size': batch, 'request_path': path,
                              'engine_batch_cap': {'max_items': 8, 'max_tokens': 3072} if base['runtime'] == 'synapse' else response['engine_batch_cap'],
                              'throughput': batch * 1000 / timing['median'], **timing})
        cell['embedding'] = workloads
    else:
        for count in (10, 100):
            candidates = base['inputs']['candidates'][:count]
            require(len(candidates) == count, 'need 100 shared candidates')
            lengths = base['inputs']['composed_lengths'][:count]
            split = split_pool(lengths, cell['inline_max_items'], row_fixture['inline']['max_tokens'])
            request = dict(base, pool_size=count, candidates=candidates, call_split=split, request_path='inline')
            timing = series(adapter, request, memory_kind)
            workloads.append({'pool_size': count, 'call_split': split, 'request_path': 'inline', **timing})
        cell['reranking'] = workloads
    # Rerank single-query latency measures one candidate, not the throughput pool.
    single = workloads[0] if base['operation'] == 'embed' else series(adapter, dict(
        base, pool_size=1, candidates=base['inputs']['candidates'][:1], call_split=[[0, 1]], request_path='inline'), memory_kind)
    cell['single_query'] = single
    cell['metrics'] = {'throughput': workloads[0].get('throughput'), 'p50': single['p50'], 'p95': single['p95'],
                       'peak_memory': max([single['peak_memory']] + [w['peak_memory'] for w in workloads]), 'cold_load': cold['elapsed_ms']}
    _, parity = adapter.call(dict(base, action='parity'))
    require(parity.get('model') == base['model'] and parity.get('manifest_digest') == base['manifest_digest']
            and parity.get('input_digest') == base['input_digest'] and parity.get('gates'), 'missing bound parity results')
    cell['parity'] = parity
    if config['precision'] == 'fp32':
        cell['label'] = 'precision-mismatch: fp32 lane vs f16 competitors'
    if base['row_id'] == 'ane-m5':
        cell['notes'] = 'ANE bursts can slow down by about 20%; consult raw series and per-sample load.'
    return cell


def dropped_cell(record):
    cause = record['drop_cause']
    require(record['row_id'] == 'ane-m5' and record['model'].startswith('qwen3-') and cause in ('parity', 'latency'), 'invalid drop')
    result = {'row_id': record['row_id'], 'model': record['model'], 'runtime': 'synapse',
              'status': 'dropped', 'drop_cause': cause, 'metrics': dict.fromkeys(METRICS), 'parity': record['parity']}
    if cause == 'latency':
        raw = record['raw_series']
        require(summarize(raw['ane'])['median'] / summarize(raw['metal'])['median'] > 3, 'invalid latency drop')
        result['raw_series'] = raw
    return result


def transition(config, manifest_digest, session, output):
    inputs = config['transition']['inputs']
    require(inputs['composed_length'] == 512, 'transition input must contain exactly 512 composed tokens')
    runs = {}
    for lane in ('ane-coreml-worker', 'ane-direct-worker'):
        cfg = config['transition'][lane]
        adapter = Adapter(cfg, output / 'raw' / 'transition' / lane)
        base = {'session_id': session, 'model': 'gte-modernbert-base', 'manifest_digest': manifest_digest,
                'operation': 'embed', 'lane': lane, 'inputs': inputs, 'batch_size': 1, 'request_path': 'inline'}
        _, prepared = adapter.call(dict(base, action='prepare_transition'))
        if prepared is None:
            raise ValueError('transition preparation failed')
        require(prepared.get('composed_length') == 512 and prepared.get('lane') == lane, 'transition lane/shape mismatch')
        runs[lane] = series(adapter, base, 'footprint')
    core, direct = (runs[k] for k in ('ane-coreml-worker', 'ane-direct-worker'))
    value = {'session_id': session, 'model': 'gte-modernbert-base', 'batch_size': 1, 'composed_length': 512,
             'coreml': core['raw_series'], 'direct': direct['raw_series'],
             'coreml_median': core['median'], 'direct_median': direct['median'],
             'machine_load': [{'coreml': c, 'direct': d} for c, d in zip(core['machine_load'], direct['machine_load'])],
             'bar_met': direct['median'] <= 1.05 * core['median']}
    save(output / 'ane-transition.json', value)
    return value


def run_row(config, root, output):
    row = config['row_id']
    require(row in ROWS, 'unknown certification row')
    manifest_path = root / 'bench/parity/models.json'
    manifest = load(manifest_path)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    require(config['manifest_digest'] == digest, 'manifest digest mismatch')
    fixture = load(root / 'bench/parity/rows' / f'{row}.json')
    session = str(uuid.uuid4())
    cells = []
    memory_kind = config['machine']['memory_kind']
    require(memory_kind in ('footprint', 'rss_plus_device'), 'missing memory architecture')
    if row in ('metal-m5', 'ane-m5'):
        require(memory_kind == 'footprint', 'Apple rows use unified memory')
    lane = LANES[row.split('-')[0]]
    for model, definition in manifest['models'].items():
        shared = config['models'][model]
        inputs = shared['inputs']
        input_digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        for runtime in ('synapse',) + COMPETITORS:
            cfg = shared['adapters'][runtime]
            profile = manifest['profiles'][f'{model}.{lane}']
            require(cfg['precision'] == ('fp32' if runtime == 'synapse' and profile['compute_dtype'] == 'f32' else 'f16'), 'precision mismatch')
            file = Path(cfg['file'])
            file_digest = hashlib.sha256(file.read_bytes()).hexdigest()
            require(file_digest == cfg['file_sha256'], 'model file digest mismatch')
            if runtime != 'synapse':
                require(file_digest == shared['f16_sha256'], 'competitors must share the same f16 file')
            record = load(shared['certification_record'])
            require(record['row_id'] == row and record['model'] == model and record['source_commit'] == config['source_commit'], 'wrong certification record')
            if runtime == 'synapse' and record['status'] == 'dropped':
                cells.append(dropped_cell(record))
                continue
            base = {'source_commit': config['source_commit'], 'session_id': session, 'row_id': row,
                    'runtime': runtime, 'lane': lane, 'model': model, 'operation': definition['operation'],
                    'manifest_digest': digest, 'input_digest': input_digest, 'file': str(file),
                    'file_sha256': file_digest, 'inputs': inputs, 'row_fixture': fixture}
            adapter = Adapter(cfg, output / 'raw' / row / model / runtime)
            cells.append(measure(adapter, base, cfg, fixture, memory_kind))
    report = {'schema': 1, 'source_commit': config['source_commit'], 'manifest_digest': digest,
              'session_id': session, 'machine': config['machine'], 'cells': cells}
    save(output / f'{row}.json', report)
    if row == 'ane-m5':
        transition(config, digest, session, output)
    return report


def assemble(root, source, output):
    reports = [load(output / f'{row}.json') for row in ROWS]
    digest = hashlib.sha256((root / 'bench/parity/models.json').read_bytes()).hexdigest()
    require(all(r['source_commit'] == source and r['manifest_digest'] == digest for r in reports), 'mixed candidate/manifest')
    cells = [c for r in reports for c in r['cells']]
    expected = {(row, model, runtime) for row in ROWS for model in load(root / 'bench/parity/models.json')['models'] for runtime in ('synapse',) + COMPETITORS}
    require(len(cells) == len(expected) and {(c['row_id'], c['model'], c['runtime']) for c in cells} == expected, 'incomplete or duplicate runtime matrix')
    report = {'schema': 1, 'source_commit': source, 'manifest_digest': digest, 'cells': cells,
              'sessions': [{'row_id': row, **{k: r[k] for k in ('session_id', 'machine')}} for row, r in zip(ROWS, reports)]}
    spec = importlib.util.spec_from_file_location('release_checks', root / 'crates/synapse-release-checks/check.py')
    if spec is None or spec.loader is None:
        raise ValueError('release checker cannot be imported')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    # The release checker matches drops by row and model, ignoring runtime.
    # Validate owned cells with drop records and competitors with support probes.
    records = [{'row_id': c['row_id'], 'model': c['model'], 'status': 'dropped'} for c in cells if c['status'] == 'dropped']
    checks.benchmark({'cells': [c for c in cells if c['runtime'] == 'synapse']}, records)
    for runtime in COMPETITORS:
        checks.benchmark({'cells': [c for c in cells if c['runtime'] == runtime]}, [])
    # Only owned cells go in cells: the release checker applies owned drop
    # records to that array. Competitor support remains independently validated.
    report['competitors'] = {runtime: [c for c in cells if c['runtime'] == runtime] for runtime in COMPETITORS}
    report['cells'] = [c for c in cells if c['runtime'] == 'synapse']
    save(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('row', 'assemble'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    require(len(args.source) == 40 and all(c in '0123456789abcdef' for c in args.source), 'source must be full SHA')
    output = ROOT / 'docs/evidence/benchmark' / args.source
    if args.action == 'row':
        config = load(args.config)
        require(config['source_commit'] == args.source, 'wrong source commit')
        run_row(config, ROOT, output)
    else:
        assemble(ROOT, args.source, output)


if __name__ == '__main__':
    main()
