#!/usr/bin/env python3
"""Normal-operation Synapse adapter with explicit inline/job completion checks."""
import json
from pathlib import Path
import subprocess
import sys
import time

from release_matrix import require


def invoke(config, request):
    def rpc(method, params):
        argv = [part.replace('{method}', method).replace('{params}', json.dumps(params)) for part in config['rpc_command']]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=config.get('timeout_seconds', 300), check=True)
        body = json.loads(result.stdout)
        # SUBC callers may return a transport envelope containing the JSON body.
        if 'body' in body:
            raw = body['body']
            body = json.loads(bytes(raw).decode() if isinstance(raw, list) else raw)
        require('error' not in body, f'{method} failed: {body}')
        return body['result']

    model = config['model_id']
    fingerprint = config['fingerprint']
    common = {'model': model, 'required_fingerprint': fingerprint}
    start = time.perf_counter_ns()
    if request['operation'] == 'embed':
        result = rpc('embed.batch', dict(common, texts=request.get('texts', [request['inputs']['text']])))
        path = 'job' if 'job_id' in result else 'inline'
        if path == 'job':
            deadline = time.monotonic() + config.get('timeout_seconds', 300)
            while True:
                status = rpc('embed.result', {'job_id': result['job_id']})
                require(status['state'] not in ('failed_transient', 'failed_permanent'), f'embed job failed: {status}')
                if status['state'] == 'done':
                    break
                require(time.monotonic() < deadline, 'embed job timeout')
                time.sleep(0.01)
            # Fetch every result page so job latency covers delivery, not enqueueing.
            count = 0
            for page_index in range(status['page_count']):
                page = rpc('embed.result', {'job_id': result['job_id'], 'page': page_index})
                require('vectors' in page, 'missing job result vectors')
                count += len(page['vectors'])
            require(count == len(request.get('texts', [request['inputs']['text']])), 'wrong job result count')
        if path == 'inline':
            require(len(result['vectors']) == len(request.get('texts', [request['inputs']['text']])), 'wrong inline vector count')
        require(path == request.get('request_path', 'inline'), 'wrong observed embed request path')
    else:
        candidates = request.get('candidates', request['inputs']['candidates'][:1])
        for begin, end in request.get('call_split', [[0, len(candidates)]]):
            result = rpc('rerank.score', dict(common, query=request['inputs']['query'], candidates=candidates[begin:end]))
            require(len(result['scores']) == end - begin, 'wrong rerank output count')
        path = 'inline'
    return (time.perf_counter_ns() - start) / 1e6, path


def bridge(config, request):
    def control(action):
        result = subprocess.run(config['controller'], input=json.dumps(dict(request, action=action)), capture_output=True,
                                text=True, timeout=config.get('timeout_seconds', 300), check=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, end='')
        return json.loads(result.stdout)

    action = request['action']
    if action == 'probe':
        provenance = control('load')
        invoke(config, request)
        return provenance
    if action == 'sample':
        elapsed, path = invoke(config, request)
        return {'elapsed_ms': elapsed, 'request_path': path, 'memory': control('memory')}
    if action == 'prepare_transition':
        return control(action)
    require(action in ('cold_load', 'parity'), 'unknown adapter action')
    return control(action)


if __name__ == '__main__':
    print(json.dumps(bridge(json.loads(Path(sys.argv[1]).read_text()), json.load(sys.stdin))))
