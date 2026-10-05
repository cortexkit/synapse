#!/usr/bin/env python3
"""HTTP call adapters for llama.cpp, LM Studio, Ollama and TEI.

This bridge invokes the deployment's controller executable (configured as an
argv array) to manage server lifecycle, f16 model loading, version discovery,
cache clearing, tokenization inspection, parity evaluation and memory telemetry.
The controller receives the bridge's JSON request on stdin, with action set to
load, memory, cold_load or parity; see RELEASE-BENCHMARK.md.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from release_matrix import COMPETITORS, require


def payload(runtime, request, texts=None, candidates=None):
    model = request['model']
    if texts is not None:
        if runtime == 'ollama':
            return '/api/embed', {'model': model, 'input': texts, 'truncate': False, 'keep_alive': -1}
        if runtime == 'tei':
            return '/embed', {'inputs': texts, 'truncate': False}
        return '/v1/embeddings', {'model': model, 'input': texts, 'encoding_format': 'float'}
    query = request['inputs']['query']
    if runtime == 'tei':
        return '/rerank', {'query': query, 'texts': candidates, 'truncate': False, 'raw_scores': True}
    return ('/api/rerank' if runtime == 'ollama' else '/v1/rerank'), {
        'model': model, 'query': query, 'documents': candidates, 'return_documents': False}


def post(endpoint, path, body, timeout):
    request = urllib.request.Request(endpoint.rstrip('/') + path, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def invoke(runtime, request, endpoint, timeout):
    responses = []
    start = time.perf_counter_ns()
    if request['operation'] == 'embed':
        path, body = payload(runtime, request, texts=request.get('texts', [request['inputs']['text']]))
        responses.append(post(endpoint, path, body, timeout))
    else:
        candidates = request.get('candidates', request['inputs']['candidates'][:1])
        for begin, end in request.get('call_split', [[0, len(candidates)]]):
            path, body = payload(runtime, request, candidates=candidates[begin:end])
            responses.append(post(endpoint, path, body, timeout))
    elapsed = (time.perf_counter_ns() - start) / 1e6
    # Require nonempty vectors or candidate scores, not just an HTTP success.
    expected = len(request.get('texts', [request['inputs'].get('text')])) if request['operation'] == 'embed' else len(request.get('candidates', request['inputs']['candidates'][:1]))
    count = 0
    for response in responses:
        if request['operation'] == 'embed':
            vectors = response if isinstance(response, list) else response.get('embeddings', response.get('data', []))
            require(all((v.get('embedding') if isinstance(v, dict) else v) for v in vectors), 'empty embedding output')
            count += len(vectors)
        else:
            scores = response if isinstance(response, list) else response.get('results', response.get('data', []))
            require(all('score' in s or 'relevance_score' in s for s in scores), 'missing rerank scores')
            count += len(scores)
    require(count == expected, 'runtime returned wrong output count')
    return elapsed


def bridge(config, request):
    runtime = request['runtime']
    require(runtime in COMPETITORS, 'unknown competitor')

    def control(action):
        result = subprocess.run(config['controller'], input=json.dumps(dict(request, action=action)),
                                capture_output=True, text=True, timeout=config.get('timeout_seconds', 300), check=False)
        if result.stderr:
            print(result.stderr, file=sys.stderr, end='')
        if result.returncode:
            # Explicit support status distinguishes hardware absence from model rejection
            # in the report, rather than silently treating every failure as unsupported.
            failure = json.loads(result.stdout)
            require(failure.get('status') in ('model_unsupported', 'hardware_unavailable'), 'unclassified controller failure')
            print(failure['cause'], file=sys.stderr)
            print(json.dumps(failure))
            raise SystemExit(result.returncode)
        return json.loads(result.stdout)

    action = request['action']
    if action == 'probe':
        provenance = control('load')
        require(provenance.get('running') is True, 'controller did not load model in running runtime')
        try:
            invoke(runtime, request, config['endpoint'], config.get('timeout_seconds', 300))
        except urllib.error.HTTPError as error:
            message = f'HTTP {error.code}: {error.read().decode(errors="replace")}'
            cause = message.splitlines()[0]
            # A live server rejecting this model/operation is unsupported. Server
            # faults and authentication failures are not evidence of model support.
            if error.code not in (400, 404, 422):
                raise
            print(message, file=sys.stderr)
            print(json.dumps({**provenance, 'status': 'model_unsupported', 'cause': cause}))
            raise SystemExit(2)
        return provenance
    if action == 'sample':
        elapsed = invoke(runtime, request, config['endpoint'], config.get('timeout_seconds', 300))
        memory = control('memory')
        return {'elapsed_ms': elapsed, 'request_path': request['request_path'], 'memory': memory,
                'machine_load': memory.get('machine_load')}
    require(action in ('cold_load', 'parity'), 'unknown adapter action')
    return control(action)


if __name__ == '__main__':
    print(json.dumps(bridge(json.loads(Path(sys.argv[1]).read_text()), json.load(sys.stdin))))
