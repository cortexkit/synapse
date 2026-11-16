"""Materialize SHA-256-pinned speech recipes, preserving original resampling inputs."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tarfile
from typing import Any


def prepare(repo, cache, output):
    sys.path.insert(0, str(repo / 'bench/parity/reference'))
    generator: Any = importlib.import_module('generate_parakeet')
    preparer: Any = importlib.import_module('prepare_parakeet')
    pq: Any = importlib.import_module('pyarrow.parquet')

    generator.ref.configure_runtime()
    parity = repo / 'bench/parity'
    document = json.loads((parity / 'fixtures/parakeet-tdt-0.6b-v3/reference.json').read_text())
    corpus = json.loads((parity / 'reference/parakeet-corpus.json').read_text())
    audio = cache / 'audio'
    audio.mkdir(parents=True, exist_ok=True)
    for name, source in corpus['sources'].items():
        if name.endswith('-transcripts'):
            continue
        wanted = {r['member']: r['audio_sha256'] for r in corpus['cases'] if r['source'] == name}
        if all((audio / digest).is_file() and hashlib.sha256((audio / digest).read_bytes()).hexdigest() == digest for digest in wanted.values()):
            continue
        artifact = preparer.fetch(source['url'], source['sha256'], cache / 'sources')
        if name.startswith('librispeech'):
            members = ((r['audio']['path'], r['audio']['bytes']) for r in pq.read_table(artifact).to_pylist())
            for member, raw in members:
                if member in wanted:
                    if hashlib.sha256(raw).hexdigest() != wanted[member]:
                        raise ValueError('corpus member digest mismatch: ' + member)
                    (audio / wanted[member]).write_bytes(raw)
        else:
            with tarfile.open(artifact, 'r:gz') as archive:
                for member in archive:
                    if member.name in wanted:
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise ValueError('corpus member is not a file: ' + member.name)
                        raw = extracted.read()
                        if hashlib.sha256(raw).hexdigest() != wanted[member.name]:
                            raise ValueError('corpus member digest mismatch: ' + member.name)
                        (audio / wanted[member.name]).write_bytes(raw)
        for digest in wanted.values():
            if hashlib.sha256((audio / digest).read_bytes()).hexdigest() != digest:
                raise ValueError('missing or corrupt corpus member: ' + digest)
    generator.CACHE = cache
    output.mkdir(parents=True, exist_ok=True)
    inputs = {}
    for row in document['cases']:
        recipe = row['input']
        sealed_generated_digest = recipe.get('generated_audio_sha256')
        samples = generator.pcm(recipe)
        if len(samples) != row['sample_count']:
            raise ValueError('recipe sample count mismatch: ' + row['id'])
        if recipe.get('recipe') == 'resampled':
            # Exercise the module's decoder/downmix/resampler, not an already
            # resampled mono surrogate that would bypass those operations.
            digest = recipe['generated_audio_sha256']
            raw = (audio / digest).read_bytes()
            if hashlib.sha256(raw).hexdigest() != sealed_generated_digest:
                raise ValueError('resampled audio digest mismatch')
        else:
            # Corpus members may be FLAC, so normalize their container to WAV.
            raw = generator.wav_bytes(samples, 16000, 'FLOAT')
        target = row['id'] + '.wav'
        (output / target).write_bytes(raw)
        inputs[row['id']] = {'path': target, 'sha256': hashlib.sha256(raw).hexdigest()}
    (output / 'inputs.json').write_text(json.dumps(inputs, sort_keys=True, indent=2) + '\n')
    print('prepared', len(inputs), 'sealed speech recipes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.repo.resolve(), args.cache.resolve(), args.output.resolve())
