import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from typing import Any
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('prepare_audio', Path(__file__).with_name('prepare-transcribe-audio.py'))
assert spec is not None and spec.loader is not None
prepare_audio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_audio)


class AudioPreparation(unittest.TestCase):
    def run_preparer(self, sealed_digest=None, sample_count=4):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parity = root / 'bench/parity'
            (parity / 'fixtures/parakeet-tdt-0.6b-v3').mkdir(parents=True)
            (parity / 'reference').mkdir()
            native = b'original stereo 48k PCM16 WAV'
            digest = hashlib.sha256(native).hexdigest()
            cases = [dict(id='tensor-silence', sample_count=4, log_mel={'sealed': True},
                          input={'id': 'tensor-silence', 'recipe': 'silence'}),
                     dict(id='resampled', sample_count=sample_count, input=dict(
                         id='resampled', recipe='resampled', generated_audio_sha256=sealed_digest or digest))]
            (parity / 'fixtures/parakeet-tdt-0.6b-v3/reference.json').write_text(json.dumps({'cases': cases}))
            (parity / 'reference/parakeet-corpus.json').write_text(json.dumps({'sources': {}, 'cases': []}))
            generator: Any = types.ModuleType('generate_parakeet')
            generator.ref = types.SimpleNamespace(configure_runtime=lambda: None)

            def pcm(recipe):
                if recipe['recipe'] == 'resampled':
                    (root / 'cache/audio' / digest).write_bytes(native)
                    recipe['generated_audio_sha256'] = digest
                return [0.0] * 4

            generator.pcm = pcm
            generator.wav_bytes = lambda *args: b'mono 16k WAV'
            preparer: Any = types.ModuleType('prepare_parakeet')
            preparer.fetch = lambda *args: self.fail('unexpected source download')
            modules = {'generate_parakeet': generator, 'prepare_parakeet': preparer,
                       'pyarrow': types.ModuleType('pyarrow'), 'pyarrow.parquet': types.ModuleType('pyarrow.parquet')}
            with patch.dict(sys.modules, modules), patch.object(sys, 'path', sys.path.copy()):
                prepare_audio.prepare(root, root / 'cache', root / 'output')
            inputs = json.loads((root / 'output/inputs.json').read_text())
            self.assertEqual(set(inputs), {'tensor-silence', 'resampled'})
            self.assertEqual((root / 'output/resampled.wav').read_bytes(), native)
            self.assertEqual(inputs['resampled']['sha256'], digest)

    def test_prepares_tensor_subset_and_preserves_original_resampled_audio(self):
        self.run_preparer()

    def test_refuses_regenerated_audio_that_differs_from_sealed_input(self):
        with self.assertRaisesRegex(ValueError, 'resampled audio digest mismatch'):
            self.run_preparer(sealed_digest='a' * 64)

    def test_refuses_recipe_sample_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'recipe sample count mismatch'):
            self.run_preparer(sample_count=5)


if __name__ == '__main__':
    unittest.main()
