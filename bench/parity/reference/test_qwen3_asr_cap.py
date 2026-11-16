"""A fake autoregressive model exercises the real reference's capped-trace path."""

from contextlib import nullcontext, redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

from qwen3_asr_corpus import PROVISIONAL_SLOPE, audio_tokens, calibration_trace, token_rate_report
from qwen3_asr_reference import Reference
from run_qwen3_asr_full import run_sequence
from test_qwen3_asr_full import policy, row

HERE = Path(__file__).resolve().parent


class Ids:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        value = self.values[index[0]][index[1]] if isinstance(index, tuple) else self.values[index]
        return Ids(value) if isinstance(value, list) else value

    def tolist(self):
        return self.values


class FloatTensor:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, _index):
        return FloatTensor(self.shape[1:])

    @property
    def T(self):
        return FloatTensor(tuple(reversed(self.shape)))

    def detach(self):
        return self

    clone = contiguous = numpy = detach

    def astype(self, _dtype, **_kwargs):
        return self

    def tobytes(self):
        return b"\0" * (math.prod(self.shape) * 4)


class Tokenizer:
    def encode(self, text, **_kwargs):
        return [1] if text == "<|im_start|>system\n" else [7]

    def decode(self, ids, **_kwargs):
        return "language Greek<asr_text>" + " name" * (len(ids) - 3)


class Processor:
    tokenizer = Tokenizer()

    def __call__(self, **_kwargs):
        count = audio_tokens(200)
        return {"input_ids": Ids([[1] + [151676] * count + [2]]),
                "input_features": FloatTensor((1, 128, 200)),
                "feature_attention_mask": SimpleNamespace(sum=lambda: 200)}


class Projection:
    hook = None

    def register_forward_hook(self, hook):
        self.hook = hook
        return SimpleNamespace(remove=lambda: setattr(self, "hook", None))


class Logits:
    def __init__(self, id):
        self.id = id

    def float(self):
        return self

    def argmax(self):
        return self.id


class FakeTorch:
    tensor = staticmethod(Ids)
    inference_mode = staticmethod(nullcontext)

    @staticmethod
    def ones_like(ids):
        return Ids([[1] * len(ids.values[0])])

    @staticmethod
    def topk(logits, count):
        assert count == 2
        return [1.0, 0.0], [logits.id, 999]


class FakeAudio(FloatTensor):
    def __init__(self):
        super().__init__((32000,))

    def __len__(self):
        return 32000


class FakeModel:
    def __init__(self, reference):
        self.reference = reference
        self.thinker = SimpleNamespace(audio_tower=SimpleNamespace(proj2=Projection()))
        self.unbounded_ids = [11528, 17860, 151704] + [10] * 1000 + [151645]

    def generate(self, input_ids, max_new_tokens, **kwargs):
        assert kwargs["do_sample"] is False and kwargs["output_scores"] is True
        assert kwargs["eos_token_id"] == [151643, 151645]
        prompt = input_ids.values[0]
        self.reference.generation_budget = {"max_new_tokens": max_new_tokens,
                                            "max_length": len(prompt) + max_new_tokens}
        self.thinker.audio_tower.proj2.hook(None, None, FloatTensor((audio_tokens(200), 2048)))
        ids = self.unbounded_ids[:max_new_tokens]
        return SimpleNamespace(sequences=Ids([prompt + ids]), scores=[[Logits(id)] for id in ids])


class CappedModelReference(unittest.TestCase):
    def test_fake_model_emits_past_cap_records_prefix_and_does_not_set_slope(self):
        reference: Any = Reference.__new__(Reference)
        reference.torch = FakeTorch()
        reference.processor = Processor()
        reference.wrapper = SimpleNamespace(_build_text_prompt=lambda *_args: "template")
        reference.stop_ids = [151643, 151645]
        reference.generation_policy = policy()
        reference.model = FakeModel(reference)
        case = {"id": "reciting-list", "split": "earnings22-test-200", "tier": 300,
                "sample_count": 32000, "ground_truth": "public", "terms": ["public term"]}
        utils = ModuleType("qwen_asr.inference.utils")
        setattr(utils, "parse_asr_output", lambda raw, user_language=None: ("Greek", raw.split("<asr_text>", 1)[1]))
        modules = {"qwen_asr": ModuleType("qwen_asr"),
                   "qwen_asr.inference": ModuleType("qwen_asr.inference"),
                   "qwen_asr.inference.utils": utils}
        with tempfile.TemporaryDirectory(dir=HERE.parent) as directory, patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
            output = Path(directory)
            run_sequence(output, [case], "shard-0", {"corpus_sha256": "a" * 64,
                         "reference_rules_sha256": "b" * 64}, lambda _directory: reference,
                         lambda _case, _cache: FakeAudio())
            trace = json.loads((output / "traces/reciting-list.json").read_text())
            self.assertEqual(trace["stop_reason"], "cap")
            self.assertEqual(trace["cap"], 114)
            self.assertEqual(trace["tokens_per_second"], PROVISIONAL_SLOPE)
            self.assertEqual(trace["generated_ids"], reference.model.unbounded_ids[:114])
            self.assertEqual(len(trace["top_two_margins"]), 114)
            self.assertGreater(len(reference.model.unbounded_ids), trace["cap"])
            report = token_rate_report([row("stopped", samples=32000, speech_tokens=6), calibration_trace(trace, case)])
            self.assertEqual(report["measured_tokens_per_second"], 6)
            self.assertEqual(report["tokens_per_second"], 25)
            self.assertEqual(report["speech_fixtures"], 1)
            self.assertEqual(report["complete_cap_checked_fixtures"], 1)
            self.assertEqual(report["capped_fixtures"]["ids"], ["reciting-list"])
            self.assertEqual(report["capped_fixtures"]["by_split"]["earnings22-test-200"]["by_tier"]["300"],
                             {"count": 1, "ids": ["reciting-list"]})


if __name__ == "__main__":
    unittest.main()
