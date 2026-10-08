"""Serving-harness controls without calling the live or scratch inference service."""
import argparse
import json
import os
import platform
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import speed


class CharacterTokenizer:
    eos_token_id = 999

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def vector():
    return [1.] + [0.] * 1023


class SpeedTests(unittest.TestCase):
    def test_load_gate_rejects_boundary_without_launch(self):
        with patch("speed.os.getloadavg", return_value=(16., 1., 1.)):
            with self.assertRaises(speed.NotQuiet):
                speed.quiet_load()
        with patch("speed.os.getloadavg", return_value=(15.999, 1., 1.)):
            self.assertEqual(speed.quiet_load(), 15.999)

    def test_busy_request_is_not_sent(self):
        client = unittest.mock.Mock()
        with patch("speed.os.getloadavg", return_value=(20., 1., 1.)):
            with self.assertRaises(speed.NotQuiet):
                speed.measure(client, [("public", [1])], "public-key")
        client.request.assert_not_called()

    def test_fixture_exact_lengths_and_terminal_eos(self):
        tokenizer = CharacterTokenizer()
        for count in (100, 128, 150, 512):
            text, ids = speed.synthetic(tokenizer, count, 3)
            self.assertEqual(len(ids), count)
            self.assertEqual(ids[-1], 999)
            self.assertTrue(text.startswith("src/cache.rs:"))

    def test_percentiles_and_memory_labels(self):
        self.assertEqual(speed.percentile([10, 20, 30], .5), 20)
        self.assertEqual(speed.percentile([10, 20, 30], .9), 28)
        self.assertEqual(speed.memory_from_time("100 maximum resident set size\n200 peak memory footprint\n"),
                         {"max_rss_bytes": 100, "peak_footprint_bytes": 200})
        with self.assertRaises(ValueError):
            speed.memory_from_time("no high water mark")

    def test_bad_vectors_fail(self):
        speed.valid_vectors([vector()], 1)
        for bad in ([], [[0.] * 1024], [[float("nan")] * 1024], [[1.] * 3]):
            with self.assertRaises(ValueError):
                speed.valid_vectors(bad, 1)

    def test_llama_ids_and_completed_tokens(self):
        client = speed.Llama.__new__(speed.Llama)
        client.url = "http://127.0.0.1:1"
        reply = {"data": [{"index": 0, "embedding": vector()}], "usage": {"prompt_tokens": 2}}
        with patch("speed.http", return_value=reply) as request:
            client.request(["public"], [[1, 2]], "unused")
            self.assertEqual(request.call_args[0][1]["input"], [[1, 2]])
        reply["usage"]["prompt_tokens"] = 1
        with patch("speed.http", return_value=reply), self.assertRaises(ValueError):
            client.request(["public"], [[1, 2]], "unused")

    def test_synapse_reads_every_job_page_by_id(self):
        client = speed.Synapse.__new__(speed.Synapse)
        client.fingerprint = None
        calls = []

        def call(method, params):
            calls.append((method, params))
            if method == "embed.batch":
                return {"job_id": "public-job"}
            page = params.get("page", 0)
            return {"state": "done", "page_count": 2, "fingerprint": "public-fingerprint",
                    "vectors": [{"id": f"item-{1 - page}", "vector": vector()}],
                    "real_token_counts": [2]}

        client.call = call
        client.request(["public-a", "public-b"], [[1, 2], [3, 4]], "public-key")
        self.assertEqual(calls[-1], ("embed.result", {"job_id": "public-job", "page": 1}))
        self.assertEqual(client.fingerprint, "public-fingerprint")
        self.assertEqual(calls[0][1]["request_key"], "public-key")
        self.assertEqual(calls[0][1]["input_type"], "document")

    @unittest.skipUnless(platform.system() == "Darwin", "Darwin time -l only")
    def test_owned_child_memory_high_water(self):
        with tempfile.TemporaryDirectory() as temp:
            child = speed.Process(Path(temp) / "child", ["/usr/bin/true"])
            child.child.wait(timeout=5)
            child.close()
            result = speed.memory_from_time(child.log.read_text())
            self.assertGreater(result["max_rss_bytes"], 0)


if __name__ == "__main__":
    print("Python", platform.python_version(), "serving harness controls", flush=True)
    unittest.main(verbosity=2)
