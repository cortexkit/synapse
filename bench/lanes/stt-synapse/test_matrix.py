"""Dispatch-only tests: fake workers never load weights or touch hardware."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]


class SynapseMatrixTests(unittest.TestCase):
    def test_synapse_dispatches_all_precisions_and_both_table_shapes(self):
        scratch = ROOT / "target" / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="stt-matrix-", dir=scratch) as tmp:
            directory = Path(tmp)
            log = directory / "calls.jsonl"
            worker = directory / "fake-bench"
            worker.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
                "with open(os.environ['CALL_LOG'], 'a') as out: out.write(json.dumps(args)+'\\n')\n"
                "Path(args['--output']).write_text('{}')\n"
            )
            worker.chmod(0o755)
            table = directory / "fake-table"
            table.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "from pathlib import Path\n"
                "assert all(Path(p).is_file() for p in sys.argv[1:] if p != '--wide')\n"
                "print('wide' if '--wide' in sys.argv else 'long')\n"
            )
            table.chmod(0o755)
            backends = "metal ane cuda-f16 cuda-f32 vulkan-f16 vulkan-f32".split()
            env = dict(os.environ, STT_LANES="synapse", STT_SYNAPSE_BACKENDS=" ".join(backends),
                       STT_BENCH=str(worker), STT_TABLE=str(table), CALL_LOG=str(log))
            for backend in backends:
                env["STT_SYNAPSE_" + backend.upper().replace("-", "_") + "_MODEL"] = f"model with spaces/{backend}"
            output = directory / "results"
            subprocess.run(["bash", "bench/stt-matrix.sh", "wavs with spaces", str(output)],
                           cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 6)
            for call, backend in zip(calls, backends):
                self.assertEqual(call["--backend"], "synapse-" + backend)
                self.assertEqual(call["--model"], f"model with spaces/{backend}")
                self.assertEqual(call["--wavs"], "wavs with spaces")
                self.assertEqual(call["--model-revision"], "541d1f99c6b0c3cd0b11a95167540bb8edefd82b")
                family = backend.split("-")[0]
                self.assertIn(f"target/stt-synapse-{family}/release/stt-synapse", call["--worker"])
            self.assertEqual((output / "matrix.tsv").read_text(), "long\n")
            self.assertEqual((output / "comparison.tsv").read_text(), "wide\n")
            rerun = subprocess.run(["bash", "bench/stt-matrix.sh", "unused", str(output)],
                                   cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("already exists", rerun.stderr)


if __name__ == "__main__":
    unittest.main()
