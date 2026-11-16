import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("prepare_timing", HERE / "prepare-timing.py")
assert spec is not None and spec.loader is not None
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class TimingPatchTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "target/scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patches = json.loads((HERE / "timing-patch.json").read_text())
        self.originals = {}
        for name, replacements in self.patches.items():
            text = "\n".join(before for before, _ in replacements)
            self.originals[name] = text.encode()
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def git(self, args):
        command = args[3:]
        if command == ["rev-parse", "HEAD"]:
            return (prepare.PIN + "\n").encode()
        if command == ["status", "--porcelain"]:
            return b""
        if command[0] == "show":
            return self.originals[command[1].split(":", 1)[1]]
        self.fail(f"unexpected git command: {args}")

    def test_exact_pin_patch_and_helper_are_applied(self):
        with patch.object(prepare.subprocess, "check_output", side_effect=self.git):
            prepare.prepare(self.root)
        for name, replacements in self.patches.items():
            text = (self.root / name).read_text()
            for _, after in replacements:
                self.assertIn(after, text)
        self.assertEqual((self.root / "Sources/FluidAudio/StageKit.swift").read_bytes(), (HERE / "StageKit.swift").read_bytes())

    def test_pin_mismatch_never_writes(self):
        with patch.object(prepare.subprocess, "check_output", return_value=b"wrong-pin\n"):
            with self.assertRaisesRegex(ValueError, "pin mismatch"):
                prepare.prepare(self.root)
        self.assertEqual({name: (self.root / name).read_bytes() for name in self.originals}, self.originals)

    def test_late_context_mismatch_never_partially_patches(self):
        name = list(self.patches)[-1]
        self.originals[name] = b"upstream context changed"
        (self.root / name).write_bytes(self.originals[name])
        with patch.object(prepare.subprocess, "check_output", side_effect=self.git):
            with self.assertRaisesRegex(ValueError, "context mismatch"):
                prepare.prepare(self.root)
        self.assertEqual({name: (self.root / name).read_bytes() for name in self.originals}, self.originals)

    def test_patch_only_adds_compile_gated_instrumentation(self):
        for replacements in self.patches.values():
            for before, after in replacements:
                stripped = re.sub(r"(?ms)^#if STAGE_KIT\n.*?^#endif\n?", "", after)
                self.assertEqual(stripped.strip(), before.strip())
        helper = (HERE / "StageKit.swift").read_text()
        self.assertTrue(helper.startswith("#if STAGE_KIT\n"))
        self.assertTrue(helper.rstrip().endswith("#endif"))


if __name__ == "__main__":
    unittest.main()
