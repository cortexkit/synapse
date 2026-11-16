#!/usr/bin/env python3
"""Apply the benchmark-only timing seam to an otherwise pristine exact source pin."""
import argparse
import json
from pathlib import Path
import subprocess
import stat

PIN = "66a239a32609d3f90b50a9634e4b2820ab977011"
HERE = Path(__file__).resolve().parent


def prepare(root):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    if git("rev-parse", "HEAD").decode().strip() != PIN:
        raise ValueError("FluidAudio source pin mismatch")
    if git("status", "--porcelain").strip():
        raise ValueError("FluidAudio checkout must be pristine (including untracked files)")
    replacements = json.loads((HERE / "timing-patch.json").read_text())
    pending = {}
    for relative, patches in replacements.items():
        path = root / relative
        original = path.read_bytes()
        if original != git("show", f"{PIN}:{relative}"):
            raise ValueError(f"source mismatch: {relative}")
        text = original.decode()
        for before, after in patches:
            if text.count(before) != 1:
                raise ValueError(f"patch context mismatch: {relative}: {before!r}")
            text = text.replace(before, after)
        pending[path] = text
    helper = root / "Sources/FluidAudio/StageKit.swift"
    if helper.exists():
        raise ValueError("StageKit helper already exists")
    pending[helper] = (HERE / "StageKit.swift").read_text()
    # Validate every source/context before modifying any dependency file.
    for path, text in pending.items():
        if path.exists():
            # SwiftPM materializes pinned source files read-only.
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        path.write_text(text)
    print(f"Applied timing-only patch to {PIN}: {len(pending)} files; enable with -DSTAGE_KIT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    prepare(parser.parse_args().checkout.resolve())
