#!/usr/bin/env python3
"""Keep tracked hand-written Rust files within their line budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

LIMIT = 5_000
EXCEPTIONS = "scripts/file-size-exceptions.json"


def exempt_path(path: str) -> bool:
    parts = PurePosixPath(path).parts
    name = parts[-1]
    return (
        "tests" in parts[:-1]
        or name == "tests.rs"
        or name.endswith("_tests.rs")
        or parts[:2] == ("bench", "spikes")
    )


def check(root: Path) -> int:
    try:
        entries = json.loads((root / EXCEPTIONS).read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError("exceptions must be a list")
        exceptions = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("each exception must be an object")
            path, maximum, reason = entry.get("path"), entry.get("max_lines"), entry.get("reason")
            if not isinstance(path, str) or not path.endswith(".rs") or path in exceptions:
                raise ValueError(f"invalid or duplicate exception path: {path!r}")
            if type(maximum) is not int or maximum <= LIMIT:
                raise ValueError(f"{path}: max_lines must be an integer over {LIMIT}")
            if not isinstance(reason, str) or not reason.strip() or len(reason.splitlines()) != 1:
                raise ValueError(f"{path}: reason must be a nonempty single line")
            exceptions[path] = maximum
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--", "*.rs"],
            check=True, stdout=subprocess.PIPE,
        )
        paths = [p.decode("utf-8", errors="surrogateescape") for p in result.stdout.split(b"\0") if p]
        errors = []
        checked = set()
        generated = []
        for path in paths:
            if exempt_path(path):
                continue
            with (root / path).open("rb") as source:
                first = source.readline()
                if first.lstrip().startswith(b"// @generated"):
                    generated.append(path)
                    continue
                lines = bool(first) + sum(1 for _ in source)
            checked.add(path)
            maximum = exceptions.get(path)
            if maximum is not None:
                if lines <= LIMIT:
                    errors.append(f"{path}: exception is stale ({lines} lines)")
                elif lines > maximum:
                    errors.append(f"{path}: {lines} lines exceeds exception maximum {maximum}")
                elif lines < maximum:
                    print(f"note: {path}: lower max_lines from {maximum} to {lines}")
            elif lines > LIMIT:
                errors.append(f"{path}: {lines} lines exceeds limit {LIMIT}")
        for path in exceptions.keys() - checked:
            errors.append(f"{path}: exception is stale (not a tracked, non-exempt file)")
        for path in generated:
            print(f"generated exemption: {path}")
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(f"checked {len(checked)} hand-written Rust files; {len(generated)} generated exemptions; {len(errors)} errors")
        return 1 if errors else 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: file-size check failed: {error}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    return check(parser.parse_args().root.resolve())


if __name__ == "__main__":
    sys.exit(main())
