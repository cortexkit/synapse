#!/usr/bin/env python3
"""Render mutations.toml for the nightly breadth audit.

Most rows set `select = "expected"` so routine replays run only their listed
tests. The nightly audit must run every test target to find unexpected
collateral failures, so this removes `select` and verifies nothing else in any
row changed.
"""
import re
import sys
import tomllib
from pathlib import Path


def prepare(source: str) -> str:
    # Refuse to rewrite anything except top-level select fields in a control.
    # Tracking multiline strings keeps Rust anchors containing TOML-looking text
    # byte-identical; changing an anchor would audit a different mutant.
    lines = []
    multiline = None
    in_control = False
    for line in source.splitlines(keepends=True):
        if multiline is None:
            if line.strip() == "[[control]]":
                in_control = True
            if in_control and re.fullmatch(
                r'\s*select\s*=\s*"expected"\s*(?:#[^\r\n]*)?\r?\n?', line
            ):
                continue
            for delimiter in ('"""', "'''"):
                if line.count(delimiter) % 2:
                    multiline = delimiter
                    break
        elif line.count(multiline) % 2:
            multiline = None
        lines.append(line)
    rendered = "".join(lines)
    expected = tomllib.loads(source)
    for control in expected.get("control", []):
        control.pop("select", None)
    if tomllib.loads(rendered) != expected:
        raise ValueError("breadth rendering must remove only expected-test selection")
    return rendered


def main() -> None:
    source, output = map(Path, sys.argv[1:])
    text = source.read_text(encoding="utf-8")
    rendered = prepare(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Prepared {len(tomllib.loads(rendered)['control'])} breadth controls without exact-test selection")


if __name__ == "__main__":
    main()
