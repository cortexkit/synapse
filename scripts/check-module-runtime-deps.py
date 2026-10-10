#!/usr/bin/env python3
"""Refuse subc-daemon and subc-presence in synapse-module's normal Cargo dependencies."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"subc-daemon", "subc-presence"}


def inspect_packages(tree):
    packages = set()
    for line in tree.splitlines():
        match = re.match(r"^([a-zA-Z0-9_-]+) v\S+(?:\s|$)", line)
        if not match:
            raise ValueError(f"unrecognized cargo tree row: {line!r}")
        packages.add(match[1])
    if "synapse-module" not in packages or len(packages) < 2:
        raise ValueError("empty or incomplete synapse-module normal dependency tree")
    print(f"Inspected {len(packages)} normal-edge packages for synapse-module", flush=True)
    forbidden = packages & FORBIDDEN
    if forbidden:
        raise ValueError(f"shipped module depends on forbidden packages: {', '.join(sorted(forbidden))}")
    return packages


def main():
    result = subprocess.run(
        ["cargo", "tree", "--locked", "-p", "synapse-module", "-e", "normal",
         "--target", "all", "--prefix", "none", "--format", "{p}"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    try:
        inspect_packages(result.stdout)
    except ValueError as error:
        print(f"module runtime dependency guard: {error}", file=sys.stderr)
        return 1
    print("module runtime dependency guard: passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
