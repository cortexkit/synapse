#!/usr/bin/env python3
"""Fast local feedback, not a substitute for the train's native CI gates."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


def cargo_test(package, *selection):
    return ["cargo", "test", "--locked", "-p", package, *selection]


def narrow_tests(metadata, changed, root):
    """Select unit targets and changed integration targets, including nested crates."""
    packages = sorted(metadata["packages"], key=lambda p: len(p["manifest_path"]), reverse=True)
    selected = {}
    for name in changed:
        path = root / name
        for package in packages:
            directory = Path(package["manifest_path"]).parent
            try:
                relative = path.relative_to(directory)
            except ValueError:
                continue
            targets = package["targets"]
            choices = selected.setdefault(package["name"], set())
            if relative.parts[0] == "tests":
                # Helpers can be imported by any integration target. If the path
                # does not identify one target, conservatively run all of them.
                matching = [t for t in targets if "test" in t["kind"] and (
                    path == Path(t["src_path"]) or (
                        Path(t["src_path"]).name in ("main.rs", "mod.rs")
                        and Path(t["src_path"]).parent in path.parents
                    )
                )]
                choices.update(("--test", t["name"]) for t in matching)
                if not matching:
                    choices.add(("--tests",))
            else:
                choices.update(("--lib",) for t in targets if "lib" in t["kind"])
                choices.update(("--bin", t["name"]) for t in targets if "bin" in t["kind"])
            break
    return [cargo_test(package, *(["--features", "test-support"] if package == "synapse-module" else []), *choice)
            for package, choices in sorted(selected.items())
            for choice in sorted(choices)]


def static_gates():
    return [
        ["cargo", "fmt", "--all", "--check"],
        ["bash", "scripts/public-banlist-check.sh"],
        ["bash", "scripts/check-cortexkit-layout.sh"],
        ["bash", "scripts/check-no-external-path-deps.sh"],
        ["bash", "scripts/check-train-preconditions.sh"],
        ["python3", "scripts/check-module-runtime-deps.py"],
        ["python3", "scripts/test-ci-e2e-output.py"],
        ["python3", "-m", "unittest", "scripts.test_land_gates", "scripts.test_mutation_ci.MutationCiTests", "-v"],
    ]


def contract_tests():
    return [
        cargo_test("synapse-core", "--test", "dev_binary_guard", "--test", "child_process"),
        # One Cargo feature resolution/build for all mandatory module targets.
        # Separate cargo-test invocations repeatedly invalidated native outputs.
        ["cargo", "nextest", "run", "--locked", "-p", "synapse-module", "--features", "test-support",
         "--lib", "--test", "wire_contract", "--test", "it", "--retries", "0", "--test-threads", "4",
         "-E", "binary(wire_contract) | test(launch_nonce_readers::) | test(store::) | test(catalog_migration::)"],
    ]


def module_fixtures():
    return [
        ["cargo", "build", "--locked", "--no-default-features", "-p", "synapse-worker-cuda", "-p", "synapse-worker-vulkan"],
        ["cargo", "build", "--locked", "-p", "synapse-module", "--features", "test-support",
         "--bin", "synapse-probe-stub", "--bin", "synapse-worker-timeout-mock"],
    ]


def full_gates():
    return [["ckdev-mutate", "check"], *module_fixtures(),
        ["cargo", "check", "--locked", "-p", "synapse-module", "--features", "test-support"],
        cargo_test("synapse-module", "--features", "test-support", "--lib", "--test", "it", "--test", "skeleton_e2e"),
        ["cargo", "clippy", "--locked", "-p", "synapse-module", "--features", "test-support", "--all-targets", "--", "-D", "warnings"],
    ]


def changed_paths(base):
    committed = subprocess.check_output(["git", "diff", "--name-only", "-z", base], text=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], text=True)
    return sorted(set((committed + untracked).strip("\0").split("\0")) - {""})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also run portable module builds, integration/e2e and clippy")
    parser.add_argument("--base", default="origin/master", help="diff base for narrow tests (includes working tree and untracked files)")
    parser.add_argument("--plan", action="store_true", help="print commands without running gates")
    args = parser.parse_args()
    root = Path.cwd()
    base = subprocess.check_output(["git", "merge-base", "HEAD", args.base], text=True).strip()
    changed = changed_paths(base)
    if not args.full and any(p in ("Cargo.toml", "Cargo.lock") for p in changed):
        parser.error("workspace dependency changes need --full")
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"], text=True))
    commands = static_gates()
    narrow = narrow_tests(metadata, changed, root)
    if args.full:
        commands += full_gates()
    else:
        if any(command[command.index("-p") + 1] == "synapse-module" for command in narrow):
            commands += module_fixtures()
    commands += contract_tests()
    if not args.full:
        commands += narrow
    # Contract tests may also be selected by the diff; do not run identical commands twice.
    commands = list(dict.fromkeys(tuple(command) for command in commands))
    if args.plan:
        print(json.dumps(commands, indent=2))
        return 0
    for tool in (["cargo", "--version"], ["rustc", "--version"], ["cargo", "nextest", "--version"]):
        subprocess.run(tool, check=True)
    if args.full:
        subprocess.run(["ckdev-mutate", "--version"], check=True)
    print("Landing mode: " + ("full portable module" if args.full else "fast; broad integration and native clippy remain CI gates"), flush=True)
    environment = dict(os.environ)
    # Nested Cargo calls must not rediscover a host sccache wrapper in a sandbox.
    environment.setdefault("RUSTC_WRAPPER", "")
    environment.setdefault("CARGO_BUILD_RUSTC_WRAPPER", "")
    for command in commands:
        print("\n$ " + shlex.join(command), flush=True)
        start = time.monotonic()
        result = subprocess.run(command, env=environment)
        print(f"TIMING {time.monotonic() - start:.2f}s exit={result.returncode}: {shlex.join(command)}", flush=True)
        if result.returncode:
            return result.returncode
    print(f"Passed {len(commands)} landing stages", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
