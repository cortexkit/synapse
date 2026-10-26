#!/usr/bin/env python3
"""Fast landing feedback on MOTOR, not a substitute for native CI gates."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.lib.remote_gate import run


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
            choices = selected.setdefault(package["name"], {})
            if relative.parts[0] == "tests":
                # Helpers can be imported by any integration target. If the path
                # does not identify one target, conservatively run all of them.
                matching = [t for t in targets if "test" in t["kind"] and (
                    path == Path(t["src_path"]) or (
                        Path(t["src_path"]).name in ("main.rs", "mod.rs")
                        and Path(t["src_path"]).parent in path.parents
                    )
                )]
                for t in matching:
                    choices[("--test", t["name"])] = t
                if not matching:
                    # Cargo skips targets whose required features are off, so
                    # --tests leaves feature-gated e2e targets out.
                    choices[("--tests",)] = None
            else:
                choices.update({("--lib",): t for t in targets if "lib" in t["kind"]})
                choices.update({("--bin", t["name"]): t for t in targets if "bin" in t["kind"]})
            break
    return [cargo_test(package, *features(package, target), *choice)
            for package, choices in sorted(selected.items())
            for choice, target in sorted(choices.items(), key=lambda item: item[0])]


def features(package, target):
    """Features a named target needs: its required features, plus test-support for the module."""
    wanted = ["test-support"] if package == "synapse-module" else []
    for feature in (target or {}).get("required-features") or []:
        if feature not in wanted:
            wanted.append(feature)
    return ["--features", ",".join(wanted)] if wanted else []


def static_gates():
    return [
        ["cargo", "fmt", "--all", "--check"],
        ["bash", "scripts/public-banlist-check.sh"],
        ["bash", "scripts/check-cortexkit-layout.sh"],
        ["bash", "scripts/check-no-external-path-deps.sh"],
        ["bash", "scripts/check-train-preconditions.sh"],
        ["python3", "scripts/check-module-runtime-deps.py"],
        ["python3", "scripts/test-ci-e2e-output.py"],
        ["python3", "-m", "unittest", "scripts.test_land_gates", "-v"],
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
    return [*module_fixtures(),
        ["cargo", "check", "--locked", "-p", "synapse-module", "--features", "test-support"],
        cargo_test("synapse-module", "--features", "test-support,e2e", "--lib", "--test", "it", "--test", "skeleton_e2e"),
        ["cargo", "clippy", "--locked", "-p", "synapse-module", "--features", "test-support,e2e", "--all-targets", "--", "-D", "warnings"],
    ]


def changed_paths(base):
    committed = subprocess.check_output(["git", "diff", "--name-only", "-z", base], text=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], text=True)
    return sorted(set((committed + untracked).strip("\0").split("\0")) - {""})


def split_commands(commands, system):
    # These package targets include macOS-gated cases. Linux omits those cases
    # and cannot verify the macOS-specific paths; ignored tests remain disabled.
    mac_packages = {"synapse-engine-owned", "synapse-worker-ane", "synapse-worker-ane-direct",
                    "synapse-worker-decode", "synapse-worker-llama"}
    portable, native = [], []
    for command in commands:
        mac_test = (system == "Darwin" and tuple(command[:2]) == ("cargo", "test")
                    and "-p" in command and command[command.index("-p") + 1] in mac_packages)
        (native if mac_test else portable).append(command)
        if (system == "Darwin" and tuple(command[:2]) == ("cargo", "test") and "-p" in command
                and command[command.index("-p") + 1] == "synapse-module"):
            native.extend(module_macos_tests(command))
    native = list(dict.fromkeys(tuple(command) for command in native))
    return portable, native


def module_macos_tests(command):
    # The broad module suite is portable, but cfg(macos) hides these active tests
    # on Linux. Retain only those cases locally, with their original opt-ins and
    # missing-model skips; this does not enable ignored hardware tests.
    tests = {
        "lib": ["direct_ane_models_reuse_one_runtime_supervisor",
                "stress_schema_rejects_missing_counts_wrong_types_and_invalid_shapes",
                "compile_duration_summary_reports_independent_even_and_odd_medians"],
        "it": ["kv_runtime_matrix_proves_warm_reuse_reclamation_and_lifecycle_faults"],
        "skeleton_e2e": ["probe_worker_death_fails_the_durable_job_with_engine_crashed",
                         "ane_probe_placement_ping_runs_outside_the_module_runtime",
                         "embed_query_loaded_owned_metal_carries_distinct_provenance_and_content_hash",
                         "probe_owned_gte_modernbert_certifies_against_family_reference",
                         "owned_gte_rerank_loads_certifies_and_serves_deterministically"],
    }
    targets, names = [], []
    for target, cases in tests.items():
        selected = ("--tests" in command or (target == "lib" and "--lib" in command)
                    or any(tuple(command[i:i + 2]) == ("--test", target) for i in range(len(command) - 1)))
        if selected:
            targets.extend(["--lib"] if target == "lib" else ["--test", target])
            names.extend(cases)
    if not names:
        return []
    fixtures = module_fixtures()[1:] if "skeleton_e2e" in targets else []
    selection = ["-p", "synapse-module", "--features",
                 "test-support,e2e" if "skeleton_e2e" in targets else "test-support", *targets]
    # A renamed or deleted test would otherwise drop out of the union filter
    # silently, so list the targets first and refuse any name that matches nothing.
    check = ["python3", "scripts/land-gates.py", "--require-tests", json.dumps(names), "--",
             "cargo", "nextest", "list", "--locked", *selection, "--message-format", "json"]
    return [*fixtures, check, ["cargo", "nextest", "run", "--locked", *selection,
             "--retries", "0", "--test-threads", "4", "-E",
             " | ".join(f"test({name})" for name in names)]]


def unmatched_tests(listing, names):
    """Names that no listed test contains, the way nextest's test() filter matches."""
    listed = [name for suite in listing.get("rust-suites", {}).values()
              for name in suite.get("testcases", {})]
    return [name for name in names if not any(name in test for test in listed)]


def require_tests(names, list_command):
    listing = json.loads(subprocess.check_output(list_command, text=True))
    missing = unmatched_tests(listing, names)
    for name in missing:
        print(f"land-gates: macOS test name matches no test: {name}", file=sys.stderr)
    return 1 if missing else 0


def main():
    if sys.argv[1:2] == ["--require-tests"] and sys.argv[3:4] == ["--"]:
        return require_tests(json.loads(sys.argv[2]), sys.argv[4:])
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
    portable, native = split_commands(commands, platform.system())
    local = []
    if args.plan:
        print(json.dumps(local + portable + native, indent=2))
        return 0
    versions = [["cargo", "--version"], ["rustc", "--version"], ["cargo", "nextest", "--version"]]
    print("Landing mode: " + ("full portable module" if args.full else "fast; broad integration and native clippy remain CI gates"), flush=True)
    if native and Path("/Applications/Xcode.app").is_dir():
        os.environ.setdefault("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")
    if native:
        local += versions + native
    return run(versions + portable, local, vcpus=4)


if __name__ == "__main__":
    raise SystemExit(main())
