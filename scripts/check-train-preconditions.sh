#!/usr/bin/env bash
# Check that CI still has its required triggers, guards, and concurrency
# isolation before platform-specific work starts.
set -euo pipefail

# CI invokes this from the workspace containing the synapse checkout. Resolve
# the checkout from this script instead of assuming the caller's current path.
script_path="${BASH_SOURCE[0]}"
case "$script_path" in
  /*) ;;
  *) script_path="$PWD/$script_path" ;;
esac
script_dir="$(cd -- "$(dirname -- "$script_path")" && pwd -P)"
repo_root="$(cd "$script_dir/.." && pwd -P)"

exec python3 - "$repo_root/.github/workflows/tests.yml" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml


# These manual GPU gates are intentionally dispatch-only. The test matrix keeps
# CPU coverage on pushes, while these expensive platform-specific builds remain
# available as explicit gates without delaying or weakening a train push.
ALLOWED_JOB_CONDITIONS = {
    "linux-llama-cuda-manual": (
        "github.event_name == 'workflow_dispatch'",
        "The CUDA build is an explicit manual gate, not a push-triggered train gate.",
    ),
    "windows-llama-vulkan-manual": (
        "github.event_name == 'workflow_dispatch'",
        "The Vulkan build is an explicit manual gate, not a push-triggered train gate.",
    ),
}
PATH_CONTEXTS = ("github.ref", "github.event_name")


def condition_text(value: object) -> str:
    """Render a YAML scalar as one line without losing its expression."""
    return " ".join(str(value).split())


def workflow_triggers(workflow: object) -> object:
    """Read `on` from PyYAML loaders that treat the YAML 1.1 key as boolean."""
    if not isinstance(workflow, dict):
        return None
    if "on" in workflow:
        return workflow["on"]
    return workflow.get(True)


def main() -> int:
    workflow_path = Path(sys.argv[1])
    failures: list[str] = []

    try:
        with workflow_path.open(encoding="utf-8") as workflow_file:
            workflow = yaml.safe_load(workflow_file)
    except (OSError, yaml.YAMLError) as error:
        print(f"train precondition failed: cannot parse {workflow_path}: {error}")
        return 1

    triggers = workflow_triggers(workflow)
    push = triggers.get("push") if isinstance(triggers, dict) else None
    branches = push.get("branches") if isinstance(push, dict) else None
    if isinstance(branches, str):
        branches = [branches]
    if not isinstance(branches, list) or "train/**" not in branches:
        failures.append("train precondition failed: push trigger does not include branch train/**")
    # master is deliberately NOT a push trigger: branch protection requires the
    # linux and windows checks on any sha reaching master, and those checks
    # attach to the sha from its train run. A master run would re-prove a
    # checked sha and double CI per landing. This only holds while protection
    # is on; if it is ever removed, the master trigger must come back as the
    # sole observer of an off-train push.
    if isinstance(branches, list) and "master" in branches:
        failures.append(
            "train precondition failed: push trigger includes master "
            "(protected branch; trains carry the checks, a master run is redundant)"
        )

    jobs = workflow.get("jobs") if isinstance(workflow, dict) else None
    if not isinstance(jobs, dict):
        jobs = {}

    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue

        job_condition = job.get("if")
        if job_condition is not None:
            rendered = condition_text(job_condition)
            if any(context in rendered for context in PATH_CONTEXTS):
                expected = ALLOWED_JOB_CONDITIONS.get(str(job_id))
                if expected is None or rendered != expected[0]:
                    failures.append(
                        "train precondition failed: path-dependent if at "
                        f"job '{job_id}': {rendered}"
                    )

        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for step_number, step in enumerate(steps, start=1):
            if not isinstance(step, dict) or step.get("if") is None:
                continue
            rendered = condition_text(step["if"])
            if any(context in rendered for context in PATH_CONTEXTS):
                step_name = step.get("name", f"step {step_number}")
                failures.append(
                    "train precondition failed: path-dependent if at "
                    f"job '{job_id}', step '{step_name}': {rendered}"
                )

    # Keep the allow-list tied to the exact manual-only jobs and expressions so
    # removing a manual gate's guard cannot silently turn it into a push job.
    for job_id, (expected_condition, _reason) in ALLOWED_JOB_CONDITIONS.items():
        job = jobs.get(job_id)
        actual_condition = (
            condition_text(job.get("if"))
            if isinstance(job, dict) and job.get("if") is not None
            else None
        )
        if actual_condition != expected_condition and not (
            actual_condition is not None
            and any(context in actual_condition for context in PATH_CONTEXTS)
        ):
            failures.append(
                "train precondition failed: manual gate allow-list drift at "
                f"job '{job_id}' (expected if: {expected_condition})"
            )

    concurrency = workflow.get("concurrency") if isinstance(workflow, dict) else None
    group = concurrency.get("group") if isinstance(concurrency, dict) else None
    if not isinstance(group, str) or "github.ref" not in group:
        failures.append(
            "train precondition failed: concurrency.group does not include github.ref"
        )

    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
