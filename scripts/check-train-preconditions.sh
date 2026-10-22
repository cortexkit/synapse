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

exec "${PYTHON:-python3}" - "$repo_root/.github/workflows/tests.yml" "$repo_root/.github/workflows/portable.yml" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml


# Only these exact event-only conditions may differ from a push. CPU contracts
# remain push-gated; whole-process fixtures and GPU builds run outside that path.
ALLOWED_JOB_CONDITIONS = {
    "e2e": (
        "github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'",
        "Whole-process fixture tests run nightly or explicitly, not on the train's push path.",
    ),
    "linux-llama-cuda-manual": (
        "github.event_name == 'workflow_dispatch'",
        "The CUDA build is an explicit manual gate, not a push-triggered train gate.",
    ),
    "windows-llama-vulkan-manual": (
        "github.event_name == 'workflow_dispatch'",
        "The Vulkan build is an explicit manual gate, not a push-triggered train gate.",
    ),
    "windows-owned-cuda-manual": (
        "github.event_name == 'workflow_dispatch'",
        "The owned-CUDA Windows build is an explicit manual gate, not a push-triggered train gate.",
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


def check(workflow: object, portable: dict) -> list[str]:
    """Every precondition the parsed workflow breaks, as failure lines."""
    failures: list[str] = []

    triggers = workflow_triggers(workflow)
    push = triggers.get("push") if isinstance(triggers, dict) else None
    branches = push.get("branches") if isinstance(push, dict) else None
    if isinstance(branches, str):
        branches = [branches]
    if not isinstance(branches, list) or "train/**" not in branches:
        failures.append("train precondition failed: push trigger does not include branch train/**")
    # Train and master use the same blocking checks. Replay is asynchronous in
    # the separate master/nightly workflows; catalogue validity still blocks both.
    if not isinstance(branches, list) or "master" not in branches:
        failures.append("train precondition failed: push trigger does not include branch master")

    jobs = workflow.get("jobs") if isinstance(workflow, dict) else None
    if not isinstance(jobs, dict):
        jobs = {}

    native_jobs = portable.get("jobs", {})
    for job_id, job in {**jobs, **native_jobs}.items():
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

    # Keep the allow-list tied to exact jobs and expressions so removing an
    # event-only guard cannot silently turn an expensive audit into a push job.
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

    for platform, runner, partitions in (
        ("linux", "ubuntu-24.04", '["1/4", "2/4", "3/4", "4/4"]'),
        ("windows", "windows-2025", '["1/1"]'),
    ):
        caller = jobs.get(f"{platform}-checks", {})
        if caller.get("uses") != "./.github/workflows/portable.yml" or caller.get("with") != {
            "platform": platform, "runner": runner, "partitions": partitions,
        } or "if" in caller or caller.get("continue-on-error", False):
            failures.append(f"train precondition failed: {platform} portable gates drifted")
        gate = jobs.get(platform, {})
        expected_needs = [f"{platform}-checks", f"{platform}-catalogue"] + (["test"] if platform == "linux" else [])
        expected_run = 'test "$NATIVE_RESULT" = success && test "$CATALOGUE_RESULT" = success'
        expected_env = {
            "NATIVE_RESULT": "${{ needs." + platform + "-checks.result }}",
            "CATALOGUE_RESULT": "${{ needs." + platform + "-catalogue.result }}",
        }
        if platform == "linux":
            expected_run += ' && test "$GUARDS_RESULT" = success'
            expected_env["GUARDS_RESULT"] = "${{ needs.test.result }}"
        gate_steps = gate.get("steps", [])
        if (gate.get("name") != platform or gate.get("if") != "${{ always() }}"
                or gate.get("needs") != expected_needs or len(gate_steps) != 1
                or gate.get("continue-on-error", False)
                or gate_steps[0].get("run") != expected_run
                or gate_steps[0].get("env") != expected_env
                or gate_steps[0].get("if") is not None
                or gate_steps[0].get("continue-on-error", False)):
            failures.append(f"train precondition failed: {platform} aggregate must require every gate")

    independent_jobs = list(native_jobs.values()) + [jobs.get(f"{p}-catalogue", {}) for p in ("linux", "windows")]
    if set(native_jobs) != {"clippy", "nextest", "llama"} or any(
        "if" in job or "needs" in job or job.get("continue-on-error", False)
        for job in independent_jobs
    ):
        failures.append("train precondition failed: native gates must run independently and unconditionally")
    nextest = native_jobs.get("nextest", {})
    if nextest.get("strategy") != {
        "fail-fast": False, "matrix": {"partition": "${{ fromJSON(inputs.partitions) }}"},
    }:
        failures.append("train precondition failed: nextest must run every partition")

    for platform in ("linux", "windows"):
        steps = jobs.get(f"{platform}-catalogue", {}).get("steps", [])
        catalogue = [s for s in steps if isinstance(s, dict) and s.get("name") == "Check mutation catalogue"]
        if len(catalogue) != 1 or catalogue[0].get("if") is not None or not any(
            line.strip() == "ckdev-mutate check"
            for line in str(catalogue[0].get("run", "")).splitlines()
        ):
            failures.append("train precondition failed: mutation catalogue check must be unconditional")

    concurrency = workflow.get("concurrency") if isinstance(workflow, dict) else None
    group = concurrency.get("group") if isinstance(concurrency, dict) else None
    if not isinstance(group, str) or "github.ref" not in group:
        failures.append(
            "train precondition failed: concurrency.group does not include github.ref"
        )

    return failures


def control_workflow() -> tuple[dict, dict]:
    """A minimal workflow that satisfies every precondition."""
    jobs: dict = {"test": {"steps": [{"name": "build", "run": "true"}]}}
    for platform, runner, partitions in (
        ("linux", "ubuntu-24.04", '["1/4", "2/4", "3/4", "4/4"]'),
        ("windows", "windows-2025", '["1/1"]'),
    ):
        jobs[f"{platform}-checks"] = {
            "uses": "./.github/workflows/portable.yml",
            "with": {"platform": platform, "runner": runner, "partitions": partitions},
        }
        jobs[f"{platform}-catalogue"] = {"steps": [
            {"name": "Check mutation catalogue", "run": "ckdev-mutate check"},
        ]}
        gate_env = {
            "NATIVE_RESULT": "${{ needs." + platform + "-checks.result }}",
            "CATALOGUE_RESULT": "${{ needs." + platform + "-catalogue.result }}",
        }
        gate_run = 'test "$NATIVE_RESULT" = success && test "$CATALOGUE_RESULT" = success'
        if platform == "linux":
            gate_env["GUARDS_RESULT"] = "${{ needs.test.result }}"
            gate_run += ' && test "$GUARDS_RESULT" = success'
        jobs[platform] = {
            "name": platform, "if": "${{ always() }}",
            "needs": [f"{platform}-checks", f"{platform}-catalogue"] + (["test"] if platform == "linux" else []),
            "steps": [{"env": gate_env, "run": gate_run}],
        }
    for job_id, (condition, _reason) in ALLOWED_JOB_CONDITIONS.items():
        jobs[job_id] = {"if": condition, "steps": [{"run": "true"}]}
    workflow = {
        True: {"push": {"branches": ["master", "train/**"]}},
        "concurrency": {"group": "ci-${{ github.ref }}"},
        "jobs": jobs,
    }
    portable = {"jobs": {name: {"steps": []} for name in ("clippy", "nextest", "llama")}}
    portable["jobs"]["nextest"]["strategy"] = {
        "fail-fast": False, "matrix": {"partition": "${{ fromJSON(inputs.partitions) }}"},
    }
    return workflow, portable


def self_test() -> list[str]:
    """Run each refusal arm against a planted violation before the real check.

    A checker whose pattern drifts until it matches nothing would pass every
    workflow, so each arm must refuse its own planted break here, and the clean
    control must pass, or the whole run fails instead of reporting clean.
    """
    manual_job = next(iter(ALLOWED_JOB_CONDITIONS))

    def no_train(w, p):
        w[True]["push"]["branches"].remove("train/**")

    def no_master(w, p):
        w[True]["push"]["branches"].remove("master")

    def no_mutations(w, p):
        w["jobs"]["linux-catalogue"]["steps"][0]["run"] = "true"

    def conditional_mutations(w, p):
        w["jobs"]["linux-catalogue"]["steps"][0]["if"] = "inputs.platform == 'linux'"

    def job_if(w, p):
        w["jobs"]["test"]["if"] = "github.ref == 'refs/heads/master'"

    def step_if(w, p):
        w["jobs"]["test"]["steps"][0]["if"] = "github.event_name == 'push'"

    def gate_drift(w, p):
        del w["jobs"][manual_job]["if"]

    def concurrency(w, p):
        w["concurrency"]["group"] = "ci-${{ github.workflow }}"

    def ignored_failure(w, p):
        w["jobs"]["linux"]["steps"][0]["run"] = "true"

    def missing_partition(w, p):
        w["jobs"]["linux-checks"]["with"]["partitions"] = '["1/4"]'

    def serial_lane(w, p):
        p["jobs"]["clippy"]["needs"] = ["nextest"]

    arms = [
        (no_train, "push trigger does not include branch train/**"),
        (no_master, "push trigger does not include branch master"),
        (no_mutations, "mutation catalogue check must be unconditional"),
        (conditional_mutations, "mutation catalogue check must be unconditional"),
        (job_if, "path-dependent if at job 'test':"),
        (step_if, "path-dependent if at job 'test', step 'build'"),
        (gate_drift, f"manual gate allow-list drift at job '{manual_job}'"),
        (concurrency, "concurrency.group does not include github.ref"),
        (ignored_failure, "linux aggregate must require every gate"),
        (missing_partition, "linux portable gates drifted"),
        (serial_lane, "native gates must run independently and unconditionally"),
    ]
    problems: list[str] = []
    clean = check(*control_workflow())
    if clean:
        problems.append(f"the clean control workflow was refused: {clean}")
    for plant, expected in arms:
        workflow, portable = control_workflow()
        plant(workflow, portable)
        found = check(workflow, portable)
        if len(found) != 1 or expected not in found[0]:
            problems.append(f"planting {plant.__name__} expected one '{expected}' refusal, got {found}")
    if not problems:
        print(f"train precondition self-test: clean control passed, {len(arms)} planted breaks refused")
    return problems


def main() -> int:
    problems = self_test()
    if problems:
        for problem in problems:
            print(f"train precondition self-test failed: {problem}")
        return 1

    workflow_path = Path(sys.argv[1])
    try:
        with workflow_path.open(encoding="utf-8") as workflow_file:
            workflow = yaml.safe_load(workflow_file)
        portable = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        print(f"train precondition failed: cannot parse {workflow_path}: {error}")
        return 1

    failures = check(workflow, portable)
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
