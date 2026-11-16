#!/usr/bin/env python3
"""Owner-only measurement driver. Builds nothing, preserves normal per-process lane locks."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from prepare import ROOT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("measure", "score"))
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--owner-paused-neural-engine", action="store_true")
    args = parser.parse_args()
    out = args.evidence.resolve()
    prepared = json.loads((out / "prepared.json").read_text())
    paths = prepared["paths"]
    if args.action == "score":
        scratch = ROOT / "target/stt-bias" / out.name
        scratch.mkdir(parents=True, exist_ok=True)
        for mode in ("off", "on"):
            source = out / f"fluid-{mode}.json"
            input_path = scratch / source.name
            score_path = scratch / f"fluid-{mode}-wer.json"
            shutil.copy2(source, input_path)
            subprocess.run([paths["scorer"], paths["wavs"], str(input_path), str(score_path)], check=True)
            shutil.copy2(score_path, out / score_path.name)
        print("Retained full per-split parity-normalized WER receipts (benchmark scores, not certification).")
        return
    if sys.platform != "darwin" or not args.owner_paused_neural_engine:
        raise ValueError("measure requires the M5 owner and an explicitly paused Neural Engine window")
    if "TMPDIR" in os.environ:
        raise ValueError("unset TMPDIR; never redirect the cross-process lane lock")
    if (out / "exits.json").exists():
        raise ValueError("never overwrite a measured run; prepare a fresh evidence directory")
    for path, expected in prepared["sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"prepared input changed: {path}")
    run_id = str(uuid.uuid4())
    exits = {}

    def run(name, command, env):
        with (out / f"{name}.stdout").open("xb") as stdout, (out / f"{name}.stderr").open("xb") as stderr:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr).returncode
        exits[name] = {"exit_code": code, "argv": command}
        (out / "exits.json").write_text(json.dumps({"run_id": run_id, "runs": exits}, indent=2) + "\n")
        print(f"{name}: exit {code}", flush=True)
        return code

    env = {k: v for k, v in os.environ.items() if not k.startswith("ANE_PARAKEET_")}
    env.update(ANE_PARAKEET_CERTIFY="1", ANE_PARAKEET_SNAPSHOT=paths["snapshot"],
        ANE_PARAKEET_PACKAGE=paths["package"], ANE_PARAKEET_MIRROR=os.environ["ANE_PARAKEET_MIRROR"],
        ANE_PARAKEET_WAVS=paths["wavs"], ANE_PARAKEET_PARITY_ROOT=str(ROOT / "bench/parity"),
        ANE_PARAKEET_MAX_DURATION_MS="81850")
    ane = [paths["ane_executable"], "--exact", "parakeet::hardware::certify_parakeet", "--ignored", "--nocapture", "--test-threads=1"]
    env.update(ANE_PARAKEET_SELF_CHECK="1", ANE_PARAKEET_REPORT=str(out / "self-check.json"))
    if run("self-check", ane, env):
        raise RuntimeError("first-load exact-reference gate failed; retained receipt, no corpus run")
    env["ANE_PARAKEET_SELF_CHECK"] = "0"
    for mode in ("off", "on"):
        env.update(ANE_PARAKEET_STAGE_RUN_ID=run_id, ANE_PARAKEET_HOST_TIMING="1" if mode == "on" else "0",
                   ANE_PARAKEET_REPORT=str(out / f"synapse-{mode}.json"))
        # Collect-all accuracy failures are retained, never converted to timing certification.
        run(f"synapse-{mode}", ane, env)
    for mode in ("off", "on"):
        fluid_env = dict(env, ANE_PARAKEET_HOST_TIMING="1" if mode == "on" else "0")
        command = [paths["fluid_executable"], "--backend", "ane-pal6", "--model", paths["fluid_model"],
            "--request", str(out / "request.json"), "--output", str(out / f"fluid-{mode}.json"),
            "--stage-run-id", run_id, "--stage-output", str(out / f"fluid-{mode}.stages.jsonl")]
        if run(f"fluid-{mode}", command, fluid_env):
            raise RuntimeError("FluidAudio failed; do not compare a partial corpus")
    print("Measurement finished; the owner can resume other ANE work. Score and compare outside the window.")


if __name__ == "__main__":
    main()
