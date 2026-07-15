"""Framework-independent TRACE reward bridge for gather final packages.

Reward shaping v1 is deliberately narrow: a naturally completed, schema-valid
package receives its cited-file F1 against the selected gold package. Invalid
or non-natural packages receive zero. The diagnostics preserve the other
mechanical measurements for later shaping work without changing this reward.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = PROJECT_ROOT / "data" / "eval-gold-rows.jsonl"


def _failure(message: str) -> dict[str, Any]:
    """Return a trainer-safe zero reward when the scoring bridge cannot run."""
    return {
        "reward": 0.0,
        "diagnostics": {
            "file_f1": 0.0,
            "line_jaccard": 0.0,
            "contract_valid": False,
            "tool_calls": 0,
            "error": message,
        },
    }


def _verdict_from_stdout(stdout: str) -> dict[str, Any] | None:
    """Read the one score-one JSON line while tolerating Bun startup notices."""
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("diagnostics"), dict) and "reward" in value:
            return value
    return None


def reward(
    trajectory_or_final_package: Any,
    job_id: str,
    *,
    gold: str | Path | None = None,
    bun_binary: str | None = None,
) -> dict[str, Any]:
    """Score one trajectory, BankedRow, or final package against a gold job.

    The candidate is serialized to a temporary JSON file because score-one is
    also useful outside Python. A bridge failure is returned as a zero reward
    with diagnostics rather than raising inside a trainer's reward callback.
    """
    if not isinstance(job_id, str) or not job_id.strip():
        return _failure("job_id must be a non-empty string")

    candidate_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as candidate_file:
            json.dump(trajectory_or_final_package, candidate_file, ensure_ascii=False)
            candidate_file.write("\n")
            candidate_path = Path(candidate_file.name)

        gold_path = Path(gold) if gold is not None else Path(os.environ.get("GATHER_DISTILL_REWARD_GOLD", DEFAULT_GOLD))
        command = [
            bun_binary or os.environ.get("BUN_BINARY", "bun"),
            "run",
            "src/cli.ts",
            "score-one",
            "--job",
            job_id,
            "--candidate-file",
            str(candidate_path),
            "--gold",
            str(gold_path),
        ]
        completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"score-one exited {completed.returncode}"
            return _failure(detail)
        verdict = _verdict_from_stdout(completed.stdout)
        if verdict is None or not isinstance(verdict.get("reward"), (int, float)):
            return _failure("score-one did not emit a valid JSON verdict")
        return {"reward": float(verdict["reward"]), "diagnostics": verdict["diagnostics"]}
    except (OSError, TypeError, ValueError) as error:
        return _failure(str(error))
    finally:
        if candidate_path is not None:
            candidate_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one TRACE candidate through gather-distill.")
    parser.add_argument("--job", required=True, help="Gold job_id to score against")
    parser.add_argument("--candidate-file", required=True, type=Path, help="JSON trajectory, BankedRow, or final package")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="Gold JSONL file")
    parser.add_argument("--bun", default=None, help="Bun executable (default: BUN_BINARY or bun)")
    args = parser.parse_args()
    try:
        candidate = json.loads(args.candidate_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(json.dumps(_failure(f"cannot parse candidate file: {error}"), separators=(",", ":")))
        return 0
    print(json.dumps(reward(candidate, args.job, gold=args.gold, bun_binary=args.bun), separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
