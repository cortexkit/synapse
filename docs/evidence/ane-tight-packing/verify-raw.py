#!/usr/bin/env python3
"""Validate published JSON without displaying input text, IDs or vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from summarize import COMPACT_FORMAT, compact_audit

FORBIDDEN = {"text", "content", "input_ids", "output", "embedding", "embeddings", "vector", "vectors", "activations"}


def fields(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from fields(item)
    elif isinstance(value, list):
        for item in value:
            yield from fields(item)


def nonfinite(_):
    raise ValueError("non-finite JSON number")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--full-raw", type=Path, help="optionally recheck the external full audit against its committed compact report")
    args = parser.parse_args()
    paths = sorted({f for p in (Path(p).expanduser() for p in args.paths) for f in (p.glob("*.json") if p.is_dir() else [p])})
    full_raw = args.full_raw.expanduser() if args.full_raw else None
    if full_raw is not None:
        paths = sorted(set(paths) | {full_raw})
    assert paths, "no JSON reports selected"
    failures = {"raw_json_files_parse": [], "raw_reports_do_not_export_inputs": [], "raw_sha256_fields_are_digests": [], "compact_audit_retains_aggregates": []}
    if full_raw is not None:
        failures["full_raw_matches_compact"] = []
    compacts = []
    for path in paths:
        try:
            report = json.loads(path.read_text(), parse_constant=nonfinite)
        except (ValueError, OSError):
            failures["raw_json_files_parse"].append(path.name)
            continue
        is_compact = isinstance(report, dict) and report.get("format") == COMPACT_FORMAT
        if path.name == "tight-audit.json" and path != full_raw:
            if not is_compact or path.stat().st_size >= 1_000_000:
                failures["compact_audit_retains_aggregates"].append(f"{path.name}: compact format and size below 1 MB required")
        if is_compact:
            compacts.append(report)
            try:
                assert len(report["audit"]) == 12
                for entry in report["audit"]:
                    result = entry["results"]
                    assert "rows" not in result
                    count = result["row_counts"]
                    summary = result["summary"]
                    assert count["audited"] == report["rows"] == summary["rows"]
                    assert count["passed"] == count["audited"] - len(summary["failed_rows"])
                    assert count["bit_identical"] == summary["bit_identical"]
                    assert len(result["worst_20_rows"]) == 20
                    assert result["worst_row"] in result["worst_20_rows"]
                    histogram = result["per_pass_fill_histogram"]
                    plan = next(p for p in report["plans"] if (p["width"], p["window"]) == (entry["width"], entry["window"]))
                    assert sum(h["passes"] for h in histogram) == plan["passes"]
                    assert sum(h["passes"] * h["used_columns"] for h in histogram) == report["tokens"]
                    assert sum(b["rows"] for b in result["position_bins"]) == report["rows"]
            except (AssertionError, KeyError, StopIteration, TypeError):
                failures["compact_audit_retains_aggregates"].append(path.name)
        for key, value in fields(report):
            if key in FORBIDDEN:
                failures["raw_reports_do_not_export_inputs"].append(f"{path.name}: forbidden key {key}")
            if key == "package_digest" and (not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value)):
                failures["raw_sha256_fields_are_digests"].append(f"{path.name}: invalid package digest")
            if key.endswith("sha256") and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
                failures["raw_sha256_fields_are_digests"].append(f"{path.name}: invalid digest field {key}")
    if full_raw is not None:
        try:
            assert len(compacts) == 1, "select exactly one compact audit"
            raw_bytes = full_raw.read_bytes()
            expected = compacts[0]["full_raw"]
            assert len(raw_bytes) == expected["bytes"], "external byte size mismatch"
            assert hashlib.sha256(raw_bytes).hexdigest() == expected["sha256"], "external SHA-256 mismatch"
            regenerated = compact_audit(json.loads(raw_bytes, parse_constant=nonfinite), raw_bytes)
            assert regenerated == compacts[0], "external records do not regenerate the committed compact audit"
        except (AssertionError, ValueError, KeyError, OSError) as error:
            failures["full_raw_matches_compact"].append(str(error))
    for name, errors in failures.items():
        print(f"{'FAIL' if errors else 'PASS'} {name}: {len(paths)} JSON files" + ("; " + "; ".join(errors) if errors else ""))
    return int(any(failures.values()))


if __name__ == "__main__":
    sys.exit(main())
