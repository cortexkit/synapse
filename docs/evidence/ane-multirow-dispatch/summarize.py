#!/usr/bin/env python3
"""Summarize multi-row experiment reports written by
`multirow::hardware::multirow_experiment` (one JSON file per arm).

Usage: summarize.py <report.json>... [--json summary.json]
Prints one table row per arm and, with --json, writes the same rows as JSON.
"""
import json
import statistics
import sys


def summarize(path):
    report = json.load(open(path))
    compile_ = report["multi_row_compile"]
    timing = report.get("timing", {})
    parity = report.get("multi_row_parity", {})
    rows = parity.get("rows", [])
    gate = report.get("single_row_fp32_gate", [])
    samples = timing.get("samples", [])
    loads = [s["load_before"] for s in samples] + [s["load_after"] for s in samples]

    def per_row(path_name):
        return [s["per_row_ms"] for s in samples if s["path"] == path_name]

    single, multi = per_row("single"), per_row("multi")
    return {
        "arm": report["arm"],
        "layout": report["layout"],
        "rows": report["rows"],
        "width": report["width"],
        "compiled": compile_["ok"],
        "compile_error": compile_["error"],
        "compile_s": compile_["ms"] / 1000.0,
        "executables": compile_["executables"],
        "single_compile_s": report["single_row_compile"]["ms"] / 1000.0,
        "single_ms_median": statistics.median(single) if single else None,
        "single_ms_range": [min(single), max(single)] if single else None,
        "multi_ms_median": statistics.median(multi) if multi else None,
        "multi_ms_range": [min(multi), max(multi)] if multi else None,
        "speedup": timing.get("speedup"),
        "repeats": len(multi),
        "load_range": [min(loads), max(loads)] if loads else None,
        "parity_rows": len(rows),
        "parity_min_cosine": parity.get("min_cosine_vs_single"),
        "parity_identical_rows": sum(1 for r in rows if r["identical_to_single"]),
        "single_fp32_min_cosine": min((g["cosine_vs_fp32"] for g in gate), default=None),
        "isolation": report.get("isolation"),
        "correctness_failures": len(report.get("correctness_failures", [])),
        "replay": report.get("replay"),
    }


def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main(argv):
    out = None
    if "--json" in argv:
        index = argv.index("--json")
        out = argv[index + 1]
        argv = argv[:index] + argv[index + 2 :]
    rows = [summarize(path) for path in argv]
    rows.sort(key=lambda r: (r["width"], r["rows"], r["layout"]))
    print("arm | compiled | compile s | exe | single ms/row | multi ms/row | speed-up | parity min cos | identical | fp32 min cos | failures | load")
    for r in rows:
        load = r["load_range"]
        print(
            " | ".join(
                [
                    r["arm"],
                    str(r["compiled"]),
                    fmt(r["compile_s"], 1),
                    str(r["executables"]),
                    fmt(r["single_ms_median"]),
                    fmt(r["multi_ms_median"]),
                    fmt(r["speedup"], 2),
                    fmt(r["parity_min_cosine"], 7),
                    f"{r['parity_identical_rows']}/{r['parity_rows']}",
                    fmt(r["single_fp32_min_cosine"], 5),
                    str(r["correctness_failures"]),
                    f"{load[0]:.2f}-{load[1]:.2f}" if load else "-",
                ]
            )
        )
    if out:
        with open(out, "w") as handle:
            json.dump(rows, handle, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
