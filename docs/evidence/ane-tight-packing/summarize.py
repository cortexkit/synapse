#!/usr/bin/env python3
"""Summarize hash-only hardware reports; never load or emit the export text."""
import argparse
import collections
import copy
import hashlib
import json
import statistics


def samples(rows):
    walls = [r["wall_ms"] for r in rows]
    return {
        "median_wall_s": statistics.median(walls) / 1000,
        "walls_s": [w / 1000 for w in walls],
        "min_cosine": min(r["parity"]["min_cosine"] for r in rows),
        "max_abs": max(r["parity"]["max_abs"] for r in rows),
        "identical": min(r["parity"].get("identical", r["parity"].get("bit_identical")) for r in rows),
        "load_range": [min(v for r in rows for v in (r["load_before"], r["load_after"])), max(v for r in rows for v in (r["load_before"], r["load_after"]))],
    }


COMPACT_FORMAT = "ane-tight-audit-compact-v1"


def compact_audit(report, raw_bytes):
    """Retain measured tables, worst rows and exact per-pass occupancy counts."""
    compact = copy.deepcopy(report)
    compact["format"] = COMPACT_FORMAT
    compact["full_raw"] = {"sha256": hashlib.sha256(raw_bytes).hexdigest(), "bytes": len(raw_bytes)}
    for entry in compact["audit"]:
        results = entry["results"]
        rows = results.pop("rows")
        summary = results["summary"]
        assert len(rows) == report["rows"] == summary["rows"]
        passed = sum(row["passed"] for row in rows)
        identical = sum(row["bit_identical"] for row in rows)
        assert passed == len(rows) - len(summary["failed_rows"])
        assert identical == summary["bit_identical"]
        assert min(1.0, min(row["cosine"] for row in rows if row["cosine"] is not None)) == summary["min_cosine"]
        assert max(row["max_abs"] for row in rows if row["max_abs"] is not None) == summary["max_abs"]
        for row in rows:
            assert row["passed"] == (row["cosine"] is not None and row["cosine"] >= 0.9999)
        results["row_counts"] = {"audited": len(rows), "passed": passed, "bit_identical": identical}
        ranked = sorted(rows, key=lambda r: (r["cosine"] if r["cosine"] is not None else -1, -(r["max_abs"] or 0), r["seq"]))
        results["worst_20_rows"] = ranked[:20]
        assert results["worst_row"] in results["worst_20_rows"]
        occupancy = {}
        for row in rows:
            pid = row["pass_index"]
            used = row["pass_tokens"]
            assert pid not in occupancy or occupancy[pid] == used
            occupancy[pid] = used
        histogram = collections.Counter(occupancy.values())
        results["per_pass_fill_histogram"] = [{"used_columns": used, "passes": count} for used, count in sorted(histogram.items())]
        plan = next(p for p in report["plans"] if (p["width"], p["window"]) == (entry["width"], entry["window"]))
        assert len(occupancy) == plan["passes"]
        assert sum(occupancy.values()) == report["tokens"]
        assert sum(occupancy.values()) / (entry["width"] * len(occupancy)) == plan["fill_ratio"]
        bins = collections.defaultdict(list)
        for row in rows:
            bins[row["position_in_pass"] // 128].append(row)
        results["position_bins"] = [{"first_column": key * 128, "last_column": key * 128 + 127, "rows": len(group), "passed": sum(r["passed"] for r in group), "bit_identical": sum(r["bit_identical"] for r in group), "min_cosine": min(r["cosine"] for r in group if r["cosine"] is not None), "max_abs": max(r["max_abs"] for r in group if r["max_abs"] is not None)} for key, group in sorted(bins.items())]
    return compact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("planner", nargs="?")
    parser.add_argument("tight", nargs="?")
    parser.add_argument("--compact-full", help="full raw audit to compact instead of generating tables")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.compact_full:
        if args.planner or args.tight:
            parser.error("--compact-full does not take planner/tight arguments")
        with open(args.compact_full, "rb") as source:
            raw_bytes = source.read()
        compact = compact_audit(json.loads(raw_bytes), raw_bytes)
        with open(args.out, "w") as out:
            json.dump(compact, out, indent=2)
            out.write("\n")
        print(f"Compacted {len(compact['audit'])} arm/width/window audits; retained 20 worst rows and pass-fill histogram per audit")
        return
    if not args.planner or not args.tight:
        parser.error("planner and tight reports are required for tables")
    planner = json.load(open(args.planner))
    tight = json.load(open(args.tight))
    assert tight.get("format") == COMPACT_FORMAT, "table generation requires the committed compact audit"
    for key in ["export_sha256", "meta_sha256", "tokenizer_sha256", "package_digest", "rows", "tokens"]:
        assert planner[key] == tight[key], key
    result = {k: planner[k] for k in ["export_sha256", "meta_sha256", "tokenizer_sha256", "package_digest", "rows", "tokens", "batches"]}
    groups = collections.defaultdict(list)
    for row in planner["samples"]:
        groups[(row["method"], row["window"])].append(row)
    result["planner"] = []
    baseline = statistics.median(r["wall_ms"] for r in groups[("single", 0)]) / 1000
    for (method, window), rows in groups.items():
        summary = samples(rows)
        summary.update(method=method, window=window, rows_per_second=planner["rows"] / summary["median_wall_s"], speedup=baseline / summary["median_wall_s"])
        result["planner"].append(summary)
    result["cold_after_reload"] = []
    cold_groups = collections.defaultdict(list)
    for row in planner.get("cold_samples", []):
        warm = next(r for r in planner["samples"] if (r["repeat"], r["method"], r["window"]) == (row["repeat"], row["method"], row["window"]))
        cold_groups[row["method"]].append({"window": row["window"], "repeat": row["repeat"], "cold_s": row["wall_ms"] / 1000, "warm_s": warm["wall_ms"] / 1000, "penalty_s": (row["wall_ms"] - warm["wall_ms"]) / 1000, "cold_shapes": row["cold_shapes"]})
    for method, rows in cold_groups.items():
        result["cold_after_reload"].append({"method": method, "median_penalty_s": statistics.median(r["penalty_s"] for r in rows), "paired_samples": rows})
    groups.clear()
    for row in tight["samples"]:
        groups[(row["width"], row["arm"], row["method"], row["window"])].append(row)
    result["tight"] = []
    for (width, arm, method, window), rows in groups.items():
        summary = samples(rows)
        baseline = statistics.median(r["wall_ms"] for r in groups[(width, arm, "single", 0)]) / 1000
        summary.update(width=width, arm=arm, method=method, window=window, rows_per_second=tight["rows"] / summary["median_wall_s"], speedup=baseline / summary["median_wall_s"])
        result["tight"].append(summary)
    result["tight_first_export"] = []
    for row in tight.get("cold_samples", []):
        warm_s = statistics.median(r["wall_ms"] for r in groups[(row["width"], row["arm"], row["method"], row["window"])]) / 1000
        result["tight_first_export"].append({"width": row["width"], "arm": row["arm"], "method": row["method"], "cold_s": row["wall_ms"] / 1000, "warm_median_s": warm_s, "penalty_s": row["wall_ms"] / 1000 - warm_s, "cold_shapes": row["cold_shapes"]})
    result["pass_cost"] = []
    for row in tight["pass_cost"]:
        medians = {v["mode"]: statistics.median(s["wall_ms"] for s in v["samples"]) for v in row["variants"]}
        result["pass_cost"].append({"width": row["width"], "segment_lengths": row["segment_lengths"], "encode_decode_ms": row["encode_decode_ms"], "median_ms": medians, "runtime_mask_over_constants": medians["runtime-mask"] / medians["constants"], "runtime_all_over_constants": medians.get("constant-compatible", 0) / medians["constants"] if "constant-compatible" in medians else None})
    result["fixed_plans"] = planner["plans"]
    result["tight_plans"] = tight["plans"]
    result["correctness"] = [{"width": r["width"], "arm": r["arm"], "window": r["window"], "summary": {k: v for k, v in r["results"]["summary"].items() if k != "failed_rows"}, "failures": len(r["results"]["summary"]["failed_rows"]), "worst_row": r["results"]["worst_row"], "fixture": r["fixture"], "isolation": r["isolation"]} for r in tight["audit"]]
    result["single_fp32"] = [planner["single_fp32"], tight["single_fp32"]]
    result["planner_setup"] = planner["setup"]
    result["tight_setup"] = tight["setup"]
    with open(args.out, "w") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    for section in ["planner", "tight", "pass_cost", "tight_plans"]:
        print(section)
        for row in result[section]:
            print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
