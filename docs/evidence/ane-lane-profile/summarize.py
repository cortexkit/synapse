"""Aggregate raw profile records by each driver call's timestamp interval."""
import json
import pathlib
import sys


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def summarize(directory):
    root = pathlib.Path(directory)
    calls = json.loads((root / "calls.json").read_text())["records"]
    worker = [r for p in root.glob("worker-*.jsonl") for r in read_lines(p)]
    module = [r for p in root.glob("module-*.jsonl") for r in read_lines(p)]
    results = []
    for call in calls:
        end = call["unix_us"]
        start = end - call["wall_ms"] * 1000
        rows = [r for r in worker if r["kind"] == "row" and start <= r["unix_us"] <= end]
        sequences = [r for r in worker if r["kind"] == "sequences" and start <= r["unix_us"] <= end]
        rpc = [r for r in worker if r["kind"] == "request" and start <= r["unix_us"] <= end]
        m = [r for r in module if start <= r["unix_us"] <= end]
        phases = {}
        for row in rows:
            for phase in row["phases"]:
                phases[phase["stage"]] = phases.get(phase["stage"], 0) + phase["ms"]
        layers = [layer for row in rows for layer in row["layers"]]
        evaluate = sum(layer.get("submit_wait_ms", layer.get("wall_ms", 0)) for layer in layers)
        hardware = sum(layer.get("hardware_ms", 0) for layer in layers)
        preparation = sum(layer.get("prepare_ms", 0) for layer in layers)
        module_phases = {}
        for phase in m:
            module_phases[phase["stage"]] = module_phases.get(phase["stage"], 0) + phase["ms"]
        rpc_ms = sum(r["wall_ms"] for r in rpc)
        results.append({**call, "measurement_status": "measured" if rows else "not_measured", "overlapping_call_window": "overlap-" in call["label"], "profile_rows": len(rows), "dispatches": len(layers),
                        "shape_counts": {str(s): sum(r["shape"] == s for r in rows) for s in sorted({r["shape"] for r in rows})},
                        "worker_row_ms": sum(r["wall_ms"] for r in rows),
                        "worker_rpc_ms": rpc_ms, "phases_ms": phases,
                        "layer_prepare_ms": preparation, "layer_submit_wait_ms": evaluate,
                        "layer_hardware_ms": hardware,
                        "layer_nonhardware_ms": evaluate - hardware if hardware else None,
                        "interrow_gaps_ms": sum(sum(r["gaps_ms"]) for r in sequences),
                        "row_evidence_io_ms": sum(sum(r["row_envelopes_ms"]) for r in sequences) - sum(r["wall_ms"] for r in rows),
                        "module_ms": module_phases,
                        "roundtrip_minus_worker_rpc_ms": module_phases.get("direct_ane_roundtrip", 0) - rpc_ms})
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    for r in results:
        print(r["label"], f'load={r["load_1m"]:.2f} wall={r["wall_ms"]:.3f}',
              f'rows={r["profile_rows"]} dispatch={r["layer_submit_wait_ms"]:.3f} hw={r["layer_hardware_ms"]:.3f}',
              "error=" + str(r["error"]))
    return results


if __name__ == "__main__":
    for directory in sys.argv[1:]:
        summarize(directory)
