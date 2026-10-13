"""Summarize NVIDIA certification, RPC timing and worker-progress logs.

An error reply measures failure handling, not inference, so every timed RPC
must succeed before its elapsed time is included in the timing report.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

raw = Path(sys.argv[1])
out = Path(sys.argv[2])
models = ["gte-modernbert-base", "gte-reranker-modernbert-base", "qwen3-embedding-0.6b", "qwen3-reranker-0.6b"]
candidate = "be6cd118a40cc8cc5e999fad9e1e880e43dcd644"
baseline = "5f6d7419fc42a5ebf6ed1b8eab2115b6cc84fe5a"
summary = {"candidate": candidate, "baseline": baseline, "cells": [], "timings": [], "progress": []}
for revision, commit in [("progress", candidate), ("baseline", baseline)]:
    backends = ["cuda", "vulkan"] if revision == "progress" else ["cuda"]
    for backend in backends:
        for model in models:
            record = json.loads((out / "records" / commit / f"{backend}-linux-nvidia" / f"{model}.json").read_text())
            expected_pin = json.loads((out / "pins.json").read_text())["pins"][model][backend]
            assert record["fingerprint"] == expected_pin, (revision, backend, model)
            summary["cells"].append({"revision": revision, "backend": backend, "model": model,
                                     "status": record["status"], "fingerprint": record["fingerprint"],
                                     "expected_pin": expected_pin, "fingerprint_equals_pin": record["fingerprint"] == expected_pin,
                                     "admission": record["admission"], "fixtures": record["parity"]["completed_fixtures"]})
            name = f"{revision}.{backend}.{model}"
            events = [json.loads(line) for line in (raw / "remote-raw" / f"timing.{name}.jsonl").read_text().splitlines()]
            load = next(x["value"] for x in events if x["step"] == "load")
            assert load["error"] is None, (name, load)
            assert load["fingerprint"] == record["fingerprint"], name
            for tokens in [512, 8192]:
                samples = [x["value"] for x in events if x["step"] == "row" and x["value"]["tokens"] == tokens]
                assert len(samples) == 6 and all(x["ok"] and x["error"] is None for x in samples), (name, samples)
                assert [x["repeat"] for x in samples] == list(range(6)), name
                warm = [x["elapsed_ms"] for x in samples[1:]]
                summary["timings"].append({"revision": revision, "backend": backend, "model": model, "tokens": tokens,
                                           "successful_samples": len(samples), "samples_attempted": len(samples),
                                           "median_warm_ms": statistics.median(warm), "first_shape_ms": samples[0]["elapsed_ms"],
                                           "warm_samples_ms": warm, "cold_load_ms": load["cold_ms_including_load"]})

fields_re = re.compile(r"([a-z_]+)=([^\s]+)")
for kind in ["certification-logs", "runtime-logs"]:
    for path in sorted((raw / kind).rglob("synapse.*.log")):
        text = path.read_text()
        if "completed work received" not in text:
            continue
        lanes = set(re.findall(r"model_id=([^\s]+)", text))
        assert len(lanes) == 1, (str(path), lanes)
        lane = lanes.pop()
        backend = lane.rsplit("-", 1)[1]
        model = lane.rsplit("-", 1)[0]
        assert model in models and backend in ["cuda", "vulkan"], lane
        revision = "progress" if ("/synapse/" in str(path) or "timing-progress." in str(path)) else "baseline"
        pending = {}
        for line in text.splitlines():
            if not any(s in line for s in ["completed work received", "worker exchange drained"]):
                continue
            f = dict(fields_re.findall(line))
            req = f["req_id"]
            if "completed work received" in line:
                sequence = pending.setdefault(req, [])
                assert int(f["completed"]) == len(sequence) + 1, (str(path), req, f)
                sequence.append(f)
            else:
                sequence = pending.pop(req, [])
                assert int(f["progress_units"]) == len(sequence), (str(path), req, f)
                # The first completed-work event measures dispatch-to-progress,
                # not an interval between two completed-work events.
                gaps = [float(event["gap_ms"]) for event in sequence[1:]]
                summary["progress"].append({"source": "certify" if kind == "certification-logs" else "timing",
                    "revision": revision, "backend": backend, "model": model, "operation": f["operation"], "req_id": req,
                    "progress_units": len(sequence), "max_between_events_ms": max(gaps) if gaps else None,
                    "host_max_gap_ms": float(f["max_gap_ms"]), "elapsed_ms": float(f["elapsed_ms"]),
                    "log": str(path.relative_to(raw))})
        assert not pending, (str(path), pending)

for filename, key, fields in [
    ("timings.csv", "timings", ["revision", "backend", "model", "tokens", "successful_samples", "samples_attempted", "median_warm_ms", "first_shape_ms"]),
    ("progress.csv", "progress", ["source", "revision", "backend", "model", "operation", "req_id", "progress_units", "max_between_events_ms", "host_max_gap_ms", "elapsed_ms", "log"]),
]:
    with (out / filename).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary[key])
exchanges = summary["progress"]
summary["progress"] = []
for source in ["certify", "timing"]:
    for backend in ["cuda", "vulkan"]:
        for model in models:
            rows = [x for x in exchanges if x["source"] == source and x["revision"] == "progress" and x["backend"] == backend and x["model"] == model]
            assert rows, (source, backend, model)
            gaps = [x["max_between_events_ms"] for x in rows if x["max_between_events_ms"] is not None]
            summary["progress"].append({"source": source, "backend": backend, "model": model,
                "exchanges": len(rows), "completed_work_events": sum(x["progress_units"] for x in rows),
                "max_between_events_ms": max(gaps),
                "max_inference_exchange_ms": max(x["host_max_gap_ms"] for x in rows if x["operation"] != "LOAD"),
                "max_load_exchange_ms": max(x["host_max_gap_ms"] for x in rows if x["operation"] == "LOAD")})
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"Reduced {len(summary['cells'])} certification cells, {len(summary['timings'])} timing shapes and {len(exchanges)} worker exchanges")
