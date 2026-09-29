#!/usr/bin/env python3
"""Measure a multi-function ModernBERT Core ML package against single-bucket packages.

Run with the spike's Python environment (the one `build.py` used) after
`build.py` has written the packages and `build_probe.sh` has built the Swift
probe. Stages can be run one at a time with `--stages`; each writes a JSON report
under `<root>/reports/`.

Before every probe process that touches the Neural Engine the driver waits until
no campaign holds `campaign_rig_claim` in the prefrontal store (opened read-only)
and the 1-minute load average is at or below 16. The latency probe repeats that
check before every timed run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ROWS_JSONL = REPO_ROOT / "bench" / "spikes" / "ane-direct-probe" / "rows.jsonl"
DEFAULT_ROOT = Path.home() / ".local" / "share" / "cortexkit" / "synapse" / "ane-multifunction"
INSTALLED_ROOT = Path.home() / ".local" / "share" / "cortexkit" / "models" / "ane-coreml"
RIG_DB = Path.home() / ".local" / "share" / "cortexkit" / "prefrontal-core" / "store.db"
MAX_LOAD = 16.0
BUCKETS = (128, 256, 512)
LONG_BUCKET = 1024
MF3 = "mf-128-256-512"
MF4 = "mf-128-256-512-1024"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def rig_claims() -> list[tuple[Any, ...]]:
    connection = sqlite3.connect(f"file:{RIG_DB}?mode=ro", uri=True)
    try:
        return connection.execute(
            "SELECT rig_key, holder_campaign_id, heartbeat_at FROM campaign_rig_claim"
        ).fetchall()
    finally:
        connection.close()


def wait_for_quiet_rig() -> dict[str, Any]:
    """Block until no campaign holds the rig and the 1-minute load is at or below 16."""
    started = time.monotonic()
    announced = False
    while True:
        claims = rig_claims()
        load = os.getloadavg()[0]
        if not claims and load <= MAX_LOAD:
            return {"waited_s": time.monotonic() - started, "load1": load, "rig_claims": 0}
        if not announced:
            log(f"waiting: rig_claims={claims} load1={load:.2f}")
            announced = True
        time.sleep(10)


def probe(root: Path, *args: str) -> dict[str, Any]:
    gate = wait_for_quiet_rig()
    completed = subprocess.run(
        [str(root / "bin" / "probe"), *args], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"probe {args} failed: {completed.stderr.strip()}")
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    result["gate"] = gate
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_bytes(path: Path) -> int:
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def bundle_files(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(entry.relative_to(path)): {"bytes": entry.stat().st_size, "sha256": sha256_file(entry)}
        for entry in sorted(path.rglob("*"))
        if entry.is_file()
    }


def write_report(root: Path, name: str, payload: Any) -> None:
    path = root / "reports" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {path.name}")


def read_report(root: Path, name: str) -> Any:
    return json.loads((root / "reports" / f"{name}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Rows


def near_length(row_id: str, bucket: int) -> int:
    """Token count of a near-limit row at `bucket`.

    rows.jsonl declares the near-limit rows only at 512/1024/2048, with token
    counts at fixed fractions of the shape (7/8, 15/16, 500/512, and shape - 1).
    The same fractions give the 128 and 256 variants.
    """
    fractions = {
        "near-climate": bucket * 7 // 8,
        "near-database": bucket * 15 // 16,
        "near-software": round(bucket * 500 / 512),
        "near-science": bucket - 1,
    }
    return fractions[row_id]


def prepare_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Per-bucket token rows derived from rows.jsonl.

    Short rows are used as declared. A near-limit row at a given length is the
    2048-shape row's content prefix wrapped with its CLS and SEP ids, which is the
    construction rows.jsonl documents; the declared 512 and 1024 rows are checked
    against that construction before it is used for 128 and 256.
    """
    source = [json.loads(line) for line in ROWS_JSONL.read_text(encoding="utf-8").splitlines() if line]
    rows: dict[str, list[dict[str, Any]]] = {}
    for bucket in (*BUCKETS, LONG_BUCKET):
        bucket_rows = []
        for row in source:
            shapes = row["input_ids_by_shape"]
            if row["id"].startswith("short-"):
                ids = shapes["512"]
            else:
                full = shapes["2048"]
                length = near_length(row["id"], bucket)
                ids = [full[0], *full[1 : length - 1], full[-1]]
                declared = shapes.get(str(bucket))
                if declared is not None and declared != ids:
                    raise RuntimeError(f"{row['id']} construction differs from declared {bucket} row")
            if len(ids) > bucket:
                raise RuntimeError(f"{row['id']} does not fit {bucket}")
            bucket_rows.append({"id": row["id"], "ids": ids})
        rows[str(bucket)] = bucket_rows
    (root / "rows.json").write_text(json.dumps(rows) + "\n", encoding="utf-8")
    return rows


# ---------------------------------------------------------------------------
# Artifacts


def installed_production() -> dict[int, dict[str, Any]]:
    """Locate the installed production bucket bundles by their materialization records."""
    found: dict[int, dict[str, Any]] = {}
    for record_path in sorted(INSTALLED_ROOT.glob("*/materialization.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        relative = record["model_relative_path"]
        if "gte-modernbert-base-seq" not in relative:
            continue
        bucket = int(relative.rsplit("seq", 1)[1].split(".", 1)[0])
        found[bucket] = {
            "bundle": record_path.parent / relative,
            "source_digest": record["source_digest"],
            "materialized_digest": record["materialized_digest"],
        }
    return found


def compiled_path(root: Path, name: str) -> Path:
    return root / "compiled" / f"{name}.mlmodelc"


def stage_compile(root: Path) -> None:
    """Compile every package to a stable path and copy the installed production bundles."""
    build = json.loads((root / "reports" / "build.json").read_text(encoding="utf-8"))
    report: dict[str, Any] = {"compile": {}, "compiled": {}, "production": {}}
    packages = {f"single-seq{bucket}": Path(entry["path"]) for bucket, entry in build["singles"].items()}
    packages.update({name: Path(entry["path"]) for name, entry in build["multis"].items()})
    for name, package in packages.items():
        destination = compiled_path(root, name)
        result = probe(root, "compile", str(package), str(destination))
        report["compile"][name] = result["compile_ms"]
        report["compiled"][name] = {"bytes": tree_bytes(destination), "files": bundle_files(destination)}
        log(f"compiled {name} in {result['compile_ms']:.0f} ms")
    for bucket, entry in installed_production().items():
        destination = compiled_path(root, f"prod-seq{bucket}")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(entry["bundle"], destination)
        rebuilt = report["compiled"].get(f"single-seq{bucket}", {}).get("files", {})
        production_files = bundle_files(destination)
        report["production"][str(bucket)] = {
            "source_digest": entry["source_digest"],
            "materialized_digest": entry["materialized_digest"],
            "bytes": tree_bytes(destination),
            "files": production_files,
            "rebuilt_matches": {
                name: rebuilt.get(name, {}).get("sha256") == info["sha256"]
                for name, info in production_files.items()
            },
        }
    write_report(root, "compile", report)


def functions_under_test(include_long: bool) -> list[tuple[str, str, str, int]]:
    """(label, bundle name, function, bucket) for every model the stages measure."""
    targets: list[tuple[str, str, str, int]] = []
    for bucket in BUCKETS:
        targets.append((f"prod-seq{bucket}", f"prod-seq{bucket}", "-", bucket))
        targets.append((f"single-seq{bucket}", f"single-seq{bucket}", "-", bucket))
        targets.append((f"{MF3}:seq{bucket}", MF3, f"seq{bucket}", bucket))
    if include_long:
        targets.append((f"single-seq{LONG_BUCKET}", f"single-seq{LONG_BUCKET}", "-", LONG_BUCKET))
        for bucket in (*BUCKETS, LONG_BUCKET):
            targets.append((f"{MF4}:seq{bucket}", MF4, f"seq{bucket}", bucket))
    return targets


def stage_placement(root: Path) -> None:
    report = {}
    for label, bundle, function, _bucket in functions_under_test(include_long=True):
        result = probe(root, "placement", str(compiled_path(root, bundle)), function)
        report[label] = result
        log(f"placement {label}: share={result['placement_share']:.6f} counts={result['preferred_device_counts']}")
    write_report(root, "placement", report)


def stage_embed(root: Path) -> None:
    vectors_dir = root / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for label, bundle, function, bucket in functions_under_test(include_long=True):
        out = vectors_dir / f"{label.replace(':', '__')}.json"
        report[label] = probe(
            root, "embed", str(compiled_path(root, bundle)), function, str(root / "rows.json"), str(bucket), str(out)
        )
        log(f"embedded {label}")
    write_report(root, "embed", report)


def stage_reference(root: Path) -> None:
    """fp32 eager PyTorch vectors for the same padded rows, as an external yardstick."""
    import importlib.util

    import torch

    converter_path = REPO_ROOT / "bench" / "spikes" / "ane-minilm" / "convert_modernbert_to_coreml.py"
    spec = importlib.util.spec_from_file_location("convert_modernbert_to_coreml", converter_path)
    assert spec is not None and spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    model_ref, _ = converter.resolve_model_ref(converter.EMBEDDER_MODEL_ID)
    wrapper = converter.load_wrapper("embedder", model_ref, False)
    rows = json.loads((root / "rows.json").read_text(encoding="utf-8"))
    out: dict[str, list[dict[str, Any]]] = {}
    with torch.inference_mode():
        for bucket, bucket_rows in rows.items():
            size = int(bucket)
            out[bucket] = []
            for row in bucket_rows:
                ids = torch.zeros((1, size), dtype=torch.long)
                mask = torch.zeros((1, size), dtype=torch.long)
                ids[0, : len(row["ids"])] = torch.tensor(row["ids"])
                mask[0, : len(row["ids"])] = 1
                vector = wrapper(ids, mask)[0].float().tolist()
                out[bucket].append({"id": row["id"], "vector": vector})
            log(f"reference {bucket}")
    (root / "vectors" / "reference-fp32.json").write_text(json.dumps(out) + "\n", encoding="utf-8")


def compare_vectors(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    rows = []
    for a, b in zip(left, right, strict=True):
        assert a["id"] == b["id"]
        x = np.asarray(a["vector"], dtype=np.float64)
        y = np.asarray(b["vector"], dtype=np.float64)
        rows.append(
            {
                "id": a["id"],
                "identical": bool(np.array_equal(x, y)),
                "max_abs": float(np.max(np.abs(x - y))),
                "cosine": float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30)),
                "finite": bool(np.all(np.isfinite(x)) and np.all(np.isfinite(y))),
            }
        )
    return {
        "rows": len(rows),
        "all_identical": all(row["identical"] for row in rows),
        "max_abs": max(row["max_abs"] for row in rows),
        "min_cosine": min(row["cosine"] for row in rows),
        "per_row": rows,
    }


def stage_parity(root: Path) -> None:
    def load(label: str) -> list[dict[str, Any]]:
        path = root / "vectors" / f"{label.replace(':', '__')}.json"
        return json.loads(path.read_text(encoding="utf-8"))["rows"]

    reference_path = root / "vectors" / "reference-fp32.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else {}
    report: dict[str, Any] = {}
    for bucket in BUCKETS:
        production = load(f"prod-seq{bucket}")
        report[f"seq{bucket}"] = {
            "single_rebuilt_vs_production": compare_vectors(load(f"single-seq{bucket}"), production),
            "mf3_vs_production": compare_vectors(load(f"{MF3}:seq{bucket}"), production),
            "mf4_vs_production": compare_vectors(load(f"{MF4}:seq{bucket}"), production),
            "production_vs_fp32_reference": compare_vectors(production, reference[str(bucket)])
            if reference
            else None,
        }
    long_single = load(f"single-seq{LONG_BUCKET}")
    report[f"seq{LONG_BUCKET}"] = {
        "mf4_vs_single": compare_vectors(load(f"{MF4}:seq{LONG_BUCKET}"), long_single),
        "single_vs_fp32_reference": compare_vectors(long_single, reference[str(LONG_BUCKET)])
        if reference
        else None,
        "mf4_vs_fp32_reference": compare_vectors(load(f"{MF4}:seq{LONG_BUCKET}"), reference[str(LONG_BUCKET)])
        if reference
        else None,
    }
    for bucket, entry in report.items():
        for name, result in entry.items():
            if result:
                log(
                    f"{bucket} {name}: identical={result['all_identical']} "
                    f"max_abs={result['max_abs']:.3g} min_cos={result['min_cosine']:.6f}"
                )
    write_report(root, "parity", report)


def fresh_copy(root: Path, bundle: str, tag: str) -> Path:
    """Copy a compiled bundle to a path Core ML has never loaded, for cold loads."""
    destination = root / "cold" / f"{bundle}-{tag}.mlmodelc"
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(compiled_path(root, bundle), destination)
    return destination


def stage_load(root: Path, trials: int, warm_repeats: int) -> None:
    """Cold first load at a never-loaded path, then warm loads from fresh processes on it.

    For the multi-function bundle the functions are loaded one after another, each
    in its own process, so each function's first load is recorded separately and
    its warm loads follow on the same path. The mlpackage-to-mlmodelc compile time
    is reported by the compile stage.
    """
    rows = str(root / "rows.json")
    samples: list[dict[str, Any]] = []
    run_id = time.strftime("%Y%m%d%H%M%S")
    for trial in range(trials):
        variants: list[tuple[str, list[tuple[str, int]]]] = [
            (f"prod-seq{bucket}", [("-", bucket)]) for bucket in BUCKETS
        ]
        variants.append((MF3, [(f"seq{bucket}", bucket) for bucket in BUCKETS]))
        if trial % 2 == 1:
            variants.reverse()
        for bundle, functions in variants:
            path = fresh_copy(root, bundle, f"{run_id}-t{trial}")
            for function, bucket in functions:
                cold = probe(root, "load", str(path), function, rows, str(bucket))
                warm = [probe(root, "load", str(path), function, rows, str(bucket)) for _ in range(warm_repeats)]
                samples.append(
                    {"trial": trial, "bundle": bundle, "function": function, "bucket": bucket, "cold": cold, "warm": warm}
                )
                log(
                    f"load t{trial} {bundle}:{function} cold={cold['load_ms']:.0f}ms "
                    f"warm_median={statistics.median(w['load_ms'] for w in warm):.0f}ms"
                )
            shutil.rmtree(path)
    write_report(root, "load", {"samples": samples})


def stage_latency(root: Path, iterations: int) -> None:
    groups = []
    for bucket in BUCKETS:
        groups.append(
            [
                {"label": f"prod-seq{bucket}", "model": str(compiled_path(root, f"prod-seq{bucket}")), "function": "-", "bucket": bucket},
                {"label": f"{MF3}:seq{bucket}", "model": str(compiled_path(root, MF3)), "function": f"seq{bucket}", "bucket": bucket},
            ]
        )
    groups.append(
        [
            {"label": f"single-seq{LONG_BUCKET}", "model": str(compiled_path(root, f"single-seq{LONG_BUCKET}")), "function": "-", "bucket": LONG_BUCKET},
            {"label": f"{MF4}:seq{LONG_BUCKET}", "model": str(compiled_path(root, MF4)), "function": f"seq{LONG_BUCKET}", "bucket": LONG_BUCKET},
        ]
    )
    plan = {"rows": str(root / "rows.json"), "groups": groups, "warmup": 3, "iterations": iterations, "max_load": MAX_LOAD}
    plan_path = root / "reports" / "latency-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    raw_path = root / "reports" / "latency-raw.json"
    probe(root, "latency", str(plan_path), str(raw_path))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {"load_ms": raw["load_ms"], "labels": {}}
    labels = sorted({sample["label"] for sample in raw["samples"]})
    for label in labels:
        runs = [sample for sample in raw["samples"] if sample["label"] == label]
        times = [sample["run_ms"] for sample in runs]
        # The probe only starts a run at load <= 16; a run during which load rose
        # above 16 is kept in the raw data but left out of the quiet median.
        quiet = [s["run_ms"] for s in runs if s["load_before"][0] <= MAX_LOAD and s["load_after"][0] <= MAX_LOAD]
        summary["labels"][label] = {
            "runs": len(times),
            "rows_per_run": runs[0]["rows"],
            "median_ms": statistics.median(times),
            "quiet_runs": len(quiet),
            "quiet_median_ms": statistics.median(quiet) if quiet else None,
            "p10_ms": sorted(times)[len(times) // 10],
            "p90_ms": sorted(times)[(len(times) * 9) // 10],
            "min_ms": min(times),
            "max_ms": max(times),
            "max_load1_seen": max(max(s["load_before"][0], s["load_after"][0]) for s in runs),
            "waited_s": sum(s["waited_s"] for s in runs),
        }
        entry = summary["labels"][label]
        log(
            f"latency {label}: median={entry['median_ms']:.2f}ms over {len(times)} runs, "
            f"quiet median={entry['quiet_median_ms']} over {len(quiet)}, max load1 {entry['max_load1_seen']:.2f}"
        )
    write_report(root, "latency", summary)


def wired_bytes() -> int:
    """System-wide wired memory from vm_stat.

    Neural Engine model buffers are not charged to the loading process, so the
    process footprint alone cannot show whether shared weights are loaded once.
    Wired memory is a noisy system-wide proxy that at least moves with them.
    """
    output = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page_size = int(output.split("page size of ", 1)[1].split(" ", 1)[0])
    for line in output.splitlines():
        if line.startswith("Pages wired down:"):
            return int(line.split(":", 1)[1].strip().rstrip(".")) * page_size
    raise RuntimeError("vm_stat has no wired line")


def held_memory_probe(root: Path, rows: str, targets: list[str]) -> dict[str, Any]:
    """Run the memory probe, sample wired memory while its models stay loaded, then release it."""
    gate = wait_for_quiet_rig()
    wired_before = wired_bytes()
    process = subprocess.Popen(
        [str(root / "bin" / "probe"), "memory", rows, *targets],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PROBE_HOLD": "1"},
    )
    assert process.stdout is not None and process.stdin is not None
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"memory probe failed: {process.communicate()[1]}")
    result = json.loads(line)
    time.sleep(1)
    wired_loaded = wired_bytes()
    process.stdin.close()
    process.wait(timeout=60)
    time.sleep(1)
    wired_after_exit = wired_bytes()
    result.update(
        gate=gate,
        wired_before_bytes=wired_before,
        wired_loaded_bytes=wired_loaded,
        wired_after_exit_bytes=wired_after_exit,
    )
    return result


def stage_memory(root: Path, repeats: int) -> None:
    rows = str(root / "rows.json")
    production = [f"{compiled_path(root, f'prod-seq{bucket}')}:-:{bucket}" for bucket in BUCKETS]
    multi = [f"{compiled_path(root, MF3)}:seq{bucket}:{bucket}" for bucket in BUCKETS]
    samples = []
    for repeat in range(repeats):
        order = [("production-3-packages", production), ("multi-function-3-functions", multi)]
        if repeat % 2 == 1:
            order.reverse()
        for name, targets in order:
            result = held_memory_probe(root, rows, targets)
            samples.append({"repeat": repeat, "variant": name, **result})
            log(
                f"memory {name}: footprint={result['after_predict_bytes'] / 2**20:.1f} MiB "
                f"resident={result['after_predict_resident_bytes'] / 2**20:.1f} MiB "
                f"wired_delta_loaded={(result['wired_loaded_bytes'] - result['wired_before_bytes']) / 2**20:.1f} MiB "
                f"wired_delta_exit={(result['wired_after_exit_bytes'] - result['wired_before_bytes']) / 2**20:.1f} MiB "
                f"load_ms={[round(value, 1) for value in result['load_ms']]}"
            )
    write_report(root, "memory", {"samples": samples})


def stage_evidence(root: Path, out: Path) -> None:
    """Condense the stage reports into one committed JSON without machine-specific paths."""
    build = read_report(root, "build")
    compiled = read_report(root, "compile")
    placement = read_report(root, "placement")
    parity = read_report(root, "parity")
    load = read_report(root, "load")["samples"]
    latency = read_report(root, "latency")
    memory = read_report(root, "memory")["samples"]

    def weights(files: dict[str, Any]) -> dict[str, Any]:
        return {name: info for name, info in files.items() if name.endswith("weight.bin")}

    loads: dict[str, Any] = {}
    for sample in load:
        key = f"{sample['bundle']}:{sample['function']}"
        entry = loads.setdefault(key, {"cold_load_ms": [], "cold_first_predict_ms": [], "warm_load_ms": []})
        entry["cold_load_ms"].append(sample["cold"]["load_ms"])
        entry["cold_first_predict_ms"].append(sample["cold"]["first_predict_ms"])
        entry["warm_load_ms"].extend(warm["load_ms"] for warm in sample["warm"])
    for entry in loads.values():
        entry["warm_load_median_ms"] = statistics.median(entry["warm_load_ms"])
    load_averages = [
        value
        for sample in load
        for run in (sample["cold"], *sample["warm"])
        for value in (run["load_before"][0], run["load_after"][0])
    ]

    memory_summary: dict[str, Any] = {}
    for variant in sorted({sample["variant"] for sample in memory}):
        runs = [sample for sample in memory if sample["variant"] == variant]
        memory_summary[variant] = {
            "samples": len(runs),
            "footprint_after_load_median_bytes": statistics.median(s["after_load_bytes"] for s in runs),
            "footprint_after_predict_median_bytes": statistics.median(s["after_predict_bytes"] for s in runs),
            "resident_after_predict_median_bytes": statistics.median(
                s["after_predict_resident_bytes"] for s in runs
            ),
            "load_all_three_warm_median_ms": statistics.median(sum(s["load_ms"]) for s in runs),
            "wired_delta_while_loaded_bytes": [s["wired_loaded_bytes"] - s["wired_before_bytes"] for s in runs],
        }

    evidence = {
        "environment": build["environment"],
        "source_model": build["source_model"],
        "source_snapshot": build["source_snapshot"],
        "packages": {
            name: {"package_bytes": entry["package_bytes"], "weights": weights(entry["files"])}
            for name, entry in {**{f"single-seq{k}": v for k, v in build["singles"].items()}, **build["multis"]}.items()
        },
        "multi_function_spec": {
            name: {
                "specification_version": entry["specification_version"],
                "functions": entry["function_descriptions"],
            }
            for name, entry in build["multis"].items()
        },
        "single_parity_gate": {
            bucket: entry.get("production_parity_gate") for bucket, entry in build["singles"].items()
        },
        "compile_ms": compiled["compile"],
        "compiled": {
            name: {"bytes": entry["bytes"], "weights": weights(entry["files"])}
            for name, entry in compiled["compiled"].items()
        },
        "production": {
            bucket: {
                key: entry[key] for key in ("source_digest", "materialized_digest", "bytes", "rebuilt_matches")
            }
            for bucket, entry in compiled["production"].items()
        },
        "placement": {
            label: {
                key: entry[key]
                for key in ("placement_share", "preferred_device_counts", "non_neural_engine_operators", "total_operations")
            }
            for label, entry in placement.items()
        },
        "parity": parity,
        "load": {"by_model": loads, "load1_min": min(load_averages), "load1_max": max(load_averages)},
        "latency": latency,
        "memory": memory_summary,
    }
    text = json.dumps(evidence, indent=2) + "\n"
    text = text.replace(str(root), "<artifact-root>").replace(str(Path.home()), "~")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    log(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["rows", "compile", "placement", "embed", "reference", "parity", "load", "latency", "memory"],
    )
    parser.add_argument("--load-trials", type=int, default=3)
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--latency-iterations", type=int, default=25)
    parser.add_argument("--memory-repeats", type=int, default=6)
    parser.add_argument(
        "--evidence-out",
        type=Path,
        default=REPO_ROOT / "docs" / "evidence" / "ane-coreml-multifunction" / "EVIDENCE.json",
    )
    args = parser.parse_args()
    root: Path = args.root
    for stage in args.stages:
        log(f"stage {stage}")
        if stage == "rows":
            rows = prepare_rows(root)
            log("rows " + ", ".join(f"{bucket}: {[len(row['ids']) for row in r]}" for bucket, r in rows.items()))
        elif stage == "compile":
            stage_compile(root)
        elif stage == "placement":
            stage_placement(root)
        elif stage == "embed":
            stage_embed(root)
        elif stage == "reference":
            stage_reference(root)
        elif stage == "parity":
            stage_parity(root)
        elif stage == "load":
            stage_load(root, args.load_trials, args.warm_repeats)
        elif stage == "latency":
            stage_latency(root, args.latency_iterations)
        elif stage == "memory":
            stage_memory(root, args.memory_repeats)
        elif stage == "evidence":
            stage_evidence(root, args.evidence_out)
        else:
            raise SystemExit(f"unknown stage {stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
