#!/usr/bin/env python3
"""Reduce archived liveness events to reviewable per-request calibration tables.

Tokens are final model-input counts, including templates/special tokens. For
ROG Ally X Windows Vulkan runs, request order comes from checksum-pinned
fixtures and the release runner's 64-item/8192-token grouping. Apple M4 Mac mini
ANE token/shape values come from its worker row profiles. Event count includes
completed-work events and the terminal drained event.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

RAW_FILES = {
    "ally/gte-embedding-liveness.txt": "gte-modernbert-base",
    "ally/gte-reranker-liveness.txt": "gte-reranker-modernbert-base",
    "ally/qwen-embedding-liveness.txt": "qwen3-embedding-0.6b",
    "ally/qwen-reranker-liveness.txt": "qwen3-reranker-0.6b",
    "m4/liveness-events.txt": "qwen3-embedding-0.6b",
}
COLUMNS = ["model", "request", "operation", "purpose", "tokens", "rows", "max_tokens", "shape",
           "total_ms", "maximum_gap_ms", "event_count", "progress_events",
           "first_fixture", "last_fixture"]


def events(path):
    counts = Counter()
    progress = Counter()
    terminal = []
    for line in path.read_text().splitlines():
        if "completed work received" not in line and "worker exchange drained" not in line:
            continue
        fields = dict(re.findall(r"(\w+)=([^ ]+)", line))
        request = fields["req_id"]
        counts[request] += 1
        if "completed work received" in line:
            progress[request] += 1
        else:
            terminal.append({"request": request, "operation": fields["operation"],
                             "total_ms": fields["elapsed_ms"],
                             "maximum_gap_ms": fields["max_gap_ms"],
                             "progress_events": int(fields["progress_units"])})
    assert len({row["request"] for row in terminal}) == len(terminal)
    for row in terminal:
        request = row["request"]
        assert progress[request] == row["progress_events"], request
        assert counts[request] == progress[request] + 1, request
        row["event_count"] = counts[request]
    return terminal


def fixture_requests(cases, operation):
    groups, grouped = [], {}
    for case in cases:
        count = len(case["input_ids"])
        key = None if count == 8192 else ("batched" if case["category"] == "batched"
                                         else case.get("pool") if operation == "rerank" else None)
        if key is None:
            groups.append([case])
        else:
            if key not in grouped:
                grouped[key] = len(groups)
                groups.append([])
            groups[grouped[key]].append(case)
    requests = []
    for group in groups:
        start = 0
        while start < len(group):
            end, used = start, 0
            while end < len(group) and end - start < 64:
                count = len(group[end]["input_ids"])
                assert 0 < count <= 8192
                if count > 8192 - used or (end > start and
                    (count == 8192 or len(group[start]["input_ids"]) == 8192)):
                    break
                used += count
                end += 1
            assert end > start
            requests.append(group[start:end])
            start = end
    return requests


def write_table(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="compare tables without modifying them")
    args = parser.parse_args()
    evidence = Path(__file__).resolve().parent
    checkout = evidence.parents[2]
    pins = json.loads((checkout / "bench/parity/fixtures/index.json").read_text())
    self_checks = json.loads((checkout / "crates/synapse-certify/src/self_check/index.json").read_text())
    hashes = {}
    for line in (evidence / "raw-files.sha256").read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        hashes[relative] = digest
    ally = []
    for relative, model in RAW_FILES.items():
        raw = args.raw_root / relative
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == hashes[relative], relative
        if relative.startswith("m4/"):
            continue
        pin = next(value for value in pins.values() if value["model"] == model)
        fixture_path = checkout / "bench/parity" / pin["path"]
        assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == pin["sha256"], model
        fixture = json.loads(fixture_path.read_text())
        requests = fixture_requests(fixture["cases"], fixture["operation"])
        check_pin = self_checks[model]
        assert check_pin["source_sha256"] == pin["sha256"], model
        cases = {case["id"]: case for case in fixture["cases"]}
        checks = [[cases[case_id]] for case_id in check_pin["case_ids"]]
        # The module compares selected outputs with pinned reference results
        # one input at a time before the runner's fixture groups. The observed
        # inference count must equal self-check inputs plus grouped fixture calls.
        sequence = [("self_check", request) for request in checks]
        sequence += [("fixture", request) for request in requests]
        rows = events(raw)
        inference = [row for row in rows if row["operation"] in ("EMBED_BATCH", "RERANK_SEQUENCES")]
        assert len(inference) == len(sequence), model
        for row, (purpose, request) in zip(inference, sequence):
            counts = [len(case["input_ids"]) for case in request]
            row.update(purpose=purpose, tokens=sum(counts), rows=len(counts), max_tokens=max(counts),
                       first_fixture=request[0]["id"], last_fixture=request[-1]["id"])
        for row in rows:
            row["model"] = model
        ally.extend(rows)
    metadata, pending = {}, None
    for line in (evidence / "m4/worker-rows.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["kind"] in ("row", "admit", "evict"):
            pending = row
        elif row["kind"] == "request" and pending is not None:
            metadata[row["req_id"]] = {key: pending[key] for key in ("tokens", "shape") if key in pending}
            pending = None
    m4 = events(args.raw_root / "m4/liveness-events.txt")
    for row in m4:
        row.update(metadata.get(row["request"], {}))
        row["model"] = "qwen3-embedding-0.6b"
        if row["operation"] == "EMBED_BATCH":
            row.update(rows=1, max_tokens=row["tokens"])
    for relative, rows in (("ally/per-request.csv", ally), ("m4/per-request.csv", m4)):
        path = evidence / relative
        if args.check:
            with path.open(newline="") as source:
                recorded = list(csv.DictReader(source))
            expected = [{column: str(row.get(column, "")) for column in COLUMNS} for row in rows]
            assert recorded == expected, relative
        else:
            write_table(path, rows)
    print(f"PASS: {len(hashes)} archived hashes, 4 fixture pins, {len(ally)} Ally and {len(m4)} M4 requests")


if __name__ == "__main__":
    main()
