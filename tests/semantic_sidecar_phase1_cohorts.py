#!/usr/bin/env python3
"""Freeze the deterministic phase-1 classify and adversarial cohorts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "semantic-sidecar-v1"
SOURCE = ROOT / "tools" / "classify-distill" / "data" / "classify-gold-v1.jsonl"
SELECTION_DOMAIN = b"semantic-sidecar-classify-cohort-v1\n"


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def classify_grammar() -> str:
    scalar_enum = lambda values: {"type": "string", "enum": values}
    spec = {
        "type": "object",
        "properties": {
            "draft_path": scalar_enum([
                ".cortexkit/alfonso/drafts/example.md",
                ".cortexkit/alfonso/drafts/plan.md",
                ".cortexkit/alfonso/drafts/spec.md",
            ]),
            "complexity_estimate": scalar_enum(["small", "medium", "large"]),
            "rigor_assessed": scalar_enum(["r1", "r2", "r3"]),
        },
        "required": ["draft_path", "complexity_estimate", "rigor_assessed"],
        "additionalProperties": False,
    }
    panel = {
        "type": "object",
        "properties": {
            "size": {"type": "integer", "enum": list(range(1, 9))},
            "iq_floor": {"type": "number", "enum": list(range(11))},
            "diversity": scalar_enum(["family", "none"]),
            "class": scalar_enum(["AUDIT", "DIAGNOSE", "EVALUATE", "EXPLAIN", "PLAN", "SPEC"]),
        },
        "required": ["size", "iq_floor", "diversity", "class"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "schema": {"type": "integer", "enum": [1]},
            "route": scalar_enum(["verdict", "campaign", "spec"]),
            "class": scalar_enum(["AUDIT", "DIAGNOSE", "EVALUATE", "EXPLAIN", "PLAN", "SPEC"]),
            "spec": spec,
            "evidence_mode": scalar_enum(["shared", "hybrid", "independent"]),
            "execution_mode": scalar_enum(["vocabulary-restricted", "full-session"]),
            "panel_spec": panel,
            "gather": {"type": "boolean"},
        },
        "required": [
            "schema",
            "route",
            "class",
            "evidence_mode",
            "execution_mode",
            "panel_spec",
            "gather",
        ],
        "additionalProperties": False,
    }
    return compact(schema)


def adversarial_schemas() -> list[str]:
    schemas: list[str] = []
    for index in range(30):
        if index % 5 == 0:
            schema = {"type": "object", "properties": {f"field_{index}": {"type": "string", "enum": [f"value_{index}"]}}, "required": [f"field_{index}"], "additionalProperties": False}
        elif index % 5 == 1:
            schema = {"type": "object", "properties": {f"outer_{index}": {"type": "object", "properties": {f"inner_{index}": {"type": "null"}}, "required": [f"inner_{index}"], "additionalProperties": False}}, "required": [f"outer_{index}"], "additionalProperties": False}
        elif index % 5 == 2:
            schema = {"type": "object", "properties": {f"flag_{index}": {"type": "boolean"}, f"code_{index}": {"type": "integer", "enum": [index]}}, "required": [f"flag_{index}", f"code_{index}"], "additionalProperties": False}
        elif index % 5 == 3:
            schema = {"type": "object", "properties": {f"items_{index}": {"type": "array", "items": {"type": "object", "properties": {f"kind_{index}": {"type": "string", "enum": [f"fixed_{index}"]}}, "required": [f"kind_{index}"], "additionalProperties": False}}}, "required": [f"items_{index}"], "additionalProperties": False}
        else:
            schema = {"type": "string", "enum": [f"allow_{index}", f"deny_{index}"]}
        schemas.append(compact(schema))
    return schemas


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    data = "".join(compact(row) + "\n" for row in rows).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    source_rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    if len(source_rows) != 999:
        raise SystemExit(f"expected 999 classify rows, found {len(source_rows)}")
    if any(row.get("valid") is not True or row.get("source") != "synthetic" for row in source_rows):
        raise SystemExit("classify source contains an ineligible row")

    ranked = sorted(
        source_rows,
        key=lambda row: hashlib.sha256(SELECTION_DOMAIN + row["id"].encode()).digest(),
    )[:100]
    grammar = classify_grammar()
    classify_rows = [
        {"request_id": row["id"], "prompt": row["request_prose"], "grammar": grammar, "max_tokens": 384}
        for row in ranked
    ]
    classify_digest = write_jsonl(EVIDENCE / "cohort-classify-v1.jsonl", classify_rows)

    adversarial_rows = []
    for index, schema in enumerate(adversarial_schemas()):
        digest = hashlib.sha256(schema.encode()).hexdigest()
        adversarial_rows.append(
            {
                "request_id": f"adversarial-{index:02d}",
                "adversarial_schema_id": f"jsonschema-source-sha256:{digest}",
                "prompt": "Return exactly one compact JSON value accepted by the supplied schema.",
                "grammar": schema,
                "max_tokens": 64,
            }
        )
    adversarial_digest = write_jsonl(EVIDENCE / "cohort-adversarial-v1.jsonl", adversarial_rows)
    print(compact({
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "classify_rows": len(classify_rows),
        "classify_sha256": classify_digest,
        "adversarial_rows": len(adversarial_rows),
        "adversarial_sha256": adversarial_digest,
    }))


if __name__ == "__main__":
    main()
