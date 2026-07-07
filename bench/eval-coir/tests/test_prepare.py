from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coir_eval import TextRow, limit_queries  # noqa: E402



def test_limit_queries_keeps_first_sorted_query_ids_and_matching_qrels() -> None:
    query_rows = [
        TextRow(id="q1", text="query one"),
        TextRow(id="q2", text="query two"),
        TextRow(id="q3", text="query three"),
    ]
    qrels_rows = [
        ("q1", "d1", 1),
        ("q2", "d2", 1),
        ("q2", "d3", 1),
        ("q3", "d4", 1),
    ]

    limited_queries, limited_qrels = limit_queries(query_rows, qrels_rows, max_queries=2)

    assert limited_queries == query_rows[:2]
    assert limited_qrels == [("q1", "d1", 1), ("q2", "d2", 1), ("q2", "d3", 1)]
