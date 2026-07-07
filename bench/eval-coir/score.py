from __future__ import annotations

import argparse
import json
from pathlib import Path

from coir_eval import DEFAULT_K, score_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CoIR retrieval vectors against qrels.")
    parser.add_argument("--corpus-vectors", type=Path, required=True)
    parser.add_argument("--query-vectors", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = score_task(
        corpus_vectors_path=args.corpus_vectors,
        query_vectors_path=args.query_vectors,
        qrels_path=args.qrels,
        k=args.k,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
