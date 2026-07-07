from __future__ import annotations

import argparse
import json
from pathlib import Path

from coir_eval import prepare_cosqa, prepare_csn_python

SUPPORTED_TASKS = ("cosqa", "csn-python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CoIR retrieval data in Synapse lane JSONL format.")
    parser.add_argument("--task", choices=SUPPORTED_TASKS, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Optionally keep only the first N sorted query ids and their matching qrels.",
    )
    args = parser.parse_args()
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.task == "cosqa":
        summary = prepare_cosqa(args.out_dir, max_queries=args.max_queries)
    elif args.task == "csn-python":
        summary = prepare_csn_python(args.out_dir, max_queries=args.max_queries)
    else:
        raise ValueError(f"unsupported task: {args.task}")
    print(json.dumps({"task": args.task, **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
