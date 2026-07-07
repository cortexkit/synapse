from __future__ import annotations

import argparse
import json
from pathlib import Path

from coir_eval import prepare_cosqa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CoIR retrieval data in Synapse lane JSONL format.")
    parser.add_argument("--task", choices=["cosqa"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task != "cosqa":
        raise ValueError(f"unsupported task: {args.task}")
    summary = prepare_cosqa(args.out_dir)
    print(json.dumps({"task": args.task, **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
