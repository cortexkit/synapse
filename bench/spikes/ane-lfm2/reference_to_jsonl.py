#!/usr/bin/env python3
"""Extract fixed-bucket Core ML runner inputs from an LFM2 reference archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.reference_npz, allow_pickle=False) as archive:
        ids = archive["ids"].tolist()
        input_ids = archive["input_ids"].astype(np.int32)
        attention_mask = archive["attention_mask"].astype(np.int32)
    if len(ids) != len(input_ids) or input_ids.shape != attention_mask.shape:
        raise ValueError("reference archive contains inconsistent input arrays")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        for identifier, row_ids, row_mask in zip(ids, input_ids, attention_mask, strict=True):
            output.write(
                json.dumps(
                    {
                        "id": str(identifier),
                        "input_ids": row_ids.tolist(),
                        "attention_mask": row_mask.tolist(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
