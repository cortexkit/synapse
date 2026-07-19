#!/usr/bin/env python3
"""Build the deterministic carrier-sentence corpus for the STT bias spike."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TECHNICAL_TEMPLATES = (
    "Then we call {term} on the buffer before the next test.",
    "The {term} change is documented in the project notes.",
    "Please review {term} during the model integration meeting.",
)
CONTROL_TEMPLATES = (
    "The {term} is ready for the next test.",
    "Please review the {term} before the meeting.",
    "We recorded the {term} in the project notes.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", type=Path, default=Path(__file__).with_name("terms.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    terms = load_jsonl(args.terms)
    rng = random.Random(args.seed)
    rows: list[dict] = []
    for index, term in enumerate(terms, 1):
        templates = TECHNICAL_TEMPLATES if term["class"] == "technical" else CONTROL_TEMPLATES
        # The fixed three template slots make every term equally represented;
        # seeded ordering prevents the corpus from clustering template styles.
        order = list(range(len(templates)))
        rng.shuffle(order)
        for carrier_index, template_index in enumerate(order, 1):
            rows.append(
                {
                    "id": f"{index:03d}-{carrier_index}",
                    "source_text": templates[template_index].format(term=term["term"]),
                    "expected_term": term["term"],
                    "term_class": term["class"],
                    "source": term["source"],
                    "excluded": term.get("excluded", False),
                    "exclusion_reason": term.get("exclusion_reason"),
                    "seed": args.seed,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} utterances from {len(terms)} terms to {args.out}")


if __name__ == "__main__":
    main()
