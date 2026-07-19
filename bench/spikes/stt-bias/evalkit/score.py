#!/usr/bin/env python3
"""Score term recovery, case fidelity, false insertions, and WER for one arm."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--asr-output", type=Path, required=True)
    parser.add_argument("--terms", type=Path, default=Path(__file__).with_name("terms.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def words(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:[./_-][\w]+)*", text.casefold(), flags=re.UNICODE)


def levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_word in enumerate(reference, 1):
        current = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1]


def contains_term(text: str, term: str, *, case_sensitive: bool) -> bool:
    flags = 0 if case_sensitive else re.IGNORECASE
    # Word guards avoid counting a short term embedded in a different word while
    # still allowing terms such as `tok/s` and `llama.cpp` to match literally.
    expression = rf"(?<!\w){re.escape(term)}(?!\w)"
    return re.search(expression, text, flags) is not None


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    args = parse_args()
    manifest = {row["id"]: row for row in load_jsonl(args.audio_manifest)}
    with args.asr_output.open(encoding="utf-8") as handle:
        asr_result = json.load(handle)
    transcripts = {row["id"]: row for row in asr_result["results"]}
    technical_terms = [
        row["term"]
        for row in load_jsonl(args.terms)
        if row["class"] == "technical" and not row.get("excluded", False)
    ]

    missing = sorted(set(manifest) - set(transcripts))
    if missing:
        raise SystemExit(f"ASR output is missing {len(missing)} manifest ids, first: {missing[0]}")

    term_hits = 0
    case_hits = 0
    technical_total = 0
    false_insertions = 0
    control_total = 0
    word_errors = 0
    reference_words = 0
    detail_rows: list[dict] = []

    for identifier, row in manifest.items():
        transcript = transcripts[identifier]["text"]
        excluded = row.get("excluded", False)
        if not excluded:
            reference = words(row["source_text"])
            hypothesis = words(transcript)
            word_errors += levenshtein(reference, hypothesis)
            reference_words += len(reference)

        detail = {
            "id": identifier,
            "expected_term": row["expected_term"],
            "term_class": row["term_class"],
            "excluded": excluded,
            "transcript": transcript,
        }
        if row["term_class"] == "technical" and not excluded:
            technical_total += 1
            term_hit = contains_term(transcript, row["expected_term"], case_sensitive=False)
            case_hit = contains_term(transcript, row["expected_term"], case_sensitive=True)
            term_hits += term_hit
            case_hits += case_hit
            detail.update(term_exact=term_hit, case_exact=case_hit)
        elif row["term_class"] == "control" and not excluded:
            control_total += 1
            inserted = [
                term
                for term in technical_terms
                if contains_term(transcript, term, case_sensitive=False)
            ]
            false_insertions += bool(inserted)
            detail.update(false_inserted_terms=inserted)
        detail_rows.append(detail)

    score = {
        "arm": args.arm,
        "technical_scored": technical_total,
        "term_exact_hits": term_hits,
        "term_exact_accuracy": rate(term_hits, technical_total),
        "case_exact_hits": case_hits,
        "case_fidelity": rate(case_hits, technical_total),
        "case_fidelity_given_term_hit": rate(case_hits, term_hits),
        "control_scored": control_total,
        "false_insertions": false_insertions,
        "false_insertion_rate": rate(false_insertions, control_total),
        "wer_errors": word_errors,
        "wer_reference_words": reference_words,
        "wer": rate(word_errors, reference_words),
        "tts_excluded_rows": sum(bool(row.get("excluded", False)) for row in manifest.values()),
        "runtime": {
            "prefill_wall_s": asr_result.get("prefill_wall_s"),
            "decode_wall_s": asr_result.get("decode_wall_s"),
            "trie_bias_wall_s": asr_result.get("trie_bias_wall_s"),
            "text_prompt_tokens": sum(
                result.get("text_prompt_tokens", 0) for result in asr_result["results"]
            ),
            "bias_prompt_tokens": sum(
                result.get("bias_prompt_tokens", 0) for result in asr_result["results"]
            ),
            "bias": asr_result.get("bias"),
        },
        "rows": detail_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    print(
        "{arm}: term exact {term:.1%}, false insertion {insert:.1%}, WER {wer:.1%}, case fidelity {case:.1%}".format(
            arm=args.arm,
            term=score["term_exact_accuracy"] or 0.0,
            insert=score["false_insertion_rate"] or 0.0,
            wer=score["wer"] or 0.0,
            case=score["case_fidelity"] or 0.0,
        )
    )


if __name__ == "__main__":
    main()
