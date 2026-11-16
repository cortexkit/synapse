"""Public-only Qwen3-ASR corpus rules, independent of the inference runtime."""

import hashlib
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import re

SLUG = "qwen3-asr-1.7b"
SEED = 20261119
TIERS = (30, 100, 300)
PROVISIONAL_SLOPE = 25
EARNINGS_REVISION = "59d7f94771cc197984d8038e494c83abeff87282"
EARNINGS_SOURCE = {
    "url": "https://huggingface.co/datasets/argmaxinc/contextual-earnings22/resolve/"
    + EARNINGS_REVISION + "/data/test-00000-of-00001.parquet",
    "sha256": "4d43471525d8df72930629da0b3dfe20d03b5442f04fd7abedf9df9d301661ec",
    "bytes": 356824211,
}


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def compact(corpus: dict) -> dict:
    """Intern repeated public recordings and term arrays without changing cases."""
    recordings, term_lists, cases = {}, {}, []
    audio_fields = {"ground_truth", "audio_sha256", "sample_count", "sample_rate", "channels",
                    "recipe", "parts", "seed", "trailing_silence_samples", "source", "member",
                    "utterance_id"}
    term_fields = {"terms", "vocabulary", "padding_terms", "retained_terms", "global_terms", "local_terms"}
    def key(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode()).hexdigest()
    for case in corpus["cases"]:
        recording = {k: v for k, v in case.items() if k in audio_fields}
        recording_id = key(recording)
        recordings[recording_id] = recording
        row = {k: v for k, v in case.items() if k not in audio_fields | term_fields}
        row["recording"] = recording_id
        row["term_lists"] = {}
        for field in sorted(term_fields & case.keys()):
            list_id = key(case[field])
            term_lists[list_id] = case[field]
            row["term_lists"][field] = list_id
        cases.append(row)
    return {**corpus, "cases": cases, "recordings": recordings, "term_lists": term_lists}


def expand(corpus: dict) -> dict:
    if "recordings" not in corpus:
        return corpus
    cases = []
    for row in corpus["cases"]:
        case = {**corpus["recordings"][row["recording"]],
                **{k: v for k, v in row.items() if k not in ("recording", "term_lists")},
                **{k: corpus["term_lists"][value] for k, value in row["term_lists"].items()}}
        cases.append(case)
    return {k: (cases if k == "cases" else v) for k, v in corpus.items()
            if k not in ("recordings", "term_lists")}


def score_normalize(text):
    """bench/lanes/stt-bias/score.py normalization; not the parity WER normalizer."""
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("-", " ")).split()
    numbers = {"1": "one", "4": "four", "28": "twenty eight", "20": "twenty"}
    return " ".join(numbers.get(word, word) for word in words)


def occurs(term, reference):
    reference = " " + score_normalize(reference) + " "
    return any(normalized and " " + normalized + " " in reference
               for normalized in map(score_normalize, term.split("|")))


def stable_seed(label):
    return int.from_bytes(hashlib.sha256(f"{SEED}:{label}".encode()).digest()[:8], "big")


def distractors(pool, references, count, label, excluded=()):
    if count == 0:
        return []
    excluded = {score_normalize(term) for term in excluded}
    candidates = sorted(set(pool))
    random.Random(stable_seed(label)).shuffle(candidates)
    selected = []
    for term in candidates:
        normalized = score_normalize(term)
        if not normalized or normalized in excluded or any(occurs(term, ref) for ref in references):
            continue
        selected.append(term)
        excluded.add(normalized)
        if len(selected) == count:
            return selected
    raise ValueError(f"insufficient public distractors for {label}: {len(selected)} < {count}")


def tier_list(row, pool, tier):
    retained = row["global_terms"][:tier]
    padding = distractors(pool, [row["ground_truth"]], tier - len(retained),
                          f"earnings:{row['id']}:{tier}", retained)
    terms = retained + padding
    vocabulary = list(dict.fromkeys(row["vocabulary"] + padding))
    return {"terms": terms, "padding_terms": padding, "vocabulary": vocabulary,
            "retained_terms": retained}


def control_lists(controls, pool, tier):
    references = [row["ground_truth"] for row in controls]
    return [distractors(pool, references, tier, f"control:{tier}:{i}") for i in range(3)]


def audio_tokens(valid_mel_frames):
    if valid_mel_frames < 0:
        raise ValueError("negative mel frame count")
    # Three stride-two padded convolutions on each 100-frame chunk.
    return (valid_mel_frames // 100) * 13 + ((valid_mel_frames % 100 + 7) // 8)


def max_new_tokens(sample_count, tokens_per_second=6):
    return 64 + math.ceil(tokens_per_second * sample_count / 16000)


def capped_fixture_report(traces):
    def summary(rows):
        ids = [row["id"] for row in rows if row["stop_reason"] == "cap"]
        return {"count": len(ids), "ids": ids}
    def tiers(rows):
        return {label: summary([row for row in rows if row.get("tier") == tier])
                for label, tier in (("none", None), ("30", 30), ("100", 100), ("300", 300))}
    if len({row["id"] for row in traces}) != len(traces) or any(
            row.get("tier") not in (None, 30, 100, 300) for row in traces):
        raise ValueError("duplicate capped-report fixture or unknown tier")
    splits = {}
    for split in sorted({row["split"] for row in traces}):
        rows = [row for row in traces if row["split"] == split]
        splits[split] = {**summary(rows), "by_tier": tiers(rows)}
    return {**summary(traces), "by_tier": tiers(traces), "by_split": splits}


def calibration_trace(row, case):
    """Use the actual request, not a tier label, to identify name-list context."""
    listed = bool(case.get("terms"))
    if row["context_tokens"] < 0 or listed != (row["context_tokens"] > 0):
        raise ValueError("request terms/context disagree: " + row["id"])
    return {**row, "has_name_list": listed}


def token_rate_report(rows):
    traces = [row for row in rows if row.get("generated_ids")]
    if not traces:
        raise ValueError("no traces for token-rate calibration")
    if any(row["stop_reason"] not in ("cap", "stop") for row in traces):
        raise ValueError("unknown reference stop reason")
    stopped = [row for row in traces if row["stop_reason"] == "stop"]
    no_list = [row for row in stopped if not row["has_name_list"] and row["context_tokens"] == 0]
    eligible = [row for row in stopped if row["sample_count"] >= 32000 and
              (row["split"].startswith(("librispeech-", "fleurs-")) or
               row["split"] in ("earnings22-test-200", "long-form"))]
    speech = [row for row in eligible if not row["has_name_list"] and row["context_tokens"] == 0]
    name_list = [row for row in eligible if row["has_name_list"]]
    if not speech:
        raise ValueError("no >=2-second speech traces for slope calibration")
    def rate(row):
        return Fraction((len(row["generated_ids"]) - row["language_prefix_tokens"]) * 16000, row["sample_count"])
    maximum = max(speech, key=rate)
    slope = max(6, math.ceil(Fraction(3, 2) * rate(maximum)))
    if slope > PROVISIONAL_SLOPE:
        raise ValueError(f"speech slope {slope} exceeds 25 at {maximum['id']}; stop before sealing")
    failures = [row["id"] for row in no_list if
                max_new_tokens(row["sample_count"], PROVISIONAL_SLOPE) < math.ceil(1.5 * len(row["generated_ids"]))]
    if failures:
        raise ValueError("complete-cap headroom failed: " + ", ".join(failures))
    template = max(len(row["prompt_ids"]) - row["span"]["len"] - row["context_tokens"] for row in traces)
    cap = max_new_tokens(600 * 16000, PROVISIONAL_SLOPE)
    ten_minute_audio = audio_tokens(600 * 100)
    name_maximum = max(name_list, key=rate) if name_list else None
    return {"highest_speech_tokens_per_second": float(rate(maximum)),
            "maximum_rate_fixture": maximum["id"], "speech_fixtures": len(speech),
            "speech_rate_excludes": "generated language prefix through <asr_text>; includes stop",
            "measured_tokens_per_second": slope,
            "highest_name_list_tokens_per_second": float(rate(name_maximum)) if name_maximum else None,
            "maximum_name_list_rate_fixture": name_maximum["id"] if name_maximum else None,
            "name_list_speech_fixtures": len(name_list),
            "highest_raw_tokens_per_second": max(len(r["generated_ids"]) * 16000 / r["sample_count"] for r in stopped),
            "rate_and_headroom_scope": "stopped, empty request terms, context_tokens=0 only; name-list traces and capped prefixes remain unchanged and are excluded from both checks",
            "capped_fixtures": capped_fixture_report(traces),
            "tokens_per_second": PROVISIONAL_SLOPE, "headroom": 1.5, "includes_stop_token": True,
            "complete_cap_checked_fixtures": len(no_list), "ten_minute_cap": cap,
            "ten_minute_audio_tokens": ten_minute_audio, "max_template_tokens": template,
            "kv_positions_without_bias": ten_minute_audio + template + cap,
            "kv_positions_at_2048_bias_tokens": ten_minute_audio + template + 2048 + cap,
            "bias_budget_note": "2048 is an upper-bound sizing scenario, not a selected engine bias limit"}


def validate_annotations(annotations, candidates):
    """Require a contiguous, seeded review prefix ending at the 50th acceptance."""
    by_id = {row["utterance_id"]: row for row in candidates}
    order = sorted(by_id)
    random.Random(SEED).shuffle(order)
    reviews = annotations["reviews"]
    if annotations["seed"] != SEED or [r["utterance_id"] for r in reviews] != order[:len(reviews)]:
        raise ValueError("curation is not the seeded review prefix")
    accepted = []
    for review in reviews:
        if "reason" in review:
            if review["reason"] not in ("none", "multiple", "ambiguous"):
                raise ValueError("unknown rejection reason")
            continue
        row = by_id[review["utterance_id"]]
        start, end = review["word_span"]
        words = row["ground_truth"].split()
        if not (0 <= start < end <= len(words)) or " ".join(words[start:end]) != review["noun"]:
            raise ValueError("curated name does not match its reference word span")
        accepted.append((row, review))
        if len(accepted) == 50 and review is not reviews[-1]:
            raise ValueError("curation continued beyond its 50th acceptance")
    if len(accepted) != 50:
        raise ValueError("exactly 50 accepted single-name utterances are required")
    return accepted
