"""Fail-closed Q8 quality and teacher-forced calibration gates on parsed text."""

from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from qwen3_asr_reference import canonical, sha256

HERE = Path(__file__).resolve().parent
SCORER_SHA256 = "d73eaa81fba20cc9492a3445a306c0b15bdfe69d322c93a9d7cf06a73aaba8f8"


def name_scorer():
    path = HERE.parents[1] / "lanes/stt-bias/score.py"
    if sha256(path) != SCORER_SHA256:
        raise ValueError("name scorer differs from the frozen score.py")
    spec = importlib.util.spec_from_file_location("quant_name_scorer", path)
    if spec is None or spec.loader is None:
        raise ValueError("name scorer could not be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def probe_failure(case, trace):
    if case["split"] not in ("near-silence", "noise", "single-name"):
        return None
    scorer = name_scorer()
    terms = case["terms"]
    if case["split"] == "single-name":
        terms = [term for term in terms if term != case["noun"]]
    occurrences = scorer.spans(scorer.norm(trace["text"]).split(), terms)
    if occurrences:
        return {"gate": "parsed-text-probe", "id": case["id"],
                "split": case["split"], "forbidden_occurrences": len(occurrences)}
    return None


def cap_failure(trace, baseline):
    """Only a sealed fp32 capped fixture may supply a bounded Q8 teacher prefix."""
    if trace["stop_reason"] == "stop":
        return None
    if (trace["stop_reason"] == "cap" and baseline["stop_reason"] == "cap"
            and trace["cap"] == baseline["cap"]
            and len(trace["generated_ids"]) == trace["cap"]
            and not any(id in (151643, 151645) for id in trace["generated_ids"])):
        return None
    return {"gate": "decode_limit_reached", "id": trace.get("id"),
            "cap": trace.get("cap")}


def calibration(cases, teachers, profiles):
    wanted = {case["id"] for case in cases if not case.get("refusal")}
    if set(teachers) != wanted or set(profiles) != wanted:
        raise ValueError("incomplete calibration fixture coverage")
    disagreements, largest = 0, 0.0
    for id in sorted(wanted):
        teacher, profile = teachers[id], profiles[id]
        ids, margins = teacher["generated_ids"], teacher["top_two_margins"]
        if (not ids or len(ids) != len(margins)
                or any(not math.isfinite(m) or m < 0 for m in margins)
                or len(profile["generated_ids"]) != len(ids)
                or profile["forced_ids"] != ids[:-1]
                or profile["mode"] != "f16-boundaries"
                or profile["prompt_ids"] != teacher["prompt_ids"]
                or profile["span"] != teacher["span"]):
            raise ValueError("invalid teacher-forced trace: " + id)
        for expected, actual, margin in zip(ids, profile["generated_ids"], margins):
            if expected != actual:
                disagreements += 1
                largest = max(largest, margin)
    tau = max(0.1, 2 * largest)
    fixtures, sets = {}, defaultdict(lambda: {"graded": 0, "exempt": 0})
    for case in cases:
        if case.get("refusal"):
            continue
        margins = teachers[case["id"]]["top_two_margins"]
        graded = sum(m > tau for m in margins)
        counts = {"graded": graded, "exempt": len(margins) - graded}
        fixtures[case["id"]] = counts
        for key in counts:
            sets[case["split"]][key] += counts[key]
    failures = []
    if tau > 1.0:
        failures.append({"gate": "tau-maximum", "tau": tau, "maximum": 1.0})
    for split, counts in sorted(sets.items()):
        total = counts["graded"] + counts["exempt"]
        if counts["graded"] * 10 < total * 9:
            failures.append({"gate": "nonvacuous-profile", "split": split, **counts})
    return {"tau": tau, "largest_disagreement_teacher_margin": largest,
            "disagreements": disagreements, "fixtures": fixtures, "fixture_sets": dict(sets),
            "failures": failures, "accepted": not failures}


def wer_counts(binary, cases, baseline, actual):
    request = [{"id": c["id"], "truth": c["ground_truth"],
                "baseline": baseline[c["id"]]["text"], "actual": actual[c["id"]]["text"]}
               for c in cases]
    result = subprocess.run([str(binary)], input=json.dumps(request), text=True,
                            capture_output=True, check=True)
    rows = json.loads(result.stdout)
    if set(rows) != {c["id"] for c in cases}:
        raise ValueError("WER scorer fixture coverage mismatch")
    return rows


def quality(cases, fp32, q8, counts, *, partial=False):
    """Grade complete matched arms; partial mode defers every incomplete aggregate."""
    wanted = {c["id"] for c in cases if not c.get("refusal")}
    ready = set(q8)
    if (set(fp32) != wanted or set(counts) != ready or not ready <= wanted
            or (not partial and ready != wanted)):
        raise ValueError("incomplete quality fixture coverage")
    scorer = name_scorer()
    failures, groups, measurements = [], defaultdict(list), {}
    tier_quality = defaultdict(list)
    for case in cases:
        if case.get("refusal"):
            continue
        id = case["id"]
        if id in ready:
            failure = probe_failure(case, q8[id])
            if failure:
                failures.append(failure)
            failure = cap_failure(q8[id], fp32[id])
            if failure:
                failures.append({**failure, "id": id})
        # Compare identical list configurations, never pool listed arms with no-list
        # speech: an easy arm must not conceal a failing tier or control list.
        arm = case.get("list_id") or ("none" if not case.get("terms") else
                                      "tier-" + str(case["tier"]) if case.get("tier") else "global")
        if case["ground_truth"]:
            groups[(case["split"], arm, case.get("tier"))].append(id)
    for (split, arm, tier), ids in sorted(groups.items(), key=lambda item: str(item[0])):
        if not set(ids) <= ready:
            continue
        words = sum(counts[id]["words"] for id in ids)
        before = sum(counts[id]["baseline_edits"] for id in ids)
        after = sum(counts[id]["actual_edits"] for id in ids)
        measurements[f"{split}/{arm}/tier-{tier}"] = {"words": words, "fp32_edits": before, "q8_edits": after}
        if not words or (after - before) * 200 > words:
            failure = {"gate": "wer-delta", "split": split, "arm": arm, "tier": tier,
                       "words": words, "fp32_edits": before, "q8_edits": after}
            (failures if tier is None else tier_quality[tier]).append(failure)

    def names(rows, traces, vocabulary=None):
        totals: dict[str, Any] = {"hits": 0, "occurrences": 0, "false": 0}
        for c in rows:
            vocab = c.get("vocabulary", c.get("terms", [])) if vocabulary is None else vocabulary(c)
            score = scorer.score_clip({"reference": c["ground_truth"], "vocabulary": vocab,
                                       "expected": []}, traces[c["id"]]["text"])
            totals["hits"] += score["term_hits"]
            totals["occurrences"] += score["term_occurrences"]
            totals["false"] += score["false_insertions"]
        totals["recall"] = totals["hits"] / totals["occurrences"] if totals["occurrences"] else None
        return totals

    earnings_all = [c for c in cases if c["split"] == "earnings22-test-200"]
    earnings = [c for c in earnings_all if c.get("terms")]
    for tier in (None, 30, 100, 300):
        for listed in (False, True):
            rows = [c for c in earnings_all if c.get("tier") == tier and bool(c.get("terms")) == listed]
            if not rows or not {c["id"] for c in rows} <= ready:
                continue
            a, b = names(rows, fp32), names(rows, q8)
            measurements[f"earnings/{tier}/{listed}"] = {"fp32": a, "q8": b}
            if a["recall"] is None or b["recall"] < a["recall"] - .01 or b["false"] > a["false"] + 2:
                failure = {"gate": "earnings-names", "tier": tier, "listed": listed, "fp32": a, "q8": b}
                (failures if tier is None else tier_quality[tier]).append(failure)

    tiers = {}
    for tier in (30, 100, 300):
        controls = [c for c in cases if c.get("tier") == tier and c.get("probe_set") == "unrelated-list"]
        rows = [c for c in earnings if c.get("tier") == tier]
        if not controls and not rows:
            continue
        tier_failures = []
        if len(controls) != 1200 or len(rows) != 200:
            raise ValueError("incomplete C4 tier")
        dependencies = ({c["id"] for c in controls + rows}
                        | {c["id"].rsplit("-tier-", 1)[0] + "-global" for c in rows}
                        | {c["id"].rsplit("-control-", 1)[0] for c in controls})
        if not dependencies <= wanted:
            raise ValueError("incomplete C4 tier dependencies")
        complete = dependencies <= ready
        insertion = names(controls, q8)["false"] if {c["id"] for c in controls} <= ready else None
        if insertion is not None and insertion * 1000 > len(controls):
            tier_failures.append({"gate": "control-insertions", "tier": tier,
                                  "pairs": len(controls), "insertions": insertion})
        # Compare retained names only, with the tier's extended vocabulary for
        # false occurrences in both the tier and the whole-call baseline.
        global_traces = {}
        name_dependencies = ({c["id"] for c in rows}
                             | {c["id"].rsplit("-tier-", 1)[0] + "-global" for c in rows})
        if name_dependencies <= ready:
            for c in rows:
                global_traces[c["id"]] = q8[c["id"].rsplit("-tier-", 1)[0] + "-global"]
            a, b = names(rows, global_traces, lambda c: c["retained_terms"]), names(rows, q8, lambda c: c["retained_terms"])
            all_a, all_b = names(rows, global_traces), names(rows, q8)
            if a["recall"] is None or b["recall"] < a["recall"] - .02 or all_b["false"] > all_a["false"] + 2:
                tier_failures.append({"gate": "tier-vs-global", "tier": tier, "global": a, "tier_names": b,
                                      "global_false": all_a["false"], "tier_false": all_b["false"]})
        for list_id in sorted({c["list_id"] for c in controls}):
            for split in sorted({c["split"] for c in controls}):
                arm = [c for c in controls if c["list_id"] == list_id and c["split"] == split]
                arm_dependencies = ({c["id"] for c in arm}
                                    | {c["id"].rsplit("-control-", 1)[0] for c in arm})
                if not arm_dependencies <= ready:
                    continue
                words = sum(counts[c["id"]]["words"] for c in arm)
                listed = sum(counts[c["id"]]["actual_edits"] for c in arm)
                unlisted = sum(counts[c["id"].rsplit("-control-", 1)[0]]["actual_edits"] for c in arm)
                if (listed - unlisted) * 200 > words:
                    tier_failures.append({"gate": "unrelated-list-wer", "tier": tier, "list": list_id,
                                          "split": split, "words": words, "listed": listed, "unlisted": unlisted})
        # WER, aligned recall and false-occurrence gates also apply with the
        # tier's identical lists in the fp32 and Q8 runs.
        inherited = tier_quality[tier]
        if partial and tier == 30:
            failures.extend(tier_failures + inherited)
        if not complete:
            continue
        tiers[str(tier)] = {"passed": not tier_failures and not inherited,
                            "pairs": len(controls), "insertions": insertion,
                            "failures": tier_failures + inherited,
                            "longest_context_tokens": max(q8[c["id"]]["context_tokens"] for c in controls + rows)}
        if tier == 30 and not tiers["30"]["passed"]:
            if not partial:
                failures.extend(tier_failures + inherited)
            break
    passing = [t for t in (30, 100, 300) if tiers.get(str(t), {}).get("passed")]
    n = min(2048, math.ceil(tiers[str(max(passing))]["longest_context_tokens"] / 64) * 64) if passing else None
    return {"accepted": not failures, "failures": failures, "measurements": measurements,
            "tiers": tiers, "provisional_bias_limit": n, "bias_limit_frozen": False}


def seal_profile(cases, teachers, profiles, fp32, wer_binary, private_aggregate):
    """Only complete public coverage and a passing local aggregate can admit engines."""
    from qwen3_asr_corpus import expand
    sealed_cases = expand(json.loads((HERE / "qwen3-asr-corpus.json").read_text()))["cases"]
    if canonical(cases) != canonical(sealed_cases):
        raise ValueError("only the entire frozen corpus can seal a profile")
    for case in cases:
        if case.get("refusal"):
            continue
        id = case["id"]
        baseline = json.loads((HERE.parent / "fixtures/qwen3-asr-1.7b/traces" / (id + ".json")).read_text())
        if baseline != fp32[id] or any(teachers[id][key] != baseline[key] for key in ("prompt_ids", "span")):
            raise ValueError("quality baseline or teacher inputs differ from the seal")
        if teachers[id]["audio_sha256"] != baseline["model_input_audio_sha256"]:
            raise ValueError("teacher audio differs from the seal")
    result = calibration(cases, teachers, profiles)
    quality_report = quality(cases, fp32, teachers, wer_counts(wer_binary,
                             [c for c in cases if not c.get("refusal")], fp32, teachers))
    private_ok = ({"correct", "total"} <= set(private_aggregate) <=
                  {"correct", "total", "clips", "false_occurrences", "names_right_percent", "passed"}
                  and type(private_aggregate["correct"]) is int and type(private_aggregate["total"]) is int
                  and 0 <= private_aggregate["correct"] <= private_aggregate["total"]
                  and private_aggregate["total"] > 0
                  and private_aggregate["correct"] * 10 >= private_aggregate["total"] * 9)
    accepted = result["accepted"] and quality_report["accepted"] and private_ok and quality_report["provisional_bias_limit"] is not None
    profile = json.loads((HERE / "qwen3-asr-quant-profile.json").read_text())
    profile.update(status="sealed" if accepted else "gate-failed", engine_slices_allowed=accepted,
                   tau=result["tau"], fixture_counts=result["fixtures"], calibration=result,
                   quality=quality_report, private_aggregate=private_aggregate,
                   teacher_sha256=hashlib.sha256(canonical(teachers)).hexdigest(),
                   forced_profile_sha256=hashlib.sha256(canonical(profiles)).hexdigest())
    return profile
