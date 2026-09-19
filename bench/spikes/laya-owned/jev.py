"""Jev on the same Athena rows, to separate the model CLASS from the open instance.

Laya scored 19.7% class-exact on 233 real Athena consult requests against a
52.4% constant-AUDIT floor, which closed the lane. That result cannot tell us
whether a System One model is wrong for this task or whether a 421M open
checkpoint is simply too small: Jev is the frontier instance of the same class,
with the same API shape, so running it on identical rows with identical criteria
answers exactly that.

TWO ARMS, because context is a confound rather than a nuisance here:

  full        Jev's natural condition. Its state budget is 32k tokens, so no row
              truncates -- where laya's English checkpoint kept roughly 320
              tokens of state and truncated 85.8% of rows.
  clipped     the same prose cut to laya's state room, so both models see the
              same bytes. This is the controlled comparison; `full` is the
              product question.

If `clipped` lands near `full`, laya's failure was capability. If `clipped`
collapses toward laya, it was context, and the verdict on the class is wrong.
"""

import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

# The production-faithful set: Athena's own classify prompt defines no class, so
# names plus one neutral sentence is at least as much as production gets. Reused
# verbatim from the laya run so the two are comparable byte for byte.
CRITERIA_V0 = {
    "AUDIT": "Inspect something against requirements or standards to identify issues.",
    "EVALUATE": "Assess the quality, value, or suitability of something.",
    "PLAN": "Organize proposed actions into a plan for achieving a goal.",
    "DIAGNOSE": "Determine the cause of a problem from the available evidence.",
    "EXPLAIN": "Clarify how or why something works or happens.",
    "SPEC": "Define the requirements and intended behavior of something.",
}

# Laya's English checkpoint leaves roughly this many characters of state after
# its 192-token question head. Measured from the laya run rather than assumed:
# the median surviving fraction there put the cut near 1300 characters.
LAYA_STATE_CHARS = 1300


def key():
    raw = (pathlib.Path.home() / ".config/jev.key").read_text().strip()
    if not raw:
        raise SystemExit("jev.key is empty")
    return raw


def rows(name):
    out = []
    for line in (DATA / name).open():
        row = json.loads(line)
        gold = (row.get("reply_json") or {}).get("class")
        if gold is None:
            continue
        out.append({"id": row["id"], "prose": row["request_prose"], "gold": gold.upper()})
    return out


def ask(api_key, state, attempt=0):
    body = json.dumps(
        {
            "state": state,
            "model": MODEL,
            "questions": {
                "kind": {
                    "type": "choice",
                    "instructions": "Which kind of work does this request ask for?",
                    "criteria": CRITERIA_V0,
                }
            },
        }
    ).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        payload = error.read().decode()[:300]
        # Their docs name 429 as the rate-limit signal and say the response may
        # carry retry-after; honour it rather than guessing a backoff.
        if error.code == 429 and attempt < 6:
            wait = float(error.headers.get("retry-after", 2 ** attempt))
            time.sleep(wait)
            return ask(api_key, state, attempt + 1)
        if 500 <= error.code < 600 and attempt < 4:
            time.sleep(2 ** attempt)
            return ask(api_key, state, attempt + 1)
        raise SystemExit(f"jev {error.code}: {payload}")


def run(arm, corpus, clip):
    api_key = key()
    out_path = ROOT / f"rows-jev-{arm}-{corpus.replace('.jsonl', '')}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open():
            done.add(json.loads(line)["id"])
        print(f"   resuming: {len(done)} rows already recorded", flush=True)

    corpus_rows = rows(corpus)
    tokens = 0
    started = time.time()
    with out_path.open("a") as fh:
        for i, row in enumerate(corpus_rows):
            if row["id"] in done:
                continue
            state = row["prose"][:clip] if clip else row["prose"]
            answer = ask(api_key, state)
            kind = answer["answers"]["kind"]
            tokens += answer.get("usage", {}).get("input_tokens", 0)
            fh.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "gold": row["gold"],
                        "predicted": kind["choice"],
                        "correct": kind["choice"] == row["gold"],
                        "probabilities": kind["probabilities"],
                        "confidence": kind["confidence"],
                        "sent_chars": len(state),
                        "prose_chars": len(row["prose"]),
                        "clipped": bool(clip) and len(state) < len(row["prose"]),
                    }
                )
                + "\n"
            )
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"   {arm}/{corpus}: {i + 1}/{len(corpus_rows)}", flush=True)

    elapsed = time.time() - started
    print(
        f"   {arm}/{corpus}: done in {elapsed:.0f}s, {tokens} input tokens "
        f"(~${tokens * 0.042 / 1e6:.4f} at their published rate)",
        flush=True,
    )


if __name__ == "__main__":
    arm = sys.argv[1]
    corpus = sys.argv[2] if len(sys.argv) > 2 else "real-gold.jsonl"
    run(arm, corpus, LAYA_STATE_CHARS if arm == "clipped" else 0)
