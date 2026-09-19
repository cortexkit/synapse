"""Compare every owned output with both CPU references; exit nonzero on gate failure."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    battery = json.loads((ROOT / "battery.json").read_text())
    owned = json.loads((ROOT / (sys.argv[1] if len(sys.argv) > 1 else "owned-f16.json")).read_text())
    refs = {name: row for case in battery["cases"] for name, row in case["rows"].items()}
    assert len(owned["rows"]) == len(refs) == 24
    assert {row["name"] for row in owned["rows"]} == set(refs)
    report = {}
    for precision in ["fp32", "f16_encoder"]:
        dp, da, ds, disagreements = [], [], [], []
        agreements = 0
        per_row = []
        for row in owned["rows"]:
            name, out = row["name"], row["output"]
            if precision not in refs[name]:
                continue
            ref = refs[name][precision]
            p = np.array(ref["probabilities"])
            q = np.array(out["probabilities"])
            assert q.shape == p.shape and np.isfinite(q).all()
            gap = float(np.sort(p)[-1] - np.sort(p)[-2])
            same = bool(p.argmax() == q.argmax())
            agreements += same
            delta_p = float(np.max(np.abs(q-p)))
            delta_a = abs(out["act_probability"] - ref["act_probability"])
            dp.append(delta_p)
            da.append(delta_a)
            score = None
            if ref["score"] is not None:
                score = abs(out["score"]-ref["score"])
                ds.append(score)
            per_row.append({"name": name, "delta_p": delta_p, "delta_act": delta_a, "delta_score": score, "top2_gap": gap, "argmax_agrees": same})
            if not same:
                disagreements.append({"name": name, "reference_top2_gap": gap, "reference_argmax": int(p.argmax()), "owned_argmax": int(q.argmax())})
        if not dp:
            report[precision] = {"unavailable": battery["f16_error"]}
            continue
        stats = lambda a: {"max": max(a), "p95": float(np.percentile(a, 95))}
        report[precision] = {"probability": stats(dp), "act_probability": stats(da), "score": stats(ds),
                             "argmax_agreement": f"{agreements}/{len(dp)}", "disagreements": disagreements,
                             "gate_passes": max(dp) <= .02 and all(d["reference_top2_gap"] < .05 for d in disagreements),
                             "rows": per_row}
    path = ROOT / f"parity-{owned['dtype']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({p: {k: v for k,v in r.items() if k != "rows"} for p,r in report.items()}, indent=2))
    assert report["fp32"]["gate_passes"], "pre_registered_fp32_probability_and_argmax_gate"
    print("pre_registered_fp32_probability_and_argmax_gate: PASS")


if __name__ == "__main__":
    main()
