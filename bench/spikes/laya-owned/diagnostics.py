"""Inspect active-token encoder/head drift, including unsaturated act logits."""
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parent
battery = json.loads((ROOT / "battery.json").read_text())
owned = {r["name"]: r["output"] for r in json.loads((ROOT / "owned-f16.json").read_text())["rows"]}
report = {}
for precision in ["fp32", "f16_encoder"]:
    rows = []
    for case in battery["cases"]:
        name = case["name"]
        ref = next(iter(case["rows"].values()))
        if precision not in ref:
            continue
        py = load_file(ROOT / "dumps" / precision / (name + ".safetensors"))
        rs = load_file(ROOT / "dumps/owned-f16" / (name + ".safetensors"))
        n = len(ref["input_ids"])
        entry = {"name": name}
        for key in ["encoder", "head"]:
            delta = py[key][:, :n].astype(np.float64) - rs[key][:, :n]
            entry[key] = {"max_absolute": float(np.abs(delta).max()), "rmse": float(np.sqrt(np.mean(delta**2)))}
        entry["act_logits_max_absolute"] = float(np.max(np.abs(np.array(ref[precision]["act_logits"]) - owned[name]["act_logits"])))
        rows.append(entry)
    report[precision] = rows
(ROOT / "tensor-drift.json").write_text(json.dumps(report, indent=2) + "\n")
for precision, rows in report.items():
    print(precision, "encoder max", max(r["encoder"]["max_absolute"] for r in rows),
          "head max", max(r["head"]["max_absolute"] for r in rows),
          "act logits max", max(r["act_logits_max_absolute"] for r in rows))
