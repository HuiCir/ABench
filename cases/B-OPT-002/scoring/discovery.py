"""Local discovery score X (rules in abench/spec/discovery.md).

For each claimed discovery:
  a  effectiveness        the benchmark operator retrains on L2 with the module replaced by ablation_fallback, e = ln(S_full / S_ablated), which must exceed z·σ
  n  application novelty  judges look up existing uses of the source in ML in the hidden snapshot: new 1 / transfer 0.5 / none 0
  c  provenance correct   1 if the source actually contains the transferred result, otherwise 0
  r  provenance distance  same_subfield 0 / ml 0.3 / cs 0.6 / other_discipline 1.0
x = a · n · c · (0.5 + 0.5 r); X = 1 - Π(1 - x), counting at most max_counted claims.
"""
from __future__ import annotations

import copy
import math

import sota_margin as sm


def _ablated_speedup(runs: dict, setting: str, abl: dict, cap: float):
    s = copy.deepcopy(runs["settings"][setting])
    s["submission"] = {"loss": abl["loss"], "time_ratio": abl["time_ratio"]}
    return sm.setting_speedup(s, cap)


def discovery_score(decls: list, runs: dict, margin_report: dict, judge: dict, cfg: dict, cap: float = 2.0):
    setting = cfg["ablation_setting"]
    full = margin_report["per_setting"][setting]
    S_full, sig_full = full["speedup"], full["sigma_log"]
    rows = []
    for d in decls:
        mod = d["module"]
        jv = judge.get("discoveries", {}).get(mod, {})
        row = {"module": mod, "source": d["source"]["id"], "source_year": d["source"].get("year"),
               "source_field": d["source"].get("field"), "transferred": d.get("transferred")}
        abl = runs.get("ablations", {}).get(mod)
        if abl is None:
            row.update(status="not ablated", x=0.0)
            rows.append(row); continue
        ab = _ablated_speedup(runs, setting, abl, cap)
        e = math.log(S_full / ab["speedup"])
        sigma = math.sqrt(sig_full ** 2 + ab["sigma_log"] ** 2)
        significant = e > cfg["z"] * sigma
        fallback_ok = jv.get("fallback_ok", False)
        a = min(1.0, max(0.0, e / math.log(cfg["full_effect_speedup"]))) if (significant and fallback_ok) else 0.0
        n = cfg["novelty_levels"][jv.get("novelty", "none")]
        c = 1.0 if jv.get("source_correct", False) else 0.0
        r = cfg["field_distance"][jv.get("field_distance", "same_subfield")]
        x = a * n * c * (0.5 + 0.5 * r)
        if not fallback_ok:
            status = "unreasonable substitute"
        elif not significant:
            status = "decorative import (no significant difference in ablation)"
        elif c == 0:
            status = "incorrect provenance"
        elif n == 0:
            status = "retrieval reuse (the same use already exists in ML)"
        elif n < 1:
            status = "transfer discovery"
        else:
            status = "local discovery"
        row.update(S_full=S_full, S_ablated=ab["speedup"], effect_ln=round(e, 4), sigma=round(sigma, 4),
                   significant=significant, a=round(a, 4), n=n, c=c, r=r, x=round(x, 4), status=status,
                   judge_note=jv.get("note"), ml_precedent=jv.get("ml_precedent"))
        rows.append(row)
    counted = sorted(rows, key=lambda z: -z["x"])[: cfg["max_counted"]]
    prod = 1.0
    for z in counted:
        prod *= 1 - z["x"]
    return {"X": round(1 - prod, 4), "declarations": rows}
