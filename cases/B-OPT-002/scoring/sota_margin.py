"""SOTA margin M: effective speedup relative to the "reference envelope".

Each reference method has a "budget–final loss" curve in each setting (WSD decay branches, averaged over seeds).
Cost axis = tokens × step-time ratio (relative to AdamW). Envelope = the lowest loss over all reference methods at each cost point.
Speedup S = cost the envelope needs to reach the submission's final loss / the submission's own cost.
"""
from __future__ import annotations

import math

import numpy as np


def _ref_curve(ref: dict, T: float, time_ratio: float):
    b = np.asarray(ref["budgets"], dtype=float)
    L = np.asarray(ref["loss"], dtype=float).mean(axis=1)      # [budget, seed] → mean over seeds
    return np.log(b * T * time_ratio), L


def envelope(refs: dict, time_ratios: dict, T: float, n_grid: int = 2000):
    curves = {k: _ref_curve(v, T, time_ratios[k]) for k, v in refs.items()}
    lo = min(c[0][0] for c in curves.values())
    hi = max(c[0][-1] for c in curves.values())
    grid = np.linspace(lo, hi, n_grid)
    env = np.full_like(grid, np.inf)
    arg = np.full(grid.shape, "", dtype=object)
    for k, (x, y) in curves.items():
        inside = (grid >= x[0]) & (grid <= x[-1])
        yi = np.interp(grid, x, y)
        better = inside & (yi < env)
        env[better] = yi[better]
        arg[better] = k
    return grid, env, arg


def cost_to_reach(loss: float, grid, env):
    """Log cost at which the envelope first drops to loss; returns None when out of range (with the direction noted)."""
    run_min = np.minimum.accumulate(env)
    idx = np.nonzero(run_min <= loss)[0]
    if len(idx) == 0:
        return None, "beyond_max"
    i = idx[0]
    if i == 0:
        return grid[0], "below_min"
    x0, x1, y0, y1 = grid[i - 1], grid[i], run_min[i - 1], run_min[i]
    t = 0.0 if y0 == y1 else (y0 - loss) / (y0 - y1)
    return x0 + t * (x1 - x0), "ok"


def setting_speedup(setting: dict, cap: float):
    T = setting["token_budget"]
    tr = setting["time_ratios"]
    grid, env, arg = envelope(setting["references"], tr, T)
    sub = setting["submission"]
    own_cost = T * sub["time_ratio"]
    per_seed, flags = [], []
    for L in sub["loss"]:
        lc, flag = cost_to_reach(L, grid, env)
        flags.append(flag)
        if lc is None:
            per_seed.append(cap)
        else:
            per_seed.append(min(cap, math.exp(lc) / own_cost))
    S = float(np.exp(np.mean(np.log(per_seed))))
    sigma = float(np.std(np.log(per_seed), ddof=1) / math.sqrt(len(per_seed))) if len(per_seed) > 1 else 0.0

    # which reference method dominates the envelope at the submission's cost (for the report)
    j = int(np.argmin(np.abs(grid - math.log(own_cost))))
    best_final = {k: float(np.mean(v["loss"][v["budgets"].index(1.0)])) for k, v in setting["references"].items()}
    return {
        "speedup": round(S, 4),
        "sigma_log": round(sigma, 4),
        "per_seed": [round(x, 4) for x in per_seed],
        "flags": flags,
        "submission_loss_mean": round(float(np.mean(sub["loss"])), 4),
        "envelope_leader_at_budget": arg[j],
        "best_ref_at_1x": min(best_final, key=best_final.get),
        "best_ref_loss_at_1x": round(min(best_final.values()), 4),
        "time_ratio": sub["time_ratio"],
    }


def sota_margin(runs: dict, weights: dict, cfg: dict) -> dict:
    cap = cfg.get("cap", 2.0)
    g = cfg["gate"]
    per = {name: setting_speedup(runs["settings"][name], cap) for name in weights}
    wsum = sum(weights.values())
    logS = sum(weights[k] * math.log(per[k]["speedup"]) for k in weights) / wsum
    S = math.exp(logS)
    sigma = math.sqrt(sum((weights[k] / wsum) ** 2 * per[k]["sigma_log"] ** 2 for k in weights))
    S_lcb = S * math.exp(-g["lcb_z"] * sigma)

    ood = {}
    ood_ok = True
    for k, v in runs.get("ood", {}).items():
        reg = v["submission"] / v["best_ref"] - 1
        ood[k] = {"submission": v["submission"], "best_ref": v["best_ref"], "regression": round(reg, 4)}
        ood_ok &= reg <= g["ood_max_regression"]

    time_ok = all(per[k]["time_ratio"] <= 1.25 for k in per)
    checks = {
        "aggregate_speedup>=%.2f" % g["min_aggregate_speedup"]: S >= g["min_aggregate_speedup"],
        "lcb>=%.2f" % g["min_lcb"]: S_lcb >= g["min_lcb"],
        "per_setting>=%.2f" % g["min_per_setting"]: all(per[k]["speedup"] >= g["min_per_setting"] for k in per),
        "ood_regression<=%.0f%%" % (100 * g["ood_max_regression"]): ood_ok,
        "step_time<=1.25x": time_ok,
    }
    full = cfg["full_marks_speedup"]
    M = 0.1 + 0.9 * min(1.0, max(0.0, math.log(S) / math.log(full)))
    return {
        "M": round(M, 4),
        "aggregate_speedup": round(S, 4),
        "aggregate_speedup_lcb": round(S_lcb, 4),
        "gate_pass": all(checks.values()),
        "gate_checks": checks,
        "per_setting": per,
        "ood": ood,
    }
