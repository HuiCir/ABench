"""U-LOOP-002 metrics.

For each task t and budget b (in units of F_ref(t), measured in stages.py):
  acc_t_b         accuracy of the answers on the 512 held-out problems
  compute_t_b     measured FLOPs per problem / F_ref(t)
  within_t_b      1 if compute_t_b <= b * 1.02
A point counts with its accuracy if it is within budget and with 0 otherwise.
Primary: budgeted_accuracy = mean of the nine counted points (three tasks x three budgets).
Diagnostics: adaptivity_t_b = Spearman correlation between the iterations a problem received and its difficulty, and
accuracy per difficulty at the middle budget."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401
import reason_tasks as T

PRIMARY = {"metric": "budgeted_accuracy", "goal": "max"}
GATES = [{"metric": "n_params", "max": 20_000_000}]
BUDGETS = (4, 8, 16)


def RESAMPLE(reference: dict) -> list:          # noqa: N802  (bootstrap groups: each task's problems)
    return [{"reference": [f"y_{t}", f"d_{t}"], "outputs": [["predict", f"pred_{t}_b{b}"] for b in BUDGETS]
             + [["predict", f"iters_{t}_b{b}"] for b in BUDGETS]} for t in T.TASKS]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(spearmanr(a, b).correlation)


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    tol = float((params or {}).get("budget_tolerance", 1.02))
    out, counted = {}, []
    for t in T.TASKS:
        y, d = np.asarray(reference[f"y_{t}"]), np.asarray(reference[f"d_{t}"])
        fref = float(need(outputs, "predict", f"fref_{t}", shape=(1,))[0])
        for b in BUDGETS:
            p = need(outputs, "predict", f"pred_{t}_b{b}", shape=(len(y),), dtype_kind="iu")
            it = need(outputs, "predict", f"iters_{t}_b{b}", shape=(len(y),), dtype_kind="iu")
            fl = float(need(outputs, "predict", f"flops_{t}_b{b}", shape=(1,))[0])
            acc = float((p == y).mean())
            comp = fl / fref
            ok = comp <= b * tol
            out[f"acc_{t}_b{b}"], out[f"compute_{t}_b{b}"], out[f"within_{t}_b{b}"] = acc, comp, float(ok)
            out[f"mean_iters_{t}_b{b}"] = float(it.mean())
            out[f"adaptivity_{t}_b{b}"] = _spearman(it.astype(float), d.astype(float))
            counted.append(acc if ok else 0.0)
        mid = BUDGETS[len(BUDGETS) // 2]
        p = np.asarray(outputs["predict"][f"pred_{t}_b{mid}"])
        for k in range(1, T.MAX_D[t] + 1):
            m = d == k
            if m.any():
                out[f"acc_{t}_b{mid}_d{k}"] = float((p[m] == y[m]).mean())
    out["budgeted_accuracy"] = float(np.mean(counted))
    out["points_within_budget"] = float(sum(out[f"within_{t}_b{b}"] for t in T.TASKS for b in BUDGETS))
    out["n_params"] = float(need(outputs, "predict", "n_params", shape=(1,))[0])
    out["load_seconds"] = float(seconds["predict"])
    return out
