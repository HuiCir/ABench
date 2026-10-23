"""C-FA-004 metrics: correctness of the output and the three gradients on every shape, peak memory relative to SDPA,
then the geometric mean of SDPA time / your time for forward + backward."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "relative_speed_geomean", "goal": "max"}
GATES = [{"metric": "all_correct", "min": 1.0}, {"metric": "max_memory_ratio", "max": 2.0},
         {"metric": "forbidden_api", "max": 0.0}]
FLOORS = (0.02, 0.05, 0.05, 0.05)              # out, dq, dk, dv
NAMES = ("out", "dq", "dk", "dv")


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    rows = need(outputs, "bench", "rows", shape=(None, 21), finite=False)
    if len(rows) == 0:
        raise OutputError("no shapes were benchmarked")
    ok = rows[:, 20] == 1
    correct = ok.copy()
    out = {}
    for i, (name, floor) in enumerate(zip(NAMES, FLOORS)):
        mx, mean, bmx, bmean = rows[:, 4 * i:4 * i + 4].T
        good = (mx <= np.maximum(floor, 4 * bmx)) & (mean <= np.maximum(floor / 10, 4 * bmean))
        correct &= good
        out[f"max_abs_error_{name}"] = float(mx.max())
    t_sub, t_ref, m_sub, m_ref = rows[:, 16:20].T
    ratio = np.where(correct & np.isfinite(t_sub) & (t_sub > 0), t_ref / np.maximum(t_sub, 1e-9), 0.0)
    mem = np.where(ok, m_sub / np.maximum(m_ref, 1.0), np.inf)
    out.update({"all_correct": float(correct.all()), "n_shapes": float(len(rows)),
                "relative_speed_geomean": float(np.exp(np.mean(np.log(np.maximum(ratio, 1e-9))))) if correct.all() else 0.0,
                "relative_speed_min": float(ratio.min()), "max_memory_ratio": float(mem.max()),
                "forbidden_api": float(need(outputs, "bench", "forbidden_api", shape=(1,))[0])})
    return out
