"""K-FA-003 metrics: correctness on every shape, then the geometric mean of SDPA time / your time."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "relative_speed_geomean", "goal": "max"}
GATES = [{"metric": "all_correct", "min": 1.0}, {"metric": "forbidden_api", "max": 0.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    rows = need(outputs, "bench", "rows", shape=(None, 7), finite=False)
    if len(rows) == 0:
        raise OutputError("no shapes were benchmarked")
    mx, mean, bmx, bmean, t_sub, t_ref, ok = rows.T
    correct = (ok == 1) & (mx <= np.maximum(0.02, 4 * bmx)) & (mean <= np.maximum(0.002, 4 * bmean))
    ratio = np.where(correct & np.isfinite(t_sub) & (t_sub > 0), t_ref / np.maximum(t_sub, 1e-9), 0.0)
    out = {"all_correct": float(correct.all()), "n_shapes": float(len(rows)),
           "relative_speed_geomean": float(np.exp(np.mean(np.log(np.maximum(ratio, 1e-9))))) if correct.all() else 0.0,
           "relative_speed_min": float(ratio.min()), "max_abs_error": float(mx.max()),
           "forbidden_api": float(need(outputs, "bench", "forbidden_api", shape=(1,))[0])}
    return out
