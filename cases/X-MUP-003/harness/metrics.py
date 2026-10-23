"""X-MUP-003 metrics: held-out loss of the width-1,024 model trained with the learning rate tuned at width 128
(primary); the transfer gap at width 512 (tuning-set loss with η* minus the best over η* · {1/4, ..., 4}) and where the
width-512 optimum lies; relative to the standard parametrisation trained by the evaluator under the same protocol."""
from __future__ import annotations

import math

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
PRIMARY = {"metric": "val_loss_w1024", "goal": "min"}
GATES = [{"metric": "protocol_seconds", "max": 3600.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out = {k: float(need(outputs, "protocol", k, shape=(1,))[0]) for k in
           ("val_loss_w1024", "val_loss_w512", "eta_star", "tune_w1024")}
    chk = need(outputs, "protocol", "tune_check_w512", shape=(len(MULTS),))
    sweep = np.asarray(outputs["protocol"]["tune_sweep_w128"], np.float64)
    out["transfer_gap_w512"] = float(chk[MULTS.index(1.0)] - chk.min())
    out["log2_optimum_shift_w512"] = float(math.log2(MULTS[int(np.argmin(chk))]))
    out["best_tune_loss_w128"] = float(sweep.min())
    out["sweep_width_w128"] = float((sweep <= sweep.min() + 0.02).sum())   # how many grid points lie within 0.02
    out["protocol_seconds"] = float(seconds["protocol"])
    return out


def relative(sub: dict, base: dict) -> dict:
    return {"val_loss_w1024_delta_vs_sp": sub["val_loss_w1024"] - base["val_loss_w1024"],
            "transfer_gap_w512_delta_vs_sp": sub["transfer_gap_w512"] - base["transfer_gap_w512"],
            "sp_val_loss_w1024": base["val_loss_w1024"]}
