"""PH-NO-001 metrics: PDEBench-style normalised RMSE of the predicted frames."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need, nrmse  # noqa: F401

PRIMARY = {"metric": "nrmse", "goal": "min"}
GATES: list = []          # validity = no crash, time limits respected, well-formed predictions


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    y = np.asarray(reference["u_out"], np.float64)
    y5 = np.asarray(reference["u_out_512"], np.float64)
    p = need(outputs, "predict", "pred", shape=y.shape).astype(np.float64)
    p5 = need(outputs, "predict", "pred_512", shape=y5.shape).astype(np.float64)
    return {"nrmse": nrmse(p, y, axes=(1, 2)),
            "nrmse_final_frame": nrmse(p[:, -1], y[:, -1], axes=(1,)),
            "nrmse_first_10_frames": nrmse(p[:, :10], y[:, :10], axes=(1, 2)),
            "nrmse_512": nrmse(p5, y5, axes=(1, 2)),
            "load_seconds": float(seconds["predict"])}
