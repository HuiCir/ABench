"""PRO-MOE-003 metrics: held-out FineWeb loss of the delivered mixture-of-experts model (primary), compute per token
and stored size as limits, audit gates."""
from __future__ import annotations

import math

import numpy as np

import pro_stages as P
from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "val_loss", "goal": "min"}
GATES = P.GATES + [{"metric": "flops_per_token", "max": 0.8e9}, {"metric": "stored_elements", "max": 2_500_000_000.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out = P.scores(outputs)
    v = float(need(outputs, "eval", "val_loss", shape=(1,), finite=False)[0])
    out["val_loss"] = v if math.isfinite(v) else float("inf")
    out["flops_per_token"] = float(need(outputs, "eval", "flops_per_token", shape=(1,))[0])
    out["stored_elements"] = float(need(outputs, "eval", "stored_elements", shape=(1,))[0])
    return out
