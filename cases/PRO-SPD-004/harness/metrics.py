"""PRO-SPD-004 metrics: wall-clock training time, validity requires the held-out loss target and the size cap."""
from __future__ import annotations

import math

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "train_seconds", "goal": "min", "aggregate": "median over runs"}
GATES = [
    {"metric": "val_loss", "max": 2.92},
    {"metric": "n_params", "max": 550_000_000.0},
    {"metric": "train_seconds", "max": 10800.0},
]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    v = float(need(outputs, "eval", "val_loss", shape=(1,), finite=False)[0])
    return {"val_loss": v if math.isfinite(v) else float("inf"),
            "n_params": float(need(outputs, "train", "n_params", shape=(1,))[0]),
            "train_seconds": float(seconds["train"])}
