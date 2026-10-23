"""B-OPT-003 metrics: held-out loss after the fixed training budget under a 1x optimizer-state cap, compared with the
AdamW baseline (which itself needs 2x state and is run only for reference) trained by the
evaluator on the same hardware, seeds and data."""
from __future__ import annotations

import math

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "val_loss", "goal": "min"}
GATES = [
    {"metric": "diverged", "max": 0.0},
    {"metric": "state_ratio", "max": 1.0},
    {"metric": "train_seconds", "max": 1500.0},
    {"metric": "step_time_ratio_vs_adamw", "max": 1.25},
]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    v = float(need(outputs, "eval", "val_loss", shape=(1,), finite=False)[0])
    div = float(need(outputs, "train", "diverged", shape=(1,))[0])
    return {"val_loss": v if math.isfinite(v) else float("inf"), "diverged": max(div, 0.0 if math.isfinite(v) else 1.0),
            "state_ratio": float(need(outputs, "train", "state_ratio", shape=(1,))[0]),
            "train_seconds": float(seconds["train"])}


def relative(sub: dict, base: dict) -> dict:
    return {"val_loss_delta_vs_adamw": sub["val_loss"] - base["val_loss"],
            "step_time_ratio_vs_adamw": sub["train_seconds"] / max(base["train_seconds"], 1e-9),
            "adamw_val_loss": base["val_loss"]}
