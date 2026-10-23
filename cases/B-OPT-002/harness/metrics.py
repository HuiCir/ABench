"""B-OPT-002 metrics: held-out loss after the fixed training budget (primary), compared with the AdamW program run by
the evaluator on the same hardware, seeds and data; novelty and stability of the program on the gradient stream."""
from __future__ import annotations

import math

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "val_loss", "goal": "min"}
GATES = [
    {"metric": "diverged", "max": 0.0},
    {"metric": "max_similarity", "max": 0.98},
    {"metric": "unstable", "max": 0.0},
    {"metric": "train_seconds", "max": 1500.0},
    {"metric": "step_time_ratio_vs_adamw", "max": 1.25},
]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    v = float(need(outputs, "eval", "val_loss", shape=(1,), finite=False)[0])
    div = float(need(outputs, "train", "diverged", shape=(1,))[0])
    sim = float(need(outputs, "screen", "max_similarity", shape=(1,), finite=False)[0])
    stable = float(need(outputs, "screen", "stable_extreme_scales", shape=(1,))[0])
    size = need(outputs, "screen", "program_size", shape=(3,))
    return {"val_loss": v if math.isfinite(v) else float("inf"), "diverged": max(div, 0.0 if math.isfinite(v) else 1.0),
            "state_ratio": float(need(outputs, "train", "state_ratio", shape=(1,))[0]),
            "train_seconds": float(seconds["train"]),
            "max_similarity": sim if math.isfinite(sim) else 1.0,
            "unstable": 1.0 - stable,
            "descent_cosine": float(need(outputs, "screen", "descent_cosine", shape=(1,), finite=False)[0]),
            "program_lines": float(size[0]), "program_calls": float(size[1]), "program_state": float(size[2])}


def relative(sub: dict, base: dict) -> dict:
    return {"val_loss_delta_vs_adamw": sub["val_loss"] - base["val_loss"],
            "step_time_ratio_vs_adamw": sub["train_seconds"] / max(base["train_seconds"], 1e-9),
            "adamw_val_loss": base["val_loss"]}
