"""B-OPT-004 metrics: held-out loss after a fixed token budget at three batch sizes (32, 128 and 512 sequences), their
mean (primary) and the large-batch gap, compared with the AdamW baseline (square-root learning-rate scaling) run by the
evaluator on the same hardware, seeds and data."""
from __future__ import annotations

import math

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

BATCHES = (32, 128, 512)
PRIMARY = {"metric": "mean_val_loss", "goal": "min"}
GATES = [
    {"metric": "diverged", "max": 0.0},
    {"metric": "max_state_ratio", "max": 3.0},
    {"metric": "max_train_seconds", "max": 1500.0},
]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, losses, div, ratios, secs = {}, [], 0.0, [], []
    for b in BATCHES:
        v = float(need(outputs, f"eval_b{b}", "val_loss", shape=(1,), finite=False)[0])
        v = v if math.isfinite(v) else float("inf")
        out[f"val_loss_b{b}"] = v
        losses.append(v)
        div = max(div, float(need(outputs, f"train_b{b}", "diverged", shape=(1,))[0]), 0.0 if math.isfinite(v) else 1.0)
        ratios.append(float(need(outputs, f"train_b{b}", "state_ratio", shape=(1,))[0]))
        secs.append(float(seconds[f"train_b{b}"]))
        out[f"train_seconds_b{b}"] = secs[-1]
    out["mean_val_loss"] = float(np.mean(losses))
    out["large_batch_gap"] = losses[-1] - losses[0]
    out["diverged"] = div
    out["max_state_ratio"] = max(ratios)
    out["max_train_seconds"] = max(secs)
    return out


def relative(sub: dict, base: dict) -> dict:
    d = {f"val_loss_b{b}_delta_vs_adamw": sub[f"val_loss_b{b}"] - base[f"val_loss_b{b}"] for b in BATCHES}
    d["mean_val_loss_delta_vs_adamw"] = sub["mean_val_loss"] - base["mean_val_loss"]
    d["adamw_large_batch_gap"] = base["large_batch_gap"]
    return d
