"""U-LOOP-001 metrics: held-out loss at 8 loops (primary) and the loop-count curve (1, 2, 4, 8), loss at 2,048 tokens
(length extrapolation), parameter and compute gates, causality; relative to the 4-layer transformer baseline (same
parameter budget, no loops) trained by the evaluator under identical conditions."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

LOOPS = (1, 2, 4, 8)
PRIMARY = {"metric": "val_loss_loops8", "goal": "min"}
GATES = [{"metric": "params_ratio", "max": 1.02}, {"metric": "flops_ratio_loops8", "max": 3.0},
         {"metric": "causal_leak", "max": 1e-3}, {"metric": "diverged", "max": 0.0},
         {"metric": "train_seconds", "max": 1800.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out = {}
    for k in LOOPS:
        out[f"val_loss_loops{k}"] = float(need(outputs, "run", f"val_loss_loops{k}", shape=(1,))[0])
        out[f"flops_ratio_loops{k}"] = float(need(outputs, "run", f"flops_ratio_loops{k}", shape=(1,))[0])
    out["loop_gain_1_to_8"] = out["val_loss_loops1"] - out["val_loss_loops8"]
    out["monotone_in_loops"] = float(all(out[f"val_loss_loops{b}"] <= out[f"val_loss_loops{a}"] + 1e-3
                                         for a, b in zip(LOOPS, LOOPS[1:])))
    for k in ("val_loss_T2048_loops8", "params_ratio", "causal_leak", "diverged", "tokens_per_second"):
        out[k] = float(need(outputs, "run", k, shape=(1,), finite=False)[0])
    out["extrapolation_penalty"] = out["val_loss_T2048_loops8"] - out["val_loss_loops8"]
    out["train_seconds"] = float(seconds["run"])
    return out


def relative(sub: dict, base: dict) -> dict:
    return {"val_loss_loops8_delta_vs_4layer": sub["val_loss_loops8"] - base["val_loss_loops8"],
            "fourlayer_val_loss": base["val_loss_loops8"]}
