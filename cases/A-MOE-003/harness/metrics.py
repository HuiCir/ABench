"""A-MOE-003 metrics: held-out loss of the trained mixture-of-experts model (primary), with compute, parameter and
causality gates; load balance from the per-expert token counts over the held-out tokens (max load / mean load per
layer, and the fraction of experts receiving at least 1% of a layer's tokens); training throughput. Relative metrics
compare with the top-2 baseline trained by the evaluator under identical conditions."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "val_loss", "goal": "min"}
GATES = [{"metric": "flops_ratio", "max": 1.05}, {"metric": "ffn_params_ratio", "max": 1.0},
         {"metric": "causality_violation", "max": 0.0}, {"metric": "batch_dependence", "max": 0.0},
         {"metric": "diverged", "max": 0.0}, {"metric": "train_seconds", "max": 1800.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    o = outputs["run"]
    out = {k: float(need(outputs, "run", k, shape=(1,))[0]) for k in
           ("val_loss", "diverged", "flops_ratio", "ffn_params_ratio", "causality_violation", "batch_dependence",
            "tokens_per_second")}
    out["train_seconds"] = float(seconds["run"])
    if "expert_counts" in o:
        c = np.asarray(o["expert_counts"], np.float64)
        if c.ndim != 2 or (c < 0).any():
            raise OutputError("run.expert_counts: expected non-negative (layers, experts) counts")
        share = c / np.maximum(c.sum(1, keepdims=True), 1.0)
        out["load_imbalance"] = float((share.max(1) * c.shape[1]).mean())       # 1.0 = perfectly balanced
        out["expert_utilization"] = float((share >= 0.01).mean())
    return out


def relative(sub: dict, base: dict) -> dict:
    return {"val_loss_delta_vs_top2": sub["val_loss"] - base["val_loss"],
            "throughput_ratio_vs_top2": sub["tokens_per_second"] / max(base["tokens_per_second"], 1e-9),
            "top2_val_loss": base["val_loss"]}
