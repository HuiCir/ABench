"""A-HYB-001 metrics: held-out language-model loss (primary) under compute, parameter, causality and long-context
gates; multi-query associative recall (MQAR) accuracy on the held-out grid, gated relative to the transformer
baseline trained by the evaluator; forward latency per token at 1k, 4k and 16k tokens (reported)."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

IGNORE = -100
PRIMARY = {"metric": "val_loss", "goal": "min"}
GATES = [{"metric": "flops_ratio_T1024", "max": 1.05}, {"metric": "flops_ratio_T16384", "max": 0.6},
         {"metric": "params_ratio", "max": 1.05}, {"metric": "causal_leak", "max": 1e-3},
         {"metric": "diverged", "max": 0.0}, {"metric": "lm_train_seconds", "max": 1800.0},
         {"metric": "mqar_accuracy_delta_vs_transformer", "min": -0.05}]
RESAMPLE = [{"reference": ["mqar_y", "mqar_cell"], "outputs": [["mqar", "pred"]]}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out = {k: float(need(outputs, "lm", k, shape=(1,), finite=False)[0]) for k in
           ("val_loss", "diverged", "flops_ratio_T1024", "flops_ratio_T16384", "params_ratio", "causal_leak",
            "tokens_per_second", "latency_per_token_T1024", "latency_per_token_T4096", "latency_per_token_T16384")}
    out["causal_leak"] = max(out["causal_leak"], float(need(outputs, "mqar", "mqar_causal_leak", shape=(1,), finite=False)[0]))
    y, cell, grid = np.asarray(reference["mqar_y"]), np.asarray(reference["mqar_cell"]), np.asarray(reference["mqar_grid"])
    pred = need(outputs, "mqar", "pred", shape=y.shape, dtype_kind="iu")
    q = y != IGNORE
    accs = []
    for c, (T, K) in enumerate(grid):
        m = q & (cell == c)[:, None]
        if m.any():
            out[f"mqar_accuracy_T{T}_K{K}"] = float((pred[m] == y[m]).mean())
            accs.append(out[f"mqar_accuracy_T{T}_K{K}"])
    out["mqar_accuracy"] = float(np.mean(accs))
    out["latency_growth_16k_over_1k"] = out["latency_per_token_T16384"] / max(out["latency_per_token_T1024"], 1e-12)
    out["lm_train_seconds"] = float(seconds["lm"])
    return out


def relative(sub: dict, base: dict) -> dict:
    return {"val_loss_delta_vs_transformer": sub["val_loss"] - base["val_loss"],
            "mqar_accuracy_delta_vs_transformer": sub["mqar_accuracy"] - base["mqar_accuracy"],
            "latency_ratio_T16384_vs_transformer": sub["latency_per_token_T16384"] / max(base["latency_per_token_T16384"], 1e-12),
            "transformer_val_loss": base["val_loss"], "transformer_mqar_accuracy": base["mqar_accuracy"]}
