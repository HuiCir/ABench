"""A-LIN-002 metrics: MQAR recall accuracy on the held-out grid, causality, sub-quadratic scaling and throughput
relative to the gated-linear-attention baseline trained by the evaluator under the same recipe and seeds."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

IGNORE = -100
PRIMARY = {"metric": "mqar_accuracy", "goal": "max"}
GATES = [
    {"metric": "causal_leak", "max": 1e-3},
    {"metric": "scaling_exponent", "max": 1.3},
    {"metric": "train_seconds", "max": 1500.0},
    {"metric": "throughput_ratio_vs_gla", "min": 0.5},
]
# examples resampled together for the bootstrap 95% interval of the primary metric (abench_metrics.bootstrap_ci)
RESAMPLE = [{"reference": ["y", "cell"], "outputs": [["fit", "pred"]]}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    y, cell, grid = np.asarray(reference["y"]), np.asarray(reference["cell"]), np.asarray(reference["grid"])
    pred = need(outputs, "fit", "pred", shape=y.shape, dtype_kind="iu")
    q = y != IGNORE
    accs = {}
    for c, (T, K) in enumerate(grid):
        m = q & (cell == c)[:, None]
        if m.any():
            accs[f"mqar_accuracy_T{T}_K{K}"] = float((pred[m] == y[m]).mean())
    out = dict(accs)
    out["mqar_accuracy"] = float(np.mean(list(accs.values())))
    out["causal_leak"] = float(need(outputs, "fit", "causal_leak", shape=(1,), finite=False)[0])
    out["scaling_exponent"] = float(need(outputs, "fit", "scaling_exponent", shape=(1,), finite=False)[0])
    out["throughput_tokens_per_second"] = float(need(outputs, "fit", "throughput", shape=(1,), finite=False)[0])
    out["train_seconds"] = float(seconds["fit"])
    return out


def relative(sub: dict, base: dict) -> dict:
    return {"throughput_ratio_vs_gla": sub["throughput_tokens_per_second"] / max(base["throughput_tokens_per_second"], 1e-9),
            "mqar_accuracy_delta_vs_gla": sub["mqar_accuracy"] - base["mqar_accuracy"],
            "gla_mqar_accuracy": base["mqar_accuracy"]}
