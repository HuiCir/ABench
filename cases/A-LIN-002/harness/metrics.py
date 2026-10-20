"""A-LIN-002 metrics: MQAR recall accuracy on the held-out grid, causality, sub-quadratic scaling and throughput
relative to the gated-linear-attention baseline, measured by the evaluator in the same process."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

IGNORE = -100
PRIMARY = {"metric": "mqar_accuracy", "goal": "max"}
GATES = [
    {"metric": "causal_leak", "max": 1e-3},
    {"metric": "scaling_exponent", "max": 1.3},
    {"metric": "throughput_ratio_vs_gla", "min": 0.5},
]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    y, cell, grid = np.asarray(reference["y"]), np.asarray(reference["cell"]), np.asarray(reference["grid"])
    pred = need(outputs, "eval", "pred", shape=y.shape, dtype_kind="iu")
    q = y != IGNORE
    accs = {}
    for c, (T, K) in enumerate(grid):
        m = q & (cell == c)[:, None]
        accs[f"mqar_accuracy_T{T}_K{K}"] = float((pred[m] == y[m]).mean())
    out = dict(accs)
    out["mqar_accuracy"] = float(np.mean(list(accs.values())))
    out["causal_leak"] = float(need(outputs, "eval", "causal_leak", shape=(1,), finite=False)[0])
    out["scaling_exponent"] = float(need(outputs, "eval", "scaling_exponent", shape=(1,), finite=False)[0])
    out["throughput_tokens_per_second"] = float(need(outputs, "eval", "throughput", shape=(1,), finite=False)[0])
    gla = float(need(outputs, "eval", "gla_throughput", shape=(1,), finite=False)[0])
    out["gla_throughput_tokens_per_second"] = gla
    out["throughput_ratio_vs_gla"] = out["throughput_tokens_per_second"] / max(gla, 1e-9)
    out["load_seconds"] = float(seconds["eval"])
    return out
