"""T-OPT-002 metrics: test accuracy of the ViT after the fixed budget, compared with the AdamW baseline run by the
evaluator on the same hardware, seeds and data."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, accuracy, need  # noqa: F401

PRIMARY = {"metric": "test_accuracy", "goal": "max"}
GATES = [
    {"metric": "diverged", "max": 0.0},
    {"metric": "state_ratio", "max": 3.0},
    {"metric": "train_seconds", "max": 1200.0},
    {"metric": "step_time_ratio_vs_adamw", "max": 1.25},
]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    y = np.asarray(reference["y"]).astype(np.int64)
    pred = need(outputs, "eval", "pred", shape=(len(y),), dtype_kind="iu")
    return {"test_accuracy": accuracy(y, pred),
            "diverged": float(need(outputs, "train", "diverged", shape=(1,))[0]),
            "state_ratio": float(need(outputs, "train", "state_ratio", shape=(1,))[0]),
            "train_seconds": float(seconds["train"])}


def relative(sub: dict, base: dict) -> dict:
    return {"accuracy_delta_vs_adamw": sub["test_accuracy"] - base["test_accuracy"],
            "step_time_ratio_vs_adamw": sub["train_seconds"] / max(base["train_seconds"], 1e-9),
            "adamw_test_accuracy": base["test_accuracy"]}
