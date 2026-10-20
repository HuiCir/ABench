"""GN-OGB-002 metrics: accuracy on the test nodes, as in the official OGB evaluator for ogbn-arxiv."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, accuracy, need  # noqa: F401

PRIMARY = {"metric": "test_accuracy", "goal": "max"}
GATES: list = []          # validity = no crash, time limits respected, well-formed scores


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    y = np.asarray(reference["y_test"]).astype(np.int64)
    S = need(outputs, "predict", "scores", shape=(len(y), params["n_classes"]))
    return {"test_accuracy": accuracy(y, S.argmax(1)), "load_seconds": float(seconds["predict"]),
            "predict_seconds": float(seconds.get("predict.finish", 0.0))}
