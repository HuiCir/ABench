"""LG-HOLD-002 metrics: macro-F1 over the five answer positions (as in the CaseHOLD paper) and accuracy."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, accuracy, macro_f1, need  # noqa: F401

PRIMARY = {"metric": "macro_f1", "goal": "max"}
GATES: list = []          # validity = no crash, time limits respected, well-formed scores


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    y = np.asarray(reference["y"]).astype(np.int64)
    S = need(outputs, "predict", "scores", shape=(len(y), params["n_choices"]))
    pred = S.argmax(1)
    return {"macro_f1": macro_f1(y, pred, params["n_choices"]), "accuracy": accuracy(y, pred),
            "load_seconds": float(seconds["predict"]),
            "predict_seconds": float(seconds.get("predict.finish", 0.0))}
