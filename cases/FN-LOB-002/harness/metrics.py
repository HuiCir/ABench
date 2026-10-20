"""FN-LOB-002 metrics: macro-F1 over {up, stationary, down} at horizon k = 10, as reported by DeepLOB."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, accuracy, macro_f1, need  # noqa: F401

PRIMARY = {"metric": "macro_f1", "goal": "max"}
GATES: list = []          # validity = no crash, time limits respected, well-formed scores


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    y = np.asarray(reference["y"]).astype(np.int64)
    P = need(outputs, "predict", "proba", shape=(len(y), params["n_classes"]))
    pred = P.argmax(1)
    fin = max(float(seconds.get("predict.finish", 0.0)), 1e-6)
    return {"macro_f1": macro_f1(y, pred, params["n_classes"]), "accuracy": accuracy(y, pred),
            "load_seconds": float(seconds["predict"]), "predict_windows_per_second": len(y) / fin}
