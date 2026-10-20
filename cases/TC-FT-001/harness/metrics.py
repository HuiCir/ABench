"""TC-FT-001 metrics (evaluator and self-check use this same file)."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, accuracy, macro_f1, need  # noqa: F401  (OutputError is part of the API)

PRIMARY = {"metric": "accuracy", "goal": "max"}
# Validity thresholds. Provisional until the operator's calibration run on the frozen data (see the case README).
GATES = [{"metric": "predict_docs_per_second", "min": 2000.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    y = np.asarray(reference["y"]).astype(np.int64)
    pred = need(outputs, "predict", "pred", shape=(len(y),))
    if pred.dtype.kind not in "iu":
        if pred.dtype.kind == "f" and np.all(pred == np.round(pred)):
            pred = pred.astype(np.int64)
        else:
            raise OutputError(f"predict.pred: expected integer class labels, got {pred.dtype}")
    if len(pred) and (pred.min() < 0 or pred.max() >= params["n_classes"]):
        raise OutputError("predict.pred: labels outside 0..3")
    fin = max(float(seconds.get("predict.finish", 0.0)), 1e-6)
    return {"accuracy": accuracy(y, pred), "macro_f1": macro_f1(y, pred, params["n_classes"]),
            "load_seconds": float(seconds["predict"]), "predict_docs_per_second": len(y) / fin}
