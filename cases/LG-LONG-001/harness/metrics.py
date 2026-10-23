"""LG-LONG-001 metrics, as in LexGLUE for ECtHR: micro- and macro-averaged F1 over the 10 articles plus an 11th
"no violation" label that is 1 when a case has no positive label (in the gold standard and, separately, in the
prediction); the primary metric is their mean."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "mean_f1", "goal": "max"}
GATES: list = []          # validity = no crash, time limits respected, well-formed outputs
RESAMPLE = [{"reference": ["y"], "outputs": [["predict", "pred"]]}]


def with_none(y: np.ndarray) -> np.ndarray:
    return np.concatenate([y, (y.sum(1, keepdims=True) == 0).astype(y.dtype)], 1)


def f1s(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    tp, fp, fn = (y & p).sum(0), ((1 - y) & p).sum(0), (y & (1 - p)).sum(0)
    micro = 2 * tp.sum() / max(1, 2 * tp.sum() + fp.sum() + fn.sum())
    per = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(1, 2 * tp + fp + fn), 0.0)
    return float(micro), float(per.mean())


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    y = np.asarray(reference["y"]).astype(np.int64)
    p = need(outputs, "predict", "pred", shape=y.shape).astype(np.int64)
    if not np.isin(p, (0, 1)).all():
        raise OutputError("predict.pred: expected 0/1 decisions")
    micro, macro = f1s(with_none(y), with_none(p))
    return {"micro_f1": micro, "macro_f1": macro, "mean_f1": (micro + macro) / 2,
            "load_seconds": float(seconds["predict"]), "predict_seconds": float(seconds.get("predict.finish", 0.0))}
