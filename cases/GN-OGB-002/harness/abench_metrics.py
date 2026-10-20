"""Metric helpers shared by the ABench standard cases (identical copy in each case's harness/ directory).

numpy only, so the judge, the self-check and the tests compute metrics with the same code.
"""
from __future__ import annotations

import numpy as np


class OutputError(ValueError):
    """The submission's outputs have the wrong shape, dtype or values; the run is invalid."""


def need(outputs: dict, stage: str, key: str, shape: tuple | None = None, finite: bool = True,
         dtype_kind: str | None = None) -> np.ndarray:
    """Fetch outputs[stage][key] and check its shape (None entries are wildcards) and finiteness."""
    if stage not in outputs or key not in outputs[stage]:
        raise OutputError(f"stage {stage!r} did not return {key!r}")
    a = np.asarray(outputs[stage][key])
    if shape is not None:
        if a.ndim != len(shape) or any(s is not None and s != t for s, t in zip(shape, a.shape)):
            raise OutputError(f"{stage}.{key}: expected shape {shape}, got {a.shape}")
    if dtype_kind is not None and a.dtype.kind not in dtype_kind:
        raise OutputError(f"{stage}.{key}: expected dtype kind {dtype_kind!r}, got {a.dtype}")
    if finite and a.dtype.kind in "fc" and not np.isfinite(a).all():
        raise OutputError(f"{stage}.{key}: contains NaN or inf")
    return a


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float((np.asarray(pred) == np.asarray(y)).mean())


def confusion(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.int64)
    np.add.at(m, (np.asarray(y, int), np.asarray(pred, int)), 1)
    return m


def macro_f1(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    m = confusion(y, pred, k)
    tp = np.diag(m).astype(float)
    prec = np.divide(tp, m.sum(0), out=np.zeros(k), where=m.sum(0) > 0)
    rec = np.divide(tp, m.sum(1), out=np.zeros(k), where=m.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return float(f1.mean())


def binary_f1(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp = float((y & pred).sum())
    denom = float(y.sum() + pred.sum())
    return 2 * tp / denom if denom else 0.0


def matthews(y: np.ndarray, pred: np.ndarray) -> float:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = float(((y == 1) & (pred == 1)).sum()); tn = float(((y == 0) & (pred == 0)).sum())
    fp = float(((y == 0) & (pred == 1)).sum()); fn = float(((y == 1) & (pred == 0)).sum())
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / d) if d else 0.0


def auc_binary(y01: np.ndarray, s: np.ndarray) -> float:
    """ROC AUC via the rank statistic, with ties given the average rank."""
    y01 = np.asarray(y01).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n1, n0 = int(y01.sum()), int((~y01).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y01].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auc_ovr(y: np.ndarray, P: np.ndarray) -> float:
    """Macro one-vs-rest ROC AUC over the classes present in y."""
    vals = [auc_binary(y == c, P[:, c]) for c in range(P.shape[1]) if (y == c).any() and (y != c).any()]
    return float(np.mean(vals))


def ndcg_recall_at_k(target: np.ndarray, topk: np.ndarray, k: int = 10) -> tuple[float, float]:
    """One relevant item per row (leave-one-out): NDCG@k and Recall@k (= hit rate)."""
    topk = np.asarray(topk)[:, :k]
    hit = topk == np.asarray(target)[:, None]
    pos = np.where(hit.any(1), hit.argmax(1), -1)
    ndcg = np.where(pos >= 0, 1.0 / np.log2(np.maximum(pos, 0) + 2.0), 0.0)
    return float(ndcg.mean()), float((pos >= 0).mean())


def nrmse(pred: np.ndarray, true: np.ndarray, axes: tuple) -> float:
    """PDEBench-style normalized RMSE: ||pred - true|| / ||true|| over `axes`, averaged over the rest."""
    num = np.sqrt(((pred - true) ** 2).sum(axis=axes))
    den = np.sqrt((true ** 2).sum(axis=axes))
    return float((num / np.maximum(den, 1e-12)).mean())


def softmax_rows(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    P = P - P.max(1, keepdims=True)
    e = np.exp(P)
    return e / e.sum(1, keepdims=True)
