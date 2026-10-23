"""X-ROBUST-009 metrics: selective-classification risk over a mix of in-scope, out-of-scope, shifted and noisy
queries (primary: area under the risk-coverage curve), plus calibration, out-of-scope detection, accuracy and the
quality of the abstain decision."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "aurc", "goal": "min"}
GATES: list = []
RESAMPLE = [{"reference": ["label", "group"], "outputs": [["predict", "label"], ["predict", "confidence"], ["predict", "abstain"]]}]


def aurc(conf: np.ndarray, err: np.ndarray) -> float:
    """Mean over coverages k/N of the error rate among the k most confident queries (ties in confidence share their
    mean error)."""
    order = np.argsort(-conf, kind="stable")
    c, e = conf[order], err[order].astype(np.float64)
    _, start, counts = np.unique(-c, return_index=True, return_counts=True)
    for s, k in zip(start, counts):                     # average errors within tied confidences
        e[s:s + k] = e[s:s + k].mean()
    risk = np.cumsum(e) / np.arange(1, len(e) + 1)
    return float(risk.mean())


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(tot)


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counted half."""
    s = np.concatenate([pos, neg])
    r = np.empty(len(s))
    order = np.argsort(s, kind="stable")
    ranks = np.arange(1, len(s) + 1, dtype=np.float64)
    ss = s[order]
    _, inv, cnt = np.unique(ss, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = csum - (cnt - 1) / 2.0
    r[order] = avg[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def macro_f1(y: np.ndarray, p: np.ndarray, k: int = 150) -> float:
    f = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    y = np.asarray(reference["label"]).astype(np.int64)
    g = np.asarray(reference["group"]).astype(np.int64)
    n = len(y)
    lab = need(outputs, "predict", "label", shape=(n,)).astype(np.int64)
    conf = need(outputs, "predict", "confidence", shape=(n,)).astype(np.float64)
    ab = need(outputs, "predict", "abstain", shape=(n,)).astype(bool)
    if conf.min() < 0 or conf.max() > 1:
        raise OutputError("confidence must lie in [0, 1]")
    ins = y >= 0
    right = ins & (lab == y)                              # the answer would be correct
    out = {"aurc": aurc(conf, ~right)}
    clean = g == 0
    out["ece_in_scope"] = ece(conf[clean], right[clean].astype(float))
    out["ood_auroc"] = auroc(conf[ins], conf[~ins]) if ins.any() and (~ins).any() else float("nan")
    out["accuracy_in_scope"] = float(right[clean].mean())
    out["macro_f1_in_scope"] = macro_f1(y[clean], lab[clean])
    decision_ok = np.where(ins, right & ~ab, ab)
    out["decision_accuracy"] = float(decision_ok.mean())
    out["over_abstention"] = float(ab[ins].mean())
    out["out_of_scope_answered"] = float((~ab[~ins]).mean()) if (~ins).any() else float("nan")
    for name, k in (("out_of_scope", 1), ("banking77", 2), ("typo", 3)):
        m = g == k
        if m.any():
            out[f"decision_accuracy_{name}"] = float(decision_ok[m].mean())
    if (g == 3).any():
        out["accuracy_typo"] = float(right[g == 3].mean())
    out["ms_per_query"] = float(need(outputs, "predict", "ms_per_query", shape=(1,))[0])
    out["checkpoint_mb"] = float(need(outputs, "predict", "checkpoint_bytes", shape=(1,))[0]) / 2**20
    return out
