"""MD-MNIST-003 metrics: accuracy and macro one-vs-rest ROC AUC per dataset, as reported by MedMNIST v2."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, accuracy, auc_ovr, need  # noqa: F401

DATASETS = {"blood": 8, "organa": 11}
PRIMARY = {"metric": "mean_accuracy", "goal": "max"}
GATES: list = []          # validity = no crash, time limits respected, well-formed scores


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, accs, aucs = {}, [], []
    for d, k in DATASETS.items():
        y = np.asarray(reference[f"y_test_{d}"]).reshape(-1).astype(np.int64)
        P = need(outputs, "predict", f"proba_{d}", shape=(len(y), k))
        out[f"accuracy_{d}"] = accuracy(y, P.argmax(1))
        out[f"auc_{d}"] = auc_ovr(y, P)
        accs.append(out[f"accuracy_{d}"])
        aucs.append(out[f"auc_{d}"])
    out.update(mean_accuracy=float(np.mean(accs)), mean_auc=float(np.mean(aucs)), load_seconds=float(seconds["predict"]))
    return out
