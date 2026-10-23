"""DS-PIPE-002 metrics (evaluator and self-check use this same file)."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, auc_binary, need  # noqa: F401

DATASETS = [44122, 44123, 44125, 44126, 44128, 45021]
PRIMARY = {"metric": "mean_accuracy", "goal": "max"}
GATES: list = []          # validity = no crash, time limits respected, well-formed outputs

# examples resampled together for the bootstrap 95% interval of the primary metric (abench_metrics.bootstrap_ci)
RESAMPLE = [{"reference": [f"y_{d}"], "outputs": [["predict", f"proba_{d}"]]} for d in DATASETS]

def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, accs, aucs = {}, [], []
    for d in DATASETS:
        y = np.asarray(reference[f"y_{d}"]).astype(np.int64)
        P = need(outputs, "predict", f"proba_{d}", shape=(len(y), 2))
        if (P < -1e-9).any():
            raise OutputError(f"predict.proba_{d}: negative probabilities")
        acc = float((P.argmax(1) == y).mean())
        auc = auc_binary(y == 1, P[:, 1] - P[:, 0])
        out[f"accuracy_{d}"], out[f"auc_{d}"] = acc, auc
        accs.append(acc)
        aucs.append(auc)
    out["mean_accuracy"] = float(np.mean(accs))
    out["mean_auc"] = float(np.mean(aucs))
    out["load_seconds"] = float(seconds["predict"])
    out["predict_seconds"] = float(seconds.get("predict.finish", 0.0))
    return out
