"""BI-DNA-002 metrics: test accuracy per dataset (the Genomic Benchmarks metric) and its mean."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, accuracy, auc_binary, need  # noqa: F401

DATASETS = ["human_nontata_promoters", "human_enhancers_cohn", "human_ocr_ensembl"]
PRIMARY = {"metric": "mean_accuracy", "goal": "max"}
GATES: list = []          # validity = no crash, time limits respected, well-formed scores


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, accs = {}, []
    for d in DATASETS:
        y = np.asarray(reference[f"y_{d}"]).astype(np.int64)
        P = need(outputs, "predict", f"proba_{d}", shape=(len(y), 2))
        out[f"accuracy_{d}"] = accuracy(y, P.argmax(1))
        out[f"auc_{d}"] = auc_binary(y == 1, P[:, 1] - P[:, 0])
        accs.append(out[f"accuracy_{d}"])
    out.update(mean_accuracy=float(np.mean(accs)), load_seconds=float(seconds["predict"]))
    return out
