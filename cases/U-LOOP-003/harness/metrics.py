"""U-LOOP-003 metrics: exact-match accuracy beyond the training lengths (21..40) and within them (1..20)."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

TASKS = ("parity", "addition", "copy")
PRIMARY = {"metric": "ood_accuracy", "goal": "max"}
GATES: list = []          # validity = no crash, time limits respected, well-formed outputs


def _expected_len(task: str, n: np.ndarray) -> np.ndarray:
    return {"parity": np.ones_like(n), "addition": n + 1, "copy": n}[task]

# examples resampled together for the bootstrap 95% interval of the primary metric (abench_metrics.bootstrap_ci)
RESAMPLE = [{"reference": [f"y_{t}", f"n_{t}"], "outputs": [["predict", f"pred_{t}"], ["predict", f"len_{t}"]]} for t in TASKS]

def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, ood, ind = {}, [], []
    for t in TASKS:
        y, n = np.asarray(reference[f"y_{t}"]), np.asarray(reference[f"n_{t}"])
        pred = need(outputs, "predict", f"pred_{t}", shape=y.shape)
        plen = need(outputs, "predict", f"len_{t}", shape=n.shape)
        exp = _expected_len(t, n)
        ok = (plen == exp) & np.all((pred == y) | (y == -1), axis=1)
        far = n > params["max_train_len"]
        out[f"ood_accuracy_{t}"] = float(ok[far].mean())
        out[f"id_accuracy_{t}"] = float(ok[~far].mean())
        ood.append(out[f"ood_accuracy_{t}"])
        ind.append(out[f"id_accuracy_{t}"])
    out.update(ood_accuracy=float(np.mean(ood)), id_accuracy=float(np.mean(ind)), load_seconds=float(seconds["predict"]))
    return out
