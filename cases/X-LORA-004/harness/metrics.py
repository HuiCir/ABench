"""X-LORA-004 metrics: GLUE validation metrics per task (SST-2, MNLI-matched and QNLI accuracy, CoLA Matthews
correlation) and their average, under a cap on task-specific parameters."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, accuracy, binary_f1, matthews, need  # noqa: F401

TASKS = ("sst2", "cola", "mnli", "qnli")
PRIMARY = {"metric": "glue_average", "goal": "max"}
GATES = [{"metric": "max_params_per_task", "max": 1_000_000.0}]

# examples resampled together for the bootstrap 95% interval of the primary metric (abench_metrics.bootstrap_ci)
RESAMPLE = [{"reference": [f"{t}_label"], "outputs": [["eval", f"pred_{t}"]]} for t in TASKS]

def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out = {}
    for t in TASKS:
        y = np.asarray(reference[f"{t}_label"]).astype(np.int64)
        p = need(outputs, "eval", f"pred_{t}", shape=(len(y),), dtype_kind="iu")
        if t == "cola":
            out["cola"] = matthews(y, p)
        else:
            out[t] = accuracy(y, p)
    out["glue_average"] = float(np.mean([out[t] for t in TASKS]))
    counts = need(outputs, "eval", "params_per_task", shape=(len(TASKS),))
    out["max_params_per_task"] = float(counts.max())
    out["load_seconds"] = float(seconds["eval"])
    return out
