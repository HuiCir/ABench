"""X-DATA-007 metrics: held-out loss of the fixed model on educational text after training on your selection
(primary), loss on general web text, duplicates in the selection, and the pipeline's CPU time."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "target_loss", "goal": "min"}
GATES = [{"metric": "diverged", "max": 0.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    sel = need(outputs, "select", "selection", shape=(None,)).astype(np.int64)
    out = {"target_loss": float(need(outputs, "train", "target_loss", shape=(1,))[0]),
           "general_loss": float(need(outputs, "train", "general_loss", shape=(1,))[0]),
           "diverged": float(need(outputs, "train", "diverged", shape=(1,))[0]),
           "selected_tokens": float(need(outputs, "select", "selected_tokens", shape=(1,))[0]),
           "selected_documents": float(len(sel)), "distinct_documents": float(len(np.unique(sel))),
           "select_cpu_seconds": float(need(outputs, "select", "select_cpu_seconds", shape=(1,))[0])}
    if "cluster" in reference:                       # planted near-duplicate clusters (held-out evaluation only)
        cl = np.asarray(reference["cluster"]).astype(np.int64)
        u = np.unique(sel)
        c = cl[u]
        c = c[c >= 0]
        if len(c):
            _, cnt = np.unique(c, return_counts=True)
            out["planted_duplicates_kept"] = float(np.sum(cnt - 1) / max(np.sum(cl >= 0) - len(np.unique(cl[cl >= 0])), 1))
        else:
            out["planted_duplicates_kept"] = 0.0
        out["duplicate_rate"] = float(np.sum(np.maximum(cnt - 1, 0)) / len(u)) if len(c) else 0.0
    return out
