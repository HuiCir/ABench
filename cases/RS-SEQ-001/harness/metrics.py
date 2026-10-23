"""RS-SEQ-001 metrics: full-ranking NDCG@10 and Recall@10 with one held-out item per user."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, ndcg_recall_at_k, need  # noqa: F401

PRIMARY = {"metric": "ndcg_at_10", "goal": "max"}
GATES: list = []          # validity = no crash, time limits respected, well-formed lists

# examples resampled together for the bootstrap 95% interval of the primary metric (abench_metrics.bootstrap_ci)
RESAMPLE = [{"reference": ["target"], "outputs": [["predict", "topk"]]}]

def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    target = np.asarray(reference["target"]).astype(np.int64)
    k = params["k"]
    top = need(outputs, "predict", "topk", shape=(len(target), k), dtype_kind="iu").astype(np.int64)
    if any(len(set(row)) != k for row in top.tolist()):
        raise OutputError("predict.topk: each row must contain k distinct item ids")
    ndcg, recall = ndcg_recall_at_k(target, top, k)
    return {"ndcg_at_10": ndcg, "recall_at_10": recall, "load_seconds": float(seconds["predict"]),
            "recommend_seconds": float(seconds.get("predict.finish", 0.0))}
