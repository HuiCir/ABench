"""RS-GRAPH-002 metrics: Recall@20 and NDCG@20 over all items as defined by NGCF and LightGCN. Per user with held-out
items T (padded with -1 in the reference): recall = |top20 ∩ T| / |T|; NDCG = DCG / IDCG with binary relevance,
DCG = Σ_{hits at rank r} 1 / log2(r + 1) and IDCG over min(|T|, 20) positions; both averaged over users."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "recall_at_20", "goal": "max"}
GATES: list = []          # validity = no crash, time limits respected, well-formed lists

# examples resampled together for the bootstrap 95% interval of the primary metric (abench_metrics.bootstrap_ci)
RESAMPLE = [{"reference": ["test"], "outputs": [["recommend", "topk"]]}]


def recall_ndcg(test: np.ndarray, top: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    test = np.asarray(test)
    n_rel = (test >= 0).sum(1)
    hit = np.zeros(top.shape, bool)
    for j in range(test.shape[1]):                      # test items per user, padded with -1
        col = test[:, j:j + 1]
        hit |= (top == col) & (col >= 0)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hit * disc).sum(1)
    idcg = np.array([disc[:min(n, k)].sum() for n in n_rel])
    return hit.sum(1) / np.maximum(n_rel, 1), dcg / np.maximum(idcg, 1e-12)


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    test = np.asarray(reference["test"]).astype(np.int64)
    k = params["k"]
    top = need(outputs, "recommend", "topk", shape=(len(test), k), dtype_kind="iu").astype(np.int64)
    srt = np.sort(top, axis=1)
    if (srt[:, 1:] == srt[:, :-1]).any():
        raise OutputError("recommend.topk: each row must contain k distinct item ids")
    rec, ndcg = recall_ndcg(test, top, k)
    return {"recall_at_20": float(rec.mean()), "ndcg_at_20": float(ndcg.mean()),
            "load_seconds": float(seconds["recommend"]), "recommend_seconds": float(seconds.get("recommend.finish", 0.0))}
