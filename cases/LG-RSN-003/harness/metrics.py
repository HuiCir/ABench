"""LG-RSN-003 metrics: balanced accuracy per LegalBench task; primary = mean over all held-out tasks (the twenty
development tasks on new rows and fifteen tasks never shown), also per group: seen, unseen, unseen from a new family.
"""
from __future__ import annotations

import json

import numpy as np

import legaltasks as LT
from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "mean_balanced_accuracy", "goal": "max"}
GATES: list = []


def RESAMPLE(reference: dict) -> list:          # noqa: N802
    """Rows are resampled jointly with their task index (tasks keep their own labels and groups)."""
    return [{"reference": ["gold", "row_task"], "outputs": [["predict", "answer"]]}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    gold = [str(g) for g in np.asarray(reference["gold"])]
    rt = np.asarray(reference["row_task"]).astype(np.int64)
    ans = need(outputs, "predict", "answer", shape=(len(gold),), finite=False)
    labels = [json.loads(s) for s in np.asarray(reference["task_labels_json"])]
    group = [str(g) for g in np.asarray(reference["task_group"])]
    names = [str(n) for n in np.asarray(reference["task_name"])]
    out, by = {}, {}
    for k in range(len(labels)):
        idx = np.flatnonzero(rt == k)
        if not len(idx):
            continue
        g = [gold[i] for i in idx]
        p = [LT.match_label(ans[i], labels[k]) or "" for i in idx]
        ba = LT.balanced_accuracy(g, p)
        out[f"bacc_{names[k]}"] = ba
        by.setdefault(group[k], []).append(ba)
        by.setdefault("all", []).append(ba)
        out.setdefault("_invalid", 0)
        out["_invalid"] += sum(1 for x in p if not x)
    out["mean_balanced_accuracy"] = float(np.mean(by["all"]))
    for gname in ("seen", "unseen", "unseen_new_family"):
        if gname in by:
            out[f"mean_balanced_accuracy_{gname}"] = float(np.mean(by[gname]))
    if "unseen_new_family" in by and "unseen" in by:
        out["mean_balanced_accuracy_unseen_all"] = float(np.mean(by["unseen"] + by["unseen_new_family"]))
    out["invalid_answer_rate"] = float(out.pop("_invalid", 0)) / max(len(gold), 1)
    if "predict" in outputs and "ms_per_row" in outputs["predict"]:
        out["ms_per_row"] = float(np.asarray(outputs["predict"]["ms_per_row"])[0])
    out["load_seconds"] = float(seconds.get("predict", float("nan")))
    return out
