"""C-KV-002 metrics: RULER-style string-match score of the answers (fraction of expected strings present) at each of
the three cache budgets (4.5, 2.5 and 2.0 stored bits per cached element); a prompt whose cache exceeded the budget
scores 0. Primary: the mean score over the three budgets (three points of the compression-accuracy curve). The score
with an uncompressed cache, and scores by task and context length, are reported."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401
import ruler_tasks

BUDGETS = (4.5, 2.5, 2.0)
PRIMARY = {"metric": "mean_budget_score", "goal": "max"}
GATES: list = []


def answers(o: dict, name: str) -> list[str]:
    b, off = np.asarray(o[f"{name}_answer_utf8"], np.uint8).tobytes(), np.asarray(o[f"{name}_answer_offsets"])
    return [b[off[i]:off[i + 1]].decode("utf-8", errors="replace") for i in range(len(off) - 1)]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    o = outputs["evaluate"]
    exp = [str(e).split("|") for e in np.asarray(reference["expected"])]
    task, length = np.asarray(reference["task"]), np.asarray(reference["length"])
    out, per_budget = {}, []
    for name, b in [("full", None)] + [(f"b{x}", x) for x in BUDGETS]:
        ans = answers(o, name)
        bits = np.asarray(o[f"{name}_bits"], np.float64)
        if len(ans) != len(exp) or len(bits) != len(exp):
            raise OutputError(f"evaluate.{name}: expected {len(exp)} answers")
        s = np.array([ruler_tasks.score(a, e) for a, e in zip(ans, exp)])
        within = np.ones(len(s), bool) if b is None else bits <= b + 1e-9
        s = np.where(within, s, 0.0)
        out[f"score_{name}"] = float(s.mean())
        out[f"max_bits_{name}"] = float(bits.max())
        out[f"within_budget_{name}"] = float(within.mean())
        for t in np.unique(task):
            out[f"score_{name}_{t}"] = float(s[task == t].mean())
        for L in np.unique(length):
            out[f"score_{name}_L{int(L)}"] = float(s[length == L].mean())
        if b is not None:
            per_budget.append(out[f"score_{name}"])
    out["mean_budget_score"] = float(np.mean(per_budget))
    out["gap_to_full"] = out["score_full"] - out["mean_budget_score"]
    return out
