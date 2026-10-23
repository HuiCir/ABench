"""FN-QA-004 metrics: execution accuracy on the held-out FinQA questions (primary), by number of reasoning steps and by
where the evidence is, program validity, and cost per question.

A question counts as answered by the submitted `answer` value; when that is empty, by the executed value of the
submitted program. Correctness is finqa.correct (FinQA's rounding to 5 decimals, small tolerance).
"""
from __future__ import annotations

import numpy as np

import finqa as F
from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "execution_accuracy", "goal": "max"}
GATES: list = []


def RESAMPLE(reference: dict) -> list:          # noqa: N802
    return [{"reference": ["gold", "steps", "has_table_evidence"],
             "outputs": [["answer", "answer"], ["answer", "program"], ["answer", "program_value"]]}]


def _gold(s: str):
    s = str(s)
    return s if s in ("yes", "no") else float(s)


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    gold = [_gold(g) for g in np.asarray(reference["gold"])]
    n = len(gold)
    ans = need(outputs, "answer", "answer", shape=(n,), finite=False)
    prog = need(outputs, "answer", "program", shape=(n,), finite=False)
    val = need(outputs, "answer", "program_value", shape=(n,), finite=False)
    ok = np.zeros(n, bool)
    for i, g in enumerate(gold):
        a = str(ans[i]) if str(ans[i]) else str(val[i])
        ok[i] = bool(a) and F.correct(a, g)
    has_prog = np.asarray([bool(str(p)) for p in prog])
    prog_ok = np.asarray([bool(str(v)) and F.correct(str(v), g) for v, g in zip(val, gold)])
    steps = np.asarray(reference["steps"])
    out = {"execution_accuracy": float(ok.mean()),
           "program_rate": float(has_prog.mean()),
           "program_valid_rate": float(np.mean([bool(str(v)) for v in val])),
           "program_accuracy": float(prog_ok.mean()),
           "accuracy_1_step": float(ok[steps == 1].mean()) if (steps == 1).any() else float("nan"),
           "accuracy_2plus_steps": float(ok[steps >= 2].mean()) if (steps >= 2).any() else float("nan")}
    tab = np.asarray(reference["has_table_evidence"]).astype(bool)
    if tab.any() and (~tab).any():
        out["accuracy_table_evidence"] = float(ok[tab].mean())
        out["accuracy_text_only_evidence"] = float(ok[~tab].mean())
    if "answer" in outputs and "ms_per_question" in outputs["answer"]:
        out["ms_per_question"] = float(np.asarray(outputs["answer"]["ms_per_question"])[0])
    out["load_seconds"] = float(seconds.get("answer", float("nan")))
    return out
