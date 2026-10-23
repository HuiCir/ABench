"""LegalBench tasks as the pipeline sees them (evaluator code, agent-visible; LG-RSN-003).

A task is a dict:
  name            LegalBench task name
  prompt_template the task's LegalBench base prompt: instructions and few-shot examples, ending in a query with
                  {{column}} slots for the row to classify
  labels          the possible answers, exactly as they must be returned (case and spacing are normalised when
                  scoring: "yes", " Yes" and "YES" all match "Yes")
  columns         the input columns of a row
  train           the task's LegalBench training rows ({column: value, ..., "answer": label}), usually 2-8
A row is {column: value} for the columns of its task.

Score of a task: balanced accuracy (mean recall over the labels present in its test rows), LegalBench's measure for
classification. An answer that matches no label counts as wrong.
"""
from __future__ import annotations

import re

SLOT = re.compile(r"\{\{(.*?)\}\}")


def fill(template: str, row: dict) -> str:
    """The prompt for one row: every {{column}} slot replaced by the row's value."""
    return SLOT.sub(lambda m: str(row.get(m.group(1), "")), template)


def norm(s) -> str:
    return " ".join(str(s).split()).lower()


def match_label(answer, labels: list) -> str | None:
    a = norm(answer)
    for l in labels:
        if norm(l) == a:
            return l
    return None


def balanced_accuracy(gold: list, pred: list) -> float:
    labs = sorted(set(gold))
    rec = []
    for l in labs:
        idx = [i for i, g in enumerate(gold) if g == l]
        rec.append(sum(1 for i in idx if pred[i] == l) / len(idx))
    return sum(rec) / len(rec) if rec else float("nan")
