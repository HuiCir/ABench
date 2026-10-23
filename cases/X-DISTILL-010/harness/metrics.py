"""X-DISTILL-010 metrics: the student's held-out loss (primary), the fraction of the gap between the starting
student and the teacher that training closed, and the architecture check."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "heldout_nll", "goal": "min"}
GATES = [{"metric": "same_architecture", "min": 1.0}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    nll = float(need(outputs, "eval", "nll", shape=(1,))[0])
    s = float(need(outputs, "eval", "nll_start", shape=(1,))[0])
    t = float(need(outputs, "eval", "nll_teacher", shape=(1,))[0])
    return {"heldout_nll": nll, "start_nll": s, "teacher_nll": t, "improvement": s - nll,
            "gap_closed": (s - nll) / (s - t) if s - t > 1e-9 else float("nan"),
            "same_architecture": float(need(outputs, "eval", "same_architecture", shape=(1,))[0])}
