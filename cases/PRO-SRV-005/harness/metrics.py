"""PRO-SRV-005 metrics: the highest request rate (requests per second) at which at least 90% of requests meet the
latency SLO (TTFT <= 2 s, TPOT <= 100 ms), interpolated on the rate ladder and averaged over the two traces
(primary); greedy consistency of the answers as a gate."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

TRACES, LADDER, ATTAIN = ("conv", "code"), (1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0), 0.90
PRIMARY = {"metric": "sustained_rate", "goal": "max", "aggregate": "median over runs"}
GATES = [{"metric": "consistency", "min": 0.97}, {"metric": "check_complete", "min": 1.0}]


def max_factor(att: np.ndarray) -> float:
    """Largest rate factor with attainment >= ATTAIN, linearly interpolated to the first factor below it."""
    att = np.asarray(att, np.float64)
    good = [i for i in range(len(att)) if att[i] >= ATTAIN]
    if not good:
        return 0.0
    i = good[-1]
    if i + 1 < len(att):
        a0, a1 = att[i], att[i + 1]
        return float(LADDER[i] + (LADDER[i + 1] - LADDER[i]) * (a0 - ATTAIN) / max(a0 - a1, 1e-9))
    return float(LADDER[i])


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, rates, comp = {}, [], []
    for t in TRACES:
        att = need(outputs, "serve", f"{t}_attainment")
        native = float(need(outputs, "serve", f"{t}_native_rate", shape=(1,))[0])
        out[f"{t}_max_factor"] = max_factor(att)
        out[f"{t}_rate"] = out[f"{t}_max_factor"] * native
        out[f"{t}_peak_tokens_per_s"] = float(np.max(need(outputs, "serve", f"{t}_tokens_per_s")))
        rates.append(out[f"{t}_rate"])
        comp.append(float(need(outputs, "serve", f"{t}_check_complete", shape=(1,))[0]))
    out["sustained_rate"] = float(np.mean(rates))
    out["check_complete"] = float(min(comp))
    c = need(outputs, "check", "consistent")
    out["consistency"] = float(np.mean(c)) if len(c) else 0.0
    out["startup_seconds"] = float(seconds.get("serve", float("nan")))
    return out
