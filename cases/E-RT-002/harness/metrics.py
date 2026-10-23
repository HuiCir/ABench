"""E-RT-002 metrics: LIBERO-Spatial success under 0, 3 and 6 control steps of injected latency (plus the measured
planning time); primary: the success rate averaged over the three latencies."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

DELAYS = (0, 3, 6)
PRIMARY = {"metric": "mean_success", "goal": "max"}
GATES = [{"metric": "checkpoint_numbers", "max": 100_000_000}]


def RESAMPLE(reference: dict) -> list:          # noqa: N802
    return [{"reference": ["task_id"], "outputs": [["rollout", k] for k in ("success", "steps", "plans", "plan_ms", "mean_delay")]}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    n = len(np.asarray(reference["task_id"]))
    s = need(outputs, "rollout", "success", shape=(n, len(DELAYS)))
    if not np.isin(s, (0.0, 1.0)).all():
        raise OutputError("rollout.success must be 0 or 1")
    out = {"mean_success": float(s.mean())}
    for j, d in enumerate(DELAYS):
        out[f"success_d{d}"] = float(s[:, j].mean())
    out["latency_drop"] = out[f"success_d{DELAYS[0]}"] - out[f"success_d{DELAYS[-1]}"]
    for k in ("steps", "plans", "plan_ms", "mean_delay"):
        out[f"{k}_mean"] = float(np.mean(need(outputs, "rollout", k, shape=(n, len(DELAYS)))))
    task = np.asarray(reference["task_id"]).astype(int)
    for t in sorted(set(task.tolist())):
        out[f"success_task{t}"] = float(s[task == t].mean())
    if "checkpoint_numbers" in outputs.get("rollout", {}):
        out["checkpoint_numbers"] = float(np.asarray(outputs["rollout"]["checkpoint_numbers"])[0])
    out["load_seconds"] = float(seconds.get("rollout", float("nan")))
    return out
