"""E-VLA-003 metrics: LIBERO-Spatial and LIBERO-Object success of the policy on the frozen VLM (primary: the mean of
the two suites' success rates), real-time cost per control step, frozen-weight check, trained size."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

SUITES = ("libero_spatial", "libero_object")
PRIMARY = {"metric": "mean_success", "goal": "max"}
GATES = [{"metric": "vlm_unchanged", "min": 1.0}, {"metric": "checkpoint_numbers", "max": 100_000_000},
         {"metric": "ms_per_step", "max": 50.0}]


def RESAMPLE(reference: dict) -> list:          # noqa: N802
    return [{"reference": ["task_id", "suite_id"], "outputs": [["rollout", k] for k in ("success", "steps", "policy_ms")]}]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    suite = np.asarray(reference["suite_id"]).astype(int)
    n = len(suite)
    s = need(outputs, "rollout", "success", shape=(n,))
    if not np.isin(s, (0.0, 1.0)).all():
        raise OutputError("rollout.success must be 0 or 1")
    steps = need(outputs, "rollout", "steps", shape=(n,))
    ms = need(outputs, "rollout", "policy_ms", shape=(n,))
    out, per = {}, []
    for i, name in enumerate(SUITES):
        m = suite == i
        if m.any():
            out[f"success_{name}"] = float(s[m].mean())
            per.append(out[f"success_{name}"])
    out["mean_success"] = float(np.mean(per))
    out["ms_per_step"] = float(np.sum(ms * steps) / max(np.sum(steps), 1))
    out["steps_mean"] = float(steps.mean())
    task = np.asarray(reference["task_id"]).astype(int)
    for i, name in enumerate(SUITES):
        for t in sorted(set(task[suite == i].tolist())):
            out[f"success_{name}_task{t}"] = float(s[(suite == i) & (task == t)].mean())
    r = outputs.get("rollout", {})
    out["vlm_unchanged"] = float(np.asarray(r.get("vlm_unchanged", [0]))[0])
    if "checkpoint_numbers" in r:
        out["checkpoint_numbers"] = float(np.asarray(r["checkpoint_numbers"])[0])
    out["load_seconds"] = float(seconds.get("rollout", float("nan")))
    return out
