"""E-ACT-001 metrics: average over episodes of the maximum reward (target-area coverage relative to the 95% success
threshold, as reported by Diffusion Policy for PushT) and success rate."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "mean_max_reward", "goal": "max"}
GATES: list = []          # validity = no crash, time limits respected, finite actions

# examples resampled together for the bootstrap 95% interval of the primary metric (abench_metrics.bootstrap_ci)
RESAMPLE = [{"reference": [], "outputs": [["rollout", "episodes"]]}]

def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    ep = need(outputs, "rollout", "episodes", shape=(None, 3))
    if len(ep) == 0:
        raise OutputError("no episodes were run")
    return {"mean_max_reward": float(ep[:, 0].mean()), "success_rate": float(ep[:, 1].mean()),
            "mean_episode_steps": float(ep[:, 2].mean()), "load_seconds": float(seconds["rollout"])}
