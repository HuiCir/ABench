"""PH-WELL-002 metrics: VRMSE of 10-frame autoregressive rollouts on three systems, for held-out trajectories of the
training parameters and for unseen parameters (primary: rollout VRMSE averaged over the six system-set pairs)."""
from __future__ import annotations

import numpy as np

import wellsurr as W
from abench_metrics import OutputError, need  # noqa: F401

SETS = ("id", "unseen")
PAIRS = [(s, t) for s in W.SYSTEMS for t in SETS]
PRIMARY = {"metric": "mean_rollout_vrmse", "goal": "min"}
GATES = [{"metric": "checkpoint_numbers", "max": 100_000_000}]


def RESAMPLE(reference: dict) -> list:          # noqa: N802
    return [{"reference": [f"{s}_{t}_y"], "outputs": [["rollout", f"pred_{s}_{t}"]]} for s, t in PAIRS]


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, vals = {}, []
    for s, t in PAIRS:
        y = np.asarray(reference[f"{s}_{t}_y"])
        p = need(outputs, "rollout", f"pred_{s}_{t}", shape=y.shape, finite=False)
        v = W.vrmse(p, y)                                   # (n, steps)
        out[f"rollout_vrmse_{s}_{t}"] = float(v.mean())
        out[f"onestep_vrmse_{s}_{t}"] = float(v[:, 0].mean())
        out[f"final_vrmse_{s}_{t}"] = float(v[:, -1].mean())
        vals.append(out[f"rollout_vrmse_{s}_{t}"])
    out["mean_rollout_vrmse"] = float(np.mean(vals))
    for t in SETS:
        out[f"mean_rollout_vrmse_{t}"] = float(np.mean([out[f"rollout_vrmse_{s}_{t}"] for s in W.SYSTEMS]))
        out[f"mean_onestep_vrmse_{t}"] = float(np.mean([out[f"onestep_vrmse_{s}_{t}"] for s in W.SYSTEMS]))
    out["unseen_gap"] = out["mean_rollout_vrmse_unseen"] - out["mean_rollout_vrmse_id"]
    r = outputs.get("rollout", {})
    if "checkpoint_numbers" in r:
        out["checkpoint_numbers"] = float(np.asarray(r["checkpoint_numbers"])[0])
    out["load_seconds"] = float(seconds.get("rollout", float("nan")))
    return out
