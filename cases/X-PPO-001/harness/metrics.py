"""X-PPO-001 metrics: per task, the mean undiscounted return of the trained policy over the 20 held-out episodes,
normalised as (R - R_random) / (R_upper - R_random), where R_random is the return of uniformly random actions from the
same initial states (computed here) and R_upper is a fixed upper bound of the task (control_envs.RETURN_UPPER).
Primary: the mean normalised return over the four tasks (per run; the judge averages over its 5 seeds)."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401
import control_envs as E

PRIMARY = {"metric": "mean_normalized_return", "goal": "max"}
GATES = [{"metric": "train_seconds", "max": 2400.0}, {"metric": "max_samples_ratio", "max": 1.0}]
BUDGET = 200_000


def random_return(name: str, init: np.ndarray) -> float:
    return float(E.rollout_returns(E.random_policy(12345), name, init).mean())


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    out, norms, ratios = {}, [], []
    budget = float((params or {}).get("samples_per_task", BUDGET))
    for name in E.ENVS:
        init = np.asarray(reference[f"init_{name}"])
        r = need(outputs, "train", f"returns_{name}", shape=(len(init),))
        lo, hi = random_return(name, init), E.RETURN_UPPER[name]
        out[f"return_{name}"] = float(r.mean())
        out[f"random_return_{name}"] = lo
        out[f"normalized_{name}"] = (float(r.mean()) - lo) / (hi - lo)
        norms.append(out[f"normalized_{name}"])
        ratios.append(float(need(outputs, "train", f"samples_{name}", shape=(1,))[0]) / budget)
    out["mean_normalized_return"] = float(np.mean(norms))
    out["max_samples_ratio"] = float(max(ratios))
    out["train_seconds"] = float(seconds["train"])
    return out


def relative(sub: dict, base: dict) -> dict:
    d = {f"normalized_{e}_delta_vs_ppo": sub[f"normalized_{e}"] - base[f"normalized_{e}"] for e in E.ENVS}
    d["mean_normalized_return_delta_vs_ppo"] = sub["mean_normalized_return"] - base["mean_normalized_return"]
    d["ppo_mean_normalized_return"] = base["mean_normalized_return"]
    return d
