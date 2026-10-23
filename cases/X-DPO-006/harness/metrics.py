"""X-DPO-006 metrics: win rate of your responses over the starting model's by a frozen reward model (primary),
length-controlled win rate, mean reward gain, KL from the starting model, implicit-reward accuracy on held-out
preference pairs, response length and trained numbers."""
from __future__ import annotations

import numpy as np

from abench_metrics import OutputError, need  # noqa: F401

PRIMARY = {"metric": "win_rate", "goal": "max"}
GATES: list = []
RESAMPLE = [{"outputs": [["judge", "reward_policy"], ["judge", "reward_base"], ["generate", "policy_words"],
                         ["generate", "base_words"], ["generate", "log_ratio_own"], ["generate", "own_tokens"]]},
            {"outputs": [["generate", "margin_pairs"]]}]


def length_controlled(win: np.ndarray, dlen: np.ndarray, iters: int = 200) -> float:
    """Win rate at equal length: logistic regression of win on log(policy words / base words), read at 0."""
    a, b = 0.0, 0.0
    for _ in range(iters):                               # Newton steps
        p = 1 / (1 + np.exp(-(a + b * dlen)))
        g = np.array([np.sum(win - p), np.sum((win - p) * dlen)])
        w = p * (1 - p) + 1e-9
        H = np.array([[w.sum(), (w * dlen).sum()], [(w * dlen).sum(), (w * dlen * dlen).sum()]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(1 / (1 + np.exp(-a)))


def score(outputs: dict, reference: dict, params: dict, seconds: dict) -> dict:
    rp = need(outputs, "judge", "reward_policy", shape=(None,), finite=False)
    rb = need(outputs, "judge", "reward_base", shape=(len(rp),), finite=False)
    pw = need(outputs, "generate", "policy_words", shape=(len(rp),))
    bw = need(outputs, "generate", "base_words", shape=(len(rp),))
    lr = need(outputs, "generate", "log_ratio_own", shape=(len(rp),))
    nt = need(outputs, "generate", "own_tokens", shape=(len(rp),))
    mg = need(outputs, "generate", "margin_pairs", shape=(None,))
    out = {"pair_accuracy": float(np.mean(mg > 0)), "kl_per_token": float(lr.sum() / max(nt.sum(), 1)),
           "kl_per_response": float(lr.mean()), "mean_words": float(pw.mean()), "length_ratio": float(pw.mean() / max(bw.mean(), 1e-9)),
           "trained_numbers": float(need(outputs, "generate", "checkpoint_numbers", shape=(1,))[0])}
    if np.isfinite(rp).all() and np.isfinite(rb).all():
        win = np.where(rp > rb, 1.0, np.where(rp == rb, 0.5, 0.0))
        out["win_rate"] = float(win.mean())
        out["reward_gain"] = float(np.mean(rp - rb))
        dl = np.log((pw + 1) / (bw + 1))
        out["win_rate_length_controlled"] = length_controlled((rp > rb).astype(float), dl)
    else:
        out["win_rate"] = float("nan")                  # self-check: no reward model in the sandbox
    return out
