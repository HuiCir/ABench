"""Behavioural screen of B-OPT-002 (evaluator code, agent-visible): how close a program's updates are to known
optimizers on a fixed stream of gradients, and whether it stays finite at extreme gradient scales.

A stream is a sequence of gradient matrices (NumPy) for one 2D parameter: a slowly rotating low-rank signal, Gaussian
noise and occasional heavy-tailed spikes (stream()). Every rule starts from zero state and the same parameter, and
sees the same gradients (the parameter is not updated, so the comparison is of update rules, not of trajectories).

similarity(rule, other) = mean over steps of the cosine between their updates.
max_similarity = the largest similarity to any rule of known_rules(): reference_rules.REFERENCE_RULES, SGD with
momentum and Signum, and AdamW, Lion, Signum and SGD with momentum over a grid of their momentum constants; a program
within 0.98 of an existing optimizer is a rediscovery, not a new rule.
"""
from __future__ import annotations

import numpy as np

from optim_api import FunctionalRule, Ops
from reference_rules import REFERENCE_RULES

SHAPE, STEPS = (64, 32), 200


class SGDM(FunctionalRule):
    def __init__(self, b=0.9):
        super().__init__(name="sgd_momentum", hp=dict(b=b))

    def init_state(self, p, ops):
        return {"m": ops.zeros_like(p)}

    def update(self, p, g, s, step, lr, ops):
        s["m"] = self.hp["b"] * s["m"] + g
        return -lr * s["m"]


class Signum(FunctionalRule):
    def __init__(self, b=0.9):
        super().__init__(name="signum", hp=dict(b=b))

    def init_state(self, p, ops):
        return {"m": ops.zeros_like(p)}

    def update(self, p, g, s, step, lr, ops):
        s["m"] = self.hp["b"] * s["m"] + (1 - self.hp["b"]) * g
        return -lr * ops.sign(s["m"])


def stream(seed: int, steps: int = STEPS, shape=SHAPE) -> list:
    rng = np.random.default_rng(seed)
    r = 4
    u, v = rng.standard_normal((shape[0], r)), rng.standard_normal((shape[1], r))
    out = []
    for t in range(steps):
        u += 0.05 * rng.standard_normal(u.shape)
        v += 0.05 * rng.standard_normal(v.shape)
        g = (u @ v.T) / np.sqrt(r) * 0.3 + rng.standard_normal(shape)
        if rng.random() < 0.05:
            g += rng.standard_t(2, size=shape) * 5
        out.append(g * np.exp(rng.normal(0, 0.3)))
    return out


def updates(rule: FunctionalRule, grads: list, p=None) -> list:
    ops = Ops("numpy")
    p = np.zeros_like(grads[0]) + 0.01 if p is None else p
    s = rule.init_state(p, ops)
    return [np.asarray(rule.update(p, g, s, t + 1, 1.0, ops), dtype=np.float64) for t, g in enumerate(grads)]


def similarity(a: list, b: list) -> float:
    c = [float(np.sum(x * y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)) for x, y in zip(a, b)]
    return float(np.mean(c))


def known_rules() -> dict:
    """The reference rules with their defaults, plus a grid of momentum constants for the simplest ones, so that a
    re-tuned copy of AdamW, Lion, Signum or SGD with momentum still counts as that optimizer."""
    rules = {k: cls() for k, cls in REFERENCE_RULES.items()}
    for b1 in (0.8, 0.9, 0.95, 0.98):
        rules[f"signum_b{b1}"] = Signum(b1)
        rules[f"sgd_momentum_b{b1}"] = SGDM(b1)
        for b2 in (0.9, 0.95, 0.99, 0.999):
            rules[f"adamw_b{b1}_{b2}"] = REFERENCE_RULES["adamw"](b1=b1, b2=b2)
            rules[f"lion_b{b1}_{b2}"] = REFERENCE_RULES["lion"](b1=b1, b2=b2)
    return rules


def screen(rule: FunctionalRule, seed: int) -> dict:
    grads = stream(seed)
    mine = updates(rule, grads)
    finite = all(np.all(np.isfinite(u)) for u in mine)
    sims = {k: similarity(mine, updates(r, grads)) if finite else float("nan") for k, r in known_rules().items()}
    descent = float(np.mean([np.sum(-u * g) / (np.linalg.norm(u) * np.linalg.norm(g) + 1e-30) for u, g in zip(mine, grads)]))
    stable = finite
    for scale in (1e-6, 1e6):
        ex = updates(rule, [g * scale for g in grads[:20]])
        stable = stable and all(np.all(np.isfinite(u)) for u in ex)
    return {"similarity": sims, "max_similarity": max(v for v in sims.values()) if finite else float("nan"),
            "closest": max(sims, key=lambda k: sims[k]) if finite else "", "descent_cosine": descent,
            "finite": float(finite), "stable_extreme_scales": float(stable)}
