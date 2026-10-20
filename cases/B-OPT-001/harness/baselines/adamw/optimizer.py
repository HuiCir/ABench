"""Baseline run by the evaluator under the same conditions as every submission: AdamW (Loshchilov & Hutter 2019),
the decoupled-weight-decay form of the anchor Adam (Kingma & Ba 2015), as a functional rule."""
from __future__ import annotations

from optim_api import FunctionalRule


class AdamW(FunctionalRule):
    def __init__(self, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
        super().__init__(name="adamw", hp=dict(b1=b1, b2=b2, eps=eps, wd=wd))

    def init_state(self, p, ops):
        return {"m": ops.zeros_like(p), "v": ops.zeros_like(p)}

    def update(self, p, g, s, step, lr, ops):
        h = self.hp
        s["m"] = h["b1"] * s["m"] + (1 - h["b1"]) * g
        s["v"] = h["b2"] * s["v"] + (1 - h["b2"]) * g * g
        mh = s["m"] / (1 - h["b1"] ** step)
        vh = s["v"] / (1 - h["b2"] ** step)
        wd = h["wd"] if ops.ndim(p) >= 2 else 0.0
        return -lr * (mh / (ops.sqrt(vh) + h["eps"]) + wd * p)


RULE = AdamW
