"""B-OPT-003 reference submission: Adafactor without momentum (Shazeer & Stern 2018) as a functional rule.

2D parameters keep only the row and column means of the squared gradient (r + c numbers for an r x c matrix) and
precondition by their rank-1 reconstruction; the update is clipped to unit RMS before the learning rate is applied.
1D parameters keep an unfactored second moment (their size is negligible). With no first moment the state is far
below the 1x cap; the room up to 1x is what entries can spend on better preconditioning.
"""
from __future__ import annotations

from optim_api import FunctionalRule


class Adafactor(FunctionalRule):
    def __init__(self, b2=0.95, eps=1e-30, clip=1.0, wd=0.1):
        super().__init__(name="adafactor", hp=dict(b2=b2, eps=eps, clip=clip, wd=wd))

    def init_state(self, p, ops):
        if ops.ndim(p) == 2:
            r, c = ops.shape(p)
            return {"vr": ops.zeros((r, 1), p), "vc": ops.zeros((1, c), p)}
        return {"v": ops.zeros_like(p)}

    def update(self, p, g, s, step, lr, ops):
        h = self.hp
        g2 = g * g + h["eps"]
        bc = 1 - h["b2"] ** step
        if ops.ndim(p) == 2:
            s["vr"] = h["b2"] * s["vr"] + (1 - h["b2"]) * ops.mean(g2, axis=1, keepdims=True)
            s["vc"] = h["b2"] * s["vc"] + (1 - h["b2"]) * ops.mean(g2, axis=0, keepdims=True)
            v = (s["vr"] / bc) * (s["vc"] / bc) / (ops.mean(s["vr"] / bc) + 1e-30)
        else:
            s["v"] = h["b2"] * s["v"] + (1 - h["b2"]) * g2
            v = s["v"] / bc
        u = g / ops.sqrt(v)
        u = u / max(1.0, ops.scalar(ops.rms(u)) / h["clip"])
        wd = h["wd"] if ops.ndim(p) >= 2 else 0.0
        return -lr * (u + wd * p)


RULE = Adafactor
