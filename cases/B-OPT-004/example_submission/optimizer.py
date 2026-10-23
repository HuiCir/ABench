"""B-OPT-004 reference submission: LAMB (You et al. 2020) as a functional rule.

An Adam direction with decoupled weight decay is rescaled per parameter tensor by the trust ratio ||p|| / ||update||,
so every layer moves by a fixed fraction of its norm regardless of batch size; 1D parameters (norm gains) use the
plain Adam direction. Together with square-root learning-rate scaling in batch size (lr_batch_exponent: 0.5 in
hparam_space.yaml) this is the anchor's large-batch recipe.
"""
from __future__ import annotations

from optim_api import FunctionalRule


class LAMB(FunctionalRule):
    def __init__(self, b1=0.9, b2=0.99, eps=1e-6, wd=0.1, max_trust=10.0):
        super().__init__(name="lamb", hp=dict(b1=b1, b2=b2, eps=eps, wd=wd, max_trust=max_trust))

    def init_state(self, p, ops):
        return {"m": ops.zeros_like(p), "v": ops.zeros_like(p)}

    def update(self, p, g, s, step, lr, ops):
        h = self.hp
        s["m"] = h["b1"] * s["m"] + (1 - h["b1"]) * g
        s["v"] = h["b2"] * s["v"] + (1 - h["b2"]) * g * g
        mh = s["m"] / (1 - h["b1"] ** step)
        vh = s["v"] / (1 - h["b2"] ** step)
        u = mh / (ops.sqrt(vh) + h["eps"])
        if ops.ndim(p) < 2:
            return -lr * u
        u = u + h["wd"] * p
        pn, un = ops.scalar(ops.norm(p)), ops.scalar(ops.norm(u))
        trust = min(pn / un, h["max_trust"]) if pn > 0 and un > 0 else 1.0
        return -lr * trust * u


RULE = LAMB
