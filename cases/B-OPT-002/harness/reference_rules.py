"""Functional update rules of the reference methods (for behavioral probes and code divergence).

Real training uses the official implementations listed in reference_pool.yaml; these simplified versions only need to match them in "update direction".
In the formal benchmark, SOAP, Sophia and Schedule-Free are wired into the probes by recording updates from their official implementations; they are not included in this file.
"""
from __future__ import annotations

from optim_api import FunctionalRule, Ops, newton_schulz5


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
        return -lr * (mh / (ops.sqrt(vh) + h["eps"]) + h["wd"] * p)


class CautiousAdamW(AdamW):
    """C-AdamW: keep only update components with the same sign as the current gradient, rescaled by the kept fraction."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.name = "cautious_adamw"

    def update(self, p, g, s, step, lr, ops):
        h = self.hp
        s["m"] = h["b1"] * s["m"] + (1 - h["b1"]) * g
        s["v"] = h["b2"] * s["v"] + (1 - h["b2"]) * g * g
        u = (s["m"] / (1 - h["b1"] ** step)) / (ops.sqrt(s["v"] / (1 - h["b2"] ** step)) + h["eps"])
        mask = (u * g > 0) * 1.0
        u = u * mask / (ops.mean(mask) + 1e-3)
        return -lr * (u + h["wd"] * p)


class Lion(FunctionalRule):
    def __init__(self, b1=0.9, b2=0.99, wd=0.1):
        super().__init__(name="lion", hp=dict(b1=b1, b2=b2, wd=wd))

    def init_state(self, p, ops):
        return {"m": ops.zeros_like(p)}

    def update(self, p, g, s, step, lr, ops):
        h = self.hp
        u = ops.sign(h["b1"] * s["m"] + (1 - h["b1"]) * g)
        s["m"] = h["b2"] * s["m"] + (1 - h["b2"]) * g
        return -lr * (u + h["wd"] * p)


class Adafactor(FunctionalRule):
    """Simplified: factored second moment + first-order momentum + update RMS clipping."""

    def __init__(self, b1=0.9, b2=0.95, eps=1e-30, clip=1.0, wd=0.1):
        super().__init__(name="adafactor", hp=dict(b1=b1, b2=b2, eps=eps, clip=clip, wd=wd))

    def init_state(self, p, ops):
        if ops.ndim(p) == 2:
            r, c = ops.shape(p)
            return {"m": ops.zeros_like(p), "vr": ops.zeros((r, 1), p), "vc": ops.zeros((1, c), p)}
        return {"m": ops.zeros_like(p), "v": ops.zeros_like(p)}

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
        u = u / ops.maximum(ops.rms(u) / h["clip"], 1.0)
        s["m"] = h["b1"] * s["m"] + (1 - h["b1"]) * u
        return -lr * (s["m"] + h["wd"] * p)


class AdEMAMix(FunctionalRule):
    def __init__(self, b1=0.9, b2=0.95, b3=0.9999, alpha=5.0, eps=1e-8, wd=0.1):
        super().__init__(name="ademamix", hp=dict(b1=b1, b2=b2, b3=b3, alpha=alpha, eps=eps, wd=wd))

    def init_state(self, p, ops):
        return {"m1": ops.zeros_like(p), "m2": ops.zeros_like(p), "v": ops.zeros_like(p)}

    def update(self, p, g, s, step, lr, ops):
        h = self.hp
        s["m1"] = h["b1"] * s["m1"] + (1 - h["b1"]) * g
        s["m2"] = h["b3"] * s["m2"] + (1 - h["b3"]) * g
        s["v"] = h["b2"] * s["v"] + (1 - h["b2"]) * g * g
        m1 = s["m1"] / (1 - h["b1"] ** step)
        v = s["v"] / (1 - h["b2"] ** step)
        return -lr * ((m1 + h["alpha"] * s["m2"]) / (ops.sqrt(v) + h["eps"]) + h["wd"] * p)


class MARSAdamW(FunctionalRule):
    def __init__(self, b1=0.95, b2=0.99, gamma=0.025, eps=1e-8, wd=0.1):
        super().__init__(name="mars", hp=dict(b1=b1, b2=b2, gamma=gamma, eps=eps, wd=wd))

    def init_state(self, p, ops):
        return {"m": ops.zeros_like(p), "v": ops.zeros_like(p), "g_prev": ops.zeros_like(p)}

    def update(self, p, g, s, step, lr, ops):
        h = self.hp
        c = g + h["gamma"] * h["b1"] / (1 - h["b1"]) * (g - s["g_prev"])
        cn = ops.norm(c)
        c = c / ops.maximum(cn, 1.0)
        s["g_prev"] = ops.copy(g)
        s["m"] = h["b1"] * s["m"] + (1 - h["b1"]) * c
        s["v"] = h["b2"] * s["v"] + (1 - h["b2"]) * c * c
        mh = s["m"] / (1 - h["b1"] ** step)
        vh = s["v"] / (1 - h["b2"] ** step)
        return -lr * (mh / (ops.sqrt(vh) + h["eps"]) + h["wd"] * p)


class Muon(FunctionalRule):
    """Muon (Nesterov momentum + Newton–Schulz orthogonalization + RMS matching from 2502.16982); non-matrix parameters fall back to AdamW."""

    def __init__(self, mu=0.95, wd=0.1):
        super().__init__(name="muon", hp=dict(mu=mu, wd=wd))
        self._fallback = AdamW(wd=wd)

    def init_state(self, p, ops):
        if ops.ndim(p) == 2:
            return {"m": ops.zeros_like(p)}
        return self._fallback.init_state(p, ops)

    def update(self, p, g, s, step, lr, ops):
        if ops.ndim(p) != 2:
            return self._fallback.update(p, g, s, step, lr, ops)
        h = self.hp
        s["m"] = h["mu"] * s["m"] + g
        nest = g + h["mu"] * s["m"]
        O = newton_schulz5(nest, ops)
        r, c = ops.shape(p)
        scale = 0.2 * max(r, c) ** 0.5
        return -lr * (scale * O + h["wd"] * p)


REFERENCE_RULES = {
    "adamw": AdamW,
    "cautious_adamw": CautiousAdamW,
    "lion": Lion,
    "adafactor": Adafactor,
    "ademamix": AdEMAMix,
    "mars": MARSAdamW,
    "muon": Muon,
}
