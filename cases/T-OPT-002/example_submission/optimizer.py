"""T-OPT-002 reference submission: Muon with update-RMS matching (the anchor's recipe) for matrices, AdamW otherwise.

2D parameters: Nesterov momentum, orthogonalised by the quintic Newton–Schulz iteration (Jordan et al. 2024), scaled by
0.2·sqrt(max(rows, cols)) so that the update RMS matches a typical AdamW update (Liu et al. 2025), plus decoupled weight
decay. 1D parameters (LayerNorm gains and biases): AdamW. This is the anchor transferred as-is to the vision backbone;
it is the starting point that entries should improve on.
"""
from __future__ import annotations

from optim_api import FunctionalRule, newton_schulz5


class MuonRMS(FunctionalRule):
    def __init__(self, mu=0.95, b1=0.9, b2=0.99, wd=0.05, eps=1e-8):
        super().__init__(name="muon_rms", hp=dict(mu=mu, b1=b1, b2=b2, wd=wd, eps=eps))

    def init_state(self, p, ops):
        if ops.ndim(p) == 2:
            return {"m": ops.zeros_like(p)}
        return {"m": ops.zeros_like(p), "v": ops.zeros_like(p)}

    def update(self, p, g, s, step, lr, ops):
        h = self.hp
        if ops.ndim(p) == 2:
            s["m"] = h["mu"] * s["m"] + g
            nesterov = g + h["mu"] * s["m"]
            rows, cols = ops.shape(p)
            o = newton_schulz5(nesterov, ops) * (0.2 * max(rows, cols) ** 0.5)
            return -lr * (o + h["wd"] * p)
        s["m"] = h["b1"] * s["m"] + (1 - h["b1"]) * g
        s["v"] = h["b2"] * s["v"] + (1 - h["b2"]) * g * g
        mh = s["m"] / (1 - h["b1"] ** step)
        vh = s["v"] / (1 - h["b2"] ** step)
        return -lr * mh / (ops.sqrt(vh) + h["eps"])


RULE = MuonRMS
