"""Control submission: an early version of RankGate (linear gate, tau=0.15). The gate is almost always 0, so it effectively degenerates to Adafactor.

Original example description: RankGate, which gates between an "orthogonalized update" and a "factored adaptive update" according to the stable rank of the momentum.

(This is an example method constructed to exercise the pipeline; it has not been validated by real training.)

Design points (see design.md for details)
  - 2D parameters: compute both a Muon-style orthogonalized update O and an Adafactor-style update normalized by the factored second moment A,
    both RMS-matched, then sigmoid-gated on a log scale by the normalized stable rank r of the momentum matrix:
        delta = -lr * ( a(r)·O + (1-a(r))·A + wd·p ),  a(r) = sigmoid(kappa·(ln r - ln tau))
    Motivation: when the momentum is low-rank, orthogonalization flattens all singular directions and amplifies noise-dominated small ones; in that case fall back to the element-wise adaptive update.
  - Stable rank r = ||M||_F^2 / ||M||_2^2 / min(m, n); the spectral norm is estimated by a persistent single-vector power iteration (1 per step).
  - State: momentum m (1×) + row/column second moments (m+n) + power-iteration vector (n), far below the 3× cap.
  - 1D parameters (norm, bias) and embedding: AdamW.
"""
from __future__ import annotations

import math

from optim_api import FunctionalRule, newton_schulz5


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class RankGate(FunctionalRule):
    def __init__(self, mu=0.95, b2=0.95, tau=0.15, kappa=20.0, wd=0.1, eps=1e-30):
        super().__init__(name="rankgate",
                         hp=dict(mu=mu, b2=b2, tau=tau, kappa=kappa, wd=wd, eps=eps))

    # ---------------- state
    def init_state(self, p, ops):
        if ops.ndim(p) == 2:
            r, c = ops.shape(p)
            return {
                "m": ops.zeros_like(p),
                "vr": ops.zeros((r, 1), p),
                "vc": ops.zeros((1, c), p),
                "u": ops.ones((c, 1), p) / c ** 0.5,   # power-iteration vector
                "gate": ops.zeros((1,), p),            # most recent gate value, kept for analysis
            }
        return {"m": ops.zeros_like(p), "v": ops.zeros_like(p)}

    # ---------------- update
    def update(self, p, g, s, step, lr, ops):
        h = self.hp
        if ops.ndim(p) != 2:
            return self._adamw(p, g, s, step, lr, ops)

        # 1) Nesterov momentum
        s["m"] = h["mu"] * s["m"] + g
        M = g + h["mu"] * s["m"]

        # 2) orthogonalized branch (Muon style), RMS-matched to 0.2
        rows, cols = ops.shape(p)
        O = newton_schulz5(M, ops) * (0.2 * max(rows, cols) ** 0.5)

        # 3) factored adaptive branch (Adafactor-style row/column second moments), RMS-matched to 0.2
        g2 = g * g + h["eps"]
        s["vr"] = h["b2"] * s["vr"] + (1 - h["b2"]) * ops.mean(g2, axis=1, keepdims=True)
        s["vc"] = h["b2"] * s["vc"] + (1 - h["b2"]) * ops.mean(g2, axis=0, keepdims=True)
        bc = 1 - h["b2"] ** step
        v = (s["vr"] / bc) * (s["vc"] / bc) / (ops.mean(s["vr"] / bc) + 1e-30)
        A = M / ops.sqrt(v)
        A = A * (0.2 / (ops.scalar(ops.rms(A)) + 1e-12))

        # 4) stable-rank gate: spectral norm estimated by a persistent one-step power iteration
        w = ops.matmul(M, s["u"])
        u_new = ops.matmul(ops.T(M), w)
        s["u"] = u_new / (ops.norm(u_new) + 1e-12)
        sigma_max_sq = ops.scalar(ops.sum(w * w))          # old u is normalized, so ||M u||^2 ≈ sigma_max^2
        fro_sq = ops.scalar(ops.sum(M * M))
        r_norm = fro_sq / (sigma_max_sq + 1e-12) / min(rows, cols)
        a = _sigmoid(h["kappa"] * (r_norm - h["tau"]))
        s["gate"] = s["gate"] * 0 + a

        return -lr * (a * O + (1 - a) * A + h["wd"] * p)

    def _adamw(self, p, g, s, step, lr, ops, b1=0.9, eps=1e-8):
        h = self.hp
        s["m"] = b1 * s["m"] + (1 - b1) * g
        s["v"] = h["b2"] * s["v"] + (1 - h["b2"]) * g * g
        mh = s["m"] / (1 - b1 ** step)
        vh = s["v"] / (1 - h["b2"] ** step)
        return -lr * (mh / (ops.sqrt(vh) + eps) + h["wd"] * p)


RULE = RankGate   # the evaluation framework loads by this name
