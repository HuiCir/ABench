"""RidgeSign: magnitude-confidence gating found by search over frozen Lion primitives.

Search finding: Lion's sign update uses the same magnitude for small opposing noise and large signals; on the fixed
gradient stream, adding to sign a confidence = |z|/(|z|+tau) built only from the finite primitives
reduces jitter while keeping the low state cost of sign. z is a fast/slow EMA extrapolation, which avoids degenerating to
a single Lion EMA. This implementation corresponds exactly to the ridge_sign AST in search_log.jsonl.
"""
from __future__ import annotations

from optim_api import FunctionalRule, Ops


class RidgeSign(FunctionalRule):
    PROGRAM_NAME = "ridge_sign_slow"
    PROGRAM_ID = "2ab6fad54414"
    def __init__(self, beta_fast=0.95, beta_slow=0.999, alpha=0.70, tau=0.05, wd=0.1, eps=1e-12):
        super().__init__(name="ridge_sign", hp=dict(beta_fast=beta_fast, beta_slow=beta_slow,
                                                     alpha=alpha, tau=tau, wd=wd, eps=eps))

    def init_state(self, p, ops: Ops):
        return {"m_fast": ops.zeros_like(p), "m_slow": ops.zeros_like(p)}

    def update(self, p, g, s, step, lr, ops: Ops):
        h = self.hp
        s["m_fast"] = h["beta_fast"] * s["m_fast"] + (1.0 - h["beta_fast"]) * g
        s["m_slow"] = h["beta_slow"] * s["m_slow"] + (1.0 - h["beta_slow"]) * g
        z = s["m_fast"] + h["alpha"] * (s["m_fast"] - s["m_slow"])
        confidence = ops.abs(z) / (ops.abs(z) + h["tau"] + h["eps"])
        # The delta-sign filter found by search: temporarily down-weight coordinates that oppose the slow momentum.
        gate = 0.5 + 0.5 * ops.sign(s["m_fast"] - s["m_slow"])
        u = ops.sign(z) * confidence * gate
        return -lr * (u + h["wd"] * p)


RULE = RidgeSign
