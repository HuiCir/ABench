"""Control submission (negative): Muon renamed, with variables renamed and a different momentum coefficient. The divergence threshold should stop it."""
from __future__ import annotations

from optim_api import FunctionalRule, newton_schulz5


class OrthoMomentumPro(FunctionalRule):
    def __init__(self, beta=0.9, decay=0.05):
        super().__init__(name="orthomomentum_pro", hp=dict(beta=beta, decay=decay))

    def init_state(self, w, ops):
        if ops.ndim(w) == 2:
            return {"buf": ops.zeros_like(w)}
        return {"m1": ops.zeros_like(w), "m2": ops.zeros_like(w)}

    def update(self, w, grad, st, k, lr, ops):
        if ops.ndim(w) != 2:
            st["m1"] = 0.9 * st["m1"] + 0.1 * grad
            st["m2"] = 0.95 * st["m2"] + 0.05 * grad * grad
            return -lr * ((st["m1"] / (1 - 0.9 ** k)) / (ops.sqrt(st["m2"] / (1 - 0.95 ** k)) + 1e-8)
                          + self.hp["decay"] * w)
        st["buf"] = self.hp["beta"] * st["buf"] + grad
        look = grad + self.hp["beta"] * st["buf"]
        a, b = ops.shape(w)
        return -lr * (0.2 * max(a, b) ** 0.5 * newton_schulz5(look, ops) + self.hp["decay"] * w)


RULE = OrthoMomentumPro
