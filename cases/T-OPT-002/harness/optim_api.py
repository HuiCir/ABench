"""ABench optimizer submission interface.

A submission implements only a "functional update rule": given (parameter, gradient, step, state), return the parameter increment.
The same rule code runs on numpy and, wrapped by TorchRuleOptimizer, in training.
Rules may use only the operators provided by Ops, so that both backends work.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict


# ---------------------------------------------------------------- backend operators
class Ops:
    """Minimal operator set shared by numpy and torch. Rule code does tensor operations only through it."""

    def __init__(self, backend: str = "numpy"):
        self.backend = backend
        if backend == "numpy":
            import numpy as xp
        elif backend == "torch":
            import torch as xp
        else:
            raise ValueError(backend)
        self.xp = xp

    # construction
    def zeros_like(self, x): return self.xp.zeros_like(x)
    def zeros(self, shape, like):
        if self.backend == "numpy":
            return self.xp.zeros(shape, dtype=like.dtype)
        return self.xp.zeros(shape, dtype=like.dtype, device=like.device)
    def ones(self, shape, like):
        if self.backend == "numpy":
            return self.xp.ones(shape, dtype=like.dtype)
        return self.xp.ones(shape, dtype=like.dtype, device=like.device)
    def copy(self, x): return x.copy() if self.backend == "numpy" else x.clone()

    # element-wise
    def sqrt(self, x): return self.xp.sqrt(x)
    def sign(self, x): return self.xp.sign(x)
    def abs(self, x): return self.xp.abs(x)
    def maximum(self, a, b): return self.xp.maximum(a, b)
    def where(self, c, a, b): return self.xp.where(c, a, b)

    # reductions
    def mean(self, x, axis=None, keepdims=False):
        if axis is None:
            return x.mean()
        if self.backend == "numpy":
            return x.mean(axis=axis, keepdims=keepdims)
        return x.mean(dim=axis, keepdim=keepdims)
    def sum(self, x): return x.sum()
    def norm(self, x): return self.xp.sqrt((x * x).sum())
    def rms(self, x): return self.xp.sqrt((x * x).mean())
    def scalar(self, x) -> float: return float(x)

    # matrices
    def T(self, x): return x.T
    def matmul(self, a, b): return a @ b
    def ndim(self, x) -> int: return x.ndim
    def shape(self, x): return tuple(x.shape)
    def numel(self, x) -> int: return int(x.size) if self.backend == "numpy" else int(x.numel())


# ---------------------------------------------------------------- rule base class
@dataclass
class FunctionalRule:
    """Common interface for all optimizers.

    Constraints (interface checks at submission time):
      - update must not modify p or g in place; state changes only through the returned new state or in-place updates of tensors in the state dict
      - total number of state tensor elements ≤ 3 × number of parameter elements
      - no access to any information other than (p, g, step, lr, state, hp)
    """
    name: str = "base"
    hp: Dict[str, Any] = field(default_factory=dict)
    owns_schedule: bool = False          # True means the rule has its own learning-rate schedule and the framework no longer multiplies by the schedule factor

    def init_state(self, p, ops: Ops) -> Dict[str, Any]:
        raise NotImplementedError

    def update(self, p, g, state: Dict[str, Any], step: int, lr: float, ops: Ops):
        """Return the parameter increment delta; the training framework executes p <- p + delta. step starts at 1."""
        raise NotImplementedError

    def eval_params(self, p, state, ops: Ops):
        """Parameters used at inference (Schedule-Free-type methods can return averaged parameters). Returned unchanged by default."""
        return p


# ---------------------------------------------------------------- common components
def newton_schulz5(G, ops: Ops, steps: int = 5, eps: float = 1e-7):
    """Quintic Newton–Schulz iteration used by Muon, approximating UV^T (coefficients from K. Jordan 2024)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (ops.norm(G) + eps)
    transposed = ops.shape(X)[0] > ops.shape(X)[1]
    if transposed:
        X = ops.T(X)
    for _ in range(steps):
        A = ops.matmul(X, ops.T(X))
        B = b * A + c * ops.matmul(A, A)
        X = a * X + ops.matmul(B, X)
    return ops.T(X) if transposed else X


def state_numel(state: Dict[str, Any], ops: Ops) -> int:
    def count(v):
        if hasattr(v, "shape") and getattr(v, "ndim", 0) > 0:
            return ops.numel(v)
        if isinstance(v, dict):
            return sum(count(x) for x in v.values())
        if isinstance(v, (list, tuple)):
            return sum(count(x) for x in v)
        return 0
    return count(state)


# ---------------------------------------------------------------- torch wrapper (for real training)
def make_torch_optimizer(rule: FunctionalRule, params, lr: float):
    """Wrap a FunctionalRule as a torch.optim.Optimizer.

    Note: the training framework uses torch 2.6 and calls the rule through this wrapper.
    """
    import torch

    ops = Ops("torch")

    class TorchRuleOptimizer(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, dict(lr=lr))

        @torch.no_grad()
        def step(self, closure=None):
            loss = closure() if closure is not None else None
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    if "rule_state" not in st:
                        st["rule_state"] = rule.init_state(p.detach().float(), ops)
                        st["step"] = 0
                    st["step"] += 1
                    delta = rule.update(p.detach().float(), p.grad.detach().float(),
                                        st["rule_state"], st["step"], group["lr"], ops)
                    p.add_(delta.to(p.dtype))
            return loss

    return TorchRuleOptimizer(params)


def cosine_lr(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
