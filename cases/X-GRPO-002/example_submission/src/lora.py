"""Low-rank adapters on the linear layers of a transformers causal LM: W x + (alpha / r) B A x, A Gaussian, B zero."""
import math

import torch

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x.to(self.A.dtype) @ self.A.T @ self.B.T * self.scale).to(x.dtype)


def attach(model, r: int = 16, alpha: float = 32.0):
    for p in model.parameters():
        p.requires_grad_(False)
    for name, mod in list(model.named_modules()):
        for t in TARGETS:
            child = getattr(mod, t, None)
            if isinstance(child, torch.nn.Linear):
                setattr(mod, t, LoRALinear(child, r, alpha).to(child.weight.device))
    return model


def state(model) -> dict:
    return {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}


def merge(model, tensors: dict, r: int = 16, alpha: float = 32.0):
    """Add the low-rank updates into the base weights in place (no adapter at inference)."""
    names = {n[:-2] for n in tensors}
    mods = dict(model.named_modules())
    for n in names:
        lin = mods[n.replace(".base", "")] if n.replace(".base", "") in mods else mods[n]
        A, B = tensors[n + ".A"].float(), tensors[n + ".B"].float()
        w = lin.weight
        w.data += ((alpha / r) * (B @ A)).to(w.dtype).to(w.device)
    return model
