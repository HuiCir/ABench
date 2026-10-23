"""Baseline: top-2 of 8 SwiGLU experts of width 1,024 (Shazeer et al. 2017; the Mixtral layout, no token dropping)
with the Switch Transformer load-balancing loss (Fedus et al. 2022), coefficient 0.01: E · Σ_e f_e · P_e, where f_e is
the fraction of routed assignments to expert e and P_e its mean router probability. Gate weights are the softmax over
the two selected logits."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Experts(nn.Module):
    def __init__(self, n: int, d: int, ff: int, std: float = 0.02):
        super().__init__()
        self.w_gate = nn.Parameter(torch.randn(n, d, ff) * std)
        self.w_up = nn.Parameter(torch.randn(n, d, ff) * std)
        self.w_down = nn.Parameter(torch.randn(n, ff, d) * std)

    def run(self, e: int, x: torch.Tensor) -> torch.Tensor:
        return (F.silu(x @ self.w_gate[e]) * (x @ self.w_up[e])) @ self.w_down[e]


def route_and_combine(experts: Experts, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, n: int):
    """x (N, d); idx, w (N, k): expert ids and weights. Each expert processes only the tokens routed to it."""
    out = torch.zeros_like(x)
    flat_idx, flat_w = idx.reshape(-1), w.reshape(-1)
    tok = torch.arange(x.shape[0], device=x.device).repeat_interleave(idx.shape[1])
    counts = torch.bincount(flat_idx, minlength=n)
    for e in range(n):
        sel = (flat_idx == e).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        t = tok[sel]
        out.index_add_(0, t, (experts.run(e, x[t]) * flat_w[sel, None]).to(out.dtype))
    return out, counts


class Top2(nn.Module):
    def __init__(self, d_model: int, layer: int, n_layers: int, n_experts: int = 8, ff: int = 1024, k: int = 2,
                 aux_coef: float = 0.01):
        super().__init__()
        self.n, self.k, self.aux_coef = n_experts, k, aux_coef
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.experts = Experts(n_experts, d_model, ff, std=0.02 / (2 * n_layers) ** 0.5)
        nn.init.normal_(self.experts.w_gate, std=0.02)
        nn.init.normal_(self.experts.w_up, std=0.02)

    def forward(self, x):
        B, T, d = x.shape
        h = x.reshape(-1, d)
        logits = self.router(h).float()
        top, idx = logits.topk(self.k, dim=-1)
        w = torch.softmax(top, dim=-1).to(h.dtype)
        y, counts = route_and_combine(self.experts, h, idx, w, self.n)
        aux = {"counts": counts}
        if self.training:
            p = torch.softmax(logits, dim=-1).mean(0)
            f = counts.float() / counts.sum()
            aux["loss"] = self.aux_coef * self.n * (f * p).sum()
        return y.reshape(B, T, d), aux


FFN = Top2
