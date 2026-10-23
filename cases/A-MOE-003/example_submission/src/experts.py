"""Expert weights and token routing of the reference submission (the same routing code as the baseline's)."""
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
