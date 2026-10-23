"""Baseline scheme: pre-normalisation with RMSNorm (Zhang & Sennrich 2019), as in Llama and the B-OPT-001 model:
x <- x + attn(RMSNorm(x)); x <- x + mlp(RMSNorm(x)); RMSNorm before the output head."""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class PreRMSNorm:
    def __init__(self, d_model: int, n_layers: int, n_heads: int, head_dim: int):
        self.d = d_model

    def make_layer(self, layer: int) -> nn.Module:
        return nn.ModuleDict({"n1": RMSNorm(self.d), "n2": RMSNorm(self.d)})

    def layer_forward(self, mod, x, attn, mlp, layer: int):
        x = x + attn(mod["n1"](x))
        return x + mlp(mod["n2"](x))

    def make_final(self) -> nn.Module:
        return RMSNorm(self.d)


SCHEME = PreRMSNorm
