"""Reference submission: pre-RMSNorm with RMS normalisation of queries and keys (QK-norm).

Changes relative to the baseline (pre-RMSNorm, Zhang & Sennrich 2019):
1. Queries and keys are RMS-normalised per head after the rotary embedding, with a learned gain per head dimension
   (initialised to sqrt(head_dim) ** 0.5 so that the initial logit scale matches unnormalised attention at unit RMS).
   This bounds attention logits, whose growth Dehghani et al. (2023) and Wortsman et al. (2023) identify as a cause of
   loss spikes at high learning rates.
2. Nothing else changes: same residual form, same final norm.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6, init: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((d,), float(init)))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class PreRMSNormQK:
    def __init__(self, d_model: int, n_layers: int, n_heads: int, head_dim: int):
        self.d, self.hd = d_model, head_dim

    def make_layer(self, layer: int) -> nn.Module:
        g = math.sqrt(math.sqrt(self.hd))
        return nn.ModuleDict({"n1": RMSNorm(self.d), "n2": RMSNorm(self.d),
                              "q": RMSNorm(self.hd, init=g), "k": RMSNorm(self.hd, init=g)})

    def qk(self, mod, q, k, layer: int):
        return mod["q"](q), mod["k"](k)

    def layer_forward(self, mod, x, attn, mlp, layer: int):
        x = x + attn(mod["n1"](x))
        return x + mlp(mod["n2"](x))

    def make_final(self) -> nn.Module:
        return RMSNorm(self.d)


SCHEME = PreRMSNormQK
