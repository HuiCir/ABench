"""A-MOE-003 language model with a pluggable feed-forward layer (evaluator code).

The model is B-OPT-001's μ2 decoder (8 layers, width 384, grouped-query attention with RoPE, pre-RMSNorm, tied
embeddings) with every MLP replaced by the submitted layer:

  ffn = FFN(d_model, layer, n_layers)          from moe.py
  y, aux = ffn(x)                              x, y: (B, T, d_model); aux: dict with optional entries
      "loss"    scalar tensor added to the training loss (auxiliary losses; their weight is yours)
      "counts"  (n_experts,) tokens routed to each expert in this call (for the load metrics)

Budgets, per layer (DENSE_FF = 1,024, the SwiGLU width of the dense μ2 model):
  total parameters   ≤ 8 · 3 · d_model · DENSE_FF  +  64 · d_model        (eight dense MLPs plus a router)
  active compute     forward FLOPs per token ≤ 2 · (dense MLP FLOPs per token) · 1.05   (top-2 of eight)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lm_model import Attention, Config, RMSNorm, rope_cache  # noqa: F401  (B-OPT-001 model components)

DENSE_FF = 1024


def dense_ffn_params(d: int) -> int:
    return 3 * d * DENSE_FF


def total_ffn_budget(d: int) -> int:
    return 8 * dense_ffn_params(d) + 64 * d


class Block(nn.Module):
    def __init__(self, c: Config, ffn: nn.Module):
        super().__init__()
        self.n1, self.attn, self.n2, self.ffn = RMSNorm(c.d_model), Attention(c), RMSNorm(c.d_model), ffn

    def forward(self, x, cos, sin, auxes: list):
        x = x + self.attn(self.n1(x), cos, sin)
        y, aux = self.ffn(self.n2(x))
        auxes.append(aux or {})
        return x + y


class LM(nn.Module):
    def __init__(self, c: Config, ffn_cls):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c, ffn_cls(c.d_model, i, c.n_layers)) for i in range(c.n_layers))
        self.norm = RMSNorm(c.d_model)
        for m in [self.embed] + [b.attn for b in self.blocks]:
            for mm in m.modules():
                if isinstance(mm, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(mm.weight, std=c.init_std)
        for b in self.blocks:
            nn.init.normal_(b.attn.o.weight, std=c.init_std / math.sqrt(2 * c.n_layers))
        self._rope = None
        self.last_aux: list = []

    def forward(self, idx):
        if self._rope is None or self._rope[0].device != idx.device:
            self._rope = rope_cache(self.c.max_seq, self.c.d_model // self.c.n_heads, self.c.rope_base, idx.device)
        cos, sin = self._rope
        x = self.embed(idx)
        auxes: list = []
        for b in self.blocks:
            x = b(x, cos, sin, auxes)
        self.last_aux = auxes
        return F.linear(self.norm(x), self.embed.weight)

    def loss(self, idx, targets):
        logits = self(idx)
        ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        extra = [a["loss"] for a in self.last_aux if a.get("loss") is not None]
        return ce + sum(extra) if extra and self.training else ce

    def ffn_parameters(self) -> list[int]:
        return [sum(p.numel() for p in b.ffn.parameters()) for b in self.blocks]
