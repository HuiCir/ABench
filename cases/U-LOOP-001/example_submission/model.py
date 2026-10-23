"""Reference submission: a prelude-core-coda recurrent-depth decoder (after Geiping et al. 2025) at the parameter
budget of a 4-layer transformer.

Layout: 2 prelude layers -> 1 core layer applied `loops` times -> 1 coda layer, all lm_model.Block layers of width 384.
Before every core iteration the state is combined with the prelude output e by a linear adapter on [h, e] (input
injection), so that later iterations still see the input. The state starts at h = e.
Training: each optimisation step draws loops uniformly from 1..8 (the same for the whole batch) and backpropagates
through all iterations, so the model is trained for every loop count it is evaluated at.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import lm_model as base

PRELUDE, CODA, MAX_TRAIN_LOOPS = 2, 1, 8


class Looped(nn.Module):
    def __init__(self, c: base.Config):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.prelude = nn.ModuleList(base.Block(c) for _ in range(PRELUDE))
        self.core = base.Block(c)
        self.coda = nn.ModuleList(base.Block(c) for _ in range(CODA))
        self.adapter = nn.Linear(2 * c.d_model, c.d_model, bias=False)
        self.norm = base.RMSNorm(c.d_model)
        depth = PRELUDE + CODA + 4                      # effective depth used for the residual init scale
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=c.init_std)
        for b in list(self.prelude) + [self.core] + list(self.coda):
            nn.init.normal_(b.attn.o.weight, std=c.init_std / math.sqrt(2 * depth))
            nn.init.normal_(b.mlp.down.weight, std=c.init_std / math.sqrt(2 * depth))
        with torch.no_grad():                           # adapter starts as "keep h" plus a small share of e
            self.adapter.weight.zero_()
            self.adapter.weight[:, : c.d_model] += torch.eye(c.d_model)
            self.adapter.weight[:, c.d_model:] += 0.1 * torch.eye(c.d_model)
        self._rope = None

    def forward(self, idx, loops: int = 4):
        if self._rope is None or self._rope[0].device != idx.device:
            self._rope = base.rope_cache(self.c.max_seq, self.c.d_model // self.c.n_heads, self.c.rope_base, idx.device)
        cos, sin = self._rope
        e = self.embed(idx)
        for b in self.prelude:
            e = b(e, cos, sin)
        h = e
        for _ in range(int(loops)):
            h = self.core(self.adapter(torch.cat([h, e], -1)), cos, sin)
        for b in self.coda:
            h = b(h, cos, sin)
        return F.linear(self.norm(h), self.embed.weight)

    def loss(self, idx, targets, loops: int | None = None):
        if loops is None:
            loops = int(torch.randint(1, MAX_TRAIN_LOOPS + 1, (1,)).item()) if self.training else 4
        logits = self(idx, loops)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))


def build(cfg):
    return Looped(cfg)
