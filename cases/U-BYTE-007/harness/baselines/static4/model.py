"""Baseline: a byte-level local-global model with static patches of 4 bytes (MegaByte, Yu et al. 2023), at the sizes
of the case: local encoder and decoder of one block of width 256 (windowed attention over 256 bytes), patch embedding
by concatenating the 4 encoder states, global transformer of 8 blocks of width 384 over patches.
logits[:, j] predicts byte j from the bytes before it."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import lm_model as base
from byte_blocks import Blocks

LOCAL = base.Config(vocab_size=257, d_model=256, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=704, max_seq=4096)
GLOBAL = base.Config(vocab_size=1, d_model=384, n_layers=8, n_heads=6, n_kv_heads=2, d_ff=1024, max_seq=4096)
P, BOS = 4, 256


class Static(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_enc = nn.Embedding(257, LOCAL.d_model)
        self.enc = Blocks(LOCAL, 1, window=256)
        self.to_patch = nn.Linear(P * LOCAL.d_model, GLOBAL.d_model, bias=False)
        self.bos = nn.Parameter(torch.zeros(GLOBAL.d_model))
        self.glob = Blocks(GLOBAL, 8)
        self.up = nn.Linear(GLOBAL.d_model, LOCAL.d_model, bias=False)
        self.emb_dec = nn.Embedding(257, LOCAL.d_model)
        self.dec = Blocks(LOCAL, 1, window=256)
        self.norm = base.RMSNorm(LOCAL.d_model)
        self.head = nn.Linear(LOCAL.d_model, 256, bias=False)
        for e in (self.emb_enc, self.emb_dec):
            nn.init.normal_(e.weight, std=0.02)
        for lin in (self.to_patch, self.up, self.head):
            nn.init.normal_(lin.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pad = (-T) % P
        xp = F.pad(x, (0, pad), value=0) if pad else x
        h = self.enc(self.emb_enc(xp.long()))
        n = xp.shape[1] // P
        patches = self.to_patch(h.reshape(B, n, P * LOCAL.d_model))
        g = self.glob(torch.cat([self.bos.expand(B, 1, -1), patches[:, :-1]], 1))     # g_p sees patches < p
        up = self.up(g).repeat_interleave(P, dim=1)[:, :T]
        prev = torch.cat([torch.full((B, 1), BOS, device=x.device, dtype=torch.long), x[:, :-1].long()], 1)
        d = self.dec(self.emb_dec(prev) + up)
        return self.head(self.norm(d))

    def loss(self, x):
        logits = self(x)
        return F.cross_entropy(logits.float().reshape(-1, 256), x.reshape(-1).long())


def build():
    return Static()
