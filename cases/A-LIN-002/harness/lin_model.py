"""A-LIN-002 model used by the evaluator: embedding, two blocks of [RMSNorm → your Mixer → residual, RMSNorm → MLP →
residual], final RMSNorm and an untied output head. Width 128, 4 heads, MLP width 256, vocabulary 8,192."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL, N_HEADS, N_LAYERS, D_FF, VOCAB = 128, 4, 2, 256, 8192


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class Block(nn.Module):
    def __init__(self, mixer_cls, layer_idx: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(D_MODEL), RMSNorm(D_MODEL)
        self.mixer = mixer_cls(D_MODEL, N_HEADS, layer_idx)
        self.fc1, self.fc2 = nn.Linear(D_MODEL, D_FF), nn.Linear(D_FF, D_MODEL)

    def forward(self, x):
        x = x + self.mixer(self.n1(x))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class Model(nn.Module):
    def __init__(self, mixer_cls):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(Block(mixer_cls, i) for i in range(N_LAYERS))
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))
