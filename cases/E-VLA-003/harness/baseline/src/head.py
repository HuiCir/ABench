"""Action head: frozen-VLM features and proprioception -> a chunk of H actions."""
from __future__ import annotations

import torch
import torch.nn as nn


class Head(nn.Module):
    def __init__(self, feat: int = 1920, horizon: int = 8, hidden: int = 1024):
        super().__init__()
        self.horizon = horizon
        self.norm = nn.LayerNorm(feat)
        self.state = nn.Linear(8, 128)
        self.mlp = nn.Sequential(nn.Linear(feat + 128, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, horizon * 7))

    def forward(self, feat, state):
        return torch.tanh(self.mlp(torch.cat([self.norm(feat), self.state(state)], -1))).view(-1, self.horizon, 7)
