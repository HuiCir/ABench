"""A small chunking policy: two camera encoders, proprioception and a task embedding -> H future actions."""
from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """128 x 128 RGB -> 256 features (four stride-2 convolutions, spatial softmax-free global pooling)."""

    def __init__(self, width: int = 32, out: int = 256):
        super().__init__()
        c = [3, width, 2 * width, 4 * width, 8 * width]
        layers = []
        for a, b in zip(c[:-1], c[1:]):
            layers += [nn.Conv2d(a, b, 3, 2, 1), nn.GroupNorm(8, b), nn.GELU()]
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Linear(8 * width, out)

    def forward(self, x):                       # x uint8 (B, 128, 128, 3)
        x = x.permute(0, 3, 1, 2).float() / 255.0 - 0.5
        return self.proj(self.conv(x).mean((2, 3)))


class ChunkPolicy(nn.Module):
    def __init__(self, n_tasks: int, horizon: int = 16, width: int = 32, hidden: int = 512):
        super().__init__()
        self.horizon = horizon
        self.agent, self.wrist = Encoder(width), Encoder(width)
        self.task = nn.Embedding(n_tasks, 64)
        self.state = nn.Linear(8, 64)
        self.head = nn.Sequential(nn.Linear(256 * 2 + 128, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Linear(hidden, horizon * 7))

    def forward(self, agent, wrist, state, task):
        h = torch.cat([self.agent(agent), self.wrist(wrist), self.task(task), self.state(state)], -1)
        return torch.tanh(self.head(h)).view(-1, self.horizon, 7)
