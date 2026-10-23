"""Baseline: the Vision Transformer of Dosovitskiy et al. (2021) with 16 x 16 patches, a class token, learned position
embeddings and pre-norm blocks: ViT-Ti/16 (width 192, 3 heads; small) and ViT-S/16 (width 384, 6 heads; large), 12
blocks with MLP ratio 4, as in DeiT (Touvron et al. 2021). At 160 pixels the sequence has 100 patches + 1 token."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d: int, heads: int, mlp: int = 4):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, mlp * d), nn.Linear(mlp * d, d)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class ViT(nn.Module):
    def __init__(self, d: int, heads: int, depth: int = 12, patch: int = 16, res: int = 160, num_classes: int = 10):
        super().__init__()
        self.patch = nn.Conv2d(3, d, patch, patch)
        n = (res // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        self.blocks = nn.Sequential(*[Block(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        return self.head(self.norm(self.blocks(x))[:, 0])


def build(variant: str, num_classes: int = 10):
    return ViT(192, 3, num_classes=num_classes) if variant == "small" else ViT(384, 6, num_classes=num_classes)
