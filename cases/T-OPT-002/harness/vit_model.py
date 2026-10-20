"""T-OPT-002 vision transformer used by the evaluator (fixed; you do not submit a model).

ViT for 32×32 images: 4×4 patches embedded by a linear map (so every weight matrix is 2D), class token, learned
position embeddings, 6 pre-LayerNorm blocks of width 192 with 3 heads and a GELU MLP of width 768, linear head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class Block(nn.Module):
    def __init__(self, d: int, h: int, f: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, f), nn.Linear(f, d)
        self.h = h

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class ViT(nn.Module):
    def __init__(self, d: int = 192, depth: int = 6, heads: int = 3, mlp: int = 768, patch: int = 4, n_classes: int = 10):
        super().__init__()
        self.patch = patch
        n = (32 // patch) ** 2
        self.embed = nn.Linear(3 * patch * patch, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads, mlp) for _ in range(depth))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)

    def forward(self, x):                         # x: float (B, 3, 32, 32), normalised
        B = x.shape[0]
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).reshape(B, -1, 3 * p * p)
        x = torch.cat([self.cls.expand(B, -1, -1), self.embed(x)], 1) + self.pos
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x[:, 0]))


def to_input(x_uint8_nhwc: torch.Tensor) -> torch.Tensor:
    """uint8 NHWC on the device -> normalised float NCHW."""
    x = x_uint8_nhwc.permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std
