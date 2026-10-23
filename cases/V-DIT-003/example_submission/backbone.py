"""Reference submission for V-DIT-003: DiT-S/2 with U-ViT's long skip connections (Bao et al. 2023) between block i
and block 11 - i (concatenation and a linear projection) and normalised queries and keys (QK-norm, Henry et al. 2020,
with LayerNorm per head); otherwise the baseline's adaLN-Zero DiT. The six skip projections add about 8% FLOPs (limit: 10%)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def sincos_2d(dim: int, grid: int) -> torch.Tensor:
    def one(d, pos):
        om = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2))
        o = np.einsum("m,d->md", pos.reshape(-1), om)
        return np.concatenate([np.sin(o), np.cos(o)], 1)
    gy, gx = np.meshgrid(np.arange(grid, dtype=np.float64), np.arange(grid, dtype=np.float64), indexing="ij")
    return torch.as_tensor(np.concatenate([one(dim // 2, gy), one(dim // 2, gx)], 1), dtype=torch.float32)


class TimestepEmbed(nn.Module):
    def __init__(self, dim: int, freq: int = 256):
        super().__init__()
        self.freq = freq
        self.mlp = nn.Sequential(nn.Linear(freq, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.freq // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        a = t.float()[:, None] * f[None]
        return self.mlp(torch.cat([torch.cos(a), torch.sin(a)], -1))


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class QKNormAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h, self.d = heads, dim // heads
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.qn, self.kn = nn.LayerNorm(self.d, eps=1e-6), nn.LayerNorm(self.d, eps=1e-6)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        o = torch.nn.functional.scaled_dot_product_attention(self.qn(q), self.kn(k), v)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, skip: bool = False):
        super().__init__()
        self.skip = nn.Linear(2 * dim, dim) if skip else None
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = QKNormAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(approximate="tanh"),
                                 nn.Linear(int(dim * mlp_ratio), dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, skip=None):
        if self.skip is not None:
            x = self.skip(torch.cat([x, skip], -1))
        s1, sc1, g1, s2, sc2, g2 = self.ada(c).chunk(6, -1)
        h = modulate(self.n1(x), s1, sc1)
        x = x + g1.unsqueeze(1) * self.attn(h)
        return x + g2.unsqueeze(1) * self.mlp(modulate(self.n2(x), s2, sc2))


class DiT(nn.Module):
    def __init__(self, in_ch=4, size=16, patch=2, dim=384, depth=12, heads=6, classes=10):
        super().__init__()
        self.patch, self.in_ch, self.grid = patch, in_ch, size // patch
        self.embed = nn.Conv2d(in_ch, dim, patch, patch)
        self.register_buffer("pos", sincos_2d(dim, self.grid)[None], persistent=False)
        self.t_embed = TimestepEmbed(dim)
        self.y_embed = nn.Embedding(classes + 1, dim)          # the last index is the null class
        self.blocks = nn.ModuleList([Block(dim, heads, skip=i >= depth // 2) for i in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, patch * patch * in_ch)
        for m in (self.ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t, y):
        B = x.shape[0]
        h = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        c = self.t_embed(t) + self.y_embed(y)
        half, saved = len(self.blocks) // 2, []
        for i, b in enumerate(self.blocks):
            if i < half:
                h = b(h, c)
                saved.append(h)
            else:
                h = b(h, c, saved.pop())
        sh, sc = self.ada(c).chunk(2, -1)
        h = self.out(modulate(self.norm(h), sh, sc))
        p, g = self.patch, self.grid
        h = h.view(B, g, g, p, p, self.in_ch).permute(0, 5, 1, 3, 2, 4)
        return h.reshape(B, self.in_ch, g * p, g * p)


def make_backbone(latent_shape=(4, 16, 16), num_classes: int = 10):
    return DiT(in_ch=latent_shape[0], size=latent_shape[1], classes=num_classes)
