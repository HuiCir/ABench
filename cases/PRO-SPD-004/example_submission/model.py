"""PRO-SPD-004 reference submission, model: GPT-2 Medium shape (24 layers, width 1,024, 16 heads, about 354M
parameters with tied embeddings), with rotary position embeddings, RMSNorm and a ReLU² MLP."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB, D, LAYERS, HEADS, SEQ = 50304, 1024, 24, 16, 1024
EVAL_SEQ_LEN = 1024


def rms(x):
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(self, dim: int, max_len: int = 65536):
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        f = torch.outer(torch.arange(max_len).float(), inv)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)

    def forward(self, x):                       # (B, H, T, Dh)
        T = x.size(-2)
        c, s = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, i: int):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.proj = nn.Linear(D, D, bias=False)
        self.fc = nn.Linear(D, 4 * D, bias=False)
        self.out = nn.Linear(4 * D, D, bias=False)
        self.rot = Rotary(D // HEADS)
        for w in (self.proj.weight, self.out.weight):
            nn.init.normal_(w, std=0.02 / math.sqrt(2 * LAYERS))

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(rms(x)).view(B, T, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        q, k = self.rot(rms(q)), self.rot(rms(k))           # QK-norm
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.relu(self.fc(rms(x))).square())


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, D)
        self.blocks = nn.ModuleList(Block(i) for i in range(LAYERS))
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)) and m.weight.std().item() > 0.05:
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.wte(idx)
        for b in self.blocks:
            x = b(x)
        return F.linear(rms(x), self.wte.weight)        # tied head


def build_model() -> nn.Module:
    return GPT()
