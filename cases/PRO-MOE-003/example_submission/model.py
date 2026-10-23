"""Reference submission for PRO-MOE-003: a DeepSeekMoE-style decoder (Dai et al. 2024): 12 layers of width 768,
every feed-forward layer has 1 shared expert and 64 fine-grained routed experts (SwiGLU, hidden 512) of which each
token uses 6 (softmax gate, top-6), with an expert-level balance loss; RoPE attention, RMSNorm, untied output layer.
About 1.0B stored parameters, about 0.17B used per token."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EVAL_SEQ_LEN = 1024
CFG = dict(vocab=50257, d=768, layers=12, heads=12, experts=64, top_k=6, shared=1, hidden=512, seq=1024)


def rope(x, base: float = 10000.0):
    T, D = x.shape[-2], x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D)
    f = torch.outer(torch.arange(T, device=x.device, dtype=torch.float32), inv)
    cos, sin = f.cos().to(x.dtype), f.sin().to(x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.qkv, self.o = h, nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, C))


class Experts(nn.Module):
    """n SwiGLU experts stored as stacked weights."""

    def __init__(self, n, d, h):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(n, d, h) / math.sqrt(d))
        self.w3 = nn.Parameter(torch.randn(n, d, h) / math.sqrt(d))
        self.w2 = nn.Parameter(torch.randn(n, h, d) / math.sqrt(h))

    def one(self, e, x):
        return (F.silu(x @ self.w1[e]) * (x @ self.w3[e])) @ self.w2[e]


class MoE(nn.Module):
    def __init__(self, d, n, k, shared, h):
        super().__init__()
        self.k, self.n = k, n
        self.router = nn.Linear(d, n, bias=False)
        self.experts = Experts(n, d, h)
        self.shared = Experts(shared, d, h)
        self.aux = torch.zeros(())

    def forward(self, x):
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        probs = torch.softmax(self.router(flat).float(), -1)
        w, idx = probs.topk(self.k, -1)
        out = sum(self.shared.one(s, flat) for s in range(self.shared.w1.shape[0]))
        order = idx.reshape(-1).argsort()
        tok = torch.arange(flat.shape[0], device=x.device).repeat_interleave(self.k)[order]
        ws = w.reshape(-1)[order].to(x.dtype)
        counts = torch.bincount(idx.reshape(-1), minlength=self.n).tolist()
        routed, s = torch.zeros_like(flat), 0
        for e, c in enumerate(counts):
            if c:
                t = tok[s:s + c]
                routed.index_add_(0, t, self.experts.one(e, flat[t]) * ws[s:s + c, None])
            s += c
        if self.training:                     # expert-level balance loss (DeepSeekMoE): N/(K T) * sum_i f_i P_i
            f = torch.bincount(idx.reshape(-1), minlength=self.n).float() * self.n / (self.k * flat.shape[0])
            self.aux = (f * probs.mean(0)).sum()
        return (out + routed).view(B, T, C)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(c["d"]), nn.RMSNorm(c["d"])
        self.attn = Attention(c["d"], c["heads"])
        self.moe = MoE(c["d"], c["experts"], c["top_k"], c["shared"], c["hidden"])

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.moe(self.n2(x))


class MoELM(nn.Module):
    def __init__(self, c=CFG):
        super().__init__()
        self.emb = nn.Embedding(c["vocab"], c["d"])
        self.blocks = nn.ModuleList([Block(c) for _ in range(c["layers"])])
        self.norm, self.head = nn.RMSNorm(c["d"]), nn.Linear(c["d"], c["vocab"], bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))

    def aux_loss(self):
        return sum(b.moe.aux for b in self.blocks) / len(self.blocks)


def build_model() -> MoELM:
    return MoELM()
