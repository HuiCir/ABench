"""A-LCTX-006 language model (fixed except for positions): B-OPT-001's μ2 decoder (RMSNorm, grouped-query
attention, SwiGLU MLP, tied embeddings, GPT-2 vocabulary padded to 50,304) in which every attention layer hands its
queries and keys to a positions module before attention: q, k = positions(q, k), with q (B, H, T, D) and k
(B, KV, T, D). The module is built once per model by make_positions(head_dim, n_heads, n_kv_heads, train_len) and
shared by all layers; it sees the current length T from the tensors and may change its behaviour beyond train_len.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 50304
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ff: int = 1024
    max_seq: int = 1024
    rope_base: float = 10000.0
    init_std: float = 0.02


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rope_cache(seq: int, dim: int, base: float, device):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):                       # x: (B, H, T, D)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.shape[-2]], sin[: x.shape[-2]]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.h, self.kv, self.hd = c.n_heads, c.n_kv_heads, c.d_model // c.n_heads
        self.q = nn.Linear(c.d_model, c.n_heads * self.hd, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * self.hd, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(c.n_heads * self.hd, c.d_model, bias=False)

    def forward(self, x, pos):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = pos(q, k)
        q, k = q.to(v.dtype), k.to(v.dtype)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.up = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n1, self.attn, self.n2, self.mlp = RMSNorm(c.d_model), Attention(c), RMSNorm(c.d_model), MLP(c)

    def forward(self, x, pos):
        x = x + self.attn(self.n1(x), pos)
        return x + self.mlp(self.n2(x))


class LM(nn.Module):
    def __init__(self, c: Config, make_positions, train_len: int = 1024):
        super().__init__()
        self.c = c
        self.pos = make_positions(c.d_model // c.n_heads, c.n_heads, c.n_kv_heads, train_len)
        self._pos_ids = {id(m) for m in self.pos.modules()} if isinstance(self.pos, nn.Module) else set()
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.d_model)
        self.apply(self._init)
        for b in self.blocks:                       # scaled init of residual output projections
            nn.init.normal_(b.attn.o.weight, std=c.init_std / math.sqrt(2 * c.n_layers))
            nn.init.normal_(b.mlp.down.weight, std=c.init_std / math.sqrt(2 * c.n_layers))

    def _init(self, m):
        if id(m) in self._pos_ids:                  # the positions module initialises its own parameters
            return
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=self.c.init_std)

    def forward(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x, self.pos)
        return F.linear(self.norm(x), self.embed.weight)        # tied output head

    def loss(self, idx, targets):
        logits = self(idx)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
