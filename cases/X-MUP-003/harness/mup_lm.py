"""X-MUP-003 language model of variable width with a pluggable parametrisation (evaluator code).

The architecture is B-OPT-001's decoder (pre-RMSNorm, grouped-query attention with RoPE, SwiGLU MLP, tied
embeddings) with 8 layers, head dimension 64 and MLP width 2.75 · width, at widths 128 (tuning), 512 and 1,024
(transfer targets). The submitted parametrisation (param.py, class PARAM) decides, per parameter tensor:

  PARAM.init_std(name, shape, width, base_width) -> float          standard deviation of N(0, σ²) at initialisation
  PARAM.lr_scale(name, shape, width, base_width) -> float          multiplier of the global learning rate η
  PARAM.wd_scale(name, shape, width, base_width) -> float          optional, multiplier of the weight decay (default 1)
and for the whole model
  PARAM.output_multiplier(width, base_width) -> float              multiplies the output logits
  PARAM.attention_scale(head_dim, width, base_width) -> float      optional, softmax scale (default 1 / sqrt(head_dim))

base_width is 128. Parameter names are those of `named_parameters()` below (embed.weight, blocks.<i>.attn.q.weight,
blocks.<i>.mlp.down.weight, blocks.<i>.n1.weight, norm.weight, ...); RMSNorm gains are 1-D.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lm_model import MLP, RMSNorm, apply_rope, rope_cache

BASE_WIDTH = 128
WIDTHS = (128, 512, 1024)


@dataclass
class Config:
    width: int = 128
    n_layers: int = 8
    head_dim: int = 64
    vocab_size: int = 50304
    max_seq: int = 1024
    rope_base: float = 10000.0

    @property
    def n_heads(self) -> int:
        return self.width // self.head_dim

    @property
    def n_kv_heads(self) -> int:
        return max(1, self.n_heads // 2)

    @property
    def d_ff(self) -> int:
        return int(round(2.75 * self.width / 32)) * 32

    @property
    def d_model(self) -> int:
        return self.width


class Attention(nn.Module):
    def __init__(self, c: Config, scale: float):
        super().__init__()
        self.h, self.kv, self.hd, self.scale = c.n_heads, c.n_kv_heads, c.head_dim, scale
        self.q = nn.Linear(c.width, self.h * self.hd, bias=False)
        self.k = nn.Linear(c.width, self.kv * self.hd, bias=False)
        self.v = nn.Linear(c.width, self.kv * self.hd, bias=False)
        self.o = nn.Linear(self.h * self.hd, c.width, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, c: Config, scale: float):
        super().__init__()
        self.n1, self.attn, self.n2, self.mlp = RMSNorm(c.width), Attention(c, scale), RMSNorm(c.width), MLP(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.mlp(self.n2(x))


class LM(nn.Module):
    def __init__(self, c: Config, param):
        super().__init__()
        self.c, self.param = c, param
        scale = float(param.attention_scale(c.head_dim, c.width, BASE_WIDTH)) if hasattr(param, "attention_scale") \
            else c.head_dim ** -0.5
        self.embed = nn.Embedding(c.vocab_size, c.width)
        self.blocks = nn.ModuleList(Block(c, scale) for _ in range(c.n_layers))
        self.norm = RMSNorm(c.width)
        self.out_mult = float(param.output_multiplier(c.width, BASE_WIDTH))
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.ndim >= 2:
                    p.normal_(0.0, float(param.init_std(name, tuple(p.shape), c.width, BASE_WIDTH)))
        self._rope = None

    def forward(self, idx):
        if self._rope is None or self._rope[0].device != idx.device:
            self._rope = rope_cache(self.c.max_seq, self.c.head_dim, self.c.rope_base, idx.device)
        cos, sin = self._rope
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x, cos, sin)
        return F.linear(self.norm(x), self.embed.weight) * self.out_mult

    def loss(self, idx, targets):
        logits = self(idx)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))

    def param_groups(self, weight_decay: float) -> list[dict]:
        """One AdamW group per parameter with its learning-rate multiplier ("lr_scale") and weight decay."""
        groups = []
        for name, p in self.named_parameters():
            sh, w = tuple(p.shape), self.c.width
            lr_s = float(self.param.lr_scale(name, sh, w, BASE_WIDTH))
            wd_s = float(self.param.wd_scale(name, sh, w, BASE_WIDTH)) if hasattr(self.param, "wd_scale") else 1.0
            groups.append({"params": [p], "lr_scale": lr_s, "weight_decay": weight_decay * wd_s if p.ndim >= 2 else 0.0})
        return groups
