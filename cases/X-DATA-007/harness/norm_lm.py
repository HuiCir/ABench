"""A-NORM-005 language model with a pluggable normalisation and residual scheme (evaluator code).

The sublayers are fixed: grouped-query attention with rotary position embedding and a SwiGLU MLP, as in B-OPT-001,
tied input/output embeddings and the GPT-2 vocabulary padded to 50,304. Everything that decides how activations are
normalised and how the residual stream is formed comes from the submitted scheme (norm.py, class SCHEME):

  scheme = SCHEME(d_model, n_layers, n_heads, head_dim)
  scheme.make_layer(layer) -> nn.Module            parameters of one layer's normalisation (gains, gates, ...)
  scheme.layer_forward(mod, x, attn, mlp, layer)   the residual update of one layer; attn(h) and mlp(h) apply the
                                                   fixed sublayers to an input of shape (B, T, d)
  scheme.make_final() -> nn.Module                 applied to the residual stream before the output head
  scheme.qk(mod, q, k, layer) -> (q, k)            optional, on queries/keys (B, H, T, head_dim) after RoPE
  scheme.embed(mod_final, x) -> x                  optional, on the token embeddings
  scheme.init(model)                               optional, after the default initialisation (may rescale weights)
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


SCALES = {"mu1": Config(d_model=256, n_layers=4, n_heads=4, n_kv_heads=2, d_ff=704),
          "mu2": Config()}


def rope_cache(seq: int, dim: int, base: float, device):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    f = torch.outer(torch.arange(seq, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.shape[-2]], sin[: x.shape[-2]]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.h, self.kv, self.hd = c.n_heads, c.n_kv_heads, c.d_model // c.n_heads
        self.q = nn.Linear(c.d_model, c.n_heads * self.hd, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * self.hd, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(c.n_heads * self.hd, c.d_model, bias=False)

    def forward(self, x, cos, sin, qk=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        if qk is not None:
            q, k = qk(q, k)
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


class LM(nn.Module):
    def __init__(self, c: Config, scheme_cls):
        super().__init__()
        self.c = c
        self.scheme = scheme_cls(c.d_model, c.n_layers, c.n_heads, c.d_model // c.n_heads)
        self.embed = nn.Embedding(c.vocab_size, c.d_model)
        self.attn = nn.ModuleList(Attention(c) for _ in range(c.n_layers))
        self.mlp = nn.ModuleList(MLP(c) for _ in range(c.n_layers))
        self.norms = nn.ModuleList(self.scheme.make_layer(i) for i in range(c.n_layers))
        self.final = self.scheme.make_final()
        for m in list(self.attn.modules()) + list(self.mlp.modules()) + [self.embed]:
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=c.init_std)
        for a, m in zip(self.attn, self.mlp):         # scaled init of residual output projections
            nn.init.normal_(a.o.weight, std=c.init_std / math.sqrt(2 * c.n_layers))
            nn.init.normal_(m.down.weight, std=c.init_std / math.sqrt(2 * c.n_layers))
        if hasattr(self.scheme, "init"):
            self.scheme.init(self)
        self._rope = None

    def forward(self, idx):
        if self._rope is None or self._rope[0].device != idx.device:
            self._rope = rope_cache(self.c.max_seq, self.c.d_model // self.c.n_heads, self.c.rope_base, idx.device)
        cos, sin = self._rope
        sch = self.scheme
        x = self.embed(idx)
        if hasattr(sch, "embed"):
            x = sch.embed(self.final, x)
        for i in range(self.c.n_layers):
            mod = self.norms[i]
            qk = (lambda q, k, mod=mod, i=i: sch.qk(mod, q, k, i)) if hasattr(sch, "qk") else None
            attn = (lambda h, i=i, qk=qk: self.attn[i](h, cos, sin, qk))
            mlp = (lambda h, i=i: self.mlp[i](h))
            x = sch.layer_forward(mod, x, attn, mlp, i)
        return F.linear(self.final(x), self.embed.weight)

    def loss(self, idx, targets):
        logits = self(idx)
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))

    def scheme_parameters(self) -> int:
        return sum(p.numel() for p in list(self.norms.parameters()) + list(self.final.parameters()))
