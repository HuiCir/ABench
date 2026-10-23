"""Reference submission: cross-layer attention (CLA, Brandon et al. 2024) with a sharing factor of 3.

Layers are grouped in threes (0-2, 3-5, ..., 27-29). The first layer of each group computes keys and values as before
(3 key-value heads) and caches them; the other two layers keep their own query and output projections but attend to
the keys and values of the group's first layer and cache nothing. Ten layers cache keys and values instead of thirty:
7,680 bytes per token in bf16, the budget, while every layer keeps three key-value heads. The shared projections start
from the first layer's pretrained weights; the other layers' key and value projections are dropped.
"""
from __future__ import annotations

import torch.nn as nn

import smol_lm

GROUP = 3


class Store:
    """Keys and values of the current group, passed from the producing layer to its consumers within one forward."""
    k = v = None


class Producer(nn.Module):
    def __init__(self, att: smol_lm.GQAttention, store: Store):
        super().__init__()
        self.att, self.store = att, store

    def forward(self, x, pos0: int = 0, cache=None):
        y, (k, v) = self.att(x, pos0, cache)
        self.store.k, self.store.v = k, v
        return y, (k, v)


class Consumer(nn.Module):
    def __init__(self, att: smol_lm.GQAttention, store: Store):
        super().__init__()
        self.c, self.store = att.c, store
        self.q_proj, self.o_proj = att.q_proj, att.o_proj

    def forward(self, x, pos0: int = 0, cache=None):
        c = self.c
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, c.n_heads, c.head_dim).transpose(1, 2)
        cos, sin = smol_lm.rope(T, c.head_dim, c.rope_theta, x.device, pos0)
        q = smol_lm.apply_rope(q, cos, sin)
        rep = c.n_heads // c.n_kv_heads
        k, v = self.store.k.repeat_interleave(rep, 1), self.store.v.repeat_interleave(rep, 1)
        y = smol_lm.F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1 and cache is None))
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1)), None


def build(base_dir: str, device: str):
    model = smol_lm.load_base(base_dir, device)
    store = None
    for i, layer in enumerate(model.layers):
        if i % GROUP == 0:
            store = Store()
            layer.mixer = Producer(layer.mixer, store)
        else:
            layer.mixer = Consumer(layer.mixer, store)
    return model
