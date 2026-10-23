"""Baseline: SmolLM2-135M with the attention of every odd layer (15 of 30) restricted to a sliding window of 512
tokens (Beltagy et al. 2020; Mistral), keeping its pretrained projections; those layers cache only the last 512 keys
and values, so their state does not grow with the context."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import smol_lm

WINDOW = 512


class SlidingWindow(nn.Module):
    def __init__(self, att: smol_lm.GQAttention, window: int = WINDOW):
        super().__init__()
        self.att, self.window = att, window

    def forward(self, x, pos0: int = 0, cache=None):
        q, k, v = self.att.qkv(x, pos0)
        T = q.shape[2]
        if cache is not None:
            k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
        S = k.shape[2]
        i = torch.arange(S - T, S, device=x.device)[:, None]
        j = torch.arange(S, device=x.device)[None, :]
        mask = (j <= i) & (j > i - self.window)
        rep = q.shape[1] // k.shape[1]
        y = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), attn_mask=mask)
        B, H, _, D = y.shape
        y = self.att.o_proj(y.transpose(1, 2).reshape(B, T, H * D))
        return y, (k[:, :, -(self.window - 1):], v[:, :, -(self.window - 1):])


def build(base_dir: str, device: str):
    model = smol_lm.load_base(base_dir, device)
    for i, layer in enumerate(model.layers):
        if i % 2 == 1:
            layer.mixer = SlidingWindow(layer.mixer)
    return model
