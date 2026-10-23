"""Blocked causal local attention for the reference submission (same projections as lm_model.Attention)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

import lm_model as base


class LocalAttention(base.Attention):
    def __init__(self, c, block: int):
        super().__init__(c)
        self.block = block

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = base.apply_rope(q, cos, sin), base.apply_rope(k, cos, sin)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        L = self.block
        if T <= L:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o(y.transpose(1, 2).reshape(B, T, -1))
        pad = (-T) % L
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        n = (T + pad) // L
        qb = q.view(B, self.h, n, L, self.hd)
        kb, vb = k.view(B, self.h, n, L, self.hd), v.view(B, self.h, n, L, self.hd)
        prev_k = torch.cat([torch.zeros_like(kb[:, :, :1]), kb[:, :, :-1]], dim=2)
        prev_v = torch.cat([torch.zeros_like(vb[:, :, :1]), vb[:, :, :-1]], dim=2)
        kk, vv = torch.cat([prev_k, kb], dim=3), torch.cat([prev_v, vb], dim=3)      # (B, H, n, 2L, hd)
        i = torch.arange(L, device=x.device)[:, None]
        j = torch.arange(2 * L, device=x.device)[None, :]
        mask = (j <= i + L)                                                          # causal within the window
        first = torch.ones(n, 1, 1, dtype=torch.bool, device=x.device)
        first[0] = False                                                             # block 0 has no previous block
        m = mask[None] & (first | (j >= L)[None])                                    # (n, L, 2L)
        H = self.h
        y = F.scaled_dot_product_attention(qb.reshape(B, H * n, L, self.hd), kk.reshape(B, H * n, 2 * L, self.hd),
                                           vv.reshape(B, H * n, 2 * L, self.hd),
                                           attn_mask=m.repeat(H, 1, 1)[None])                # (B, H·n, L, hd)
        y = y.reshape(B, H, n * L, self.hd)[:, :, :T]
        return self.o(y.transpose(1, 2).reshape(B, T, -1))
