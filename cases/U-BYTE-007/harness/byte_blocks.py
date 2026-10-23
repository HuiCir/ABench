"""U-BYTE-007 building blocks you may use (evaluator code; the baseline is built from them).

  Blocks(cfg, n, window)   n pre-RMSNorm transformer blocks of lm_model (grouped-query attention with RoPE, SwiGLU),
                           with causal attention restricted to a window of `window` positions (blocked: each block of
                           `window` positions attends to itself and the previous block), or full causal attention if
                           window is None
  segment_mean(x, seg, n)  mean of x (B, T, d) over positions with the same segment id seg (B, T) -> (B, n, d)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import lm_model as base


class WindowAttention(base.Attention):
    def __init__(self, c, window):
        super().__init__(c)
        self.window = window

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        q, k = base.apply_rope(q, cos, sin), base.apply_rope(k, cos, sin)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        L = self.window
        if L is None or T <= L:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o(y.transpose(1, 2).reshape(B, T, -1))
        pad = (-T) % L
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        n, H = (T + pad) // L, self.h
        qb, kb, vb = (t.view(B, H, n, L, self.hd) for t in (q, k, v))
        pk = torch.cat([torch.zeros_like(kb[:, :, :1]), kb[:, :, :-1]], 2)
        pv = torch.cat([torch.zeros_like(vb[:, :, :1]), vb[:, :, :-1]], 2)
        kk, vv = torch.cat([pk, kb], 3), torch.cat([pv, vb], 3)
        i = torch.arange(L, device=x.device)[:, None]
        j = torch.arange(2 * L, device=x.device)[None, :]
        first = torch.ones(n, 1, 1, dtype=torch.bool, device=x.device)
        first[0] = False
        m = (j <= i + L)[None] & (first | (j >= L)[None])
        y = F.scaled_dot_product_attention(qb.reshape(B, H * n, L, self.hd), kk.reshape(B, H * n, 2 * L, self.hd),
                                           vv.reshape(B, H * n, 2 * L, self.hd), attn_mask=m.repeat(H, 1, 1)[None])
        y = y.reshape(B, H, n * L, self.hd)[:, :, :T]
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class Blocks(nn.Module):
    def __init__(self, cfg: base.Config, n: int, window: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(base.Block(cfg) for _ in range(n))
        for b in self.blocks:
            if window is not None:
                wa = WindowAttention(cfg, window)
                wa.load_state_dict(b.attn.state_dict())
                b.attn = wa
            for m in b.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=cfg.init_std)
            nn.init.normal_(b.attn.o.weight, std=cfg.init_std / (2 * max(n, 1)) ** 0.5)
            nn.init.normal_(b.mlp.down.weight, std=cfg.init_std / (2 * max(n, 1)) ** 0.5)
        self._rope = None

    def forward(self, x):
        if self._rope is None or self._rope[0].device != x.device or self._rope[0].shape[0] < x.shape[1]:
            self._rope = base.rope_cache(max(self.cfg.max_seq, x.shape[1]), self.cfg.d_model // self.cfg.n_heads,
                                         self.cfg.rope_base, x.device)
        cos, sin = self._rope
        for b in self.blocks:
            x = b(x, cos, sin)
        return x


def segment_mean(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """x (B, T, d), seg (B, T) int64 in [0, n) -> (B, n, d) mean over each segment (0 for empty segments)."""
    B, T, d = x.shape
    out = torch.zeros(B, n, d, device=x.device, dtype=x.dtype)
    cnt = torch.zeros(B, n, 1, device=x.device, dtype=x.dtype)
    out.scatter_add_(1, seg[..., None].expand(-1, -1, d), x)
    cnt.scatter_add_(1, seg[..., None], torch.ones(B, T, 1, device=x.device, dtype=x.dtype))
    return out / cnt.clamp(min=1.0)
