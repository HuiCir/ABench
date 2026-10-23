"""TTT-Linear mixer of the reference submission (mini-batch dual form; checked against sequential updates in NumPy)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MINI = 16


class TTTLinear(nn.Module):
    def __init__(self, att):
        super().__init__()
        c = att.c
        self.h, self.d = c.n_heads, c.head_dim
        rep = c.n_heads // c.n_kv_heads
        self.q_proj, self.o_proj = att.q_proj, att.o_proj
        self.k_proj = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)
        with torch.no_grad():                       # expand the 3 pretrained key-value heads to 9
            for new, old in ((self.k_proj, att.k_proj), (self.v_proj, att.v_proj)):
                w = old.weight.view(c.n_kv_heads, c.head_dim, c.d_model).repeat_interleave(rep, 0)
                new.weight.copy_(w.reshape(-1, c.d_model))
        self.W0 = nn.Parameter(torch.zeros(self.h, self.d, self.d))
        self.log_eta = nn.Parameter(torch.full((self.h,), math.log(0.1)))
        self.gate = nn.Parameter(torch.full((self.h,), 0.1))

    def _heads(self, lin, x):
        B, T, _ = x.shape
        return lin(x).view(B, T, self.h, self.d).transpose(1, 2).float()       # (B, H, T, d)

    def forward(self, x, pos0: int = 0, cache=None):
        B, T, _ = x.shape
        q = F.normalize(self._heads(self.q_proj, x), dim=-1)
        k = F.normalize(self._heads(self.k_proj, x), dim=-1)
        v = self._heads(self.v_proj, x)
        eta = self.log_eta.exp()[None, :, None, None]
        if cache is None:
            W = self.W0.float()[None].expand(B, -1, -1, -1).contiguous()
            bk = bv = x.new_zeros(B, self.h, 0, self.d, dtype=torch.float32)
        else:
            W, bk, bv = cache
        outs = []
        t = 0
        while t < T:
            n = min(MINI - bk.shape[2], T - t)                # fill the current mini-batch
            kk, vv, qq = torch.cat([bk, k[:, :, t:t + n]], 2), torch.cat([bv, v[:, :, t:t + n]], 2), q[:, :, t:t + n]
            m = kk.shape[2]
            E = kk @ W.transpose(-1, -2) - vv                      # (B, H, m, d): W k_s - v_s
            A = qq @ kk.transpose(-1, -2)                          # (B, H, n, m): k_s · q_t
            A = A * torch.tril(torch.ones(m, m, device=x.device, dtype=A.dtype))[m - n:]
            outs.append(qq @ W.transpose(-1, -2) - 2 * eta * (A @ E))
            if m == MINI:                                          # mini-batch complete: commit the update
                W = W - 2 * eta * (E.transpose(-1, -2) @ kk)
                bk = bv = kk[:, :, :0]
            else:
                bk, bv = kk, vv
            t += n
        z = torch.cat(outs, 2) * self.gate[None, :, None, None]
        y = self.o_proj(z.transpose(1, 2).reshape(B, T, self.h * self.d).to(x.dtype))
        return y, (W, bk, bv)
