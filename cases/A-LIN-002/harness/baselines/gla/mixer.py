"""Baseline mixer run by the evaluator under the same conditions as every submission: a gated linear attention layer
(Yang et al. 2024, arXiv:2312.06635) in plain PyTorch, chunkwise-parallel form.

Per head, with key dimension d_k and value dimension d_v:
  S_t = diag(α_t) S_{t-1} + k_tᵀ v_t,   o_t = q_t S_t,   α_t = σ(W_α x_t)^(1/16)
followed by an output gate o ← (o ⊙ swish(W_r x)) W_o. Simplifications relative to the paper: a full-rank gate
projection instead of a low-rank one, no normalisation of o before gating, chunk size 64, fp32 inside the layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CHUNK = 64


class Mixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layer_idx: int):
        super().__init__()
        self.h, self.dk, self.dv = n_heads, d_model // n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.g = nn.Linear(d_model, d_model)            # log-decay gate, one value per key channel
        self.r = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h, dk, dv = self.h, self.dk, self.dv
        q = self.q(x).float().view(B, T, h, dk).transpose(1, 2) * dk ** -0.5      # (B, h, T, dk)
        k = self.k(x).float().view(B, T, h, dk).transpose(1, 2)
        v = self.v(x).float().view(B, T, h, dv).transpose(1, 2)
        log_a = F.logsigmoid(self.g(x).float()).view(B, T, h, dk).transpose(1, 2) / 16.0
        pad = (-T) % CHUNK
        if pad:
            q, k, v, log_a = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v, log_a))
        n = (T + pad) // CHUNK
        q, k, v, log_a = (t.reshape(B, h, n, CHUNK, -1) for t in (q, k, v, log_a))
        g = log_a.cumsum(3)                                                       # within-chunk cumulative log decay
        S = torch.zeros(B, h, dk, dv, device=x.device)
        mask = torch.tril(torch.ones(CHUNK, CHUNK, dtype=torch.bool, device=x.device))
        outs = []
        for c in range(n):
            qc, kc, vc, gc = q[:, :, c], k[:, :, c], v[:, :, c], g[:, :, c]       # (B, h, C, d)
            inter = (qc * gc.exp()) @ S                                          # contribution of earlier chunks
            # intra-chunk: A_ij = Σ_d q_id k_jd exp(g_id − g_jd) for j ≤ i, computed pairwise for stability
            diff = (gc.unsqueeze(3) - gc.unsqueeze(2)).masked_fill(~mask[..., None], float("-inf"))
            A = (qc.unsqueeze(3) * kc.unsqueeze(2) * diff.exp()).sum(-1)          # (B, h, C, C)
            outs.append(inter + A @ vc)
            last = gc[:, :, -1:]                                                  # (B, h, 1, dk)
            S = S * last.transpose(-1, -2).exp() + (kc * (last - gc).exp()).transpose(-1, -2) @ vc
        o = torch.cat(outs, 2)[:, :, :T].transpose(1, 2).reshape(B, T, D)
        o = o * F.silu(self.r(x).float())
        return self.o(o.to(x.dtype))
