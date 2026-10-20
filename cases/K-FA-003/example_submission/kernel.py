"""K-FA-003 reference submission: blockwise attention with an online softmax, in plain PyTorch.

Follows the FlashAttention algorithm (Dao et al. 2022) at the level of tensor operations: queries are processed in
blocks, keys and values are streamed in blocks, and a running maximum and normaliser keep the softmax exact without
materialising the T × T score matrix. Without a fused kernel it is slower than SDPA's flash backend; it shows the
interface and the numerics that a Triton or CUDA kernel has to reproduce.
"""
from __future__ import annotations

import torch

BLOCK_Q, BLOCK_K = 256, 256


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    B, H, T, D = q.shape
    scale = D ** -0.5
    out = torch.empty_like(q)
    for qs in range(0, T, BLOCK_Q):
        qe = min(qs + BLOCK_Q, T)
        qb = q[:, :, qs:qe].float() * scale
        m = torch.full((B, H, qe - qs, 1), float("-inf"), device=q.device)
        l = torch.zeros((B, H, qe - qs, 1), device=q.device)
        acc = torch.zeros((B, H, qe - qs, D), device=q.device)
        k_end = qe if causal else T
        for ks in range(0, k_end, BLOCK_K):
            ke = min(ks + BLOCK_K, k_end)
            s = qb @ k[:, :, ks:ke].float().transpose(-1, -2)
            if causal and ke > qs:
                qi = torch.arange(qs, qe, device=q.device)[:, None]
                kj = torch.arange(ks, ke, device=q.device)[None, :]
                s = s.masked_fill(kj > qi, float("-inf"))
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            p = torch.exp(s - m_new)
            corr = torch.exp(m - m_new)
            l = l * corr + p.sum(-1, keepdim=True)
            acc = acc * corr + p @ v[:, :, ks:ke].float()
            m = m_new
        out[:, :, qs:qe] = (acc / l).to(q.dtype)
    return out
