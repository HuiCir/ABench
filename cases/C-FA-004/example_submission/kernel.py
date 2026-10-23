"""C-FA-004 reference submission: exact attention with a memory-bounded backward pass, in plain PyTorch.

A torch.autograd.Function processes query blocks of all (batch, head) pairs at once. The forward pass keeps only the
output and the log-sum-exp of every query row; the backward pass recomputes each block's probabilities from them, as
in FlashAttention (Dao et al. 2022), so no T x T matrix is ever stored. The block length is chosen so that one block
of fp32 scores is no larger than q itself. The arithmetic is fp32 with bf16 inputs and outputs; it is not IO-aware
(every block is a separate matrix product that goes through global memory), which is what entries can improve.
"""
from __future__ import annotations

import torch


def _block(q: torch.Tensor) -> int:
    B, H, T, D = q.shape
    blk = (q.numel() * q.element_size()) // max(1, B * H * T * 4)
    return int(max(16, min(256, blk)))


def _scores(qb, k, scale, start, causal):
    s = torch.matmul(qb.float(), k.float().transpose(-1, -2)) * scale
    if causal:
        qi = torch.arange(start, start + qb.shape[-2], device=qb.device)[:, None]
        kj = torch.arange(k.shape[-2], device=qb.device)[None, :]
        s = s.masked_fill(kj > qi, float("-inf"))
    return s


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        scale = q.shape[-1] ** -0.5
        T, blk = q.shape[-2], _block(q)
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        vf = v.float()
        for i in range(0, T, blk):
            s = _scores(q[..., i:i + blk, :], k, scale, i, causal)
            m = s.amax(-1, keepdim=True)
            p = torch.exp(s - m)
            l = p.sum(-1, keepdim=True)
            out[..., i:i + blk, :] = (torch.matmul(p, vf) / l).to(q.dtype)
            lse[..., i:i + blk] = (m + torch.log(l)).squeeze(-1)
            del s, p
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal, ctx.scale, ctx.blk = causal, scale, blk
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors
        scale, blk, T = ctx.scale, ctx.blk, q.shape[-2]
        dout = dout.contiguous()
        delta = (dout.float() * out.float()).sum(-1)                  # rowsum(dO * O)
        dq = torch.empty_like(q)
        dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
        dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
        kf, vf = k.float(), v.float()
        for i in range(0, T, blk):
            qb, dob = q[..., i:i + blk, :].float(), dout[..., i:i + blk, :].float()
            p = torch.exp(_scores(qb, k, scale, i, ctx.causal) - lse[..., i:i + blk, None])
            dv += torch.matmul(p.transpose(-1, -2), dob)
            ds = p * (torch.matmul(dob, vf.transpose(-1, -2)) - delta[..., i:i + blk, None])
            del p
            dq[..., i:i + blk, :] = (torch.matmul(ds, kf) * scale).to(q.dtype)
            dk += torch.matmul(ds.transpose(-1, -2), qb) * scale
            del ds
        return dq, dk.to(k.dtype), dv.to(v.dtype), None


def attention(q, k, v, causal: bool = False):
    """Exact softmax attention with scale 1/sqrt(D); inputs and output bf16 of shape (B, H, T, D)."""
    return _Attention.apply(q, k, v, bool(causal))
