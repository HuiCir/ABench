# Problem: problem_softmax.py
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _softmax_online(X, Y, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x_ptr = X + row * stride
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    s = tl.zeros([BLOCK], tl.float32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m - m_new))    # lanes that have seen only padding
        p = tl.where(x == float("-inf"), 0.0, tl.exp(x - m_new))
        s = s * alpha + p
        m = m_new
    m_row = tl.max(m, 0)
    s_row = tl.sum(tl.where(m == float("-inf"), 0.0, s * tl.exp(m - m_row)), 0)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
        tl.store(Y + row * stride + cols, tl.exp(x - m_row) / s_row, mask=cols < n_cols)


class ModelNew(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        _softmax_online[(x.shape[0],)](x, y, x.shape[1], x.stride(0), BLOCK=4096)
        return y
