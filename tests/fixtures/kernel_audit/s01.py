# Problem: problem_relu.py
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _relu(X, Y, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(X + offs, mask=m)
    tl.store(Y + offs, tl.maximum(x, 0.0), mask=m)


class ModelNew(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        y = torch.empty_like(x)
        n = x.numel()
        _relu[(triton.cdiv(n, 4096),)](x, y, n, BLOCK=4096)
        return y
