# C-FA-004 interface

## Entry point: `kernel.py`

```python
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """Exact softmax attention, softmax(q kᵀ / sqrt(D) + mask) v, differentiable with respect to q, k and v.

    q, k, v  contiguous bfloat16 CUDA tensors of shape (B, H, T, D) with requires_grad=True, D in {64, 128},
             T not necessarily a multiple of 64
    causal   if True, query position i attends only to key positions 0..i
    returns  bfloat16 tensor of shape (B, H, T, D); out.backward(dout) must fill q.grad, k.grad, v.grad (bf16)
    """
```

- The evaluator imports `kernel.py` once per seed (compilation and autotuning happen then), and for every shape calls `attention` followed by `backward` with fresh tensors: once for accuracy, 3 + 10 times for timing and once for peak memory.

## Evaluator code you can read

- `stages.py`: the shapes of the self-check, the fp32 reference with autograd, the timing loop, the memory measurement and the forbidden-API scan.
- `metrics.py`, `abench_metrics.py`: correctness rule, memory ratio and relative speed; `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton, `torch.utils.cpp_extension`), NumPy. Nothing can be installed.
