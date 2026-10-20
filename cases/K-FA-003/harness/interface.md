# K-FA-003 interface

## Entry point: `kernel.py`

```python
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """Exact softmax attention, softmax(q kᵀ / sqrt(D) + mask) v.

    q, k, v  contiguous bfloat16 CUDA tensors of shape (B, H, T, D), D in {64, 128}, T not necessarily a multiple of 64
    causal   if True, query position i attends only to key positions 0..i
    returns  bfloat16 tensor of shape (B, H, T, D)
    """
```

- The evaluator imports `kernel.py` once per seed (compilation and autotuning happen then), and calls `attention` with fresh tensors for every shape, first for accuracy and then 5 + 20 times for timing.

## Evaluator code you can read

- `stages.py`: the shapes of the self-check, the fp32 reference, the timing loop and the forbidden-API scan.
- `metrics.py`, `abench_metrics.py`: correctness rule and relative speed; `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton, `torch.utils.cpp_extension`), NumPy. Nothing can be installed.
