# A-LCTX-006 interface

## Entry point: `positions.py`

```python
def make_positions(head_dim: int, n_heads: int, n_kv_heads: int, train_len: int) -> "torch.nn.Module":
    """forward(q, k) -> (q, k); q: (B, n_heads, T, head_dim), k: (B, n_kv_heads, T, head_dim)."""
```

The model (`ctx_model.py`) calls it in every attention layer right before scaled dot-product attention; head_dim
is 64, n_heads 6, n_kv_heads 2, train_len 1,024.

## Evaluator code you can read

`ctx_model.py` (the model), `lm_train.py` (recipe), `longctx.py` (long-context loss and copy test), `stages.py`,
`metrics.py`, `token_data.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, the RoPE baseline under
`baselines/rope/` and the reference under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), NumPy. Nothing can be installed.
