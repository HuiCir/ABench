# X-DATA-007 interface

## Entry point: `pipeline.py`

```python
def select(pool_dir: str, budget_tokens: int, workdir: str) -> "np.ndarray":
    """Return document indices into the pool in training order (repeats allowed), at least 20M tokens in all.
    pool_dir holds pool_text.npz, pool_tokens.bin, pool_index.npz and target_examples.npz."""
```

## Evaluator code you can read

`stages.py` (the two stages), `lm_train.py`, `norm_lm.py`, `fixed_norm.py` (the fixed model and recipe),
`metrics.py`, `token_data.py`, `textpack.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the
reference submission's code under `baseline/`.

## Available software

Python 3, NumPy, SciPy, scikit-learn in the select stage; PyTorch (CUDA) in the sandbox for your own experiments.
Nothing can be installed.
