# U-KAN-009 interface

## Entry point: `layer.py`

```python
def make_layer(d_in: int, d_out: int, final: bool) -> "torch.nn.Module":
    """A module mapping (B, d_in) float32 to (B, d_out); final=True for the output layer."""
```

- The evaluator calls `make_layer` many times while choosing the width (with `torch.manual_seed` set); it must be cheap
  and deterministic in its parameter count.
- Initialise your parameters inside the module; the evaluator does not re-initialise them.

## Evaluator code you can read

`layerbench.py` (networks, budgets, width choice, recipe, timing), `stages.py`, `metrics.py`, `abench_metrics.py`,
`abench_stage.py`, `selfcheck.py`, and the reference submission under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), NumPy, SciPy, scikit-learn. Nothing can be installed.
