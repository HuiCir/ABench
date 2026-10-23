# U-DLM-004 interface

## Entry point: `recipe.py`

```python
import dlm                          # interface/dlm.py is on sys.path

def alpha(t, seq):                  # optional; t: (B,) float tensor -> (B, seq) probability of being unmasked
    return dlm.linear_alpha(t, seq)

def loss(model, x):                 # x: int64 (B, 1024) clean tokens -> scalar tensor
    return dlm.mdlm_loss(model, x, alpha_fn=alpha)
```

- `model(z, t) -> logits (B, T, 50304)`; `dlm.corrupt(x, t, alpha)` masks; `dlm.alpha_and_rate` returns alpha and the
  bound's weight −alpha'/(1 − alpha); `dlm.nelbo_per_token` is the evaluator's bound; `dlm.check_schedule(alpha, 1024,
  min_window=0.2)` is the evaluator's schedule check.

## Evaluator code you can read

`dlm.py`, `stages.py` (the two runs), `metrics.py`, `lm_train.py` (recipe, gradient accumulation), `lm_model.py`,
`token_data.py`, `baselines/mdlm/recipe.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML. Nothing can be installed.
