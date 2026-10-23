# U-KAN-009 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
U-KAN-009 pipeline: the evaluator trains the fixed three-layer networks of layerbench.py with your layer and with the
MLP layer at equal parameter budgets, on held-out function-fitting tasks and on Fashion-MNIST, and scores both on
held-out test points (evaluator and self-check use this same file).

Two stages, each in a new process on one GPU (both import layer.py):
  functions  before: the training and validation points of the held-out functions (held_pre.npz, untimed);
             timed: for each function, layerbench.fit_task for your layer and for the MLP (width per budget, learning
             rate chosen on validation, training); finish: predictions on the test points (given after the timed phase)
  image      setup: the public Fashion-MNIST training images (55,000 for training, the last 5,000 of a fixed
             permutation for validation); timed: fit_task for both layers; finish: predictions on the held-out
             test images
The run's seed (ctx.seed) seeds every training run.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/layer.py`
- `interface.md`
- `layerbench.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
