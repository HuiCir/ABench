# DS-GBM-001 interface

## Entry point: `model.py`

```python
def fit(x: numpy.ndarray, y: numpy.ndarray, task: str, budget_seconds: float, seed: int):
    """Train on one dataset and return a model (any Python object).

    x float32 (n, F), NaN allowed; y int64 0/1 (task == "classification") or float64 (task == "regression");
    budget_seconds is advisory (the total over the 20 datasets is what counts); seed differs per dataset and run."""

def predict(model, x: numpy.ndarray) -> numpy.ndarray:
    """Return (n,) floats: the probability of class 1, or the predicted target."""
```

- The evaluator calls `fit` once per held-out dataset inside the timed phase, then `predict` once per dataset after it.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Evaluator code you can read

- `stages.py`: the stage (how training splits are given before the clock starts, the fit loop, the prediction step) and the dev setting of the self-check.
- `metrics.py`, `abench_metrics.py`: per-dataset accuracy, AUC and R², the mean score, and the comparison with the baseline.
- `baselines/hgb/model.py`: the HistGradientBoosting baseline trained alongside your learner; it is also a working starting point.
- `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, NumPy, SciPy and scikit-learn, CPU only. Nothing can be installed.
