# DS-GBM-001 submission interface

## Data

`$ABENCH_DATA_DIR/train.npz`: `x` is a float32 feature matrix of shape `[n, 12]`; `y` holds int64 class labels in `{0, 1}`.
This is all the data you may use. If you need a validation set, split it off yourself.

## Submission directory

```
submission/
├── model.py          # required, interface below
├── src/              # optional, other code model.py depends on
├── checkpoint/       # required, trained parameters in a format of your choice
├── report/index.html # required, design report (following the structure of templates/report_template.html)
├── references.yaml   # required, reference registry
└── manifest.yaml     # optional, dependencies and random seeds
```

## model.py interface

```python
def load(checkpoint_dir: str):
    """Read parameters from checkpoint_dir and return a model object."""

def predict_proba(model, X: "numpy.ndarray") -> "numpy.ndarray":
    """X has shape [n, 12]; return class probabilities of shape [n, 2], each row non-negative."""
```

- Runtime: Python 3 with numpy available; list any other dependency in `manifest.yaml`.
- Loading and prediction run on CPU, single-threaded, without network, and must finish within 5 minutes.
- `model.py` must not read files outside `checkpoint/`.
