# DS-PIPE-002 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str):
    """Load your six fitted models from checkpoint/ and return them (any Python object)."""

def predict_proba(model, dataset: int, X: numpy.ndarray) -> numpy.ndarray:
    """Return (M, 2) non-negative class probabilities for the rows of X of dataset `dataset` (OpenML id).

    X  float32 (M, F); NaN marks a missing value
    """
```

- The evaluator calls `load` once inside the timed phase, then `predict_proba` once per dataset on its held-out rows.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from the files in `$ABENCH_DATA_DIR` (`d<id>.npz`, one fitted model per dataset), for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `model.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

- `stages.py`: the pipeline (one stage, `predict`), the dev split used by the self-check and the time limits.
- `metrics.py`, `abench_metrics.py`: how the metrics are computed.
- `abench_stage.py`: the process and timing protocol shared by all ABench standard cases; `selfcheck.py` runs the dev split with your checkpoint.

## Available software

Python 3, NumPy, SciPy and scikit-learn, CPU only. Nothing can be installed.
