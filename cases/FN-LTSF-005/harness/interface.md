# FN-LTSF-005 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str):
    """Load your trained models from checkpoint/ and return them (any Python object)."""

def predict(state, dataset: str, history: numpy.ndarray, horizon: int) -> numpy.ndarray:
    """dataset in ("etth1", "etth2", "ettm1", "ettm2", "electricity"); history float32 (B, 720, C), standardised;
    horizon in (96, 192, 336, 720). Return float32 (B, horizon, C) on the same standardised scale."""
```

- Use as much of the 720-step history as you like; every horizon is asked for separately.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"` must reproduce the
checkpoint. The reference implementation under `interface/baseline/` (`model.py`, `train.py`, `src/linear.py`) is a
working starting point: it is the reference family with a lookback search.

## Evaluator code you can read

`stages.py` (pipeline, self-check split, time limits), `metrics.py`, `ltsf.py` (splits, origins, the reference),
`abench_metrics.py`, `abench_stage.py` and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA), NumPy, SciPy and scikit-learn. Nothing can be installed.
