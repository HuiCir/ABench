# TC-FT-001 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str):
    """Load your trained classifier from checkpoint/ and return it (any Python object)."""

def predict(model, texts: list[str]) -> numpy.ndarray:
    """Return integer labels (M,) in 0..4 (0 = 1 star ... 4 = 5 stars), one per text, in order."""
```

- The evaluator calls `load` inside the timed phase and `predict` after it, once, on the held-out texts.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from the files in `$ABENCH_DATA_DIR`, for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `model.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

- `stages.py`: the pipeline (one stage, `predict`), the dev split used by the self-check and the time limits.
- `textpack.py`: `unpack(utf8, offsets[, idx])` → list of str, for the packed texts in `train.npz`.
- `metrics.py` and `abench_metrics.py`: how the metrics are computed from your outputs.
- `abench_stage.py`: the process and timing protocol shared by all ABench standard cases.
- `selfcheck.py`: runs the files above on the dev split with your checkpoint.

## Available software

Python 3, NumPy, SciPy and scikit-learn, CPU only. Nothing can be installed.
