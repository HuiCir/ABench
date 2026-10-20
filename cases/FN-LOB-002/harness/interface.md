# FN-LOB-002 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str):
    """Load your trained classifier from checkpoint/ and return it (any Python object). device is "cuda" at evaluation."""

def predict(model, windows: numpy.ndarray) -> numpy.ndarray:
    """windows float32 (Q, 100, 40), oldest first; the label belongs to the last event of each window.
    Return float (Q, 3) scores for up, stationary, down."""
```

- The label of event t at horizon k compares the mean mid-price of the next k events with the mid-price at t (threshold as defined by FI-2010).
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from the files in `$ABENCH_DATA_DIR` (`train.npz`: `events` float32 (E, 40) and `labels` int8 (E, 5), as described in the task), for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `model.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

`stages.py` (pipeline, window construction, dev split, time limits), `metrics.py` and `abench_metrics.py` (macro-F1),
`abench_stage.py` (process and timing protocol) and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, SciPy. Nothing can be installed.
