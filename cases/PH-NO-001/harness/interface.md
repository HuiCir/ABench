# PH-NO-001 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str):
    """Load your trained surrogate from checkpoint/ and return it (any Python object). device is "cuda" at evaluation."""

def predict(model, u_in: numpy.ndarray) -> numpy.ndarray:
    """u_in float32 (M, 10, X), X in {256, 512}: the first 10 frames on a uniform periodic grid of X points
    over the same domain. Return float32 (M, 31, X): frames 10..40."""
```

- `predict` is called twice after loading: once with X = 256, once with X = 512.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from the files in `$ABENCH_DATA_DIR` (`train.npz`: `u` float32 (N, 41, 256), trajectories with the frame index first; `x` (256,); `t` (41,); `dev_idx`), for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `model.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

`stages.py` (pipeline, dev split, time limits), `metrics.py` and `abench_metrics.py` (nRMSE),
`abench_stage.py` (process and timing protocol) and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, SciPy. Nothing can be installed.
