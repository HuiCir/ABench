# MD-MNIST-003 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str):
    """Load your trained classifiers for both datasets from checkpoint/ (any Python object). device is "cuda" at evaluation."""

def predict(model, name: str, x: numpy.ndarray) -> numpy.ndarray:
    """name is "blood" (8 classes, x uint8 (M, 28, 28, 3)) or "organa" (11 classes, x uint8 (M, 28, 28));
    return float (M, n_classes) scores."""
```

- Class indices follow the MedMNIST label dictionaries of each dataset.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from the files in `$ABENCH_DATA_DIR` (`blood.npz`, `organa.npz`: `x_train`, `y_train`, `x_val`, `y_val`), for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `model.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

`stages.py` (pipeline, dev split, time limits), `metrics.py` and `abench_metrics.py` (accuracy and AUC),
`abench_stage.py` (process and timing protocol) and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, SciPy. Nothing can be installed.
