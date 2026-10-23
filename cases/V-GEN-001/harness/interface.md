# V-GEN-001 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str) -> torch.nn.Module:
    """Rebuild your trained network from checkpoint/ only and return it (a torch.nn.Module)."""

def generate(net, n: int, steps: int, generator: torch.Generator, device: str) -> torch.Tensor:
    """Return n images, float tensor (n, 3, 32, 32) in [-1, 1], calling net at most `steps` times (25, 100)."""
```

  - `load` runs inside the timed phase; it may read only `checkpoint/`.
  - The evaluator registers a forward hook on the returned module and counts its calls during each `generate()`; all network computation must go through that module's `forward` (do not call submodules directly to avoid the count).
  - `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that produces `checkpoint/` from `$ABENCH_DATA_DIR/cifar10_train.npz`, for example
`python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not run it;
it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation under
`interface/baseline/` (its `model.py` and `train.py`) is a working starting point you may improve on or replace.

## Evaluator code you can read

- `stages.py`: the sampling loop with the call counter, image conversion (round((x + 1) · 127.5)) and the Inception features.
- `metrics.py` (FID, diversity ratio), `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, SciPy, diffusers, pytorch-fid (offline). Nothing can be installed.
