# PH-MLIP-003 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str):
    """Return the trained model(s) (any Python object)."""

def predict(model, molecule: str, z: numpy.ndarray, pos: numpy.ndarray):
    """molecule in ("aspirin", "ethanol", "malonaldehyde", "naphthalene", "toluene"); z: int64 (N,);
    pos: float64 (B, N, 3) in Å. Return (energy (B,) in kcal/mol, forces (B, N, 3) in kcal/mol/Å)."""
```

- Calls: batches of up to 100 held-out configurations, then 20,000 calls with batches of 4 during the dynamics.
- `model.py` may import modules under `src/` and next to it; the submission directory is on `sys.path`.

## Evaluator code you can read

`md.py` (integrator, bonds, stability), `stages.py` (prediction and dynamics, the dev split), `metrics.py`,
`abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference submission's code under `baseline/`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, SciPy. Nothing can be installed.
