# RS-GRAPH-002 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str):
    """Return the trained model (any Python object)."""

def recommend(model, train: dict, users: numpy.ndarray, k: int = 20) -> numpy.ndarray:
    """train["indptr"], train["items"], train["n_items"]: the public training interactions (CSR).
    Return int (len(users), k): for each user, k distinct item ids, best first."""
```

- The evaluator calls `load` inside the timed phase and `recommend` once with all user ids 0..29,857 after it.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Evaluator code you can read

`stages.py` (the stage and the dev split), `metrics.py` (Recall@20, NDCG@20), `abench_metrics.py`,
`abench_stage.py`, `selfcheck.py`, and the reference submission's code under `baseline/`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, SciPy. Nothing can be installed.
