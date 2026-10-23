# PH-WELL-002 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, device: str):
    """Return an object with predict(system, history, params) -> next frame."""
```

- `history`: float32 (B, 4, C, 64, 64), physical units, oldest frame first; `params`: float32 (B, P) in the order
  of `wellsurr.SYSTEMS[system]["params"]`. The evaluator calls `predict` 10 times per window, feeding each
  prediction back as the newest frame.
- `train.py --data "$ABENCH_DATA_DIR" --out checkpoint` must reproduce the checkpoint.

## Evaluator code you can read

`wellsurr.py` (systems, channels, rollout, VRMSE), `stages.py`, `metrics.py`, `abench_metrics.py`,
`abench_stage.py`, `selfcheck.py`, and the reference under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), NumPy, SciPy, safetensors. Nothing can be installed.
