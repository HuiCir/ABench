# V-3DG-005 interface

## Entry point: `method.py`

```python
def fit(views: dict, budget_seconds: float, device: str):
    """views: images uint8 (24, H, W, 3), K (24, 3, 3), w2c (24, 4, 4) (OpenCV: x right, y down, z forward).
    Return any state; at most budget_seconds (600) of wall-clock time."""

def render(state, cameras: dict) -> "uint8 array (M, H, W, 3)":
    """cameras: K (M, 3, 3), w2c (M, 4, 4), height, width."""
```

## Evaluator code you can read

`nvs.py` (camera convention, view split, PSNR, SSIM), `stages.py`, `metrics.py`, `abench_metrics.py`,
`abench_stage.py`, `selfcheck.py`, the 3DGS baseline under `baselines/gs3d/` and the reference under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), gsplat, NumPy, SciPy. Nothing can be installed.
