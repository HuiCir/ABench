# V-DIT-003 interface

## Entry point: `backbone.py`

```python
def make_backbone(latent_shape=(4, 16, 16), num_classes: int = 10) -> "torch.nn.Module":
    """forward(x, t, y) -> predicted noise; x (B, 4, 16, 16), t (B,) long in [0, 1000),
    y (B,) long in [0, num_classes] (num_classes = null class)."""
```

## Evaluator code you can read

`ldm.py` (schedule, training, EMA, DDIM with guidance, decoding, FLOPs), `stages.py`, `metrics.py`,
`abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, DiT-S/2 under `baselines/dit_s/` and the reference under
`baseline/`.

## Available software

Python 3, PyTorch (CUDA), NumPy, SciPy, diffusers, pytorch-fid. Nothing can be installed.
