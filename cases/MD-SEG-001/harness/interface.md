# MD-SEG-001 interface

## Entry point: `network.py`

```python
def make_network(in_channels: int, num_classes: int, patch_size: tuple, spacing: tuple) -> "torch.nn.Module":
    """forward(x): (B, in_channels, D, H, W) -> logits (B, num_classes, D, H, W), or in training mode a list of
    logits at full and lower resolutions (full resolution first) for deep supervision."""
```

Configurations of the three public tasks (`seg3d.TASKS`): hippocampus 1 channel, 3 classes, spacing (1, 1, 1),
patch (40, 56, 40), batch 8; prostate 2 channels, 3 classes, spacing (3.6, 1, 1), patch (20, 192, 160), batch 2;
heart 1 channel, 2 classes, spacing (2, 2, 2), patch (64, 128, 128), batch 2. The unseen task's configuration has
the same fields.

## Evaluator code you can read

`seg3d.py` (recipe, sampling, augmentation, loss, inference, measures), `stages.py`, `metrics.py`,
`abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, the U-Net baseline under `baselines/unet/` and the
reference under `baseline/`.

## Available software

Python 3, PyTorch (CUDA), NumPy, SciPy. Nothing can be installed.
