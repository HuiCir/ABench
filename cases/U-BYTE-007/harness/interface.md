# U-BYTE-007 interface

## Entry point: `model.py`

```python
import torch.nn as nn

def build() -> nn.Module:
    """model(x) -> logits (B, T, 256), logits[:, j] predicting x[:, j] from x[:, :j]; model.loss(x) -> scalar."""
```

- Training calls `model.loss(x)` on int64 (32, 4096) windows; evaluation calls it on (8, 4096) windows; the compute
  and causality checks call `model(x)` on one window.
- `byte_blocks.Blocks(cfg, n, window)` builds pre-RMSNorm transformer blocks (from `lm_model.py`) with windowed or
  full causal attention; `byte_blocks.segment_mean` pools byte states into patches.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Evaluator code you can read

`stages.py` (the run, bits per byte, noise, FLOPs, causality), `metrics.py`, `lm_train.py` (recipe), `byte_data.py`
(file format, noise), `byte_blocks.py`, `lm_model.py`, `baselines/static4/model.py`, `abench_metrics.py`,
`abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML. Nothing can be installed.
