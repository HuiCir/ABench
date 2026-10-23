# A-LIN-002 interface

## Entry point: `mixer.py`

```python
import torch.nn as nn

class Mixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layer_idx: int): ...
    def forward(self, x):          # x: (B, T, d_model), bf16 under autocast or fp32
        """Return (B, T, d_model). Output at position t may depend only on x[:, :t + 1]."""
```

- The evaluator constructs the model of `lin_model.py` (embedding, two blocks of RMSNorm → Mixer → residual and RMSNorm → MLP → residual, untied head) with your class, seeds `torch.manual_seed(seed)` before construction, and trains it with the recipe `PARAMS` in `stages.py` on batches from `mqar.batch` on `mqar.TRAIN_GRID`.
- `mixer.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Evaluator code you can read

- `mqar.py`: the task generator; `lin_model.py`: the fixed model; `stages.py`: the fixed training recipe, held-out prediction, causality and throughput measurements.
- `baselines/gla/mixer.py`: the GLA baseline, trained and measured alongside your mixer; copy it into your submission as a working starting point.
- `metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy. Nothing can be installed.
