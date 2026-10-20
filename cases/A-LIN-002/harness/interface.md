# A-LIN-002 interface

## Entry point: `mixer.py`

```python
import torch.nn as nn

class Mixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layer_idx: int): ...
    def forward(self, x):          # x: (B, T, d_model), bf16 under autocast or fp32
        """Return (B, T, d_model). Output at position t may depend only on x[:, :t + 1]."""
```

- The evaluator constructs the model of `lin_model.py` (embedding, two blocks of RMSNorm → Mixer → residual and RMSNorm → MLP → residual, untied head) with your class and loads `checkpoint/model.pt` into it (`torch.load(..., weights_only=True)`, `load_state_dict(..., strict=True)`).
- `mixer.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Training program: `train.py`

Write a `train.py` that trains `lin_model.Model(Mixer)` on batches from `mqar.batch(rng, n, T, K)` with (T, K) in
`mqar.TRAIN_GRID` (import both from `$ABENCH_VIEW_DIR/interface`) and saves `model.state_dict()` to
`checkpoint/model.pt`, for example `python3 train.py --out "$ABENCH_SUBMISSION_DIR/checkpoint"`. The evaluator does not
run it; it is kept with the submission so the checkpoint can be audited and reproduced. The reference implementation
under `interface/baseline/` (its `mixer.py`, the GLA layer, and `train.py`) is a working starting point you may improve
on or replace.

## Evaluator code you can read

- `mqar.py`: the task generator; `lin_model.py`: the fixed model; `stages.py`: loading, held-out prediction, causality and throughput measurements.
- `baselines/gla/mixer.py`: the GLA baseline whose fresh copy is timed alongside your model.
- `metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy. Nothing can be installed.
