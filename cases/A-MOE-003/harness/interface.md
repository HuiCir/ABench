# A-MOE-003 interface

## Entry point: `moe.py`

```python
import torch.nn as nn

class MyMoE(nn.Module):
    def __init__(self, d_model: int, layer: int, n_layers: int): ...
    def forward(self, x):                      # x: (B, T, d_model), after the block's RMSNorm
        return y, {"loss": aux_loss, "counts": tokens_per_expert}     # both entries optional

FFN = MyMoE
```

- d_model = 384, 8 layers; the dense reference MLP is SwiGLU of width 1,024 (3 · 384 · 1,024 parameters).
- Per layer: parameters ≤ 8 · 3 · 384 · 1,024 + 64 · 384; forward FLOPs per token ≤ 1.05 × the top-2 dense equivalent.
- `aux["loss"]` is added to the cross-entropy during training only; `aux["counts"]` is summed over the held-out pass.

## Evaluator code you can read

- `moe_lm.py`: the model and the budgets; `lm_model.py`: attention, RMSNorm, RoPE; `lm_train.py`: the fixed recipe;
  `stages.py`: the run, the compute and causality checks; `token_data.py`: shard format.
- `baselines/top2/moe.py`: the baseline layer and a routing helper you may copy; `metrics.py`, `abench_metrics.py`,
  `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML. Nothing can be installed.
