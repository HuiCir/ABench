# V-BB-006 interface

## Entry point: `model.py`

```python
import torch.nn as nn

def build(variant: str, num_classes: int = 10) -> nn.Module:
    """variant "small" (≤ 1.4 GFLOPs) or "large" (≤ 4.8 GFLOPs) per 160 x 160 image; logits (B, num_classes)."""
```

- Inputs are normalised with ImageNet statistics, channels-last, bf16 autocast during training and evaluation.
- FLOPs are counted by `torch.utils.flop_counter.FlopCounterMode` on one image in eval mode (convolutions and
  matrix multiplications; normalisation and activations are not counted).

## Evaluator code you can read

`img_train.py` (recipe, augmentation, prediction, FLOP count), `stages.py`, `metrics.py`, `baselines/vit/model.py`,
`abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy. Nothing can be installed.
