# U-LOOP-001 interface

## Entry point: `model.py`

```python
import torch.nn as nn
from lm_model import Config          # interface/lm_model.py is on sys.path

def build(cfg: Config) -> nn.Module:
    """cfg = Config(n_layers=4, max_seq=2048): width 384, 6 heads, 2 key-value heads, MLP 1,024, vocabulary 50,304.
    model(idx, loops=k) -> logits (B, T, vocab); model.loss(idx, targets, loops=None) -> scalar."""
```

- Training calls `model.loss(x, y)` (loops=None) in train mode; evaluation calls `model.loss(x, y, loops=k)` and
  `model(x, loops=k)` in eval mode for k in 1, 2, 4, 8.
- `lm_model.Block`, `RMSNorm`, `rope_cache` are the baseline's components; you may reuse them.

## Evaluator code you can read

`stages.py` (the run, the loop-count evaluation, FLOP count, causality), `metrics.py`, `lm_train.py` (recipe),
`lm_model.py`, `token_data.py`, `baselines/fourlayer/model.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML. Nothing can be installed.
