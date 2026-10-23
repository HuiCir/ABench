# A-HYB-001 interface

## Entry point: `model.py`

```python
import torch.nn as nn
from lm_model import Config          # interface/lm_model.py is on sys.path

def build(cfg: Config) -> nn.Module:
    """Return a causal model: model(idx) -> logits (B, T, cfg.vocab_size); model.loss(idx, targets) -> scalar."""
```

- Called with `LM_CONFIG = Config(max_seq=16384)` (width 384, 8 layers, 6 heads, 2 key-value heads, MLP 1,024,
  vocabulary 50,304) and `MQAR_CONFIG` (vocabulary 8,192, width 128, 4 layers, 2 heads, MLP 384, max_seq 2,048).
- The baseline is `lm_model.LM(cfg)`; `baselines/transformer/model.py` shows the one-line build.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Evaluator code you can read

`stages.py` (both stages, FLOP and latency measurement, causality checks), `metrics.py`, `lm_train.py` (recipe),
`lm_model.py`, `mqar.py`, `token_data.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML. Nothing can be installed.
