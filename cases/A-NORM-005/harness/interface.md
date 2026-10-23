# A-NORM-005 interface

## Entry point: `norm.py`

```python
import torch.nn as nn

class MyScheme:
    def __init__(self, d_model, n_layers, n_heads, head_dim): ...
    def make_layer(self, layer: int) -> nn.Module: ...            # parameters of one layer's scheme
    def layer_forward(self, mod, x, attn, mlp, layer: int): ...   # returns the new residual stream
    def make_final(self) -> nn.Module: ...                        # applied before the tied output head
    # optional
    def qk(self, mod, q, k, layer: int): ...                      # (B, H, T, head_dim) after RoPE -> (q, k)
    def embed(self, final, x): ...                                # on the token embeddings
    def init(self, model): ...                                    # after the default initialisation

SCHEME = MyScheme
```

- `attn(h)` and `mlp(h)` apply the layer's fixed sublayers to a (B, T, d_model) input; the baseline's
  `layer_forward` is `x = x + attn(n1(x)); return x + mlp(n2(x))`.
- Scales: μ1 = 4 layers, width 256, 4 heads (head_dim 64); μ2 = 8 layers, width 384, 6 heads (head_dim 64).

## Evaluator code you can read

- `norm_lm.py`: the model and where the scheme is called; `lm_train.py`: the fixed recipe, held-out loss, spike
  counter and FLOP count; `stages.py`: the four runs; `token_data.py`: shard format.
- `baselines/prenorm/norm.py`: the baseline scheme; `metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML. Nothing can be installed.
