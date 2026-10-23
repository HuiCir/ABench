# A-KV-004 interface

## Entry point: `model.py`

```python
import smol_lm                       # interface/smol_lm.py is on sys.path

def build(base_dir: str, device: str) -> smol_lm.SmolLM:
    model = smol_lm.load_base(base_dir, device)        # pretrained SmolLM2-135M, fp32 weights
    for i, layer in enumerate(model.layers):
        layer.mixer = MyMixer(layer.mixer)              # forward(x, pos0, cache) -> (y, new_cache)
    return model
```

- `x` is the normalised input (B, T, 576); `pos0` the position of `x[:, 0]`; `cache` None (full forward or prefill)
  or what the mixer returned for the previous tokens. Decoding calls `model.step` with one token per sequence.
- The default mixer `smol_lm.GQAttention` caches keys and values of shape (B, 3, T, 64) per layer.

## Evaluator code you can read

`smol_lm.py` (model, cache size, decoding check), `stages.py`, `metrics.py`, `lm_train.py` (recipe),
`token_data.py`, `baselines/mqa/model.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), safetensors, NumPy, PyYAML. Nothing can be installed.
