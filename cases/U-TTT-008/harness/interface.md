# U-TTT-008 interface

## Entry point: `model.py`

```python
import smol_lm                       # interface/smol_lm.py is on sys.path

def build(base_dir: str, device: str) -> smol_lm.SmolLM:
    model = smol_lm.load_base(base_dir, device)        # pretrained SmolLM2-135M, fp32 weights
    for i in range(1, 30, 2):
        model.layers[i].mixer = MyLayer(model.layers[i].mixer)   # forward(x, pos0, cache) -> (y, new_cache)
    return model
```

- `x` is the normalised input (B, T, 576); `pos0` the position of `x[:, 0]`; `cache` None (full forward or prefill)
  or what the mixer returned for the previous tokens; decoding calls `model.step` with one token per sequence.
- `smol_lm.cache_bytes` is how cache sizes are counted.

## Evaluator code you can read

`smol_lm.py` (model, caches, decoding check), `stages.py` (training, book losses, passkeys, layer count, throughput),
`metrics.py`, `lm_train.py`, `token_data.py`, `baselines/swa/model.py`, `abench_metrics.py`, `abench_stage.py`,
`selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), safetensors, NumPy, PyYAML. Nothing can be installed.
