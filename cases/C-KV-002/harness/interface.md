# C-KV-002 interface

## Entry point: `cache.py`

```python
from transformers import DynamicCache

class MyCache(DynamicCache):
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """key_states, value_states: (1, 2, T_new, 128) bf16. Store them compressed; return the keys and values
        (1, 2, T_seen, 128) the attention of this layer should use."""
    def get_seq_length(self, layer_idx=0) -> int: ...

def make_cache(budget_bits: float, config):        # budget_bits in (4.5, 2.5, 2.0)
    return MyCache(...)
```

- The evaluator loads the model itself and runs `model(input_ids=..., past_key_values=cache, use_cache=True)` for the
  prefill and for 32 greedy steps; checkpoint files are under `checkpoint/` next to `cache.py`.

## Evaluator code you can read

`ruler_tasks.py` (tasks, answer matching), `stages.py` (decoding loop, byte count), `metrics.py`, `abench_metrics.py`,
`abench_stage.py`, `selfcheck.py`, and the reference submission's code under `baseline/`.

## Available software

Python 3, PyTorch (CUDA, Triton), transformers, safetensors, NumPy. Nothing can be installed.
