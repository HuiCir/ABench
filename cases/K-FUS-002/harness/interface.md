# K-FUS-002 interface

## Entry points: `kernels/<problem>.py`

```python
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, *init_inputs):      # the same arguments as the problem's Model (get_init_inputs())
        ...
    def forward(self, *inputs):            # the same inputs as Model.forward (get_inputs()); same outputs
        ...
```

- `<problem>` is a name from `$ABENCH_DATA_DIR/problems.json`, for example `L2_76_Gemm_Add_ReLU`.
- The evaluator builds Model and ModelNew after the same `torch.manual_seed`, copies Model's parameters and buffers into ModelNew where names and shapes match, calls `.eval()` and runs `forward` under `torch.no_grad()`. Register the weights you use under Model's parameter names: they are overwritten by the copy after `__init__`, so anything derived from them inside `__init__` would come from your own initialisation; derive it in `forward` or lazily on the first call.
- The submission directory is on `sys.path`: `from src.mymodule import ...` works from a kernel file.

## Evaluator code you can read

- `kb_bench.py`: problem loading with overrides, the five input distributions, the input-modification check, the correctness rule, timing, memory and the torch entry-point check.
- `stages.py`: the two stages; `metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton, `torch.utils.cpp_extension`), NumPy. Nothing can be installed.
