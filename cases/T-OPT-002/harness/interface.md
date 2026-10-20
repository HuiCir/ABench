# T-OPT-002 interface

## Entry point: `optimizer.py`

```python
from optim_api import FunctionalRule, newton_schulz5   # optim_api.py is on sys.path

class MyRule(FunctionalRule):
    def __init__(self, **hp): super().__init__(name="myrule", hp=hp)
    def init_state(self, p, ops): ...                    # returns a dict of tensors (built with ops)
    def update(self, p, g, state, step, lr, ops): ...    # returns delta; the framework does p <- p + delta

RULE = MyRule
```

- `p` and `g` are fp32 tensors of one parameter: 2D for every weight matrix (patch embedding, attention, MLP, head), 3D for the position embedding (1, 65, 192) and the class token (1, 1, 192), 1D for LayerNorm gains and all biases. Rules must handle every shape, for example by treating non-matrix parameters element-wise.
- `update` must not modify `p` or `g` in place; state is kept only in `state`. State elements ≤ 3× parameter elements.

## `hparam_space.yaml`

```yaml
rule: MyRule
tunable:            # at most 4; the evaluator uses the defaults only
  lr: {default: 0.002, range: [0.0002, 0.01], scale: log}     # lr is required: peak learning rate of parameters with ndim >= 2
fixed: {}           # passed to MyRule(**fixed, **tunable_defaults_without_lr)
owns_schedule: false
```

- Parameters with fewer than 2 dimensions use `lr × 0.15`; parameters with 2 or more dimensions use `lr`. With `owns_schedule: false` the framework multiplies both by the warm-up + cosine factor.

## Evaluator code you can read

- `stages.py`: the full training loop (sampling, augmentation, precision, schedule) and the test-set prediction.
- `vit_model.py`: the fixed model; `optim_api.py`: `FunctionalRule`, `Ops`, the torch wrapper.
- `baselines/adamw/`: the AdamW baseline the evaluator runs alongside your rule.
- `metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML. Nothing can be installed.
