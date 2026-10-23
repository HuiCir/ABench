# B-OPT-003 interface

## Entry point: `optimizer.py`

```python
from optim_api import FunctionalRule, newton_schulz5   # optim_api.py is on sys.path

class MyRule(FunctionalRule):
    def __init__(self, **hp): super().__init__(name="myrule", hp=hp)
    def init_state(self, p, ops): ...                    # returns a dict of tensors (built with ops)
    def update(self, p, g, state, step, lr, ops): ...    # returns delta; the framework does p <- p + delta

RULE = MyRule
```

- `p` and `g` are fp32 tensors of one parameter (2D for linear and embedding weights, 1D for RMSNorm gains); `step` starts at 1 and counts the updates of that parameter; `lr` is the scheduled learning rate of its group.
- `update` must not modify `p` or `g` in place; state is kept only in `state` (tensors may be updated in place).
- The total number of state elements must not exceed 1× the number of parameter elements.

## `hparam_space.yaml`

```yaml
rule: MyRule
tunable:            # at most 4; the evaluator uses the defaults only
  lr: {default: 0.003, range: [0.0003, 0.01], scale: log}     # lr is required: peak learning rate of 2D parameters
fixed: {}           # passed to MyRule(**fixed, **tunable_defaults_without_lr)
owns_schedule: false
```

- 1D parameters use `lr × 0.15`. With `owns_schedule: false` the framework multiplies both learning rates by the warm-up + cosine factor in `stages.py`; with `true` it passes the constant peak values and the rule schedules itself.

## Evaluator code you can read

- `stages.py`: the full training loop (data sampling, precision, schedule) and the held-out loss computation.
- `lm_model.py`: the fixed model; `token_data.py`: shard format; `optim_api.py`: `FunctionalRule`, `Ops`, the torch wrapper.
- `baselines/adamw/`: the AdamW baseline the evaluator runs alongside your rule.
- `metrics.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML. Nothing can be installed.
