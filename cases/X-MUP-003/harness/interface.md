# X-MUP-003 interface

## Entry point: `param.py`

```python
class MyParam:
    @staticmethod
    def init_std(name, shape, width, base_width) -> float: ...        # matrices (ndim >= 2) only
    @staticmethod
    def lr_scale(name, shape, width, base_width) -> float: ...        # every tensor
    @staticmethod
    def output_multiplier(width, base_width) -> float: ...
    # optional
    @staticmethod
    def wd_scale(name, shape, width, base_width) -> float: ...        # matrices; default 1
    @staticmethod
    def attention_scale(head_dim, width, base_width) -> float: ...    # default 1 / sqrt(head_dim)

PARAM = MyParam
```

- Names: `embed.weight` (input and tied output), `blocks.<i>.attn.{q,k,v,o}.weight`, `blocks.<i>.mlp.{gate,up,down}.weight`,
  `blocks.<i>.n1.weight`, `blocks.<i>.n2.weight`, `norm.weight`. Widths: 128 (tuning), 512, 1,024; base_width = 128.

## Evaluator code you can read

`mup_lm.py` (model and where the parametrisation applies), `stages.py` (the protocol), `lm_train.py` (recipe),
`metrics.py`, `lm_model.py`, `token_data.py`, `baselines/sp/param.py`, `abench_metrics.py`, `abench_stage.py`, `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, PyYAML. Nothing can be installed.
