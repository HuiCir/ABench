# B-OPT-002 interface

## `program.yaml`

```yaml
state: [m, v]
constants: {b1: 0.9, b2: 0.99, wd: 0.1}
lr: 3.0e-4
program:
  - c = interp(m, g, b1)
  - m = interp(m, g, b2)
  - update = add(sign(c), mul(wd, p))      # this is Lion: it would fail the novelty threshold
```

Primitives: add, sub, mul, div, max, min, neg, abs, sign, square, sqrt, rsqrt, clip, interp, bias, rms, normalize,
orth, rownorm (definitions in `primitive_dsl.py`).

## Evaluator code you can read

`primitive_dsl.py` (language and compiler), `behaviour.py` (stream, known optimizers, similarity), `stages.py`,
`metrics.py`, `optim_api.py`, `reference_rules.py`, `lm_model.py`, `token_data.py`, `abench_metrics.py`,
`abench_stage.py`, `selfcheck.py`, the AdamW program under `baselines/adamw/` and the reference under `baseline/`.

## Available software

Python 3, NumPy, PyYAML, PyTorch (CUDA). Nothing can be installed.
