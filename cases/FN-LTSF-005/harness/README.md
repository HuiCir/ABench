# FN-LTSF-005 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
FN-LTSF-005 pipeline: long-horizon forecasts of five multivariate series at four horizons, compared with a fitted
linear reference (evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir, device) -> state                          (killed at 300 s)
  finish  for every dataset and horizon: predict(state, dataset, history, horizon) on the held-out windows in batches
          of at most 64 -> forecasts (B, horizon, C), standardised scale                   (all within 900 s)
Histories are the 720 standardised steps before each origin (ltsf.py); the reference's errors on the same windows are
part of the held-out reference and never enter the sandbox.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/src/linear.py`
- `baseline/train.py`
- `interface.md`
- `ltsf.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
