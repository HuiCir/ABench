# DS-GBM-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
DS-GBM-001 pipeline: the evaluator trains your tabular learner on datasets it has never seen (evaluator and
self-check use this same file).

One stage, "fit" (imports model.py):
  before   the training splits of the held-out datasets are placed in ctx.pre_dir (held_pre.npz); loading them is
           not timed
  timed    for each dataset: fit(x_train, y_train, task, budget_seconds, seed) -> model; the whole loop is timed by
           the parent's clock (gate: at most PARAMS["total_fit_seconds"])
  finish   the test inputs arrive (held_out.npz); predict(model, x_test) for each dataset              (≤ 300 s)
The evaluator runs the HistGradientBoosting baseline (baselines/hgb) through the same stage on the same seeds.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/hgb/model.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
