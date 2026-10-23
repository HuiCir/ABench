# DS-PIPE-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
DS-PIPE-002 pipeline: score a submitted set of tabular classifiers on six OpenML datasets (evaluator and self-check
use this same file).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir) -> model                       (load time, killed at 120 s)
  finish  for each dataset, predict_proba(model, dataset_id, X) -> (M, 2)   (≤ 120 s in total)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
