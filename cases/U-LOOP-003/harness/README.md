# U-LOOP-003 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
U-LOOP-003 pipeline: length generalisation on parity, binary addition and copy (evaluator and self-check).

The evaluator does not train (training data comes only from algo_tasks.Sampler, lengths 1..20, in your train.py).
One stage, "predict":
  timed   import model.py, load(checkpoint_dir, device) -> model                          (load time, killed at 120 s)
  finish  for each task, call predict(model, task, inputs) on lengths 1..40                (≤ 300 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `algo_tasks.py`
- `baseline/model.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
