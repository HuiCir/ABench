# LG-RSN-003 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
LG-RSN-003 pipeline: LegalBench classification tasks answered by a submitted pipeline around the frozen
Qwen2.5-1.5B-Instruct, on tasks seen during development and on tasks it has never seen (evaluator and self-check).

One stage, "predict" (the frozen model is under models/Qwen2.5-1.5B-Instruct):
  timed   import pipeline.py, load(checkpoint_dir, models_dir, device) -> state                   (killed at 600 s)
  finish  for each held-out task, in a fixed shuffled order of tasks: predict(state, task, rows) on its test rows in
          batches of 64 -> one answer per row (legaltasks.py describes tasks and rows)               (≤ 3,600 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/pipeline.py`
- `baseline/train.py`
- `interface.md`
- `legaltasks.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
