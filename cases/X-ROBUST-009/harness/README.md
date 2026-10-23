# X-ROBUST-009 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
X-ROBUST-009 pipeline: an intent classifier that knows when to abstain, scored on in-scope, out-of-scope, shifted
(Banking77) and noisy queries (evaluator and self-check).

One stage, "predict" (the provided roberta-base is available under models/):
  timed   import model.py and call load(checkpoint_dir, models_dir, device)          (≤ 600 s)
  finish  predict(state, texts) on all held-out queries, in batches of 512 in a fixed shuffled order, timed per query
          → labels (best in-scope intent), confidence, abstain                         (≤ 900 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/src/features.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `textpack.py`

The held-out data, keys and judge configuration are never in this directory.
