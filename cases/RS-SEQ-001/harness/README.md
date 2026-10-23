# RS-SEQ-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
RS-SEQ-001 pipeline: next-item recommendation on MovieLens-1M (evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict":
  setup   load train.npz: every user's interaction sequence without its final item (untimed)
  timed   import model.py, load(checkpoint_dir, device) -> model                        (load time, killed at 120 s)
  finish  recommend(model, histories, users, k=10) -> (U, 10) item ids, for every user  (≤ 120 s)
The held-out target of each user is the final item of their full sequence; it never leaves the evaluator.
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
