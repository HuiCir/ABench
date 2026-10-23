# RS-GRAPH-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
RS-GRAPH-002 pipeline: top-20 recommendation over all items on Gowalla (evaluator and self-check use this same file).

The evaluator does not train. One stage, "recommend":
  setup   load train.npz: every user's training items (CSR: indptr, items) (untimed)
  timed   import model.py, load(checkpoint_dir, device) -> model                       (load time, killed at 300 s)
  finish  recommend(model, train, users, k=20) -> (U, 20) item ids for every user, ranked over all items the user has
          not interacted with in train                                                    (≤ 600 s)
The held-out items of each user (the LightGCN test split) never leave the evaluator.
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
