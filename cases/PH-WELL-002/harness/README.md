# PH-WELL-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
PH-WELL-002 pipeline: your trained surrogate predicts held-out trajectories of three physical systems, rolled out
by the evaluator (evaluator and self-check use this same file).

One stage, "rollout" (only checkpoint/ is available):
  timed   import model.py; model = load(checkpoint_dir, device)
  finish  for each system and each held-out set (in-distribution, unseen parameters): wellsurr.rollout of
          model.predict from 4 given frames over 10 frames; predictions as float16; the numbers stored in checkpoint/
                                                                                                     (≤ 1,800 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/model.py`
- `baseline/src/fno.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `wellsurr.py`

The held-out data, keys and judge configuration are never in this directory.
