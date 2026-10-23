# PH-NO-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
PH-NO-001 pipeline: 1D Burgers (ν = 0.001) surrogate from PDEBench (evaluator and self-check use this same file).

Trajectories are on a 256-point periodic grid with 41 frames (PDEBench's 1024 × 201 subsampled by 4 in space and 5 in
time). Given the first 10 frames, the model predicts the remaining 31.
The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir, device) -> model                                  (load time, killed at 120 s)
  finish  predict(model, u_in) for the held-out inputs at 256 points, then at 512 points          (≤ 120 s)
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
