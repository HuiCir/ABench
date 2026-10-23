# V-3DG-005 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
V-3DG-005 pipeline: for each held-out scene your method is fitted to 24 training views within a time budget and
renders the test cameras (evaluator and self-check use this same file).

Stages "scene0" ... "scene5" (imports method.py), one per held-out Mip-NeRF 360 scene:
  setup   the scene's training views (images, K, w2c) arrive in ctx.pre_dir (held_pre.npz) (untimed)
  timed   state = fit(views, budget_seconds, device), at most BUDGET seconds
  finish  the test cameras arrive (held_out.npz); render(state, cameras) -> uint8 images             (≤ 300 s)
The evaluator runs plain 3D Gaussian splatting (baselines/gs3d) through the same stages.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/method.py`
- `baseline/src/gs3d.py`
- `baselines/gs3d/method.py`
- `interface.md`
- `metrics.py`
- `nvs.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
