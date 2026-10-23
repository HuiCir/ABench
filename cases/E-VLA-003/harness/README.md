# E-VLA-003 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
E-VLA-003 pipeline: your action head on the frozen SmolVLM2 backbone controls LIBERO-Spatial and LIBERO-Object
robots; the evaluator runs every episode (evaluator and self-check use this same file).

One stage, "rollout" (models: SmolVLM2-500M-Video-Instruct, smolvla_base):
  timed   start the policy's process: the evaluator loads and freezes the VLM there (vla_frozen.before) and calls
          policy.py's load_policy(checkpoint_dir, models_dir, device, vlm, processor); ready within 900 s
  finish  one synchronous episode (libero_eval.run_sync) per held-out initial state of both suites; per episode:
          success, steps, mean policy time per control step; whether the VLM's weights are unchanged afterwards
          (vla_frozen.after); the numbers stored in checkpoint/                                         (≤ 7,200 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/policy.py`
- `baseline/src/features.py`
- `baseline/src/head.py`
- `baseline/train.py`
- `interface.md`
- `libero_eval.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`
- `vla_frozen.py`

The held-out data, keys and judge configuration are never in this directory.
