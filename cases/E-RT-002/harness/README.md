# E-RT-002 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
E-RT-002 pipeline: your trained policy controls a LIBERO-Spatial robot asynchronously while its plans arrive late;
the evaluator runs every episode (evaluator and self-check use this same file).

One stage, "rollout" (no pretrained models):
  timed   start two processes from policy.py: load_planner(checkpoint_dir, device) on the GPU and
          make_controller(checkpoint_dir) on the CPU; both must be ready within 600 s
  finish  for each injected latency of DELAYS control steps and each held-out initial state, one episode of
          libero_eval.run_async (the planner sees observations, the controller executes actions; they communicate only
          through requests and chunks); per episode: success, steps, plans, planning time, effective delay; and the
          numbers stored in checkpoint/                                                                (≤ 5,400 s)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/policy.py`
- `baseline/src/demos.py`
- `baseline/src/net.py`
- `baseline/train.py`
- `interface.md`
- `libero_eval.py`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
