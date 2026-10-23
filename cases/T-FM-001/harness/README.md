# T-FM-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
T-FM-001 pipeline: imitation learning on PushT from keypoint observations under a per-step latency budget (the
setting of E-ACT-001; evaluator and self-check).

The evaluator does not train. One stage, "rollout":
  timed   import policy.py, load(checkpoint_dir, device) -> model                                 (load time, killed at 120 s)
  finish  closed-loop rollouts in gym-pusht (obs_type "environment_state_agent_pos"), one episode per held-out
          reset seed, at most 300 steps each; per episode the maximum reward (coverage / 0.95, clipped to 1) and
          whether the block reached 95% coverage; the wall time of every act() call is measured (CUDA synchronised)
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baseline/policy.py`
- `baseline/train.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
