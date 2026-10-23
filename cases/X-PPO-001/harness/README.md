# X-PPO-001 harness

Evaluator code of the case; the self-check (`selfcheck.py`) runs the same stages on the dev split.

## Pipeline (`stages.py`)

```
X-PPO-001 pipeline: the evaluator runs your policy-optimisation algorithm on four small continuous-control tasks
with a fixed number of environment samples, then evaluates the trained policies (evaluator and self-check use this
same file).

One stage, "train" (imports agent.py):
  timed   for each task in control_envs.ENVS: train(make_env, spec, seed) -> policy, where make_env(n) creates n
          parallel environments that share the task's budget of PARAMS["samples_per_task"] environment steps
          (an extra step raises BudgetExceeded)                                                 (all four ≤ 2,400 s)
  finish  the held-out initial states arrive (held_out.npz); policy.act(obs) is rolled out for one episode from each
          (20 per task); the undiscounted returns are reported                                  (≤ 300 s)
The evaluator repeats the run for 5 seeds and runs the PPO baseline (baselines/ppo) the same way.
```

## Files

- `abench_metrics.py`
- `abench_stage.py`
- `baselines/ppo/agent.py`
- `control_envs.py`
- `interface.md`
- `metrics.py`
- `sandbox_guide.md`
- `selfcheck.py`
- `stages.py`
- `submission_rules.json`

The held-out data, keys and judge configuration are never in this directory.
