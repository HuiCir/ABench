# X-PPO-001 interface

## Entry point: `agent.py`

```python
def train(make_env, spec: dict, seed: int):
    """spec = {"name", "obs_dim", "act_dim" (1), "max_episode_steps", "budget" (200000), "action_low" (-1.0),
    "action_high" (1.0)}. Return an object with act(obs) -> actions."""

    env = make_env(8)                # 8 parallel environments; at most 64 per task in total
    obs = env.reset()                # float32 (8, obs_dim)
    obs, reward, terminated, truncated = env.step(actions)   # actions (8, 1); each call uses 8 of the budget
    env.final_obs                    # last observation of the episodes that just ended (auto-reset)
```

- All environments made for a task share its budget of 200,000 steps; `step` raises `BudgetExceeded` beyond it.
- Finished environments reset automatically; `terminated` marks a true end, `truncated` the episode limit.
- After `train` returns, `act` is called on float32 arrays of 20 observations and must return a (20, 1) array.
- The evaluator calls `train` for `pendulum`, `mountain_car`, `cartpole_swing` and `acrobot` in that order, with seeds
  `run_seed`, `run_seed + 1`, ...; `agent.py` may import modules under `src/` (the submission directory is on `sys.path`).

## Evaluator code you can read

`control_envs.py` (the four tasks), `stages.py` (budget, the stage, the held-out rollouts), `metrics.py`,
`baselines/ppo/agent.py`, `abench_metrics.py`, `abench_stage.py` and `selfcheck.py`.

## Available software

Python 3, PyTorch (CUDA, Triton), NumPy, SciPy. Nothing can be installed.
