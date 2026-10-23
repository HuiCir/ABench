"""E-ACT-001 pipeline: imitation learning on PushT from keypoint observations (evaluator and self-check).

The evaluator does not train. One stage, "rollout":
  timed   import policy.py, load(checkpoint_dir, device) -> model                                 (load time, killed at 120 s)
  finish  closed-loop rollouts in gym-pusht (obs_type "environment_state_agent_pos"), one episode per held-out
          reset seed, at most 300 steps each; per episode the maximum reward (coverage / 0.95, clipped to 1) and
          whether the block reached 95% coverage
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

MAX_STEPS = 300
PARAMS = {"env": "gym_pusht/PushT-v0", "obs_type": "environment_state_agent_pos", "max_steps": MAX_STEPS}
HELD_OUT_KEYS = ["reset_seeds"]
PIPELINE = [{"stage": "rollout", "timeout": 120 + 2700, "finish_cap": 2700, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("policy.py", required=("load", "make_policy"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), ctx.device)


def rollout(policy, seed: int, max_steps: int = MAX_STEPS) -> tuple[float, float, int]:
    import gymnasium as gym
    import gym_pusht  # noqa: F401  (registers the environment)
    env = gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos")
    obs, _ = env.reset(seed=int(seed))
    policy.reset()
    best, success, steps = 0.0, 0.0, 0
    for steps in range(1, max_steps + 1):
        a = np.asarray(policy.act({"agent_pos": np.asarray(obs["agent_pos"], np.float32),
                                   "environment_state": np.asarray(obs["environment_state"], np.float32)}),
                       np.float32).reshape(2)
        if not np.isfinite(a).all():
            raise SystemExit("policy returned a non-finite action")
        obs, reward, terminated, truncated, info = env.step(np.clip(a, 0, 512))
        best = max(best, float(reward))
        if terminated:
            success = 1.0
            break
        if truncated:
            break
    env.close()
    return best, success, steps


def _finish(ctx: Context, state):
    mod, model = state
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        seeds = z["reset_seeds"]
    rows = []
    for s in seeds:
        policy = mod.make_policy(model)
        rows.append(rollout(policy, int(s)))
    return {"episodes": np.asarray(rows, np.float64)}


STAGES = {"rollout": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev setting for selfcheck.py: roll out your checkpoint on public reset seeds (10, or 3 with --quick)."""
    n = 3 if quick else 10
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", reset_seeds=np.arange(100000, 100000 + n, dtype=np.int64))
    return {"public": public, "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}
