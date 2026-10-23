"""X-PPO-001 pipeline: the evaluator runs your policy-optimisation algorithm on four small continuous-control tasks
with a fixed number of environment samples, then evaluates the trained policies (evaluator and self-check use this
same file).

One stage, "train" (imports agent.py):
  timed   for each task in control_envs.ENVS: train(make_env, spec, seed) -> policy, where make_env(n) creates n
          parallel environments that share the task's budget of PARAMS["samples_per_task"] environment steps
          (an extra step raises BudgetExceeded)                                                 (all four ≤ 2,400 s)
  finish  the held-out initial states arrive (held_out.npz); policy.act(obs) is rolled out for one episode from each
          (20 per task); the undiscounted returns are reported                                  (≤ 300 s)
The evaluator repeats the run for 5 seeds and runs the PPO baseline (baselines/ppo) the same way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import control_envs as E

N_EVAL = 20
PARAMS = {"samples_per_task": 200_000, "tasks": list(E.ENVS), "eval_episodes": N_EVAL, "train_seconds": 2400.0,
          "max_envs": 64}
HELD_OUT_KEYS = [f"init_{e}" for e in E.ENVS]
PIPELINE = [{"stage": "train", "timeout": 2400 + 120, "finish_cap": 300, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]
DEV_SEED = 1707


class BudgetExceeded(RuntimeError):
    pass


class BudgetedEnv(E.VecEnv):
    """A VecEnv whose step() draws from a budget shared by every environment of the same task."""

    def __init__(self, name: str, n: int, seed: int, counter: dict):
        super().__init__(name, n, seed)
        self._counter = counter

    def step(self, action):
        if self._counter["used"] + self.n > self._counter["budget"]:
            raise BudgetExceeded(f"{self.name}: the budget of {self._counter['budget']} environment steps is used up")
        self._counter["used"] += self.n
        return super().step(action)


def env_factory(name: str, budget: int, seed: int, max_envs: int):
    counter = {"used": 0, "budget": int(budget), "made": 0}

    def make_env(n: int = 1) -> BudgetedEnv:
        n = int(n)
        if n < 1 or counter["made"] + n > max_envs:
            raise ValueError(f"at most {max_envs} environments per task")
        counter["made"] += n
        return BudgetedEnv(name, n, seed * 1000 + counter["made"], counter)
    return make_env, counter


def task_spec(name: str, budget: int) -> dict:
    return {"name": name, "obs_dim": E.OBS_DIM[name], "act_dim": E.ACT_DIM, "max_episode_steps": E.MAX_STEPS[name],
            "budget": int(budget), "action_low": -1.0, "action_high": 1.0}


def _overrides(ctx: Context) -> dict:
    import json
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _timed(ctx: Context, _):
    mod = ctx.import_submission("agent.py", required=("train",))
    prm = dict(PARAMS, **_overrides(ctx))
    budget = int(prm["samples_per_task"])
    policies, used = {}, {}
    for i, name in enumerate(prm["tasks"]):
        make_env, counter = env_factory(name, budget, ctx.seed + 7919 * i, int(prm["max_envs"]))
        policies[name] = mod.train(make_env, task_spec(name, budget), ctx.seed + i)
        used[name] = counter["used"]
        if not hasattr(policies[name], "act"):
            raise TypeError(f"{name}: train() must return an object with act(obs)")
    return policies, used


def _finish(ctx: Context, state):
    policies, used = state
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for name, pol in policies.items():
            out[f"returns_{name}"] = E.rollout_returns(pol.act, name, z[f"init_{name}"])
            out[f"samples_{name}"] = np.asarray([used[name]], np.float64)
    return out


STAGES = {"train": Stage(timed=_timed, finish=_finish)}


def held_out_states(seed: int, n: int = N_EVAL) -> dict:
    rng = np.random.default_rng(seed)
    return {f"init_{e}": E.DYNAMICS[e]().init(rng, n) for e in E.ENVS}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev setting for selfcheck.py: 20 initial states per task from DEV_SEED play the held-out role; --quick trains
    with a tenth of the sample budget."""
    out.mkdir(parents=True, exist_ok=True)
    held = held_out_states(DEV_SEED)
    np.savez(out / "held_out.npz", **held)
    pub = out / "public"
    pub.mkdir(exist_ok=True)
    if quick:
        import json
        (pub / "overrides.json").write_text(json.dumps({"samples_per_task": PARAMS["samples_per_task"] // 10}))
    return {"public": pub, "held_out": out / "held_out.npz", "reference": dict(held), "time_scale": 1.0}
