"""Reference submission: PPO (the baseline's implementation, imported from src/ppo_core.py) with three changes.

1. Time-limit truncation is bootstrapped with the value of the final observation instead of being treated as a
   terminal state (Pardo et al. 2018); Pendulum has no terminal states at all, so the baseline's targets are biased.
2. 8 parallel environments with rollouts of 256 steps (same 2,048 samples per update), which decorrelates the batch.
3. Each update stops its epochs early once the approximate KL divergence to the rollout policy exceeds 0.02.
"""
from __future__ import annotations

from src.ppo_core import train as _train


def train(make_env, spec, seed):
    return _train(make_env, spec, seed, n_envs=8, n_steps=256, bootstrap_truncation=True, target_kl=0.02)
