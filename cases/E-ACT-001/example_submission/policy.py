"""E-ACT-001 reference submission, inference side: k-nearest-neighbour behaviour cloning with action chunks (NumPy only).

Observations (agent position + 8 block keypoints, 18 numbers) are standardised. For the current observation the k = 5
nearest demonstration frames are found, and the policy executes the average of their next 8 demonstrated actions
(open loop for 8 steps, then queries again). Action chunking follows the ACT/Diffusion Policy observation that
committing to short action sequences reduces compounding errors; the k-NN regressor is a deliberately simple stand-in
for a generative action head. train.py builds the standardised table and the chunks (checkpoint/knn.npz).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

K, CHUNK = 5, 8


def load(checkpoint_dir, device="cpu"):
    with np.load(Path(checkpoint_dir) / "knn.npz", allow_pickle=False) as z:
        return {k: z[k] for k in ("X", "chunks", "mu", "sd")}


class _Policy:
    def __init__(self, m):
        self.m, self.queue = m, []

    def reset(self):
        self.queue = []

    def act(self, obs):
        if not self.queue:
            x = (np.concatenate([obs["agent_pos"], obs["environment_state"]]) - self.m["mu"]) / self.m["sd"]
            d = ((self.m["X"] - x) ** 2).sum(1)
            nn = np.argpartition(d, K)[:K]
            self.queue = list(self.m["chunks"][nn].mean(0))
        return self.queue.pop(0)


def make_policy(model):
    return _Policy(model)
