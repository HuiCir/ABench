"""E-ACT-001: PushT demonstrations with keypoint observations (lerobot/pusht_keypoints, Apache-2.0; 206 episodes,
25,650 frames) and secret environment reset seeds for gym-pusht (Apache-2.0).

The dataset's parquet file has per-frame columns observation.state (agent position, 2), observation.environment_state
(8 block keypoints, 16), action (target position, 2), episode_index and frame_index; this script checks those shapes.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from prepare.common import Frozen, check

REV = "ace8c161a68bc025c21a5f29f85b86a9a2c5e64b"
SOURCES = [{"name": "data/chunk-000/file-000.parquet",
            "url": f"https://huggingface.co/datasets/lerobot/pusht_keypoints/resolve/{REV}/data/chunk-000/file-000.parquet",
            "license": "Apache-2.0"},
           {"name": "gym-pusht", "url": "https://github.com/huggingface/gym-pusht", "license": "Apache-2.0",
            "note": "installed in the sandbox and judge environments (pip package gym-pusht, version frozen at activation)"}]
N_EPISODES = 50


def _stack(col) -> np.ndarray:
    return np.stack([np.asarray(v, np.float32) for v in col])


def build(raw: Path) -> Frozen:
    import pandas as pd                       # operator environment only
    p = raw / "pusht_keypoints" / SOURCES[0]["name"]
    sums = {SOURCES[0]["name"]: check(p)}
    df = pd.read_parquet(p).sort_values(["episode_index", "frame_index"])
    agent, envs, act = _stack(df["observation.state"]), _stack(df["observation.environment_state"]), _stack(df["action"])
    assert agent.shape[1] == 2 and envs.shape[1] == 16 and act.shape[1] == 2, (agent.shape, envs.shape, act.shape)
    assert df["episode_index"].nunique() == 206 and len(df) == 25650, (df["episode_index"].nunique(), len(df))
    seed = int(os.environ.get("ABENCH_E_ACT_SEED", "0"))
    if not seed:
        raise SystemExit("set ABENCH_E_ACT_SEED to a secret integer (environment reset seeds)")
    seeds = np.random.default_rng(seed).choice(10**8, size=N_EPISODES, replace=False).astype(np.int64)
    return Frozen(public={"demos.npz": {"agent_pos": agent, "environment_state": envs, "action": act,
                                        "episode": df["episode_index"].to_numpy(np.int64),
                                        "frame": df["frame_index"].to_numpy(np.int64)}},
                  hidden={"reset_seeds": seeds}, sources_sha256=sums)


def synthetic(n: int) -> Frozen:
    rng = np.random.default_rng(0)
    ep = np.repeat(np.arange(max(2, n // 100)), 100)[:n]
    return Frozen(public={"demos.npz": {"agent_pos": rng.uniform(0, 512, (n, 2)).astype(np.float32),
                                        "environment_state": rng.uniform(0, 512, (n, 16)).astype(np.float32),
                                        "action": rng.uniform(0, 512, (n, 2)).astype(np.float32),
                                        "episode": ep.astype(np.int64), "frame": (np.arange(n) % 100).astype(np.int64)}},
                  hidden={"reset_seeds": np.arange(3, dtype=np.int64)}, sources_sha256={"synthetic": f"n{n}"})
