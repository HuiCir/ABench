"""Read LIBERO demonstration files (robomimic hdf5) into arrays."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def instruction_of(path: Path) -> str:
    """LIBERO names each file after its instruction: words joined by '_' and the suffix '_demo.hdf5'."""
    return path.name[: -len("_demo.hdf5")].replace("_", " ")


def load(files: list) -> dict:
    import h5py
    agent, wrist, state, act, task, ep_start = [], [], [], [], [], []
    n = 0
    for ti, f in enumerate(files):
        with h5py.File(f, "r") as h:
            for k in sorted(h["data"].keys(), key=lambda s: int(s.split("_")[1])):
                d = h["data"][k]
                o = d["obs"]
                T = len(d["actions"])
                agent.append(np.asarray(o["agentview_rgb"]))
                wrist.append(np.asarray(o["eye_in_hand_rgb"]))
                state.append(np.concatenate([o["ee_pos"], o["ee_ori"], o["gripper_states"]], 1).astype(np.float32))
                act.append(np.asarray(d["actions"], np.float32))
                task.append(np.full(T, ti, np.int64))
                ep_start.append(np.r_[n, np.full(T - 1, -1)])
                n += T
    return {"agent": np.concatenate(agent), "wrist": np.concatenate(wrist), "state": np.concatenate(state),
            "actions": np.concatenate(act), "task": np.concatenate(task),
            "episode_end": _ends(np.concatenate(ep_start), n)}


def _ends(starts: np.ndarray, n: int) -> np.ndarray:
    """For every frame, the index one past the last frame of its episode."""
    s = np.flatnonzero(starts >= 0).tolist() + [n]
    end = np.empty(n, np.int64)
    for a, b in zip(s[:-1], s[1:]):
        end[a:b] = b
    return end


def chunks(actions: np.ndarray, end: np.ndarray, idx: np.ndarray, horizon: int) -> np.ndarray:
    """actions[t : t + horizon] for each t in idx, repeating the episode's last action past its end."""
    j = idx[:, None] + np.arange(horizon)[None]
    j = np.minimum(j, end[idx][:, None] - 1)
    return actions[j]
