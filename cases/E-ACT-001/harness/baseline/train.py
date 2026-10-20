"""E-ACT-001 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--quick]

Standardises the demonstration observations (agent position + block keypoints) and stores, for every demonstration
frame, the next CHUNK demonstrated actions (clamped at the end of its episode). Saves checkpoint/knn.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from policy import CHUNK


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)          # deterministic; kept for a uniform command line
    ap.add_argument("--quick", action="store_true")         # nothing to shorten
    a = ap.parse_args()
    with np.load(Path(a.data) / "demos.npz", allow_pickle=False) as z:
        obs = np.concatenate([z["agent_pos"], z["environment_state"]], 1).astype(np.float64)
        act, ep = z["action"].astype(np.float64), z["episode"]
    mu, sd = obs.mean(0), obs.std(0) + 1e-6
    chunks = np.zeros((len(obs), CHUNK, 2))
    for i in range(len(obs)):
        j = np.minimum(np.arange(i, i + CHUNK), len(obs) - 1)
        same = ep[j] == ep[i]
        chunks[i] = act[np.where(same, j, j[same][-1] if same.any() else i)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "knn.npz", X=(obs - mu) / sd, chunks=chunks, mu=mu, sd=sd)


if __name__ == "__main__":
    main()
