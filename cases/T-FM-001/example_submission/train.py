"""T-FM-001 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--seed 0] [--device cuda] [--quick]

For every demonstration frame, the target is the next 16 demonstrated actions (clamped at the end of its episode).
Observations and actions are standardised. The conditional flow-matching loss with the linear (rectified-flow) path:
x_t = (1 − t) z + t a, target velocity a − z, t ~ U(0, 1), z ~ N(0, I). AdamW (lr 3e-4, weight decay 1e-4), batch 256,
30,000 steps with cosine decay. Saves checkpoint/flow.pt.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from policy import ACT_DIM, HORIZON, Velocity


def chunks_of(act: np.ndarray, ep: np.ndarray) -> np.ndarray:
    out = np.zeros((len(act), HORIZON, ACT_DIM), np.float32)
    for i in range(len(act)):
        j = np.minimum(np.arange(i, i + HORIZON), len(act) - 1)
        same = ep[j] == ep[i]
        last = j[same][-1]
        out[i] = act[np.where(same, j, last)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--quick", action="store_true", help="50 steps (smoke test)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    with np.load(Path(a.data) / "demos.npz", allow_pickle=False) as z:
        obs = np.concatenate([z["agent_pos"], z["environment_state"]], 1).astype(np.float32)
        act, ep = z["action"].astype(np.float32), z["episode"]
    ch = chunks_of(act, ep)
    obs_mu, obs_sd = obs.mean(0), obs.std(0) + 1e-6
    act_mu, act_sd = act.mean(0), act.std(0) + 1e-6
    O = torch.from_numpy((obs - obs_mu) / obs_sd).to(a.device)
    A = torch.from_numpy((ch - act_mu) / act_sd).to(a.device)
    net = Velocity().to(a.device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = 50 if a.quick else a.steps
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = 3e-4 * 0.5 * (1 + math.cos(math.pi * step / steps))
        idx = torch.from_numpy(rng.integers(0, len(O), 256)).to(a.device)
        o, x1 = O[idx], A[idx]
        x0 = torch.randn_like(x1)
        t = torch.rand(len(idx), device=a.device)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
        loss = (net(xt, t, o) - (x1 - x0)).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"net": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "obs_mu": torch.from_numpy(obs_mu), "obs_sd": torch.from_numpy(obs_sd),
                "act_mu": torch.from_numpy(act_mu), "act_sd": torch.from_numpy(act_sd)}, out / "flow.pt")


if __name__ == "__main__":
    main()
