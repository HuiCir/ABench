"""V-GEN-001 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--seed 0] [--max-hours 3.0] [--quick]

The DDPM recipe of Ho et al. (2020) for CIFAR-10: ε-prediction with the simple MSE loss on uniformly drawn timesteps,
random horizontal flips, Adam at lr 2e-4 with 5,000 warm-up steps, gradient clipping at 1.0, batch 128, EMA of the
weights with decay 0.9999, here with bf16 autocast. Training stops after --steps steps or --max-hours hours, whichever
comes first; the EMA weights are saved to checkpoint/ema_unet.pt.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import ALPHAS_CUMPROD, build


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--quick", action="store_true", help="20 steps of batch 8 (smoke test)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    with np.load(Path(a.data) / "cifar10_train.npz", allow_pickle=False) as z:
        data = torch.from_numpy(z["x"]).permute(0, 3, 1, 2).contiguous()          # uint8 (N, 3, 32, 32)
    net = build().to(a.device).train()
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    ac = torch.tensor(ALPHAS_CUMPROD, dtype=torch.float32, device=a.device)
    steps, bs = (20, 8) if a.quick else (a.steps, 128)
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = 2e-4 * min(1.0, (step + 1) / 5000)
        idx = torch.from_numpy(rng.integers(0, len(data), bs))
        x0 = data[idx].to(a.device).float() / 127.5 - 1
        flip = torch.from_numpy(rng.random(bs) < 0.5).to(a.device)
        x0 = torch.where(flip[:, None, None, None], x0.flip(-1), x0)
        t = torch.from_numpy(rng.integers(0, 1000, bs)).to(a.device)
        eps = torch.randn_like(x0)
        xt = ac[t].sqrt()[:, None, None, None] * x0 + (1 - ac[t]).sqrt()[:, None, None, None] * eps
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda")):
            pred = net(xt, t).sample
        loss = F.mse_loss(pred.float(), eps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.9999).add_(p.detach(), alpha=1 - 0.9999)
        if time.time() - t0 > a.max_hours * 3600:
            break
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in ema.state_dict().items()}, out / "ema_unet.pt")


if __name__ == "__main__":
    main()
