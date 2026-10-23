"""V-DST-004 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --models "$ABENCH_MODELS_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"
                   [--seed 0] [--max-hours 6.0] [--quick]

Consistency distillation (Song et al. 2023) with squared-error distance: on a grid of N = 40 timesteps
0 = t_0 < … < t_N = 999, a training image x is noised to t_{n+1}; one deterministic DDIM step of the frozen teacher
gives the neighbouring point at t_n; the student's consistency function at t_{n+1} is pulled towards the target
network's at t_n. The student and the target network start from the teacher's weights; the target network is an EMA of
the student with decay 0.999. Adam at lr 1e-4 (1,000 warm-up steps), batch 128, random horizontal flips, bf16
autocast for the student. Saves the student's weights to checkpoint/student.pt.
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

import numpy as np
import torch

from model import ALPHAS_CUMPROD, build, consistency

N_GRID = 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default=os.environ.get("ABENCH_MODELS_DIR"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--quick", action="store_true", help="10 steps of batch 8 (smoke test)")
    a = ap.parse_args()
    from diffusers import UNet2DModel
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    with np.load(Path(a.data) / "cifar10_train.npz", allow_pickle=False) as z:
        data = torch.from_numpy(z["x"]).permute(0, 3, 1, 2).contiguous()
    teacher = UNet2DModel.from_pretrained(f"{a.models}/ddpm-cifar10-32").to(a.device).eval().requires_grad_(False)
    student = build().to(a.device)
    student.load_state_dict(teacher.state_dict(), strict=True)
    student.train()
    target = copy.deepcopy(student).eval().requires_grad_(False)
    opt = torch.optim.Adam(student.parameters(), lr=1e-4)
    ac = torch.tensor(ALPHAS_CUMPROD, dtype=torch.float64, device=a.device)
    grid = torch.from_numpy(np.round(np.linspace(0, 999, N_GRID + 1)).astype(np.int64)).to(a.device)
    steps, bs = (10, 8) if a.quick else (a.steps, 128)
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = 1e-4 * min(1.0, (step + 1) / 1000)
        idx = torch.from_numpy(rng.integers(0, len(data), bs))
        x0 = data[idx].to(a.device).float() / 127.5 - 1
        flip = torch.from_numpy(rng.random(bs) < 0.5).to(a.device)
        x0 = torch.where(flip[:, None, None, None], x0.flip(-1), x0)
        n = torch.from_numpy(rng.integers(0, N_GRID, bs)).to(a.device)
        t_hi, t_lo = grid[n + 1], grid[n]
        a_hi, a_lo = ac[t_hi].float()[:, None, None, None], ac[t_lo].float()[:, None, None, None]
        x_hi = a_hi.sqrt() * x0 + (1 - a_hi).sqrt() * torch.randn_like(x0)
        with torch.no_grad():
            eps_t = teacher(x_hi, t_hi).sample                                    # teacher's DDIM step t_hi -> t_lo
            x0_t = (x_hi - (1 - a_hi).sqrt() * eps_t) / a_hi.sqrt()
            x_lo = a_lo.sqrt() * x0_t + (1 - a_lo).sqrt() * eps_t
            tgt = consistency(target, x_lo, t_lo, ac)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda")):
            pred = consistency(student, x_hi, t_hi, ac)
        loss = (pred.float() - tgt).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for pt, ps in zip(target.parameters(), student.parameters()):
                pt.mul_(0.999).add_(ps.detach(), alpha=0.001)
        if time.time() - t0 > a.max_hours * 3600:
            break
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in student.state_dict().items()}, out / "student.pt")


if __name__ == "__main__":
    main()
