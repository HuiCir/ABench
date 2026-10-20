"""A-LIN-002 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --out "$ABENCH_SUBMISSION_DIR/checkpoint" [--seed 0] [--device cuda] [--quick]

Builds the fixed model of the case (interface/lin_model.py) with mixer.Mixer and trains it on MQAR batches drawn by
interface/mqar.py from the training grid only (mqar.TRAIN_GRID), with the recipe the baseline was tuned for:
8,000 steps of batch 128, AdamW (lr 1e-3, betas 0.9/0.95, weight decay 0.1 on matrices), 400 warm-up steps then
cosine decay, gradient clipping at 1.0, bf16 autocast with fp32 weights. Saves the state_dict to checkpoint/model.pt.
(--data is accepted for a uniform command line; this case has no data files.)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if os.environ.get("ABENCH_VIEW_DIR"):
    sys.path.insert(0, str(Path(os.environ["ABENCH_VIEW_DIR"]) / "interface"))
import mqar  # noqa: E402
from lin_model import Model  # noqa: E402

from mixer import Mixer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=os.environ.get("ABENCH_DEVICE", "cuda"))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--quick", action="store_true", help="20 steps of batch 8 (smoke test)")
    a = ap.parse_args()
    steps, bs, warmup = (20, 8, 4) if a.quick else (a.steps, 128, 400)
    lr, wd = 1e-3, 0.1
    torch.manual_seed(a.seed)
    model = Model(Mixer).to(a.device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    rng = np.random.default_rng(a.seed)
    grid = [tuple(g) for g in mqar.TRAIN_GRID]
    for step in range(steps):
        f = (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr * f
        T, K = grid[step % len(grid)]
        x, y = mqar.batch(rng, bs, T, K)
        x, y = torch.from_numpy(x).to(a.device), torch.from_numpy(y).to(a.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda")):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=mqar.IGNORE)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 500 == 0:
            print(f"step {step} loss {float(loss):.4f}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model.pt")


if __name__ == "__main__":
    main()
