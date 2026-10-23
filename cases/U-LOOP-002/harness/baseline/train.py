"""Reference submission (training): the recurrent-depth model of src/looped.py on the three tasks jointly.

Each optimisation step draws, per task, a batch of 128 problems with difficulties uniform in 1..MAX_D and a number of
iterations r uniform in 1..24; the model is unrolled r times without gradient for the first r - 8 iterations
(truncated backpropagation, as in Geiping et al. 2025) and the cross-entropy is taken at the last iteration and at one
random earlier iteration, so that intermediate answers are meaningful for early exit.

  python3 train.py --out checkpoint             # about 2 GPU-hours on the reference GPU
  python3 train.py --out checkpoint --quick     # interface check (a few steps, on CPU)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(os.environ.get("ABENCH_VIEW_DIR", HERE)) / "interface"))
import reason_tasks as T  # noqa: E402
from src.looped import LoopedModel  # noqa: E402

STEPS, BATCH, MAX_R, BPTT, LR = 30000, 128, 24, 8, 1e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None, help="unused (this case has no data files)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() and not a.quick else "cpu"
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    sampler = T.Sampler(a.seed)
    net = LoopedModel().to(dev)
    steps, batch, max_r = (3, 8, 4) if a.quick else (STEPS, BATCH, MAX_R)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.05, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
    for it in range(steps):
        loss = 0.0
        for t in T.TASKS:
            ds = rng.integers(1, T.MAX_D[t] + 1, batch)
            xs, ys = [], []
            for d in np.unique(ds):
                x, y = sampler.sample(t, int(d), int((ds == d).sum()))
                xs.append(x); ys.append(y)
            x = torch.as_tensor(np.concatenate(xs), device=dev)
            y = torch.as_tensor(np.concatenate(ys), device=dev)
            r = int(rng.integers(1, max_r + 1))
            mid = int(rng.integers(1, r + 1))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                e = net.embed(t, x)
                h = e
                for k in range(1, r + 1):
                    if k <= r - BPTT:
                        with torch.no_grad():
                            h = net.core(h, e)
                    else:
                        h = net.core(h, e)
                    if k == mid and k < r and k > r - BPTT:
                        loss = loss + 0.5 * F.cross_entropy(net.head(t, h).float(), y)
                loss = loss + F.cross_entropy(net.head(t, h).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 1000 == 0:
            print(f"step {it} loss {float(loss):.4f}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, out / "model.pt")
    (out / "config.json").write_text(json.dumps({"model": {"d": 256, "heads": 4, "mlp": 1024}, "tau": 0.95}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
