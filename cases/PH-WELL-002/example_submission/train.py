"""Train the shared FNO on the three public systems jointly: one-step prediction of normalised frames (per-system,
per-channel mean and standard deviation of the training split), MSE, equal sampling of the three systems,
AdamW with cosine decay, plus a 2-step unrolled loss in the second half of training (pushforward-style).

  python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint [--steps 40000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from fno import FNO, HISTORY, SYSTEMS   # noqa: E402

BATCH, LR = 16, 1e-3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="checkpoint")
    ap.add_argument("--steps", type=int, default=40000)
    a = ap.parse_args()
    dev = "cuda"
    data, stats = {}, {}
    for s in SYSTEMS:
        with np.load(Path(a.data) / f"{s}.npz") as z:
            tr = z["trajectories"][z["split"] == 0]
            pr = z["params"][z["split"] == 0]
        mu = tr.astype(np.float32).mean((0, 1, 3, 4))
        sd = tr.astype(np.float32).std((0, 1, 3, 4)) + 1e-6
        pm, ps = pr.mean(0), pr.std(0) + 1e-6
        stats[s] = {"mu": mu.tolist(), "sd": sd.tolist(), "pm": pm.tolist(), "ps": ps.tolist()}
        data[s] = (torch.as_tensor(tr), torch.as_tensor((pr - pm) / ps, dtype=torch.float32))
    net = FNO().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    rng = np.random.default_rng(0)
    names = list(SYSTEMS)
    for step in range(a.steps):
        s = names[step % len(names)]
        tr, pr = data[s]
        unroll = 2 if step > a.steps // 2 else 1
        i = rng.integers(0, len(tr), BATCH)
        t = rng.integers(0, tr.shape[1] - HISTORY - unroll + 1, BATCH)
        win = torch.stack([tr[b, k:k + HISTORY + unroll] for b, k in zip(i, t)]).to(dev).float()
        mu = torch.tensor(stats[s]["mu"], device=dev).view(1, 1, -1, 1, 1)
        sd = torch.tensor(stats[s]["sd"], device=dev).view(1, 1, -1, 1, 1)
        win = (win - mu) / sd
        p = pr[i].to(dev)
        hist, loss = win[:, :HISTORY], 0.0
        for u in range(unroll):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = net(s, hist, p).float()
            loss = loss + ((y - win[:, HISTORY + u]) ** 2).mean()
            hist = torch.cat([hist[:, 1:], (y.detach() if u == 0 and unroll > 1 else y)[:, None]], 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(step, s, float(loss), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, out / "fno.pt")
    (out / "stats.json").write_text(json.dumps(stats))


if __name__ == "__main__":
    main()
