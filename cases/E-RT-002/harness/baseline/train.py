"""Train the chunking policy on the LIBERO-Spatial demonstrations by behaviour cloning (L1 on action chunks).

  python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint [--steps 30000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import demos                                    # noqa: E402
from net import ChunkPolicy                     # noqa: E402

HORIZON, BATCH, LR = 16, 64, 3e-4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="checkpoint")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = sorted((Path(a.data) / "libero_spatial").glob("*_demo.hdf5"))
    d = demos.load(files)
    instr = [demos.instruction_of(f) for f in files]
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ChunkPolicy(len(files), HORIZON).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    n = len(d["actions"])
    for step in range(a.steps):
        idx = rng.integers(0, n, BATCH)
        tgt = torch.as_tensor(demos.chunks(d["actions"], d["episode_end"], idx, HORIZON), device=dev)
        shift = rng.integers(-4, 5, (2,))                     # random-shift augmentation of both cameras
        ag = torch.as_tensor(np.roll(d["agent"][idx], shift, (1, 2)), device=dev)
        wr = torch.as_tensor(np.roll(d["wrist"][idx], shift, (1, 2)), device=dev)
        pred = net(ag, wr, torch.as_tensor(d["state"][idx], device=dev), torch.as_tensor(d["task"][idx], device=dev))
        loss = (pred - tgt).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(step, float(loss), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.detach().cpu().contiguous() for k, v in net.state_dict().items()}, str(out / "policy.safetensors"))
    (out / "config.json").write_text(json.dumps({"instructions": instr, "horizon": HORIZON}))


if __name__ == "__main__":
    main()
