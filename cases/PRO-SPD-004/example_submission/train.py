"""PRO-SPD-004 reference submission, training: data-parallel AdamW pretraining of model.GPT on the GPT-2 token shards.

Launched by the evaluator as `torchrun --standalone --nproc_per_node=4 train.py --data DIR --out DIR --seed S`.
Recipe: 0.5M tokens per step (4 GPUs × micro-batch 16 × 1,024 tokens × 8 accumulation steps), AdamW (β = 0.9, 0.95,
weight decay 0.1 on matrices), peak learning rate 1.5e-3 with 250 warm-up steps and linear decay to 0 over STEPS,
bf16 autocast, torch.compile. STEPS is the number of steps planned to reach the target; it has not been calibrated on
the reference hardware (see README).
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SEQ, build_model

STEPS, WARMUP, LR, MICRO, ACCUM = 9000, 250, 1.5e-3, 16, 8


def shard_tokens(path: str) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1, path
    return np.memmap(path, dtype=np.uint16, mode="r", offset=1024, shape=(int(header[2]),))


class Loader:
    """Each rank reads its own contiguous slices of the shards in order (no shuffling across documents)."""

    def __init__(self, files, rank, world):
        self.files, self.rank, self.world = files, rank, world
        self.fi, self.pos = 0, rank * MICRO * SEQ
        self.toks = shard_tokens(files[0])

    def next(self, device):
        need = MICRO * SEQ + 1
        if self.pos + need > len(self.toks):
            self.fi = (self.fi + 1) % len(self.files)
            self.toks, self.pos = shard_tokens(self.files[self.fi]), self.rank * MICRO * SEQ
        buf = torch.from_numpy(np.asarray(self.toks[self.pos:self.pos + need], np.int64))
        self.pos += MICRO * SEQ * self.world
        x, y = buf[:-1].view(MICRO, SEQ), buf[1:].view(MICRO, SEQ)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local}"
    torch.cuda.set_device(device)
    torch.manual_seed(a.seed)
    steps = 50 if a.quick else STEPS
    model = build_model().to(device)
    cmodel = torch.compile(model)
    ddp = DDP(cmodel, device_ids=[local])
    decay = [p for p in model.parameters() if p.ndim >= 2]
    other = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=LR, betas=(0.9, 0.95), fused=True)
    files = sorted(glob.glob(os.path.join(a.data, "fineweb_train_*.bin")))
    loader = Loader(files, rank, world)
    t0 = time.time()
    for step in range(steps):
        lr = LR * min(1.0, (step + 1) / WARMUP) * max(0.0, 1.0 - step / steps)
        for g in opt.param_groups:
            g["lr"] = lr
        for micro in range(ACCUM):
            x, y = loader.next(device)
            ddp.require_backward_grad_sync = micro == ACCUM - 1
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = ddp(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)) / ACCUM
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if rank == 0 and step % 100 == 0:
            print(f"step {step} loss {loss.item() * ACCUM:.4f} lr {lr:.2e} {time.time() - t0:.0f}s", flush=True)
    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, os.path.join(a.out, "model.pt"))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
