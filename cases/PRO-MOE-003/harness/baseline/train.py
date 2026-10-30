"""Public training baseline for PRO-MOE-003 (DeepSeekMoE-style fine-grained experts with a balance loss).

Launch (registered with the supervisor, from this directory):
  python3 pro_audit.py run --name moe --code . -- torchrun --standalone --nproc_per_node=4 train.py

Data parallel on 4 GPUs (DDP), sequences of 1,024 tokens, 512 sequences per step (128 per GPU in micro-batches of
16), AdamW (lr 1.5e-3, betas 0.9/0.95, weight decay 0.1), 1,000 warm-up steps and cosine decay over 18,000 steps
(about 9.4B tokens), balance-loss weight 0.01, bfloat16 autocast. A checkpoint (model/model.pt, optimiser, data
position, DONE) every 500 steps; loss printed every 10 steps.
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import build_model  # noqa: E402

STEPS, WARM, LR, SEQ, GLOBAL, MICRO, AUX, SAVE_EVERY = 18000, 1000, 1.5e-3, 1024, 512, 16, 0.01, 500


def shards(d: Path):
    out = []
    for p in sorted(d.glob("fineweb_train_*.bin")):
        head = np.fromfile(p, dtype=np.int32, count=256)
        out.append(np.memmap(p, dtype=np.uint16, mode="r", offset=1024, shape=(int(head[2]),)))
    return out


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    data = shards(Path(os.environ["ABENCH_DATA_DIR"]))
    stream = np.concatenate([np.asarray(s) for s in data[rank::world]]) if len(data) >= world else np.asarray(data[0])
    ck_root, resume = Path(os.environ["ABENCH_CKPT_DIR"]), os.environ.get("ABENCH_RESUME_FROM")
    torch.manual_seed(1234)
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    step0, pos = 0, 0
    if resume:
        model.load_state_dict(torch.load(Path(resume) / "model" / "model.pt", map_location=dev, weights_only=True))
        st = torch.load(Path(resume) / f"state_rank{rank}.pt", map_location=dev, weights_only=False)
        opt.load_state_dict(st["opt"])
        step0, pos = st["step"], st["pos"]
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    per_rank = GLOBAL // world
    for step in range(step0, STEPS):
        t0 = time.time()
        lr = LR * min(1.0, (step + 1) / WARM) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / STEPS, 1.0))))
        for g in opt.param_groups:
            g["lr"] = lr
        tot = 0.0
        for m in range(per_rank // MICRO):
            n = MICRO * (SEQ + 1)
            if pos + n > len(stream):
                pos = 0
            chunk = torch.as_tensor(stream[pos:pos + n].astype(np.int64), device=dev).view(MICRO, SEQ + 1)
            pos += n
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = ddp(chunk[:, :-1])
                loss = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), chunk[:, 1:].reshape(-1))
                full = loss + AUX * model.aux_loss()
            (full / (per_rank // MICRO)).backward()
            tot += float(loss) / (per_rank // MICRO)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        t = torch.tensor([tot], device=dev)
        dist.all_reduce(t)
        if rank == 0 and step % 10 == 0:
            print(f"ABENCH step={step} loss={float(t) / world:.5f} lr={lr:.6f} seconds={time.time() - t0:.2f}", flush=True)
        if (step + 1) % SAVE_EVERY == 0 or step + 1 == STEPS:
            d = ck_root / f"step_{step + 1:06d}"
            if rank == 0:
                (d / "model").mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), d / "model" / "model.pt")
            dist.barrier()
            torch.save({"opt": opt.state_dict(), "step": step + 1, "pos": pos}, d / f"state_rank{rank}.pt")
            dist.barrier()
            if rank == 0:
                (d / "DONE").write_text("")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
