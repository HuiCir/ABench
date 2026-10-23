"""Reference training for X-DISTILL-010: continue training SmolLM2-135M on the public FineWeb-Edu shard with a mix of
next-token cross-entropy and the forward KL to SmolLM2-1.7B's distribution (temperature 1, weight 0.5 each),
30M tokens in windows of 1,024, AdamW (lr 3e-4, warm-up 200 steps, cosine to 10%), batch 32 windows, bf16 autocast.
The last 1,048,577 tokens of the shard are left out: the self-check uses them.

  python3 train.py --data "$ABENCH_DATA_DIR" --out checkpoint [--quick] [--tokens N]
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

INTERFACE = Path(os.environ.get("ABENCH_VIEW_DIR", "")) / "interface"
sys.path.insert(0, str(INTERFACE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=30_000_000)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    from safetensors.torch import save_model
    from token_data import read_shard
    from transformers import AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    models = os.environ.get("ABENCH_MODELS_DIR", "models")
    student = AutoModelForCausalLM.from_pretrained(f"{models}/SmolLM2-135M", local_files_only=True, torch_dtype=torch.float32).to(dev)
    teacher = AutoModelForCausalLM.from_pretrained(f"{models}/SmolLM2-1.7B", local_files_only=True, torch_dtype=torch.bfloat16).to(dev).eval()
    toks = read_shard(Path(a.data) / "fwedu_train_000001.bin")[:-1_048_577]
    T, B = 1024, 32
    steps = 2 if a.quick else a.tokens // (T * B)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    lr_at = lambda s: s / 200 if s < 200 else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - 200) / max(1, steps - 200)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = np.random.default_rng(0)
    student.train()
    for s in range(steps):
        starts = rng.integers(0, len(toks) - T - 1, B)
        x = torch.from_numpy(np.stack([toks[i:i + T + 1].astype(np.int64) for i in starts])).to(dev)
        with torch.no_grad():
            t_logits = teacher(input_ids=x[:, :-1]).logits.float()
        with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16):
            s_logits = student(input_ids=x[:, :-1]).logits.float()
        ce = torch.nn.functional.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]), x[:, 1:].reshape(-1))
        kl = torch.nn.functional.kl_div(torch.log_softmax(s_logits, -1), torch.log_softmax(t_logits, -1), log_target=True,
                                        reduction="batchmean") / T
        loss = 0.5 * ce + 0.5 * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if s % 100 == 0:
            print(f"step {s}/{steps} ce {ce.item():.3f} kl {kl.item():.3f}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    save_model(student.cpu(), str(Path(a.out) / "student.safetensors"))      # handles the tied embedding


if __name__ == "__main__":
    main()
