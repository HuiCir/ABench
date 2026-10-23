"""X-LORA-004 reference submission, training program (run in the sandbox; the evaluator only loads the checkpoint).

  python3 train.py --data "$ABENCH_DATA_DIR" --models "$ABENCH_MODELS_DIR" --out "$ABENCH_SUBMISSION_DIR/checkpoint"
                   [--seed 0] [--device cuda] [--include-dev] [--quick]

For each task: rebuild roberta-base with LoRA (peft.build), train the LoRA factors and the head with AdamW, linear
warm-up over 6% of the steps then linear decay, bf16 autocast. Saves {task: {name: tensor}} to checkpoint/states.pt.
By default the dev rows (<task>_dev_idx) are left out so that selfcheck.py gives an unbiased dev estimate.
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

from peft import batches, build, trainable

if os.environ.get("ABENCH_VIEW_DIR"):
    sys.path.insert(0, str(Path(os.environ["ABENCH_VIEW_DIR"]) / "interface"))
from textpack import unpack  # noqa: E402

TASKS = {"sst2": ("sentence", None, 2), "cola": ("sentence", None, 2),
         "mnli": ("premise", "hypothesis", 3), "qnli": ("question", "sentence", 2)}
SCHEDULE = {"sst2": (2, 5e-4), "cola": (8, 4e-4), "mnli": (1, 5e-4), "qnli": (2, 5e-4)}   # epochs, peak lr
BATCH = 32


def train_task(d, models_dir, task, seed, device, epochs, lr):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok, enc, head = build(models_dir, device, d["n_classes"])
    params = list(trainable(enc, head).values())
    for p in params:
        p.requires_grad_(True)
    n = len(d["label"])
    steps = max(1, epochs * math.ceil(n / BATCH))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    enc.train(); head.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            s1 = [d["sentence1"][j] for j in idx]
            s2 = [d["sentence2"][j] for j in idx] if d["sentence2"] is not None else None
            b = batches(tok, s1, s2, device)
            y = torch.as_tensor(d["label"][idx], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                logits = head(enc(**b).last_hidden_state)
            F.cross_entropy(logits.float(), y).backward()
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    return {k: v.detach().cpu().contiguous() for k, v in trainable(enc, head).items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default=os.environ.get("ABENCH_MODELS_DIR"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=os.environ.get("ABENCH_DEVICE", "cuda"))
    ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--quick", action="store_true", help="200 examples and one epoch per task (smoke test)")
    a = ap.parse_args()
    with np.load(Path(a.data) / "glue_train.npz", allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}
    states = {}
    for t, (c1, c2, k) in TASKS.items():
        keep = np.ones(len(raw[f"{t}_label"]), bool)
        if not a.include_dev:
            keep[raw[f"{t}_dev_idx"]] = False
        idx = np.flatnonzero(keep)[:200] if a.quick else np.flatnonzero(keep)
        d = {"sentence1": unpack(raw[f"{t}_{c1}_utf8"], raw[f"{t}_{c1}_offsets"], idx),
             "sentence2": unpack(raw[f"{t}_{c2}_utf8"], raw[f"{t}_{c2}_offsets"], idx) if c2 else None,
             "label": raw[f"{t}_label"][idx].astype(np.int64), "n_classes": k}
        epochs, lr = SCHEDULE[t]
        states[t] = train_task(d, a.models, t, a.seed, a.device, 1 if a.quick else epochs, lr)
        torch.cuda.empty_cache()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(states, out / "states.pt")


if __name__ == "__main__":
    main()
