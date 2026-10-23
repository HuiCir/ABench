"""A-LIN-002 pipeline: train the fixed two-layer model (lin_model.py) with your Mixer on MQAR using the fixed recipe
below, then test recall on longer sequences and more key-value pairs, and measure training throughput.

The recipe is fixed so that differences in recall come from the mixer, not from training compute. One stage, "fit"
(imports mixer.py):
  timed   build the model with the seed, train PARAMS["steps"] steps on MQAR batches from mqar.TRAIN_GRID
  finish  (1) predictions at the query positions of the held-out MQAR sequences (scored by the evaluator)
          (2) causality check: changing tokens after position t must not change outputs up to t
          (3) forward+backward time per token at T = 1,024 and T = 4,096 (B·T fixed), giving throughput and the
              empirical scaling exponent of time in T
The evaluator runs the GLA baseline (baselines/gla) through the same stage on the same seeds.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import mqar

PARAMS = {"steps": 8000, "batch": 128, "lr": 1e-3, "weight_decay": 0.1, "warmup": 400, "train_grid": mqar.TRAIN_GRID,
          "throughput_tokens": 32768, "throughput_lengths": [1024, 4096], "precision": "bf16 autocast, fp32 weights"}
TRAIN_TIMEOUT = 1500.0
HELD_OUT_KEYS = ["x", "cell", "t_len"]         # x: (N, Tmax) zero-padded; cell: grid index; t_len: true T
PIPELINE = [{"stage": "fit", "timeout": TRAIN_TIMEOUT + 120, "finish_cap": 300, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _timed(ctx: Context, _):
    import torch
    import torch.nn.functional as F
    from lin_model import Model
    prm = dict(PARAMS, **_overrides(ctx))
    mod = ctx.import_submission("mixer.py", required=("Mixer",))
    torch.manual_seed(ctx.seed)
    model = Model(mod.Mixer).to(ctx.device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": prm["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=prm["lr"], betas=(0.9, 0.95))
    rng = np.random.default_rng(ctx.seed)
    grid = [tuple(g) for g in prm["train_grid"]]
    for step in range(prm["steps"]):
        f = (step + 1) / prm["warmup"] if step < prm["warmup"] else \
            0.5 * (1 + math.cos(math.pi * (step - prm["warmup"]) / max(1, prm["steps"] - prm["warmup"])))
        for g in opt.param_groups:
            g["lr"] = prm["lr"] * f
        T, K = grid[step % len(grid)]
        x, y = mqar.batch(rng, prm["batch"], T, K)
        x, y = torch.from_numpy(x).to(ctx.device), torch.from_numpy(y).to(ctx.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=mqar.IGNORE)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 500 == 0:
            ctx.meta.setdefault("loss_curve", []).append((step, float(loss.detach())))
    return {"model": model, "prm": prm}


def _finish(ctx: Context, st):
    import torch
    model, prm = st["model"], st["prm"]
    model.eval()
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        X, cell, t_len = z["x"], z["cell"], z["t_len"]
    preds = np.zeros_like(X)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
        for c in np.unique(cell):
            idx = np.flatnonzero(cell == c)
            T = int(t_len[idx[0]])
            for i in range(0, len(idx), 64):
                xb = torch.from_numpy(X[idx[i:i + 64], :T]).to(ctx.device)
                preds[idx[i:i + 64], :T] = model(xb).float().argmax(-1).cpu().numpy()
    # causality: outputs up to position t must not depend on later tokens
    g = torch.Generator(device="cpu").manual_seed(0)
    xa = torch.randint(1, mqar.VOCAB, (2, 512), generator=g)
    xb = xa.clone()
    xb[:, 256:] = torch.randint(1, mqar.VOCAB, (2, 256), generator=g)
    with torch.no_grad():
        ya, yb = model(xa.to(ctx.device)).float(), model(xb.to(ctx.device)).float()
    leak = float((ya[:, :256] - yb[:, :256]).abs().max() / (ya[:, :256].abs().max() + 1e-6))
    # throughput: forward + backward at fixed B·T, wall time with device synchronisation
    model.train()
    times = []
    L = prm["throughput_lengths"]
    for T in L:
        B = max(1, prm["throughput_tokens"] // T)
        x = torch.randint(1, mqar.VOCAB, (B, T), device=ctx.device)
        for rep in range(4):
            ctx.sync()
            t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
                out = model(x)
            out.float().mean().backward()
            model.zero_grad(set_to_none=True)
            ctx.sync()
            if rep > 0:
                times.append((T, time.perf_counter() - t0))
    t_short = np.median([t for T, t in times if T == L[0]])
    t_long = np.median([t for T, t in times if T == L[1]])
    return {"pred": preds, "causal_leak": np.asarray([leak]),
            "throughput": np.asarray([prm["throughput_tokens"] / t_long]),
            "scaling_exponent": np.asarray([1.0 + math.log(t_long / t_short) / math.log(L[1] / L[0])])}


STAGES = {"fit": Stage(timed=_timed, finish=_finish)}

DEV_GRID = [(256, 16), (256, 64), (512, 32), (512, 128), (1024, 64), (1024, 256)]


def build_split(seed: int, grid, per_cell: int) -> dict:
    rng = np.random.default_rng(seed)
    Tmax = max(T for T, _ in grid)
    xs, ys, cells = [], [], []
    for c, (T, K) in enumerate(grid):
        x, y = mqar.batch(rng, per_cell, T, K)
        xs.append(np.pad(x, ((0, 0), (0, Tmax - T))))
        ys.append(np.pad(y, ((0, 0), (0, Tmax - T)), constant_values=mqar.IGNORE))
        cells.append(np.full(per_cell, c))
    cell = np.concatenate(cells).astype(np.int64)
    return {"x": np.concatenate(xs), "y": np.concatenate(ys), "cell": cell,
            "t_len": np.asarray(grid, np.int64)[cell, 0], "grid": np.asarray(grid, np.int64)}


def make_dev(public, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: MQAR sequences on DEV_GRID with a public seed; --quick trains 800 steps."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "public").mkdir(exist_ok=True)
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"steps": 800, "warmup": 80}))
    d = build_split(4242, DEV_GRID, 16 if quick else 64)
    np.savez(out / "held_out.npz", x=d["x"], cell=d["cell"], t_len=d["t_len"])
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": d, "time_scale": 1.0}
