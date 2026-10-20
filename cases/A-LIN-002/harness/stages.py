"""A-LIN-002 pipeline: score your Mixer inside the fixed two-layer model (lin_model.py) from your trained checkpoint:
recall on longer sequences and more key-value pairs than in training, causality and training throughput.

The evaluator does not train. One stage, "eval" (imports mixer.py):
  timed   build lin_model.Model(Mixer), load checkpoint/model.pt (a state_dict, loaded with weights_only=True)
  finish  (1) predictions at the query positions of the held-out MQAR sequences (scored by the evaluator)
          (2) causality check: changing tokens after position t must not change outputs up to t
          (3) forward+backward time per token at T = 1,024 and T = 4,096 (B·T fixed), giving throughput and the
              empirical scaling exponent of time in T, for your model and for a freshly initialised copy of the
              gated-linear-attention baseline (baselines/gla) measured in the same process
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage, load_module
import mqar

HERE = Path(__file__).resolve().parent
PARAMS = {"train_grid": mqar.TRAIN_GRID, "throughput_tokens": 32768, "throughput_lengths": [1024, 4096],
          "precision": "bf16 autocast, fp32 weights"}
HELD_OUT_KEYS = ["x", "cell", "t_len"]         # x: (N, Tmax) zero-padded; cell: grid index; t_len: true T
PIPELINE = [{"stage": "eval", "timeout": 120 + 600, "finish_cap": 600, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]


def _timed(ctx: Context, _):
    import torch
    from lin_model import Model
    mod = ctx.import_submission("mixer.py", required=("Mixer",))
    model = Model(mod.Mixer)
    state = torch.load(ctx.submission / "checkpoint" / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return {"model": model.to(ctx.device)}


def _throughput(ctx: Context, model) -> tuple[float, float]:
    """(tokens per second at the longer length, scaling exponent of time in T) for forward + backward."""
    import torch
    model.train()
    times = []
    L = PARAMS["throughput_lengths"]
    for T in L:
        B = max(1, PARAMS["throughput_tokens"] // T)
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
    return PARAMS["throughput_tokens"] / t_long, 1.0 + math.log(t_long / t_short) / math.log(L[1] / L[0])


def _finish(ctx: Context, st):
    import torch
    from lin_model import Model
    model = st["model"]
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
    thr, expo = _throughput(ctx, model)
    del model
    gla = load_module(HERE / "baselines" / "gla" / "mixer.py", "abench_gla_mixer")
    torch.manual_seed(0)
    base_thr, _ = _throughput(ctx, Model(gla.Mixer).to(ctx.device))
    return {"pred": preds, "causal_leak": np.asarray([leak]), "throughput": np.asarray([thr]),
            "scaling_exponent": np.asarray([expo]), "gla_throughput": np.asarray([base_thr])}


STAGES = {"eval": Stage(timed=_timed, finish=_finish)}

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
    """Dev split for selfcheck.py: MQAR sequences on DEV_GRID with a public seed (16 per cell with --quick)."""
    out.mkdir(parents=True, exist_ok=True)
    d = build_split(4242, DEV_GRID, 16 if quick else 64)
    np.savez(out / "held_out.npz", x=d["x"], cell=d["cell"], t_len=d["t_len"])
    return {"public": None, "held_out": out / "held_out.npz", "reference": d, "time_scale": 1.0}
