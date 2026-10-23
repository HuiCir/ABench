"""PH-WELL-002 pipeline: your trained surrogate predicts held-out trajectories of three physical systems, rolled out
by the evaluator (evaluator and self-check use this same file).

One stage, "rollout" (only checkpoint/ is available):
  timed   import model.py; model = load(checkpoint_dir, device)
  finish  for each system and each held-out set (in-distribution, unseen parameters): wellsurr.rollout of
          model.predict from 4 given frames over 10 frames; predictions as float16; the numbers stored in checkpoint/
                                                                                                     (≤ 1,800 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import wellsurr as W
from abench_stage import Context, Stage

SETS = ("id", "unseen")
MAX_NUMBERS = 100_000_000
PARAMS = {"history": W.HISTORY, "rollout": W.ROLLOUT, "resolution": W.RES, "systems": list(W.SYSTEMS),
          "max_checkpoint_numbers": MAX_NUMBERS}
HELD_OUT_KEYS = [f"{s}_{t}_{k}" for s in W.SYSTEMS for t in SETS for k in ("x", "params")]
PIPELINE = [{"stage": "rollout", "timeout": 600 + 1800, "finish_cap": 1800, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load",))
    model = mod.load(str(ctx.submission / "checkpoint"), ctx.device)
    if not callable(getattr(model, "predict", None)):
        raise SystemExit("load() must return an object with predict(system, history, params)")
    return model


def checkpoint_numbers(ck: Path) -> int:
    n = 0
    for p in sorted(Path(ck).rglob("*")):
        if p.suffix == ".safetensors":
            import json
            with open(p, "rb") as f:
                head = json.loads(f.read(int.from_bytes(f.read(8), "little")))
            n += sum(int(np.prod(v["shape"])) for k, v in head.items() if k != "__metadata__")
        elif p.suffix in (".pt", ".pth", ".bin"):
            import torch
            stack = [torch.load(p, map_location="cpu", weights_only=True)]
            while stack:
                o = stack.pop()
                if torch.is_tensor(o):
                    n += o.numel()
                elif isinstance(o, dict):
                    stack += list(o.values())
                elif isinstance(o, (list, tuple)):
                    stack += list(o)
        elif p.suffix in (".npz", ".npy"):
            z = np.load(p, allow_pickle=False)
            n += sum(int(z[k].size) for k in z.files) if p.suffix == ".npz" else int(z.size)
    return n


def _finish(ctx: Context, model):
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for s in W.SYSTEMS:
            for t in SETS:
                if f"{s}_{t}_x" in z.files:
                    out[f"pred_{s}_{t}"] = W.rollout(model.predict, s, z[f"{s}_{t}_x"], z[f"{s}_{t}_params"])
    out["checkpoint_numbers"] = np.asarray([checkpoint_numbers(ctx.submission / "checkpoint")], np.int64)
    return out


STAGES = {"rollout": Stage(timed=_timed, finish=_finish)}


def held_out_from(trajs: np.ndarray, params: np.ndarray, starts: list) -> tuple:
    """Windows of several trajectories: (x, params, y) with starts[i] the start frames of trajectory i."""
    xs, ps, ys = [], [], []
    for tr, pr, st in zip(trajs, params, starts):
        x, y = W.windows(tr, st)
        xs.append(x)
        ys.append(y)
        ps.append(np.repeat(np.asarray(pr, np.float32)[None], len(st), 0))
    return np.concatenate(xs).astype(np.float16), np.concatenate(ps), np.concatenate(ys).astype(np.float16)


DEV_UNSEEN = {"turbulent_radiative": 1, "active_matter": 3, "gray_scott": 2}     # index among public parameter sets


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the public validation trajectories; "unseen" is played by the validation trajectories of one public
    parameter set per system (DEV_UNSEEN), which is optimistic unless you leave that set out of training. Two windows
    per trajectory (starts 0 and the middle); --quick: at most 8 windows per set."""
    held, ref = {}, {}
    for s in W.SYSTEMS:
        with np.load(public / f"{s}.npz", allow_pickle=False) as z:
            traj, prm, split = z["trajectories"], z["params"], z["split"]
        uniq = np.unique(prm, axis=0)
        un = np.all(prm == uniq[DEV_UNSEEN[s] % len(uniq)], axis=1)
        for t, m in (("id", (split == 1) & ~un), ("unseen", (split == 1) & un)):
            idx = np.flatnonzero(m)
            T = traj.shape[1]
            starts = [[0, (T - W.HISTORY - W.ROLLOUT) // 2]] * len(idx)
            x, p, y = held_out_from(traj[idx], prm[idx], starts)
            if quick:
                x, p, y = x[:8], p[:8], y[:8]
            held[f"{s}_{t}_x"], held[f"{s}_{t}_params"], ref[f"{s}_{t}_y"] = x, p, y
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}
