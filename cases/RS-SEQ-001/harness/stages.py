"""RS-SEQ-001 pipeline: next-item recommendation on MovieLens-1M (evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict":
  setup   load train.npz: every user's interaction sequence without its final item (untimed)
  timed   import model.py, load(checkpoint_dir, device) -> model                        (load time, killed at 120 s)
  finish  recommend(model, histories, users, k=10) -> (U, 10) item ids, for every user  (≤ 120 s)
The held-out target of each user is the final item of their full sequence; it never leaves the evaluator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

K = 10
PARAMS = {"k": K}
HELD_OUT_KEYS: list[str] = []                     # nothing is revealed: recommendations use the public histories only
PIPELINE = [{"stage": "predict", "timeout": 120 + 120, "finish_cap": 120, "tail_slack": 5, "threads": 8,
             "held_out": False, "models": False}]


def load_sequences(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ("items", "times", "offsets", "n_items")}


def _setup(ctx: Context):
    return load_sequences(ctx.public / "train.npz")


def _timed(ctx: Context, data):
    mod = ctx.import_submission("model.py", required=("load", "recommend"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), ctx.device), data


def _finish(ctx: Context, state):
    mod, model, data = state
    users = np.arange(len(data["offsets"]) - 1, dtype=np.int64)
    return {"topk": np.asarray(mod.recommend(model, {k: v.copy() for k, v in data.items()}, users, k=K))}


STAGES = {"predict": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: hold out the final item of every public sequence (the evaluator holds out the next one).

    For an unbiased dev estimate, train your checkpoint without each user's final public item (the reference train.py
    does this unless --include-dev is given)."""
    d = load_sequences(public / "train.npz")
    off = d["offsets"]
    n_users = len(off) - 1
    keep = np.ones(len(d["items"]), bool)
    keep[off[1:] - 1] = False
    target = d["items"][off[1:] - 1]
    lengths = np.diff(off) - 1
    new_off = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    (out / "public").mkdir(parents=True, exist_ok=True)
    np.savez(out / "public" / "train.npz", items=d["items"][keep], times=d["times"][keep], offsets=new_off,
             n_items=d["n_items"])
    _ = n_users
    return {"public": out / "public", "held_out": None, "reference": {"target": target}, "time_scale": 1.0}
