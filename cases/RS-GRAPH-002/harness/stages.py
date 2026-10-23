"""RS-GRAPH-002 pipeline: top-20 recommendation over all items on Gowalla (evaluator and self-check use this same file).

The evaluator does not train. One stage, "recommend":
  setup   load train.npz: every user's training items (CSR: indptr, items) (untimed)
  timed   import model.py, load(checkpoint_dir, device) -> model                       (load time, killed at 300 s)
  finish  recommend(model, train, users, k=20) -> (U, 20) item ids for every user, ranked over all items the user has
          not interacted with in train                                                    (≤ 600 s)
The held-out items of each user (the LightGCN test split) never leave the evaluator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

K = 20
PARAMS = {"k": K}
HELD_OUT_KEYS: list[str] = []                    # nothing is revealed: recommendations use the public interactions
PIPELINE = [{"stage": "recommend", "timeout": 300 + 600, "finish_cap": 600, "tail_slack": 10, "threads": 8,
             "held_out": False, "models": False}]
DEV_SEED, DEV_FRACTION = 2020, 0.2


def load_train(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {"indptr": z["indptr"].astype(np.int64), "items": z["items"].astype(np.int64),
                "n_items": int(z["n_items"])}


def _setup(ctx: Context):
    return load_train(ctx.public / "train.npz")


def _timed(ctx: Context, data):
    mod = ctx.import_submission("model.py", required=("load", "recommend"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), ctx.device), data


def _finish(ctx: Context, state):
    mod, model, data = state
    users = np.arange(len(data["indptr"]) - 1, dtype=np.int64)
    view = {"indptr": data["indptr"].copy(), "items": data["items"].copy(), "n_items": data["n_items"]}
    return {"topk": np.asarray(mod.recommend(model, view, users, k=K))}


STAGES = {"recommend": Stage(setup=_setup, timed=_timed, finish=_finish)}


def pad_lists(indptr: np.ndarray, items: np.ndarray) -> np.ndarray:
    """CSR lists -> (U, max_len) int64 array padded with -1."""
    lens = np.diff(indptr)
    out = np.full((len(lens), max(1, int(lens.max()))), -1, np.int64)
    for u in range(len(lens)):
        out[u, :lens[u]] = items[indptr[u]:indptr[u + 1]]
    return out


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: 20% of each user's training items (at least one) play the held-out role, as in the
    80/20 split behind the evaluation. Train your dev checkpoint on the remaining 80% (the reference train.py does this
    with --dev)."""
    d = load_train(public / "train.npz")
    rng = np.random.default_rng(DEV_SEED)
    keep_lists, held_lists = [], []
    ip, it = d["indptr"], d["items"]
    for u in range(len(ip) - 1):
        row = it[ip[u]:ip[u + 1]]
        n_hold = max(1, int(round(DEV_FRACTION * len(row)))) if len(row) > 1 else 0
        perm = rng.permutation(len(row))
        held_lists.append(np.sort(row[perm[:n_hold]]))
        keep_lists.append(np.sort(row[perm[n_hold:]]))
    kp = np.concatenate([[0], np.cumsum([len(r) for r in keep_lists])]).astype(np.int64)
    hp = np.concatenate([[0], np.cumsum([len(r) for r in held_lists])]).astype(np.int64)
    (out / "public").mkdir(parents=True, exist_ok=True)
    np.savez(out / "public" / "train.npz", indptr=kp, items=np.concatenate(keep_lists).astype(np.int32),
             n_items=np.asarray(d["n_items"]))
    return {"public": out / "public", "held_out": None,
            "reference": {"test": pad_lists(hp, np.concatenate(held_lists))}, "time_scale": 1.0}
