"""U-LOOP-003 pipeline: length generalisation on parity, binary addition and copy (evaluator and self-check).

The evaluator does not train (training data comes only from algo_tasks.Sampler, lengths 1..20, in your train.py).
One stage, "predict":
  timed   import model.py, load(checkpoint_dir, device) -> model                          (load time, killed at 120 s)
  finish  for each task, call predict(model, task, inputs) on lengths 1..40                (≤ 300 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import algo_tasks as T

PER_LENGTH = 50
PAD = -1
PARAMS = {"tasks": list(T.TASKS), "max_train_len": T.MAX_TRAIN_LEN, "max_test_len": T.MAX_TEST_LEN}
HELD_OUT_KEYS = [f"x_{t}" for t in T.TASKS] + [f"n_{t}" for t in T.TASKS]
PIPELINE = [{"stage": "predict", "timeout": 120 + 300, "finish_cap": 300, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]
OUT_W = T.MAX_TEST_LEN + 1


def pack(seqs: list[np.ndarray], width: int) -> np.ndarray:
    out = np.full((len(seqs), width), PAD, np.int64)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out


def build_split(seed: int, lengths, per_length: int) -> dict:
    rng = np.random.default_rng(seed)
    d = {}
    for t in T.TASKS:
        xs, ys, ns = [], [], []
        for n in lengths:
            x, y = T.make(t, n, per_length, rng)
            xs += x; ys += y; ns += [n] * per_length
        d[f"x_{t}"] = pack(xs, 2 * T.MAX_TEST_LEN)
        d[f"y_{t}"] = pack(ys, OUT_W)
        d[f"n_{t}"] = np.asarray(ns, np.int64)
    return d


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), ctx.device)


def _finish(ctx: Context, state):
    mod, model = state
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for t in T.TASKS:
            x, n = z[f"x_{t}"], z[f"n_{t}"]
            width = np.where(n > 0, n * (2 if t == "addition" else 1), 0)
            inputs = [x[i, :width[i]].copy() for i in range(len(n))]
            preds = mod.predict(model, t, inputs)
            if len(preds) != len(inputs):
                raise SystemExit(f"predict returned {len(preds)} outputs for {len(inputs)} inputs")
            arr = [np.asarray(p, np.int64).reshape(-1) for p in preds]
            out[f"len_{t}"] = np.asarray([len(a) for a in arr], np.int64)
            out[f"pred_{t}"] = pack([a[:OUT_W] for a in arr], OUT_W)
    return out


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def make_dev(public, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: freshly generated examples at lengths 1..40 (same distribution as the evaluation,
    different seed)."""
    out.mkdir(parents=True, exist_ok=True)
    d = build_split(12345, range(1, T.MAX_TEST_LEN + 1), 10 if quick else PER_LENGTH)
    np.savez(out / "held_out.npz", **{k: v for k, v in d.items() if k in HELD_OUT_KEYS})
    return {"public": None, "held_out": out / "held_out.npz",
            "reference": {k: v for k, v in d.items() if k.startswith(("y_", "n_"))}, "time_scale": 1.0}
