"""FN-LOB-002 pipeline: mid-price direction from limit-order-book windows, FI-2010 (evaluator and self-check use this file).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir, device) -> model                             (load time, killed at 120 s)
  finish  read the held-out windows (Q, 100, 40), call predict(model, windows) -> (Q, 3)    (≤ 120 s)
The evaluator cuts each held-out window from the test-day events in its own process, so the submission never sees
events after the end of a window.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

WINDOW, HORIZON_INDEX, N_CLASSES = 100, 4, 3          # horizon k = 10 events is the 5th label column
PARAMS = {"window": WINDOW, "horizon_events": 10, "n_classes": N_CLASSES}
HELD_OUT_KEYS = ["events", "query_idx"]               # raw test events stay with the evaluator; see held_out_inputs
PIPELINE = [{"stage": "predict", "timeout": 120 + 120, "finish_cap": 120, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]
DEV_FRACTION = 0.2                                    # the self-check holds out the last 20% of the training events


def windows(events: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    """Window ending at (and including) event t, for every t in query_idx."""
    offs = np.arange(-WINDOW + 1, 1)
    return events[query_idx[:, None] + offs[None, :]].astype(np.float32)


def held_out_inputs(hidden: dict) -> dict:
    return {"x": windows(hidden["events"], hidden["query_idx"])}


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), ctx.device)


def _finish(ctx: Context, state):
    mod, model = state
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        x = z["x"]
    return {"proba": np.asarray(mod.predict(model, x), dtype=np.float32)}


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: windows from the last 20% of the training events (a later period) play the held-out
    role. For an unbiased dev estimate, train your checkpoint on the first 80% only (the reference train.py does this
    unless --include-dev is given)."""
    with np.load(public / "train.npz", allow_pickle=False) as z:
        ev, lab = z["events"], z["labels"]
    cut = int(len(ev) * (1 - DEV_FRACTION))
    q = np.arange(cut + WINDOW - 1, len(ev))
    rng = np.random.default_rng(0)
    q = np.sort(rng.choice(q, size=min(len(q), 2000 if quick else 15000), replace=False))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", x=windows(ev, q))
    return {"public": public, "held_out": out / "held_out.npz",
            "reference": {"y": lab[q, HORIZON_INDEX].astype(np.int64)}, "time_scale": 1.0}
