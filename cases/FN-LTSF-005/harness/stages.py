"""FN-LTSF-005 pipeline: long-horizon forecasts of five multivariate series at four horizons, compared with a fitted
linear reference (evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir, device) -> state                          (killed at 300 s)
  finish  for every dataset and horizon: predict(state, dataset, history, horizon) on the held-out windows in batches
          of at most 64 -> forecasts (B, horizon, C), standardised scale                   (all within 900 s)
Histories are the 720 standardised steps before each origin (ltsf.py); the reference's errors on the same windows are
part of the held-out reference and never enter the sandbox.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import ltsf as L
from abench_stage import Context, Stage

PARAMS = {"datasets": list(L.DATASETS), "horizons": list(L.HORIZONS), "lookback_max": L.LOOKBACK_MAX,
          "stride": L.STRIDE, "batch": 64, "reference": "ridge, lookback 336, last-value normalisation (ltsf.py)"}
HELD_OUT_KEYS = [f"x_{d}" for d in L.DATASETS]
PIPELINE = [{"stage": "predict", "timeout": 300 + 900, "finish_cap": 900, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), ctx.device)


def _finish(ctx: Context, st):
    mod, state = st
    out = {}
    t0 = time.perf_counter()
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for d in L.DATASETS:
            if f"x_{d}" not in z.files:
                continue
            x = z[f"x_{d}"].astype(np.float32)
            for h in L.HORIZONS:
                parts = []
                for s in range(0, len(x), PARAMS["batch"]):
                    f = np.asarray(mod.predict(state, d, x[s:s + PARAMS["batch"]].copy(), h), np.float32)
                    parts.append(f)
                out[f"f_{d}_{h}"] = np.concatenate(parts) if parts else np.zeros((0, h, x.shape[2]), np.float32)
    out["predict_seconds"] = np.asarray([time.perf_counter() - t0])
    ck = ctx.submission / "checkpoint"
    out["checkpoint_bytes"] = np.asarray([sum(p.stat().st_size for p in ck.rglob("*") if p.is_file())
                                          if ck.is_dir() else 0], np.int64)
    return out


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def reference_errors(series: np.ndarray, train_end: int, val_end: int, orig: np.ndarray, d: str) -> dict:
    """Held-out reference of one dataset: targets and the linear reference's per-window errors."""
    w, lam = L.ridge_fit(series, train_end, val_end)
    x, y = L.windows(series, orig)
    ref = {f"y_{d}": y}
    for h in L.HORIZONS:
        ref[f"ref_mse_{d}_{h}"] = L.window_mse(L.ridge_predict(w, x, h), y[:, :h])
    return {"x": x, "reference": ref, "lambda": lam}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the validation rows play the test role; the reference chooses its penalty on the last quarter of
    the training rows and is refitted on all training rows. Train on the training rows only for an unbiased estimate."""
    out.mkdir(parents=True, exist_ok=True)
    held, ref = {}, {}
    for d in L.DATASETS:
        p = public / f"{d}.npz"
        if not p.is_file():
            continue
        with np.load(p, allow_pickle=False) as z:
            s, tr, va = z["series"], int(z["train_end"]), int(z["val_end"])
        inner = tr - (tr - L.LOOKBACK_MAX) // 4             # validation inside the training rows, for the penalty
        span = va - tr - L.HORIZONS[-1]
        orig = L.origins(tr, va, max(1, min(L.STRIDE[d], span // 12 or 1)))
        if quick:
            orig = orig[:6]
        r = reference_errors(s[:va], inner, tr, orig, d)
        held[f"x_{d}"] = r["x"]
        ref.update(r["reference"])
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}
